"""Yaw-scheduled trajectory-linearized MPC through the common inner cascade."""

import argparse
import csv
import pickle
import time
from pathlib import Path

import casadi as ca
import numpy as np
import osqp
import scipy.sparse as sp

from edmdc_mpc import (
    STATE_DIM, hover_yaw_command_reference_array,
    outer_command_reference_array, reference_state_array,
    reference_yaw_arrays, wrap_angle_pi,
)
from outer_command_mpc import (
    Q_PHYSICAL, R_PHYSICAL, RD_PHYSICAL, plant_attitude_step,
    trajectory_metrics,
)
from outer_nmpc import OuterCommandNMPC
from Simulation import quad_sim


def reference_linearizations(sim, state_reference, command_reference, yaw_rate):
    """Use automatic differentiation to linearize the exact nominal step map."""
    symbolic = OuterCommandNMPC(
        sim, horizon_seconds=sim.dt, control_horizon_seconds=sim.dt,
        max_iterations=1,
    )
    x = ca.MX.sym("x", STATE_DIM)
    u = ca.MX.sym("u", 4)
    r = ca.MX.sym("r")
    next_state = symbolic._rk4(x, u, r)
    function = ca.Function(
        "outer_step_linearization", [x, u, r],
        [next_state, ca.jacobian(next_state, x), ca.jacobian(next_state, u)],
    ).map(len(state_reference))
    result = function(
        np.asarray(state_reference).T,
        np.asarray(command_reference).T,
        np.asarray(yaw_rate).reshape(1, -1),
    )
    predicted = np.asarray(result[0]).T
    A_raw = np.asarray(result[1])
    B_raw = np.asarray(result[2])
    A = np.stack([
        A_raw[:, k*STATE_DIM:(k+1)*STATE_DIM]
        for k in range(len(state_reference))
    ])
    B = np.stack([
        B_raw[:, k*4:(k+1)*4]
        for k in range(len(state_reference))
    ])
    next_reference = np.vstack((state_reference[1:], state_reference[-1:]))
    defect = predicted - next_reference
    defect[:, 8] = wrap_angle_pi(defect[:, 8])
    return A, B, defect


def hover_lti_linearizations(sim, state_reference, command_reference):
    """Paper-style single fixed linearization at level hover and zero yaw."""
    equilibrium_state = np.zeros((1, STATE_DIM))
    equilibrium_command = np.array([[sim.quad.m*sim.quad.g, 0.0, 0.0, 0.0]])
    A0, B0, _ = reference_linearizations(
        sim, equilibrium_state, equilibrium_command, np.zeros(1)
    )
    A = np.repeat(A0, len(state_reference), axis=0)
    B = np.repeat(B0, len(state_reference), axis=0)
    predicted = (
        equilibrium_state[0]
        + np.einsum("ij,nj->ni", A0[0], state_reference-equilibrium_state[0])
        + np.einsum("ij,nj->ni", B0[0], command_reference-equilibrium_command[0])
    )
    next_reference = np.vstack((state_reference[1:], state_reference[-1:]))
    defect = predicted-next_reference
    defect[:, 8] = wrap_angle_pi(defect[:, 8])
    return A, B, defect


def yaw_scheduled_hover_linearizations(sim, state_reference, command_reference):
    """Level-hover linearization scheduled only by the reference yaw.

    This is a physics-only linear baseline.  It does not linearize along the
    trajectory's velocity, acceleration, attitude, or yaw rate; scheduling the
    hover input map by yaw only prevents a zero-yaw body/world-axis mismatch.
    """
    equilibrium_state = np.zeros_like(state_reference)
    equilibrium_state[:, :3] = state_reference[:, :3]
    equilibrium_state[:, 8] = state_reference[:, 8]
    equilibrium_command = np.zeros_like(command_reference)
    equilibrium_command[:, 0] = sim.quad.m * sim.quad.g
    equilibrium_command[:, 3] = state_reference[:, 8]
    A, B, _ = reference_linearizations(
        sim, equilibrium_state, equilibrium_command,
        np.zeros(len(state_reference)),
    )
    predicted = (
        equilibrium_state
        + np.einsum("nij,nj->ni", A, state_reference-equilibrium_state)
        + np.einsum("nij,nj->ni", B, command_reference-equilibrium_command)
    )
    next_reference = np.vstack((state_reference[1:], state_reference[-1:]))
    defect = predicted-next_reference
    defect[:, 8] = wrap_angle_pi(defect[:, 8])
    return A, B, defect


class LTVOuterCommandMPC:
    def __init__(self, sim, horizon_seconds=0.5,
                 control_horizon_seconds=0.1, r_scale=10.0,
                 thrust_trust_n=1.0, attitude_trust_rad=0.05,
                 yaw_trust_rad=0.08, q_position_scale=1.0,
                 q_velocity_scale=1.0, q_yaw_scale=1.0,
                 terminal_scale=3.0, yaw_feedforward_only=False,
                 q_attitude_scale=1.0, q_rate_scale=1.0,
                 attitude_error_max_rad=None):
        self.N = int(round(horizon_seconds / sim.dt))
        self.NC = int(round(control_horizon_seconds / sim.dt))
        self.nu = 4; self.nvar = self.NC*4
        if not 1 <= self.NC <= self.N:
            raise ValueError("Require 1 <= control horizon <= prediction horizon")
        q_physical = Q_PHYSICAL.copy()
        q_physical[:3] *= float(q_position_scale)
        q_physical[3:6] *= float(q_velocity_scale)
        q_physical[6:8] *= float(q_attitude_scale)
        q_physical[9:12] *= float(q_rate_scale)
        q_physical[[8, 11]] *= float(q_yaw_scale)
        if yaw_feedforward_only:
            q_physical[[8, 11]] = 0.0
        self.Qbar = np.kron(np.eye(self.N), np.diag(q_physical))
        self.Qbar[-STATE_DIM:, -STATE_DIM:] *= float(terminal_scale)
        self.Rbar = np.kron(np.eye(self.NC), np.diag(r_scale*R_PHYSICAL))
        self.D = np.zeros(((self.NC-1)*4, self.nvar))
        for k in range(self.NC-1):
            self.D[k*4:(k+1)*4, k*4:(k+1)*4] = -np.eye(4)
            self.D[k*4:(k+1)*4, (k+1)*4:(k+2)*4] = np.eye(4)
        self.Rdbar = np.kron(np.eye(max(self.NC-1, 0)), np.diag(r_scale*RD_PHYSICAL))
        c = sim.controller_PX4
        self.command_lower = np.array([c.thrust_min, -c.tilt_max, -c.tilt_max, -np.inf])
        self.command_upper = np.array([c.thrust_max, c.tilt_max, c.tilt_max, np.inf])
        self.trust = np.array([thrust_trust_n, attitude_trust_rad, attitude_trust_rad, yaw_trust_rad])
        if yaw_feedforward_only:
            self.trust[3] = 0.0
        self.attitude_error_max_rad = (
            None if attitude_error_max_rad is None
            else float(attitude_error_max_rad)
        )
        self.rows, self.cols = np.triu_indices(self.nvar)
        initial = sp.csc_matrix(
            (np.ones(len(self.rows)), (self.rows, self.cols)),
            shape=(self.nvar, self.nvar),
        )
        self.problem = osqp.OSQP()
        self.problem.setup(
            P=initial, q=np.zeros(self.nvar), A=sp.eye(self.nvar, format="csc"),
            l=-np.ones(self.nvar), u=np.ones(self.nvar),
            warm_start=True, verbose=False, polish=False,
        )
        self.previous = np.zeros(self.nvar)
        self.last_status = "not run"

    def compute(self, state, state_reference, command_reference, A, B, defect):
        sensitivity = np.zeros((STATE_DIM, self.nvar))
        free = np.asarray(state) - state_reference[0]
        free[8] = wrap_angle_pi(free[8])
        free_horizon = np.zeros((self.N, STATE_DIM))
        Su = np.zeros((self.N*STATE_DIM, self.nvar))
        for j in range(self.N):
            move = min(j, self.NC-1)
            injection = np.zeros((4, self.nvar))
            injection[:, move*4:(move+1)*4] = np.eye(4)
            free = A[j] @ free + defect[j]
            sensitivity = A[j] @ sensitivity + B[j] @ injection
            free_horizon[j] = free
            Su[j*STATE_DIM:(j+1)*STATE_DIM] = sensitivity
        P = Su.T @ self.Qbar @ Su + self.Rbar
        q = Su.T @ (self.Qbar @ free_horizon.reshape(-1))
        nominal = command_reference[:self.NC].copy()
        nominal[:, 3] = np.unwrap(np.r_[state[8], nominal[:, 3]])[1:]
        if self.NC > 1:
            P += self.D.T @ self.Rdbar @ self.D
            q += self.D.T @ (self.Rdbar @ (self.D @ nominal.reshape(-1)))
        P = 0.5*(P+P.T) + 1e-9*np.eye(self.nvar)
        upper_matrix = sp.csc_matrix(
            (P[self.rows, self.cols], (self.rows, self.cols)),
            shape=(self.nvar, self.nvar),
        )
        lower = np.maximum(
            np.tile(self.command_lower, self.NC)-nominal.reshape(-1),
            -np.tile(self.trust, self.NC),
        )
        upper = np.minimum(
            np.tile(self.command_upper, self.NC)-nominal.reshape(-1),
            np.tile(self.trust, self.NC),
        )
        if self.attitude_error_max_rad is not None:
            # The applied outer command drives the nonlinear attitude/rate
            # cascade. Limit its instantaneous attitude error so the inner
            # torque request remains inside the motor-feasible envelope.
            for raw_index, state_index in ((1, 6), (2, 7)):
                lower[raw_index] = max(
                    lower[raw_index],
                    state[state_index] - self.attitude_error_max_rad
                    - nominal[0, raw_index],
                )
                upper[raw_index] = min(
                    upper[raw_index],
                    state[state_index] + self.attitude_error_max_rad
                    - nominal[0, raw_index],
                )
        self.problem.update(Px=upper_matrix.data, q=q, l=lower, u=upper)
        self.problem.warm_start(x=self.previous)
        result = self.problem.solve(); self.last_status = str(result.info.status)
        if self.last_status in ("solved", "solved inaccurate"):
            self.previous = np.asarray(result.x).copy()
        return nominal[0] + self.previous[:4]


def run(reference, steps, horizon_seconds=0.5,
        control_horizon_seconds=0.1, r_scale=10.0,
        linearization_mode="trajectory_ltv", thrust_trust_n=1.0,
        attitude_trust_rad=0.05, yaw_trust_rad=0.08, sim=None,
        model_sim=None, q_position_scale=1.0, q_velocity_scale=1.0,
        q_yaw_scale=1.0, terminal_scale=3.0,
        yaw_feedforward_only=False, reference_mode="inverse_dynamics",
        q_attitude_scale=1.0, q_rate_scale=1.0,
        use_reference_defect=True, preview_reference=True, initial_yaw=None,
        attitude_error_max_rad=None):
    sim = quad_sim() if sim is None else sim
    model_sim = quad_sim() if model_sim is None else model_sim
    ref = reference[:steps]
    controller = LTVOuterCommandMPC(
        sim, horizon_seconds, control_horizon_seconds, r_scale,
        thrust_trust_n, attitude_trust_rad, yaw_trust_rad,
        q_position_scale, q_velocity_scale, q_yaw_scale, terminal_scale,
        yaw_feedforward_only, q_attitude_scale, q_rate_scale,
        attitude_error_max_rad,
    )
    if reference_mode == "inverse_dynamics":
        state_ref = reference_state_array(ref, sim.dt, model_sim.quad)
        command_ref = outer_command_reference_array(ref, model_sim.quad)
    elif reference_mode == "kinematic_hover_yaw":
        state_ref = reference_state_array(ref, sim.dt, quad=None)
        command_ref = hover_yaw_command_reference_array(
            ref, model_sim.quad.m * model_sim.quad.g
        )
    else:
        raise ValueError(f"Unknown reference mode {reference_mode!r}")
    _, yaw_rate = reference_yaw_arrays(ref, dt=sim.dt)
    if linearization_mode == "trajectory_ltv":
        A, B, defect = reference_linearizations(
            model_sim, state_ref, command_ref, yaw_rate
        )
    elif linearization_mode == "hover_lti":
        A, B, defect = hover_lti_linearizations(model_sim, state_ref, command_ref)
    elif linearization_mode == "yaw_scheduled_hover":
        A, B, defect = yaw_scheduled_hover_linearizations(
            model_sim, state_ref, command_ref
        )
    else:
        raise ValueError(f"Unknown linearization mode {linearization_mode!r}")
    if not use_reference_defect:
        defect = np.zeros_like(defect)
    state = np.zeros(STATE_DIM)
    state[8] = float(ref[0].get("yaw", 0.0) if initial_yaw is None else initial_yaw)
    states = np.zeros((steps, STATE_DIM)); states[0] = state
    wrench = np.zeros((steps, 4)); commands = np.zeros((steps, 4))
    times, statuses = [], []
    altered = 0; max_error = 0.0
    sim.controller_PX4.fct_reset()
    for k in range(steps-1):
        idx = (
            np.minimum(np.arange(k, k+controller.N), steps-1)
            if preview_reference else np.full(controller.N, k, dtype=int)
        )
        started = time.perf_counter()
        command = controller.compute(
            state, state_ref[idx], command_ref[idx],
            A[idx], B[idx], defect[idx],
        )
        times.append(time.perf_counter()-started); statuses.append(controller.last_status)
        state, applied = plant_attitude_step(sim, sim.controller_PX4, state, command, yaw_rate[k])
        error = float(np.max(np.abs(
            sim.controller_PX4.last_requested_wrench
            - sim.controller_PX4.last_allocated_wrench
        )))
        altered += int(error > 1e-8); max_error=max(max_error,error)
        states[k+1]=state; wrench[k]=applied; commands[k]=command
    if steps>1: wrench[-1]=wrench[-2]; commands[-1]=commands[-2]
    return states, wrench, commands, {
        "solve_times": times,
        "mean_solve_ms": float(1e3*np.mean(times)),
        "p95_solve_ms": float(1e3*np.percentile(times,95)),
        "p99_solve_ms": float(1e3*np.percentile(times,99)),
        "max_solve_ms": float(1e3*np.max(times)),
        "failed_solves": sum(s not in ("solved","solved inaccurate") for s in statuses),
        "allocator_altered_steps": altered,
        "allocator_max_abs_wrench_error": max_error,
        "yaw_feedforward_only": bool(yaw_feedforward_only),
        "reference_mode": reference_mode,
        "use_reference_defect": bool(use_reference_defect),
        "preview_reference": bool(preview_reference),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--indices",default="39,59,129,155,210")
    parser.add_argument("--steps",type=int,default=0)
    parser.add_argument("--horizon-seconds",type=float,default=0.5)
    parser.add_argument("--control-horizon-seconds",type=float,default=0.1)
    parser.add_argument("--r-scale",type=float,default=10.0)
    parser.add_argument("--thrust-trust-n",type=float,default=1.0)
    parser.add_argument("--attitude-trust-rad",type=float,default=0.05)
    parser.add_argument("--yaw-trust-rad",type=float,default=0.08)
    parser.add_argument(
        "--linearization-mode",
        choices=("hover_lti", "yaw_scheduled_hover", "trajectory_ltv"),
        default="trajectory_ltv",
    )
    parser.add_argument(
        "--reference-mode",
        choices=("inverse_dynamics", "kinematic_hover_yaw"),
        default="inverse_dynamics",
        help=(
            "Nominal command/reference construction. Use kinematic_hover_yaw "
            "for a simple baseline without acceleration-derived thrust or attitude."
        ),
    )
    parser.add_argument(
        "--no-reference-defect", action="store_true",
        help="Disable affine reference-defect forcing in the prediction model.",
    )
    parser.add_argument(
        "--current-reference-only", action="store_true",
        help="Disable future reference preview (normally MPC receives the known path).",
    )
    parser.add_argument(
        "--yaw-feedforward-only", action="store_true",
        help="Remove yaw/yaw-rate error from the QP while retaining inner-loop yaw feedforward.",
    )
    parser.add_argument("--q-position-scale", type=float, default=1.0)
    parser.add_argument("--q-velocity-scale", type=float, default=1.0)
    parser.add_argument("--q-yaw-scale", type=float, default=1.0)
    parser.add_argument("--terminal-scale", type=float, default=3.0)
    args=parser.parse_args()
    with args.data.open("rb") as stream:data=pickle.load(stream)
    args.output_dir.mkdir(parents=True,exist_ok=True); rows=[]
    for index in [int(v) for v in args.indices.split(",") if v.strip()]:
        reference=data["ref_traj_list"][index]
        steps=len(reference) if args.steps==0 else min(args.steps,len(reference))
        states,wrench,commands,diagnostics=run(
            reference,steps,args.horizon_seconds,args.control_horizon_seconds,
            args.r_scale,args.linearization_mode,args.thrust_trust_n,
            args.attitude_trust_rad,args.yaw_trust_rad,
            q_position_scale=args.q_position_scale,
            q_velocity_scale=args.q_velocity_scale,
            q_yaw_scale=args.q_yaw_scale,
            terminal_scale=args.terminal_scale,
            yaw_feedforward_only=args.yaw_feedforward_only,
            reference_mode=args.reference_mode,
            use_reference_defect=not args.no_reference_defect,
            preview_reference=not args.current_reference_only,
        )
        family=data["family_labels"][index]
        row={"index":index,"family":family,"controller_period_s":0.01,
             "controller_rate_hz":100.0,"prediction_horizon_s":args.horizon_seconds,
             "control_horizon_s":args.control_horizon_seconds,"r_scale":args.r_scale,
             "thrust_trust_n":args.thrust_trust_n,
             "attitude_trust_rad":args.attitude_trust_rad,
             "yaw_trust_rad":args.yaw_trust_rad,
             "linearization_mode":args.linearization_mode,
             "reference_mode":args.reference_mode,
             "use_reference_defect":not args.no_reference_defect,
             "preview_reference":not args.current_reference_only,
             "yaw_feedforward_only":args.yaw_feedforward_only,
             "q_position_scale":args.q_position_scale,
             "q_velocity_scale":args.q_velocity_scale,
             "q_yaw_scale":args.q_yaw_scale,
             "terminal_scale":args.terminal_scale,
             **trajectory_metrics(states,reference[:steps],float(data["sim_dt"])),**diagnostics}
        rows.append(row)
        np.savez_compressed(args.output_dir/f"run_{index}_{family}.npz",states=states,
                            wrench=wrench,outer_command=commands,controller_period_s=0.01,
                            controller_rate_hz=100.0)
        print(row)
    with (args.output_dir/"metrics.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=="__main__":main()
