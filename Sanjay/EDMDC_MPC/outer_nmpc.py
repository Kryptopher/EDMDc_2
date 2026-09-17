"""Nonlinear MPC through the same outer attitude-command/inner-loop boundary."""

import argparse
import csv
import pickle
import time
from pathlib import Path

import casadi as ca
import numpy as np

from edmdc_mpc import (
    STATE_DIM, build_ref_horizon, outer_command_reference_array,
    reference_state_array, reference_yaw_arrays,
)
from outer_command_mpc import (
    Q_PHYSICAL, R_PHYSICAL, RD_PHYSICAL, plant_attitude_step,
    trajectory_metrics,
)
from PID_Mixer import pid_mixer
from Simulation import quad_sim


def clip(value, lower, upper):
    return ca.fmin(ca.fmax(value, lower), upper)


def rotation(phi, theta, psi):
    cp, sp = ca.cos(phi), ca.sin(phi)
    ct, st = ca.cos(theta), ca.sin(theta)
    cy, sy = ca.cos(psi), ca.sin(psi)
    return ca.vertcat(
        ca.horzcat(cy*ct, cy*st*sp-sy*cp, cy*st*cp+sy*sp),
        ca.horzcat(sy*ct, sy*st*sp+cy*cp, sy*st*cp-cy*sp),
        ca.horzcat(-st, ct*sp, ct*cp),
    )


class OuterCommandNMPC:
    def __init__(self, sim, horizon_seconds=0.5,
                 control_horizon_seconds=0.1, max_iterations=15,
                 r_scale=10.0):
        self.sim = sim
        self.dt = float(sim.dt)
        self.N = int(round(horizon_seconds / self.dt))
        self.NC = int(round(control_horizon_seconds / self.dt))
        if not 1 <= self.NC <= self.N:
            raise ValueError("Require 1 <= control horizon <= prediction horizon")
        self.nu = 4
        self.nvar = self.NC * self.nu
        controller = sim.controller_PX4
        self.command_lower = np.array([
            controller.thrust_min, -controller.tilt_max,
            -controller.tilt_max, -np.inf,
        ])
        self.command_upper = np.array([
            controller.thrust_max, controller.tilt_max,
            controller.tilt_max, np.inf,
        ])
        self.delta_limit = np.array([1.0, 0.05, 0.05, 0.08])

        delta = ca.MX.sym("delta", self.nvar)
        parameter_size = STATE_DIM + self.N*STATE_DIM + self.NC*4 + self.N
        parameter = ca.MX.sym("parameter", parameter_size)
        cursor = 0
        state = parameter[cursor:cursor+STATE_DIM]; cursor += STATE_DIM
        reference = ca.reshape(
            parameter[cursor:cursor+self.N*STATE_DIM], STATE_DIM, self.N
        ).T; cursor += self.N*STATE_DIM
        command_reference = ca.reshape(
            parameter[cursor:cursor+self.NC*4], 4, self.NC
        ).T; cursor += self.NC*4
        yaw_rate = parameter[cursor:cursor+self.N]
        delta_matrix = ca.reshape(delta, 4, self.NC).T

        objective = 0
        constraints = []
        Q = ca.DM(Q_PHYSICAL)
        R = ca.DM(float(r_scale) * R_PHYSICAL)
        Rd = ca.DM(float(r_scale) * RD_PHYSICAL)
        previous_command = command_reference[0, :].T
        for k in range(self.N):
            move = min(k, self.NC - 1)
            command = command_reference[move, :].T + delta_matrix[move, :].T
            constraints.append(command)
            state = self._rk4(state, command, yaw_rate[k])
            error = state - reference[k, :].T
            error[8] = ca.atan2(ca.sin(error[8]), ca.cos(error[8]))
            scale = 3.0 if k == self.N - 1 else 1.0
            objective += scale * ca.dot(Q*error, error)
            dcommand = command - command_reference[move, :].T
            objective += ca.dot(R*dcommand, dcommand)
            if k < self.NC:
                change = command - previous_command
                objective += ca.dot(Rd*change, change)
                previous_command = command
        nlp = {"x": delta, "p": parameter, "f": objective, "g": ca.vertcat(*constraints)}
        options = {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": int(max_iterations),
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 5e-3,
            "ipopt.warm_start_init_point": "yes",
        }
        self.solver = ca.nlpsol("outer_nmpc", "ipopt", nlp, options)
        self.lower_delta = np.tile(-self.delta_limit, self.NC)
        self.upper_delta = np.tile(self.delta_limit, self.NC)
        self.lower_command = np.tile(self.command_lower, self.N)
        self.upper_command = np.tile(self.command_upper, self.N)
        self.previous = np.zeros(self.nvar)
        self.last_status = "not run"

    def _applied_wrench(self, x, command, yaw_rate):
        sim, quad, controller = self.sim, self.sim.quad, self.sim.controller_PX4
        thrust = clip(command[0], controller.thrust_min, controller.thrust_max)
        phi_d = clip(command[1], -controller.tilt_max, controller.tilt_max)
        theta_d = clip(command[2], -controller.tilt_max, controller.tilt_max)
        psi_d = command[3]
        phi, theta, psi = x[6], x[7], x[8]
        rates = x[9:12]
        R = rotation(phi, theta, psi)
        Rd = rotation(phi_d, theta_d, psi_d)
        skew = R.T @ Rd - Rd.T @ R
        attitude_error = 0.5 * ca.vertcat(skew[2,1], skew[0,2], skew[1,0])
        rate_sp = ca.DM(controller.att_p) * attitude_error
        rate_sp += Rd.T @ ca.vertcat(0, 0, yaw_rate)
        rate_sp = ca.vertcat(*[
            clip(rate_sp[i], -controller.rate_sp_max[i], controller.rate_sp_max[i])
            for i in range(3)
        ])
        torque = ca.DM(controller.rate_p) * (rate_sp - rates)
        torque = ca.vertcat(*[
            clip(torque[i], -controller.torque_max[i], controller.torque_max[i])
            for i in range(3)
        ])
        wrench_requested = ca.vertcat(thrust, torque)
        allocation = ca.DM(pid_mixer.fct_allocation_matrix(quad.kT, quad.kD, quad.l))
        force_max = ca.DM(pid_mixer.fct_max_motor_forces(
            quad.kT, sim.max_speed, quad.prop_efficiency
        ))
        forces = allocation @ wrench_requested
        forces = ca.vertcat(*[clip(forces[i], 0.0, force_max[i]) for i in range(4)])
        arm = quad.l / np.sqrt(2.0)
        yaw_ratio = quad.kD / quad.kT
        wrench = ca.vertcat(
            ca.sum1(forces),
            arm*(-forces[0]-forces[1]+forces[2]+forces[3]),
            arm*(forces[0]-forces[1]-forces[2]+forces[3]),
            yaw_ratio*(forces[0]-forces[1]+forces[2]-forces[3]),
        )
        return wrench

    def _plant_derivative(self, x, wrench):
        quad = self.sim.quad
        phi, theta, psi = x[6], x[7], x[8]
        rates = x[9:12]
        R = rotation(phi, theta, psi)
        velocity = x[3:6]
        acceleration = (
            R @ ca.vertcat(0, 0, wrench[0]) / quad.m
            - quad.k_drag_linear * velocity / quad.m
            + ca.vertcat(0, 0, -quad.g)
        )
        I = ca.DM(quad.I); I_inv = ca.DM(np.linalg.inv(quad.I))
        angular_acceleration = I_inv @ (
            wrench[1:4] - quad.k_drag_angular*rates
            - ca.cross(rates, I @ rates)
        )
        W = ca.vertcat(
            ca.horzcat(1, ca.sin(phi)*ca.tan(theta), ca.cos(phi)*ca.tan(theta)),
            ca.horzcat(0, ca.cos(phi), -ca.sin(phi)),
            ca.horzcat(0, ca.sin(phi)/ca.cos(theta), ca.cos(phi)/ca.cos(theta)),
        )
        return ca.vertcat(velocity, acceleration, W @ rates, angular_acceleration)

    def _derivative(self, x, command, yaw_rate):
        return self._plant_derivative(
            x, self._applied_wrench(x, command, yaw_rate)
        )

    def _rk4(self, state, command, yaw_rate):
        dt = self.dt
        # The inner controller/allocator is evaluated once per controller
        # tick. The realized wrench is held while the plant integrates over
        # that 0.01-second interval, matching plant_attitude_step exactly.
        wrench = self._applied_wrench(state, command, yaw_rate)
        k1 = self._plant_derivative(state, wrench)
        k2 = self._plant_derivative(state + 0.5*dt*k1, wrench)
        k3 = self._plant_derivative(state + 0.5*dt*k2, wrench)
        k4 = self._plant_derivative(state + dt*k3, wrench)
        return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6

    def compute(self, state, reference, command_reference, yaw_rate):
        commands = np.asarray(command_reference[:self.NC], dtype=float).copy()
        commands[:, 3] = np.unwrap(np.r_[state[8], commands[:, 3]])[1:]
        parameter = np.concatenate([
            np.asarray(state), np.asarray(reference).reshape(-1),
            commands.reshape(-1), np.asarray(yaw_rate).reshape(-1),
        ])
        result = self.solver(
            x0=self.previous, p=parameter,
            lbx=self.lower_delta, ubx=self.upper_delta,
            lbg=self.lower_command, ubg=self.upper_command,
        )
        self.previous = np.asarray(result["x"]).reshape(-1)
        self.last_status = self.solver.stats()["return_status"]
        delta = self.previous.reshape(self.NC, 4)[0]
        return commands[0] + delta


def run(reference, steps, horizon_seconds=0.5,
        control_horizon_seconds=0.1, max_iterations=15):
    sim = quad_sim()
    controller = OuterCommandNMPC(
        sim, horizon_seconds, control_horizon_seconds, max_iterations
    )
    ref = reference[:steps]
    state_ref = reference_state_array(ref, sim.dt, sim.quad)
    command_ref = outer_command_reference_array(ref, sim.quad)
    _, yaw_rate = reference_yaw_arrays(ref, dt=sim.dt)
    state = np.zeros(STATE_DIM); state[8] = float(ref[0].get("yaw", 0.0))
    states = np.zeros((steps, STATE_DIM)); states[0] = state
    wrench = np.zeros((steps, 4)); commands = np.zeros((steps, 4))
    times, statuses = [], []
    altered = 0; max_error = 0.0
    sim.controller_PX4.fct_reset()
    for k in range(steps - 1):
        xref = build_ref_horizon(state_ref, k, controller.N)
        uref = build_ref_horizon(command_ref, k, controller.N)
        rref = build_ref_horizon(yaw_rate[:, None], k, controller.N).ravel()
        started = time.perf_counter()
        command = controller.compute(state, xref, uref, rref)
        times.append(time.perf_counter() - started); statuses.append(controller.last_status)
        state, applied = plant_attitude_step(
            sim, sim.controller_PX4, state, command, yaw_rate[k]
        )
        error = float(np.max(np.abs(sim.controller_PX4.last_requested_wrench-applied)))
        altered += int(error > 1e-8); max_error = max(max_error, error)
        states[k+1] = state; wrench[k] = applied; commands[k] = command
    if steps > 1: wrench[-1] = wrench[-2]; commands[-1] = commands[-2]
    diagnostics = {
        "mean_solve_ms": float(1e3*np.mean(times)),
        "p95_solve_ms": float(1e3*np.percentile(times, 95)),
        "failed_solves": sum(s not in ("Solve_Succeeded", "Solved_To_Acceptable_Level") for s in statuses),
        "allocator_altered_steps": altered,
        "allocator_max_abs_wrench_error": max_error,
    }
    return states, wrench, commands, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--indices", default="39,59,129,155,210")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--horizon-seconds", type=float, default=0.5)
    parser.add_argument("--control-horizon-seconds", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=15)
    args = parser.parse_args()
    with args.data.open("rb") as stream: data = pickle.load(stream)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in [int(v) for v in args.indices.split(",") if v.strip()]:
        reference = data["ref_traj_list"][index]
        steps = len(reference) if args.steps == 0 else min(args.steps, len(reference))
        states, wrench, commands, diagnostics = run(
            reference, steps, args.horizon_seconds,
            args.control_horizon_seconds, args.max_iterations,
        )
        family = data["family_labels"][index]
        row = {
            "index": index, "family": family,
            "controller_period_s": 0.01, "controller_rate_hz": 100.0,
            "prediction_horizon_s": args.horizon_seconds,
            "control_horizon_s": args.control_horizon_seconds,
            **trajectory_metrics(states, reference[:steps], float(data["sim_dt"])),
            **diagnostics,
        }
        rows.append(row)
        np.savez_compressed(
            args.output_dir / f"run_{index}_{family}.npz",
            states=states, wrench=wrench, outer_command=commands,
            controller_period_s=0.01, controller_rate_hz=100.0,
        )
        print(row)
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
