"""Closed-loop MPC for the nonlinear outer desired-attitude EDMDc model."""

import argparse
import csv
import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from edmdc_mpc import (
    OUTER_ATTITUDE_INPUT_LIFT_TYPE,
    OuterCommandEDMDcMPC_SQP,
    STATE_DIM,
    build_ref_horizon,
    hover_yaw_command_reference_array,
    lifted_state_from_x,
    outer_command_reference_array,
    precompute_ref_std,
    reference_yaw_arrays,
    wrap_angle_pi,
)
from Simulation import quad_sim


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "artifacts" / "acc_balanced_waypoint_v2" / "runs_mixed_n300_waypoint_v2.pkl"
DEFAULT_MODEL = SCRIPT_DIR / "artifacts" / "acc_balanced_waypoint_v2" / "edmdc_outer_attitude_error_dt001.pkl"
DEFAULT_TEST_INDICES = (39, 59, 129, 155, 210)
Q_PHYSICAL = np.array([
    45.0, 45.0, 55.0, 4.0, 4.0, 5.0,
    0.2, 0.2, 4.0, 0.05, 0.05, 1.0,
])
R_PHYSICAL = np.array([0.05, 1.0, 1.0, 0.5])
RD_PHYSICAL = np.array([0.02, 0.5, 0.5, 0.25])


def load_pickle(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def compact_model_matrices(model):
    full_A = np.asarray(model["A"], dtype=float)
    full_B = np.asarray(model["B"], dtype=float)
    active = np.asarray(
        model.get("active_observable_indices", np.arange(full_A.shape[0])),
        dtype=int,
    )
    if full_A.shape[0] != len(active):
        A = full_A[np.ix_(active, active)]
        B = full_B[active]
    else:
        A, B = full_A, full_B
    lookup = {int(raw): local for local, raw in enumerate(active)}
    missing = [index for index in range(STATE_DIM) if index not in lookup]
    if missing:
        raise ValueError(f"Model omits physical-state observables {missing}")
    Cz = np.zeros((STATE_DIM, len(active)), dtype=float)
    for state_index in range(STATE_DIM):
        Cz[state_index, lookup[state_index]] = 1.0
    return A, B, Cz, active


def build_controller(model, sim, horizon_seconds, control_horizon_seconds,
                     offboard_period, iterations, r_scale=1.0,
                     thrust_trust_n=1.0, attitude_trust_rad=0.05,
                     yaw_trust_rad=0.08, q_position_scale=1.0,
                     q_velocity_scale=1.0, q_attitude_scale=1.0,
                     q_yaw_scale=1.0, q_rate_scale=1.0,
                     terminal_scale=3.0, yaw_feedforward_only=False,
                     attitude_error_max_rad=None, rd_scale=1.0,
                     first_move_rd_scale=1.0,
                     command_slew_max_raw=None):
    if model.get("input_source") != "outer_command":
        raise ValueError("Outer-command MPC requires input_source=outer_command")
    if model.get("input_lift_type") != OUTER_ATTITUDE_INPUT_LIFT_TYPE:
        raise ValueError("Checkpoint must use the attitude-error outer input lift")
    model_dt = float(model["dt"])
    N = max(1, int(round(horizon_seconds / model_dt)))
    move_block = max(1, int(round(offboard_period / model_dt)))
    if not np.isclose(move_block * model_dt, offboard_period):
        raise ValueError("Offboard period must be an integer multiple of model dt")
    NC = max(1, int(np.ceil(control_horizon_seconds / offboard_period)))
    A, B, Cz, active = compact_model_matrices(model)
    state_scale = np.asarray(model["scaler"].scale_[:STATE_DIM], dtype=float)
    input_scale = np.asarray(model["u_scaler"].scale_[:4], dtype=float)
    q_physical = Q_PHYSICAL.copy()
    q_physical[:3] *= float(q_position_scale)
    q_physical[3:6] *= float(q_velocity_scale)
    q_physical[6:8] *= float(q_attitude_scale)
    q_physical[[8]] *= float(q_yaw_scale)
    q_physical[9:12] *= float(q_rate_scale)
    if yaw_feedforward_only:
        q_physical[[8, 11]] = 0.0
        yaw_trust_rad = 0.0
    controller = OuterCommandEDMDcMPC_SQP(
        A=A,
        B=B,
        Cz=Cz,
        N=N,
        NC=NC,
        Q=np.diag(q_physical * state_scale**2),
        R=np.diag(float(r_scale) * R_PHYSICAL * input_scale**2),
        Rd=np.diag(float(r_scale) * float(rd_scale) * RD_PHYSICAL * input_scale**2),
        Q_terminal=np.diag(float(terminal_scale) * q_physical * state_scale**2),
        state_scaler=model["scaler"],
        u_scaler=model["u_scaler"],
        u_min_raw=np.array([
            sim.controller_PX4.thrust_min,
            -sim.controller_PX4.tilt_max,
            -sim.controller_PX4.tilt_max,
            -np.inf,
        ]),
        u_max_raw=np.array([
            sim.controller_PX4.thrust_max,
            sim.controller_PX4.tilt_max,
            sim.controller_PX4.tilt_max,
            np.inf,
        ]),
        du_max_raw=np.array([
            thrust_trust_n, attitude_trust_rad,
            attitude_trust_rad, yaw_trust_rad,
        ]),
        max_iterations=iterations,
        move_block_steps=move_block,
        attitude_error_max_raw=attitude_error_max_rad,
        first_move_rd_scale=first_move_rd_scale,
        command_slew_max_raw=command_slew_max_raw,
    )
    controller.model_dt = model_dt
    controller.active_observable_indices = active
    controller.offboard_period = offboard_period
    return controller


def plant_attitude_step(sim, controller, state, command, yaw_rate_ref):
    omega, allocated = controller.fct_attitude_step(
        state, command, sim.dt, yaw_rate_ref=yaw_rate_ref
    )
    omega_plant = sim.quad.fct_apply_motor_dynamics(omega, sim.dt)
    rotor_thrust, rotor_drag = sim.quad.fct_rotor_forces(omega_plant)
    physical_thrust, physical_torque = sim.quad.fct_Rotor_torque(
        rotor_thrust, rotor_drag
    )
    applied = np.r_[physical_thrust[2], physical_torque]
    controller.last_actual_wrench = applied
    controller.last_allocated_wrench = allocated
    start = float(getattr(sim, "plant_time", 0.0))
    solution = solve_ivp(
        lambda t, x: sim.quad.fct_dynamics(t, x, omega_plant),
        [start, start + sim.dt],
        state,
        method="RK45",
    )
    sim.plant_time = start + sim.dt
    return solution.y[:, -1], applied


def run_outer_mpc(model, sim, ref, steps, horizon_seconds=2.0,
                  control_horizon_seconds=0.1, offboard_period=0.01,
                  iterations=2, r_scale=1.0, thrust_trust_n=1.0,
                  attitude_trust_rad=0.05, yaw_trust_rad=0.08,
                  thrust_feedforward_offset_n=0.0,
                  reference_mode="inverse_dynamics",
                  yaw_feedforward_only=False, q_position_scale=1.0,
                  q_velocity_scale=1.0, q_attitude_scale=1.0,
                  q_yaw_scale=1.0, q_rate_scale=1.0,
                  terminal_scale=3.0, attitude_error_max_rad=None,
                  rd_scale=1.0, first_move_rd_scale=1.0,
                  command_slew_max_raw=None, initial_yaw=None):
    controller = build_controller(
        model, sim, horizon_seconds, control_horizon_seconds,
        offboard_period, iterations, r_scale=r_scale,
        thrust_trust_n=thrust_trust_n,
        attitude_trust_rad=attitude_trust_rad,
        yaw_trust_rad=yaw_trust_rad,
        q_position_scale=q_position_scale,
        q_velocity_scale=q_velocity_scale,
        q_attitude_scale=q_attitude_scale,
        q_yaw_scale=q_yaw_scale,
        q_rate_scale=q_rate_scale,
        terminal_scale=terminal_scale,
        yaw_feedforward_only=yaw_feedforward_only,
        attitude_error_max_rad=attitude_error_max_rad,
        rd_scale=rd_scale,
        first_move_rd_scale=first_move_rd_scale,
        command_slew_max_raw=command_slew_max_raw,
    )
    model_dt = controller.model_dt
    model_stride = int(round(model_dt / sim.dt))
    update_stride = int(round(offboard_period / sim.dt))
    if model_stride < 1 or update_stride < 1:
        raise ValueError("Model and offboard periods must not be faster than the plant")

    ref_used = ref[:steps]
    nominal_quad = sim.controller_PX4.quad
    if reference_mode == "inverse_dynamics":
        ref_std = precompute_ref_std(
            ref_used, model["scaler"], dt=sim.dt, quad=nominal_quad
        )[::model_stride]
        outer_feedforward = outer_command_reference_array(
            ref_used, nominal_quad
        )[::model_stride]
        outer_feedforward[:, 0] += float(thrust_feedforward_offset_n)
    elif reference_mode == "kinematic_hover_yaw":
        ref_std = precompute_ref_std(
            ref_used, model["scaler"], dt=sim.dt, quad=None
        )[::model_stride]
        outer_feedforward = hover_yaw_command_reference_array(
            ref_used,
            nominal_quad.m * nominal_quad.g + float(thrust_feedforward_offset_n),
        )[::model_stride]
    else:
        raise ValueError(f"Unknown reference mode {reference_mode!r}")
    _, yaw_rate = reference_yaw_arrays(ref_used, dt=sim.dt)
    state = np.zeros(STATE_DIM, dtype=float)
    state[8] = float(
        ref_used[0].get("yaw", 0.0) if initial_yaw is None else initial_yaw
    )
    states = np.zeros((steps, STATE_DIM), dtype=float)
    wrench = np.zeros((steps, 4), dtype=float)
    outer = np.zeros((steps, 4), dtype=float)
    states[0] = state
    command = outer_feedforward[0].copy()
    solve_times = []
    statuses = []
    iteration_counts = []
    allocator_altered_steps = 0
    allocator_max_abs_error = 0.0
    plant_wrench_mismatch_steps = 0
    plant_wrench_mismatch_max_abs_error = 0.0
    sim.controller_PX4.fct_reset()

    for k in range(steps - 1):
        if k % update_stride == 0:
            z = lifted_state_from_x(state, model["scaler"])
            active = controller.active_observable_indices
            if len(active) < len(z):
                z = z[active]
            model_index = k // model_stride
            state_horizon = build_ref_horizon(ref_std, model_index, controller.N)
            command_horizon = build_ref_horizon(
                outer_feedforward, model_index, controller.N
            )
            started = time.perf_counter()
            command = controller.compute(z, state_horizon, command_horizon)
            solve_times.append(time.perf_counter() - started)
            statuses.append(controller.last_status)
            iteration_counts.append(controller.last_iterations)

        state, applied = plant_attitude_step(
            sim, sim.controller_PX4, state, command, yaw_rate[k]
        )
        allocation_error = float(np.max(np.abs(
            sim.controller_PX4.last_requested_wrench
            - sim.controller_PX4.last_allocated_wrench
        )))
        plant_wrench_error = float(np.max(np.abs(
            applied - sim.controller_PX4.last_allocated_wrench
        )))
        allocator_altered_steps += int(allocation_error > 1e-8)
        plant_wrench_mismatch_steps += int(plant_wrench_error > 1e-8)
        allocator_max_abs_error = max(
            allocator_max_abs_error, allocation_error
        )
        plant_wrench_mismatch_max_abs_error = max(
            plant_wrench_mismatch_max_abs_error, plant_wrench_error
        )
        states[k + 1] = state
        wrench[k] = applied
        outer[k] = sim.controller_PX4.last_outer_command
    if steps > 1:
        wrench[-1] = wrench[-2]
        outer[-1] = outer[-2]
    sim.controller_PX4.fct_reset()
    diagnostics = {
        "statuses": statuses,
        "iterations": iteration_counts,
        "solve_times": solve_times,
        "mean_solve_ms": float(1e3 * np.mean(solve_times)) if solve_times else np.nan,
        "p95_solve_ms": float(1e3 * np.percentile(solve_times, 95)) if solve_times else np.nan,
        "p99_solve_ms": float(1e3 * np.percentile(solve_times, 99)) if solve_times else np.nan,
        "max_solve_ms": float(1e3 * np.max(solve_times)) if solve_times else np.nan,
        "allocator_altered_steps": allocator_altered_steps,
        "allocator_max_abs_wrench_error": allocator_max_abs_error,
        "plant_wrench_mismatch_steps": plant_wrench_mismatch_steps,
        "plant_wrench_mismatch_max_abs_error": plant_wrench_mismatch_max_abs_error,
        "reference_mode": reference_mode,
        "yaw_feedforward_only": bool(yaw_feedforward_only),
        "attitude_error_max_rad": attitude_error_max_rad,
    }
    return states, wrench, outer, diagnostics


def trajectory_metrics(states, ref, dt):
    position_ref = np.asarray([point["pos"] for point in ref], dtype=float)
    velocity_ref = np.asarray([point.get("vel", np.zeros(3)) for point in ref])
    yaw_ref, yaw_rate_ref = reference_yaw_arrays(ref, dt=dt)
    return {
        "position_rmse_m": float(np.sqrt(np.mean((states[:, :3] - position_ref)**2))),
        "velocity_rmse_mps": float(np.sqrt(np.mean((states[:, 3:6] - velocity_ref)**2))),
        "yaw_rmse_rad": float(np.sqrt(np.mean(wrap_angle_pi(states[:, 8] - yaw_ref)**2))),
        "yaw_rate_rmse_radps": float(np.sqrt(np.mean((states[:, 11] - yaw_rate_ref)**2))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--indices", default=",".join(map(str, DEFAULT_TEST_INDICES)))
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--horizon-seconds", type=float, default=2.0)
    parser.add_argument("--control-horizon-seconds", type=float, default=0.1)
    parser.add_argument(
        "--offboard-period", type=float, default=0.01,
        help="Outer-controller update period (default: 0.01 s = 100 Hz).",
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--r-scale", type=float, default=10.0)
    parser.add_argument("--thrust-trust-n", type=float, default=1.0)
    parser.add_argument("--attitude-trust-rad", type=float, default=0.05)
    parser.add_argument("--yaw-trust-rad", type=float, default=0.08)
    parser.add_argument(
        "--reference-mode",
        choices=("inverse_dynamics", "kinematic_hover_yaw"),
        default="inverse_dynamics",
        help=(
            "Use kinematic_hover_yaw for the common-information comparison "
            "without acceleration-derived thrust or attitude feedforward."
        ),
    )
    parser.add_argument("--yaw-feedforward-only", action="store_true")
    parser.add_argument("--q-position-scale", type=float, default=1.0)
    parser.add_argument("--q-velocity-scale", type=float, default=1.0)
    parser.add_argument("--q-attitude-scale", type=float, default=1.0)
    parser.add_argument("--q-yaw-scale", type=float, default=1.0)
    parser.add_argument("--q-rate-scale", type=float, default=1.0)
    parser.add_argument("--terminal-scale", type=float, default=3.0)
    parser.add_argument("--rd-scale", type=float, default=1.0)
    parser.add_argument(
        "--first-move-rd-scale", type=float, default=1.0,
        help="Extra rate penalty between the previous applied command and the new first move.",
    )
    parser.add_argument(
        "--command-slew-max", default="",
        help=("Optional per-update physical slew limits as "
              "thrust,roll,pitch,yaw; e.g. 0.15,0.0065,0.0065,0.008."),
    )
    args = parser.parse_args()

    command_slew_max_raw = None
    if args.command_slew_max.strip():
        command_slew_max_raw = np.asarray(
            [float(value) for value in args.command_slew_max.split(",")],
            dtype=float,
        )
        if command_slew_max_raw.shape != (4,) or np.any(command_slew_max_raw <= 0):
            parser.error("--command-slew-max requires four positive values")

    model = load_pickle(args.model)
    data = load_pickle(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in [int(value) for value in args.indices.split(",") if value.strip()]:
        ref = data["ref_traj_list"][index]
        steps = len(ref) if args.steps == 0 else min(args.steps, len(ref))
        sim = quad_sim()
        states, wrench, outer, diagnostics = run_outer_mpc(
            model, sim, ref, steps,
            horizon_seconds=args.horizon_seconds,
            control_horizon_seconds=args.control_horizon_seconds,
            offboard_period=args.offboard_period,
            iterations=args.iterations,
            r_scale=args.r_scale,
            thrust_trust_n=args.thrust_trust_n,
            attitude_trust_rad=args.attitude_trust_rad,
            yaw_trust_rad=args.yaw_trust_rad,
            reference_mode=args.reference_mode,
            yaw_feedforward_only=args.yaw_feedforward_only,
            q_position_scale=args.q_position_scale,
            q_velocity_scale=args.q_velocity_scale,
            q_attitude_scale=args.q_attitude_scale,
            q_yaw_scale=args.q_yaw_scale,
            q_rate_scale=args.q_rate_scale,
            terminal_scale=args.terminal_scale,
            rd_scale=args.rd_scale,
            first_move_rd_scale=args.first_move_rd_scale,
            command_slew_max_raw=command_slew_max_raw,
        )
        metrics = trajectory_metrics(states, ref[:steps], sim.dt)
        family = data.get("family_labels", [str(index)] * data["n"])[index]
        row = {
            "index": index,
            "family": family,
            "plant_dt_s": float(sim.dt),
            "model_dt_s": float(model["dt"]),
            "controller_period_s": float(args.offboard_period),
            "controller_rate_hz": float(1.0 / args.offboard_period),
            "prediction_horizon_s": float(args.horizon_seconds),
            "control_horizon_s": float(args.control_horizon_seconds),
            "r_scale": float(args.r_scale),
            "iterations": int(args.iterations),
            "thrust_trust_n": float(args.thrust_trust_n),
            "attitude_trust_rad": float(args.attitude_trust_rad),
            "yaw_trust_rad": float(args.yaw_trust_rad),
            "reference_mode": args.reference_mode,
            "yaw_feedforward_only": bool(args.yaw_feedforward_only),
            "q_position_scale": float(args.q_position_scale),
            "q_velocity_scale": float(args.q_velocity_scale),
            "q_attitude_scale": float(args.q_attitude_scale),
            "q_yaw_scale": float(args.q_yaw_scale),
            "q_rate_scale": float(args.q_rate_scale),
            "terminal_scale": float(args.terminal_scale),
            "rd_scale": float(args.rd_scale),
            "first_move_rd_scale": float(args.first_move_rd_scale),
            "command_slew_max_raw": (
                "" if command_slew_max_raw is None else
                ",".join(map(str, command_slew_max_raw))
            ),
            **metrics,
            "mean_solve_ms": diagnostics["mean_solve_ms"],
            "p95_solve_ms": diagnostics["p95_solve_ms"],
            "failed_solves": sum(
                status not in ("solved", "solved inaccurate")
                for status in diagnostics["statuses"]
            ),
            "allocator_altered_steps": diagnostics["allocator_altered_steps"],
            "allocator_max_abs_wrench_error": diagnostics[
                "allocator_max_abs_wrench_error"
            ],
        }
        rows.append(row)
        np.savez_compressed(
            args.output_dir / f"run_{index}_{family}.npz",
            states=states, wrench=wrench, outer_command=outer,
            plant_dt_s=float(sim.dt),
            model_dt_s=float(model["dt"]),
            controller_period_s=float(args.offboard_period),
            controller_rate_hz=float(1.0 / args.offboard_period),
            prediction_horizon_s=float(args.horizon_seconds),
            control_horizon_s=float(args.control_horizon_seconds),
            r_scale=float(args.r_scale),
            iterations=int(args.iterations),
            thrust_trust_n=float(args.thrust_trust_n),
            attitude_trust_rad=float(args.attitude_trust_rad),
            yaw_trust_rad=float(args.yaw_trust_rad),
            reference_mode=args.reference_mode,
            yaw_feedforward_only=bool(args.yaw_feedforward_only),
            q_position_scale=float(args.q_position_scale),
            q_velocity_scale=float(args.q_velocity_scale),
            q_attitude_scale=float(args.q_attitude_scale),
            q_yaw_scale=float(args.q_yaw_scale),
            q_rate_scale=float(args.q_rate_scale),
            terminal_scale=float(args.terminal_scale),
            rd_scale=float(args.rd_scale),
            first_move_rd_scale=float(args.first_move_rd_scale),
            command_slew_max_raw=(
                np.array([], dtype=float) if command_slew_max_raw is None
                else command_slew_max_raw
            ),
        )
        reference = np.asarray([point["pos"] for point in ref[:steps]])
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(*reference.T, "k", linewidth=2, label="Reference")
        ax.plot(*states[:, :3].T, color="#2ca02c", label="Outer EDMDc-MPC")
        ax.set_title(f"{family} run {index}: {metrics['position_rmse_m']:.3f} m RMSE")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.legend()
        fig.savefig(args.output_dir / f"run_{index}_{family}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(row)

    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
