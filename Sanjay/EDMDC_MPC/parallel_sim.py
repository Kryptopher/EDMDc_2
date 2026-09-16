"""Generate reproducible yaw-aware EDMDc simulation data with dual inputs.

The default ``paper`` profile recreates the original paper's data regime:
100-second runs at 100 Hz, with 50/50/50/50/30 trajectory-family runs and
70 PRBS runs.  It intentionally retains the corrected applied-wrench logs and
the yaw-aware closed-loop excitation introduced for the ACC revision.

The ``acc_balanced`` profile retains the same 300-run composition and held-out
index convention, but uses 60-second runs and speed-balanced references.
"""

import argparse
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path

import numpy as np

from Simulation import ACC_BALANCED_PROFILE_CONFIG, quad_sim


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_RUN_COUNTS = ((1, 50), (2, 50), (3, 50), (4, 50), (5, 30)) # 1-helix, 2-figure-8, 3-Lissajous, 4-#waypoint, 5-#hover excitation
PAPER_PRBS_RUNS = 70
PAPER_DURATION_SECONDS = 100.0
ACC_BALANCED_RUN_COUNTS = PAPER_RUN_COUNTS
ACC_BALANCED_PRBS_RUNS = PAPER_PRBS_RUNS
ACC_BALANCED_DURATION_SECONDS = ACC_BALANCED_PROFILE_CONFIG["duration_seconds"]
INPUT_LABELS = ["thrust", "tau_roll", "tau_pitch", "tau_yaw"]
OUTER_INPUT_LABELS = ["thrust", "phi_des", "theta_des", "psi_des"]

PAPER_PROFILE_CONFIG = {
    "version": "paper_yaw_v1",
    "duration_seconds": PAPER_DURATION_SECONDS,
    "parametric_ramp_fraction": 0.3,
    "family_run_counts": dict(PAPER_RUN_COUNTS),
    "yaw_prbs_runs": PAPER_PRBS_RUNS,
    "family_parameters": {
        "helix": {"radius_m": [3.0, 10.0], "z_end_m": [3.0, 10.0], "turns": 1.0},
        "figure8": {
            "a_m": [25.0, 35.0], "b_m": [25.0, 35.0],
            "loops": 1.0, "tilt_deg": [10.0, 80.0],
        },
        "lissajous": {
            "xy_amplitude_m": [15.0, 25.0], "z_amplitude_m": [3.0, 7.0],
            "center_z_m": [1.0, 4.0], "axis_frequencies": [1.0, 2.0, 3.0],
        },
        "waypoint": {
            "count": [5, 15], "xy_range_m": [15.0, 25.0],
            "z_lower_m": 0.5, "z_upper_m": [3.0, 8.0],
            "gaussian_sigma_samples": [25, 50],
        },
        "hover_excitation": {
            "xy_amplitude_m": [2.0, 4.0], "z_amplitude_m": [1.0, 2.0],
            "xy_base_frequency_hz": [0.05, 0.12],
            "z_base_frequency_hz": [0.06, 0.15],
            "yaw_amplitude_deg": [2.0, 8.0], "sine_count": [2, 4],
        },
        "yaw_prbs": {
            "yaw_rate_radps": [-0.8, 0.8],
            "hold_seconds_at_100hz": [0.4, 1.2], "seed_start": 7000,
        },
    },
}


def configure_duration(sim, duration):
    if duration is None:
        return
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    sim.time = np.arange(0.0, duration, sim.dt)


def resolve_profile_duration(profile, duration):
    """Keep direct function calls consistent with fixed named profiles."""
    fixed_duration = {
        "paper": PAPER_DURATION_SECONDS,
        "acc_balanced": ACC_BALANCED_DURATION_SECONDS,
    }.get(profile)
    if fixed_duration is None:
        return duration
    if duration is None:
        return fixed_duration
    if not np.isclose(duration, fixed_duration):
        raise ValueError(
            f"profile={profile!r} requires duration={fixed_duration:g} seconds"
        )
    return duration


def default_worker_count(n_tasks):
    return max(1, min(n_tasks, os.cpu_count() or 1, 4))


def map_runs(worker, tasks, workers):
    """Map deterministic, independent simulations without oversubscribing RAM."""
    workers = default_worker_count(len(tasks)) if workers is None else workers
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers == 1:
        return [worker(task) for task in tasks]
    with mp.Pool(processes=min(workers, len(tasks))) as pool:
        return pool.map(worker, tasks)


def run_trajectory_single(task):
    """Top-level worker so that the task is pickleable on every platform."""
    traj, run_index, duration, profile = task
    sim = quad_sim()
    configure_duration(sim, duration)
    return sim.fct_run_single_simulation(traj, run_index, profile=profile)


def reference_metrics(ref):
    """Return compact physical-coverage diagnostics for one saved reference."""
    positions = np.asarray([point["pos"] for point in ref], dtype=float)
    velocities = np.asarray([point["vel"] for point in ref], dtype=float)
    accelerations = np.asarray([point["acc"] for point in ref], dtype=float)
    yaw_rates = np.asarray([point.get("yaw_rate", 0.0) for point in ref], dtype=float)
    speed = np.linalg.norm(velocities, axis=1)
    acceleration = np.linalg.norm(accelerations, axis=1)
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    return {
        "speed_mps_p50": float(np.percentile(speed, 50.0)),
        "speed_mps_p95": float(np.percentile(speed, 95.0)),
        "speed_mps_max": float(speed.max()),
        "acceleration_mps2_p95": float(np.percentile(acceleration, 95.0)),
        "acceleration_mps2_max": float(acceleration.max()),
        "abs_yaw_rate_radps_max": float(np.max(np.abs(yaw_rates))),
        "path_length_m": path_length,
    }


def simulation_metrics(states, applied, requested, ref, allocation_tolerance=1e-8):
    """Return tracking, numerical, and actuator diagnostics for one run."""
    states = np.asarray(states, dtype=float)
    applied = np.asarray(applied, dtype=float)
    requested = np.asarray(requested, dtype=float)
    ref_position = np.asarray([point["pos"] for point in ref], dtype=float)
    ref_velocity = np.asarray([point["vel"] for point in ref], dtype=float)
    ref_yaw = np.asarray([point["yaw"] for point in ref], dtype=float)
    position_error = states[:, :3] - ref_position
    velocity_error = states[:, 3:6] - ref_velocity
    yaw_error = (states[:, 8] - ref_yaw + np.pi) % (2.0 * np.pi) - np.pi
    allocation_error = applied - requested
    altered = np.max(np.abs(allocation_error), axis=1) > allocation_tolerance
    return {
        "finite": bool(
            np.isfinite(states).all()
            and np.isfinite(applied).all()
            and np.isfinite(requested).all()
        ),
        "initial_yaw_error_rad": float(abs(yaw_error[0])),
        "position_error_vector_rmse_m": float(
            np.sqrt(np.mean(np.sum(position_error**2, axis=1)))
        ),
        "velocity_error_vector_rmse_mps": float(
            np.sqrt(np.mean(np.sum(velocity_error**2, axis=1)))
        ),
        "yaw_error_rmse_rad": float(np.sqrt(np.mean(yaw_error**2))),
        "max_abs_roll_deg": float(np.rad2deg(np.max(np.abs(states[:, 6])))),
        "max_abs_pitch_deg": float(np.rad2deg(np.max(np.abs(states[:, 7])))),
        "allocation_altered_steps": int(altered.sum()),
        "allocation_altered_fraction": float(np.mean(altered)),
        "allocation_max_abs_wrench_error": np.max(
            np.abs(allocation_error), axis=0
        ).tolist(),
        "applied_max_abs_wrench": np.max(np.abs(applied), axis=0).tolist(),
        "requested_max_abs_wrench": np.max(np.abs(requested), axis=0).tolist(),
    }


def profile_config(profile, duration=None):
    if profile == "paper":
        return dict(PAPER_PROFILE_CONFIG)
    if profile == "acc_balanced":
        config = dict(ACC_BALANCED_PROFILE_CONFIG)
        config["family_run_counts"] = dict(ACC_BALANCED_RUN_COUNTS)
        config["yaw_prbs_runs"] = ACC_BALANCED_PRBS_RUNS
        return config
    return {
        "version": "custom",
        "duration_seconds": duration,
        "trajectory_parameters": "paper_ranges",
    }


def save_runs(traj, n, filename, duration=None, workers=None, profile="custom"):
    """Save one family with aligned applied-wrench and outer-command logs."""
    if n < 1:
        raise ValueError("n must be at least 1")
    duration = resolve_profile_duration(profile, duration)
    print(f"Running trajectory {traj}, n={n}...")
    start = time.perf_counter()
    results = map_runs(
        run_trajectory_single,
        [(traj, run_index, duration, profile) for run_index in range(n)],
        workers,
    )

    t = np.stack([entry[0] for entry in results])
    states = np.stack([entry[1] for entry in results])
    U = np.stack([entry[2] for entry in results])
    U_requested = np.stack([entry[3] for entry in results])
    U_outer = np.stack([entry[4] for entry in results])
    refs = [entry[5] for entry in results]
    ref_metrics = [reference_metrics(ref) for ref in refs]
    sim_metrics = [
        simulation_metrics(states[k], U[k], U_requested[k], refs[k])
        for k in range(n)
    ]
    sim_dt = quad_sim.dt
    time_vector = np.arange(0.0, duration, sim_dt) if duration is not None else quad_sim().time

    with open(filename, "wb") as f:
        pickle.dump(
            {
                "traj": traj,
                "n": n,
                "sim_dt": sim_dt,
                "time": time_vector,
                "t": t,
                "states": states,
                "U": U,
                "U_requested": U_requested,
                "U_outer": U_outer,
                "ref_traj_list": refs,
                "reference_metrics": ref_metrics,
                "simulation_metrics": sim_metrics,
                "run_seeds": [1000 * traj + run_index for run_index in range(n)],
                "input_type": "applied_wrench",
                "input_labels": INPUT_LABELS,
                "outer_input_type": "desired_attitude",
                "outer_input_labels": OUTER_INPUT_LABELS,
                "dataset_profile": profile,
                "trajectory_profile_config": profile_config(profile, duration),
                "schema_version": "yaw_dual_input_v1",
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Saved {filename} in {time.perf_counter() - start:.1f} s")


def make_yaw_prbs_reference(time_vector, seed):
    """Create bounded closed-loop yaw PRBS, replacing old raw-angle PRBS.

    The run count and deterministic seed sequence match the original setup.
    The old input log contained desired roll/pitch angles rather than plant
    inputs, so it cannot be reused for applied-wrench EDMDc identification.
    """
    rng = np.random.default_rng(seed)
    time_vector = np.asarray(time_vector, dtype=float)
    n_samples = len(time_vector)
    dt = float(time_vector[1] - time_vector[0]) if n_samples > 1 else 0.01

    yaw_rate = np.zeros(n_samples)
    k = 0
    while k < n_samples:
        hold_steps = int(rng.integers(40, 121))  # 0.4--1.2 s at 100 Hz
        yaw_rate[k:k + hold_steps] = rng.uniform(-0.8, 0.8)
        k += hold_steps
    yaw = np.cumsum(yaw_rate) * dt

    # Keep a small translational component so the PRBS family covers a local
    # flight regime rather than only a perfectly stationary hover.
    ramp_duration = min(4.0, max(time_vector[-1] if n_samples else 0.0, dt))
    ramp = np.clip(time_vector / max(ramp_duration, dt), 0.0, 1.0)
    rise = ramp * ramp * (3.0 - 2.0 * ramp)
    rise_dot = np.where(
        ramp < 1.0,
        6.0 * ramp * (1.0 - ramp) / max(ramp_duration, dt),
        0.0,
    )
    x = 0.4 * np.sin(0.35 * time_vector)
    y = 0.3 * np.sin(0.27 * time_vector + 0.5)
    z = 1.5 * rise + 0.08 * np.sin(0.45 * time_vector)
    vx = 0.14 * np.cos(0.35 * time_vector)
    vy = 0.081 * np.cos(0.27 * time_vector + 0.5)
    vz = 1.5 * rise_dot + 0.036 * np.cos(0.45 * time_vector)
    ax = -0.049 * np.sin(0.35 * time_vector)
    ay = -0.02187 * np.sin(0.27 * time_vector + 0.5)
    az = -0.0162 * np.sin(0.45 * time_vector)

    return [
        {
            "pos": np.array([x[i], y[i], z[i]], dtype=float),
            "vel": np.array([vx[i], vy[i], vz[i]], dtype=float),
            "acc": np.array([ax[i], ay[i], az[i]], dtype=float),
            "yaw": float(yaw[i]),
            "yaw_rate": float(yaw_rate[i]),
        }
        for i in range(n_samples)
    ]


def run_yaw_prbs_single(task):
    run_index, duration = task
    sim = quad_sim()
    configure_duration(sim, duration)
    ref = make_yaw_prbs_reference(sim.time, seed=7000 + run_index)
    init_state = np.zeros(12)
    init_state[8] = float(ref[0]["yaw"])
    t, states, _, U, U_requested, U_outer = sim.sim_PID.fct_simulate(
        sim.time, sim.dt, ref, init_state,
        return_requested=True, return_outer=True,
    )
    return t, states, U, U_requested, U_outer, ref


def save_prbs_runs(n, filename, duration=None, workers=None, profile="custom"):
    """Save bounded yaw-PRBS flights with the old 7000+i seed convention."""
    if n < 1:
        raise ValueError("n must be at least 1")
    duration = resolve_profile_duration(profile, duration)
    print(f"Running yaw PRBS excitation, n={n}...")
    start = time.perf_counter()
    results = map_runs(
        run_yaw_prbs_single,
        [(run_index, duration) for run_index in range(n)],
        workers,
    )
    t = np.stack([entry[0] for entry in results])
    states = np.stack([entry[1] for entry in results])
    U = np.stack([entry[2] for entry in results])
    U_requested = np.stack([entry[3] for entry in results])
    U_outer = np.stack([entry[4] for entry in results])
    refs = [entry[5] for entry in results]
    ref_metrics = [reference_metrics(ref) for ref in refs]
    sim_metrics = [
        simulation_metrics(states[k], U[k], U_requested[k], refs[k])
        for k in range(n)
    ]
    sim_dt = quad_sim.dt
    time_vector = np.arange(0.0, duration, sim_dt) if duration is not None else quad_sim().time

    with open(filename, "wb") as f:
        pickle.dump(
            {
                "traj": "yaw_prbs",
                "n": n,
                "sim_dt": sim_dt,
                "time": time_vector,
                "t": t,
                "states": states,
                "U": U,
                "U_requested": U_requested,
                "U_outer": U_outer,
                "ref_traj_list": refs,
                "reference_metrics": ref_metrics,
                "simulation_metrics": sim_metrics,
                "run_seeds": [7000 + run_index for run_index in range(n)],
                "input_type": "applied_wrench",
                "input_labels": INPUT_LABELS,
                "outer_input_type": "desired_attitude",
                "outer_input_labels": OUTER_INPUT_LABELS,
                "dataset_profile": profile,
                "trajectory_profile_config": profile_config(profile, duration),
                "schema_version": "yaw_dual_input_v1",
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Saved {filename} in {time.perf_counter() - start:.1f} s")


def resolve_generation_spec(parser, args):
    if args.profile in ("paper", "acc_balanced"):
        if any(value is not None for value in (
            args.runs_per_family, args.prbs_runs, args.duration,
        )):
            description = "100 s" if args.profile == "paper" else "60 s"
            parser.error(
                f"The {args.profile} profile has fixed {description} and "
                "50/50/50/50/30/70 "
                "counts. Use --profile compact for a custom run."
            )
        if args.profile == "paper":
            return PAPER_RUN_COUNTS, PAPER_PRBS_RUNS, PAPER_DURATION_SECONDS
        return (
            ACC_BALANCED_RUN_COUNTS,
            ACC_BALANCED_PRBS_RUNS,
            ACC_BALANCED_DURATION_SECONDS,
        )

    runs_per_family = 50 if args.runs_per_family is None else args.runs_per_family
    prbs_runs = 0 if args.prbs_runs is None else args.prbs_runs
    if runs_per_family < 1:
        parser.error("--runs-per-family must be at least 1")
    if prbs_runs < 0:
        parser.error("--prbs-runs cannot be negative")
    return tuple((traj, runs_per_family) for traj in (1, 2, 3)), prbs_runs, args.duration


def main():
    parser = argparse.ArgumentParser(
        description="Generate deterministic yaw-aware EDMDc training trajectories."
    )
    parser.add_argument(
        "--profile", choices=("paper", "acc_balanced", "compact"), default="paper",
        help=(
            "paper recreates the old 300-run regime (default); acc_balanced "
            "uses speed-balanced 60 s runs; compact is custom."
        ),
    )
    parser.add_argument(
        "--runs-per-family", type=int, default=None,
        help="Compact profile only: runs for each of trajectories 1--3 (default: 50).",
    )
    parser.add_argument(
        "--prbs-runs", type=int, default=None,
        help="Compact profile only: bounded yaw-PRBS runs (default: 0).",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Compact profile only: duration in seconds (default: simulator 45 s).",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Independent simulation workers (default: up to 4).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR,
        help="Directory for generated pickle files (default: this script's directory).",
    )
    args = parser.parse_args()
    family_counts, prbs_runs, duration = resolve_generation_spec(parser, args)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for traj, n in family_counts:
        save_runs(
            traj=traj,
            n=n,
            filename=output_dir / f"runs_traj{traj}_n{n}.pkl",
            duration=duration,
            workers=args.workers,
            profile=args.profile,
        )
    if prbs_runs:
        save_prbs_runs(
            n=prbs_runs,
            filename=output_dir / f"runs_prbs_n{prbs_runs}.pkl",
            duration=duration,
            workers=args.workers,
            profile=args.profile,
        )
    print(f"Total time: {(time.perf_counter() - start) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
