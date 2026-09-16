"""Screen where EDMDc overtakes fixed and yaw-scheduled hover-linear MPC.

The plant remains nominal.  Only reference amplitude, velocity, and
acceleration are scaled.  Every controller receives the same kinematic
position/velocity preview and hover-thrust/yaw command reference.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import pickle
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from edmdc_mpc import reference_state_array
from outer_command_mpc import run_outer_mpc, trajectory_metrics
from outer_ltv_mpc import run as run_linear_mpc
from publication_scenarios import tracking_reference
from reactive_pid_tracking import run_reactive_pid
from Simulation import quad_sim


CONTROLLERS = ("reactive_pid", "fixed_hover_lti", "yaw_scheduled_hover", "edmdc")
LABELS = {
    "reactive_pid": "Reactive cascaded PID",
    "fixed_hover_lti": "Fixed hover LTI-MPC",
    "yaw_scheduled_hover": "Yaw-scheduled hover-linear MPC",
    "edmdc": "EDMDc-MPC",
}
COLORS = {
    "reactive_pid": "#009E73",
    "fixed_hover_lti": "#56B4E9",
    "yaw_scheduled_hover": "#0072B2",
    "edmdc": "#D55E00",
}
FAMILIES = ("helix", "figure8", "lissajous", "waypoint")


def runtime_environment():
    """Return the execution details needed to reproduce timing results."""
    packages = {}
    for name in ("numpy", "scipy", "osqp"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cpu_model = platform.processor()
    if platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                cpu_model = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except (ImportError, OSError):
            pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "package_versions": packages,
        "osqp_settings": {
            "warm_starting": True,
            "verbose": False,
            "polishing": False,
            "remaining_settings": "OSQP package defaults",
        },
        "timing_scope": "controller optimization call only",
    }


def write_csv(path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_aggressiveness(reference, gravity):
    velocity = np.asarray([point.get("vel", np.zeros(3)) for point in reference])
    acceleration = np.asarray([point.get("acc", np.zeros(3)) for point in reference])
    horizontal = np.linalg.norm(acceleration[:, :2], axis=1)
    vertical_specific_force = np.maximum(1e-6, gravity + acceleration[:, 2])
    implied_tilt = np.arctan2(horizontal, vertical_specific_force)
    return {
        "reference_peak_speed_mps": float(np.max(np.linalg.norm(velocity, axis=1))),
        "reference_peak_acceleration_mps2": float(np.max(np.linalg.norm(acceleration, axis=1))),
        "reference_p95_implied_tilt_deg": float(np.degrees(np.percentile(implied_tilt, 95))),
        "reference_peak_implied_tilt_deg": float(np.degrees(np.max(implied_tilt))),
    }


def response_activity(states, commands):
    physical_tilt = np.maximum(np.abs(states[:, 6]), np.abs(states[:, 7]))
    commanded_tilt = np.maximum(np.abs(commands[:, 1]), np.abs(commands[:, 2]))
    delta = np.diff(commands, axis=0)
    attitude_step = np.linalg.norm(delta[:, 1:3], axis=1)
    body_rate_norm = np.linalg.norm(states[:, 9:12], axis=1)
    return {
        "actual_p95_tilt_deg": float(np.degrees(np.percentile(physical_tilt, 95))),
        "actual_peak_tilt_deg": float(np.degrees(np.max(physical_tilt))),
        "command_peak_tilt_deg": float(np.degrees(np.max(commanded_tilt))),
        "body_rate_rms_radps": float(np.sqrt(np.mean(states[:, 9:12] ** 2))),
        "body_rate_p99_radps": float(np.percentile(body_rate_norm, 99)),
        "body_rate_max_radps": float(np.max(body_rate_norm)),
        "roll_pitch_step_rms_rad": float(np.sqrt(np.mean(delta[:, 1:3] ** 2))),
        "roll_pitch_step_p99_rad": float(np.percentile(attitude_step, 99)),
        "roll_pitch_step_max_rad": float(np.max(attitude_step)),
    }


def run_one(task):
    reference = tracking_reference(
        task["family"], task["run_index"], task["duration_seconds"],
        speed_scale=task["aggressiveness_scale"],
    )
    sim = quad_sim()
    if task["controller"] == "reactive_pid":
        gains = task.get("reactive_gains")
        pid_sim, states, applied, requested, commands = run_reactive_pid(
            reference, gains
        )
        allocation_error = np.max(np.abs(applied - requested), axis=1)
        diagnostics = {
            "failed_solves": 0,
            "allocator_altered_steps": int(np.sum(allocation_error > 1e-8)),
            "mean_solve_ms": 0.0,
        }
        sim = pid_sim
    elif task["controller"] in ("fixed_hover_lti", "yaw_scheduled_hover"):
        config = task["linear_config"]
        mode = "hover_lti" if task["controller"] == "fixed_hover_lti" else "yaw_scheduled_hover"
        states, _, commands, diagnostics = run_linear_mpc(
            reference, len(reference), config["horizon_seconds"],
            config["control_horizon_seconds"], config["r_scale"], mode,
            config["thrust_trust_n"], config["attitude_trust_rad"], 0.0,
            sim=sim, model_sim=quad_sim(),
            q_position_scale=config["q_position_scale"],
            q_velocity_scale=config["q_velocity_scale"],
            q_attitude_scale=0.0, q_yaw_scale=0.0, q_rate_scale=0.0,
            terminal_scale=config["terminal_scale"], yaw_feedforward_only=True,
            reference_mode="kinematic_hover_yaw", use_reference_defect=False,
            preview_reference=True,
            attitude_error_max_rad=config.get("attitude_error_max_rad"),
        )
    else:
        with Path(task["model_path"]).open("rb") as stream:
            model = pickle.load(stream)
        config = task["edmd_config"]
        hover = float(model.get("identified_hover_thrust_n", sim.quad.m * sim.quad.g))
        states, _, commands, diagnostics = run_outer_mpc(
            model, sim, reference, len(reference),
            horizon_seconds=config["horizon_seconds"],
            control_horizon_seconds=config["control_horizon_seconds"],
            offboard_period=0.01, iterations=1, r_scale=config["r_scale"],
            rd_scale=config.get("rd_scale", 1.0),
            first_move_rd_scale=config.get("first_move_rd_scale", 0.0),
            command_slew_max_raw=config.get("command_slew_max_raw"),
            thrust_trust_n=config["thrust_trust_n"],
            attitude_trust_rad=config["attitude_trust_rad"], yaw_trust_rad=0.0,
            thrust_feedforward_offset_n=hover - sim.quad.m * sim.quad.g,
            reference_mode="kinematic_hover_yaw", yaw_feedforward_only=True,
            q_position_scale=config["q_position_scale"],
            q_velocity_scale=config["q_velocity_scale"],
            q_attitude_scale=0.0, q_yaw_scale=0.0, q_rate_scale=0.0,
            terminal_scale=config["terminal_scale"],
            attitude_error_max_rad=(
                config.get("attitude_error_max_rad")
                if (config.get("attitude_error_max_rad") is not None and
                    config.get("attitude_error_max_rad") > 0) else None
            ),
        )
    position_ref = np.asarray([point["pos"] for point in reference])
    result = {
        "controller": task["controller"], "family": task["family"],
        "replicate": task["replicate"], "run_index": task["run_index"],
        "aggressiveness_scale": task["aggressiveness_scale"],
        **reference_aggressiveness(reference, sim.quad.g),
        **trajectory_metrics(states, reference, sim.dt),
        "position_vector_rmse_m": float(np.sqrt(np.mean(np.sum(
            (states[:, :3] - position_ref) ** 2, axis=1
        )))),
        **response_activity(states, commands),
        "failed_solves": int(diagnostics.get("failed_solves", sum(
            status not in ("solved", "solved inaccurate")
            for status in diagnostics.get("statuses", ())
        ))),
        "allocator_altered_steps": int(diagnostics["allocator_altered_steps"]),
        "mean_solve_ms": float(diagnostics["mean_solve_ms"]),
        "p95_solve_ms": float(diagnostics.get("p95_solve_ms", 0.0)),
        "p99_solve_ms": float(diagnostics.get("p99_solve_ms", 0.0)),
        "max_solve_ms": float(diagnostics.get("max_solve_ms", 0.0)),
    }
    # Preserve experiment identifiers inside the worker result.  Callers use
    # imap_unordered, so assigning these from task-list order after collection
    # silently corrupts tuning rankings.
    for key in ("candidate_id", "stage", "scenario_id"):
        if key in task:
            result[key] = task[key]
    if task.get("trace_path"):
        trace_path = Path(task["trace_path"])
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trace_path,
            states=states,
            commands=commands,
            reference=reference_state_array(reference, sim.dt, quad=None),
            dt=np.array(sim.dt),
        )
    return result


def summarize(rows, scales, controllers):
    output = []
    for scale in scales:
        for controller in controllers:
            selected = [row for row in rows if (
                row["aggressiveness_scale"] == scale and row["controller"] == controller
            )]
            output.append({
                "aggressiveness_scale": scale, "controller": controller,
                "episodes": len(selected),
                "mean_reference_p95_implied_tilt_deg": float(np.mean([
                    row["reference_p95_implied_tilt_deg"] for row in selected
                ])),
                "mean_position_rmse_m": float(np.mean([
                    row["position_rmse_m"] for row in selected
                ])),
                "mean_position_vector_rmse_m": float(np.mean([
                    row["position_vector_rmse_m"] for row in selected
                ])),
                "mean_velocity_rmse_mps": float(np.mean([
                    row["velocity_rmse_mps"] for row in selected
                ])),
                "mean_actual_p95_tilt_deg": float(np.mean([
                    row["actual_p95_tilt_deg"] for row in selected
                ])),
                "max_actual_peak_tilt_deg": float(np.max([
                    row["actual_peak_tilt_deg"] for row in selected
                ])),
                "failed_solves": int(sum(row["failed_solves"] for row in selected)),
                "allocator_altered_steps": int(sum(
                    row["allocator_altered_steps"] for row in selected
                )),
                "mean_solve_ms": float(np.mean([
                    row["mean_solve_ms"] for row in selected
                ])),
                "p95_sample_solve_ms": float(np.max([
                    row["p95_solve_ms"] for row in selected
                ])),
                "p99_sample_solve_ms": float(np.max([
                    row["p99_solve_ms"] for row in selected
                ])),
                "worst_sample_solve_ms": float(np.max([
                    row["max_solve_ms"] for row in selected
                ])),
            })
    return output


def make_plots(summary, scales, output_dir, controllers):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for controller in controllers:
        selected = [row for row in summary if row["controller"] == controller]
        axes[0].plot(
            scales, [row["mean_position_vector_rmse_m"] for row in selected],
            marker="o", color=COLORS[controller], label=LABELS[controller],
        )
        axes[1].plot(
            [row["mean_reference_p95_implied_tilt_deg"] for row in selected],
            [row["mean_position_vector_rmse_m"] for row in selected],
            marker="o", color=COLORS[controller], label=LABELS[controller],
        )
    axes[0].set_xlabel("Reference aggressiveness scale")
    axes[1].set_xlabel("Mean reference 95th-percentile implied tilt [deg]")
    for axis in axes:
        axis.set_yscale("log"); axis.set_ylabel("Position vector RMSE [m]")
        axis.grid(alpha=.3); axis.legend()
    fig.suptitle("Nominal-model aggressiveness crossover screen")
    fig.tight_layout()
    fig.savefig(output_dir / "aggressiveness_crossover.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "aggressiveness_crossover.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--linear-config", type=Path, required=True)
    parser.add_argument("--edmd-config", type=Path, required=True)
    parser.add_argument(
        "--reactive-config", type=Path,
        help="Validation-selected reactive PID configuration JSON.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scales", default="1.0,1.5,2.0,2.5")
    parser.add_argument("--runs-per-family", type=int, default=1)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--run-index-start", type=int, default=130000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--experiment-role", choices=("screening", "confirmation"),
        default="screening",
        help="Label a preselected, fresh-seed run as confirmation.",
    )
    parser.add_argument(
        "--controllers", default=",".join(CONTROLLERS),
        help=("Comma-separated subset of reactive_pid,fixed_hover_lti,"
              "yaw_scheduled_hover,edmdc."),
    )
    args = parser.parse_args()
    scales = [float(value) for value in args.scales.split(",")]
    controllers = tuple(value.strip() for value in args.controllers.split(",") if value.strip())
    unknown = set(controllers) - set(CONTROLLERS)
    if unknown:
        parser.error(f"Unknown controllers: {sorted(unknown)}")
    linear = json.loads(args.linear_config.read_text())["selected"]["hover_linear"]
    edmd = json.loads(args.edmd_config.read_text())["selected"]["edmdc"]
    reactive_gains = None
    if args.reactive_config is not None:
        reactive_gains = json.loads(args.reactive_config.read_text())["gain_multipliers"]
    tasks = []
    for scale_index, scale in enumerate(scales):
        for family_index, family in enumerate(FAMILIES):
            for replicate in range(args.runs_per_family):
                # Hold the random trajectory realization fixed across scale.
                # Otherwise a trend with aggressiveness is confounded by a
                # different path draw at every level.
                run_index = args.run_index_start + 1000 * family_index + replicate
                for controller in controllers:
                    tasks.append({
                        "controller": controller, "family": family,
                        "replicate": replicate, "run_index": run_index,
                        "aggressiveness_scale": scale,
                        "duration_seconds": args.duration_seconds,
                        "model_path": str(args.model.resolve()),
                        "linear_config": linear, "edmd_config": edmd,
                        "reactive_gains": reactive_gains,
                    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with mp.Pool(min(args.workers, len(tasks))) as pool:
        for row in pool.imap_unordered(run_one, tasks):
            rows.append(row); print(f"completed {len(rows)}/{len(tasks)}", flush=True)
    rows.sort(key=lambda row: (
        row["aggressiveness_scale"], row["family"], row["controller"]
    ))
    write_csv(args.output_dir / "episodes.csv", rows)
    summary = summarize(rows, scales, controllers)
    write_csv(args.output_dir / "summary.csv", summary)
    make_plots(summary, scales, args.output_dir, controllers)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "experiment_role": args.experiment_role,
        "screening_only": args.experiment_role == "screening",
        "nominal_plant": True,
        "reference_information_contract": {
            "reactive_pid": "current position, velocity, and yaw reference only",
            "predictive_controllers": "identical future position, velocity, and yaw preview",
            "equal_preview_between_mpc_controllers": True,
            "reactive_pid_has_future_preview": False,
        },
        "reference_mode": "kinematic_hover_yaw",
        "preview_seconds": 0.5, "scales": scales,
        "run_index_start": args.run_index_start,
        "run_index_end": args.run_index_start + 1000 * (len(FAMILIES) - 1) + args.runs_per_family - 1,
        "model_path": str(args.model.resolve()), "model_sha256": sha256(args.model),
        "linear_config_path": str(args.linear_config.resolve()),
        "linear_config_sha256": sha256(args.linear_config),
        "edmd_config_path": str(args.edmd_config.resolve()),
        "edmd_config_sha256": sha256(args.edmd_config),
        "reactive_config_path": (str(args.reactive_config.resolve()) if args.reactive_config else None),
        "reactive_config_sha256": (sha256(args.reactive_config) if args.reactive_config else None),
        "paired_trajectory_seeds_across_scales": True,
        "families": list(FAMILIES), "runs_per_family": args.runs_per_family,
        "controllers": list(controllers),
        "duration_seconds": args.duration_seconds,
        "runtime_environment": runtime_environment(),
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
