"""Paired moving-target interception for the current attitude-command stack."""

import argparse
import json
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np

from Intercept_comparison import interception_metrics, scenario_family
from aggressiveness_crossover import write_csv
from outer_command_mpc import run_outer_mpc
from outer_ltv_mpc import run as run_linear_mpc
from reactive_pid_tracking import run_reactive_pid
from Simulation import quad_sim
from model_mismatch_utils import make_hidden_plant_sim


FAMILIES = ("straight", "accelerating", "helix", "weaving")
CONTROLLERS = ("PID", "Linear MPC", "EDMDc-MPC")
TRACE_KEYS = {"PID": "pid", "Linear MPC": "linear_mpc", "EDMDc-MPC": "edmd_mpc"}


def qualified_capture_metrics(states, target_position, target_velocity, dt,
                              radius, dwell_seconds, speed_limits=(2.0, 3.0, 5.0)):
    separation = np.linalg.norm(states[:, :3] - target_position, axis=1)
    relative_speed = np.linalg.norm(states[:, 3:6] - target_velocity, axis=1)
    # N consecutive samples span (N - 1) * dt.  Include the sample at the
    # dwell endpoint so that the elapsed interval is at least dwell_seconds.
    dwell_steps = max(1, int(np.ceil(float(dwell_seconds) / dt - 1e-9)) + 1)
    output = {
        "qualified_dwell_s": float(dwell_seconds),
        "qualified_time_reference": "dwell_end",
        "qualified_sample_convention": "elapsed_interval",
    }
    for speed_limit in speed_limits:
        qualified = (separation <= radius) & (relative_speed <= speed_limit)
        start = None
        if len(qualified) >= dwell_steps:
            window = np.convolve(
                qualified.astype(int), np.ones(dwell_steps, dtype=int), mode="valid"
            )
            matches = np.flatnonzero(window == dwell_steps)
            if matches.size:
                start = int(matches[0])
        suffix = str(speed_limit).replace(".", "p")
        output[f"qualified_capture_speed_{suffix}_mps"] = int(start is not None)
        # Qualification is established at the END of the dwell window, not
        # when the first eligible sample enters the capture sphere.
        output[f"qualified_capture_time_speed_{suffix}_mps_s"] = (
            float((start + dwell_steps - 1) * dt)
            if start is not None else np.nan
        )
    return output


def run_scenario(task):
    rng = np.random.default_rng(task["seed"])
    target = scenario_family(task["family"], rng)
    dt = float(quad_sim.dt)
    steps = int(round(task["tmax"] / dt)) + 1
    times = np.arange(steps, dtype=float) * dt
    current_reference = [target.reference(t) for t in times]
    predictive_reference = [target.reference(t + task["lead"]) for t in times]
    initial_yaw = float(current_reference[0].get("yaw", 0.0))
    target_position = np.asarray([target.position(t) for t in times])
    target_velocity = np.asarray([target.velocity(t) for t in times])
    target_acceleration = np.asarray([target.acceleration(t) for t in times])

    requested_controllers = tuple(task.get("controllers", CONTROLLERS))
    model = None
    if "EDMDc-MPC" in requested_controllers:
        with Path(task["model_path"]).open("rb") as stream:
            model = pickle.load(stream)
    linear = task["linear_config"]
    edmd = task["edmd_config"]
    traces = {}

    def plant_sim():
        plant_id = task.get("plant_id", "nominal")
        return quad_sim() if plant_id == "nominal" else make_hidden_plant_sim(plant_id)

    if "PID" in requested_controllers:
        sim, states, applied, _, commands = run_reactive_pid(
            current_reference, task["reactive_gains"], initial_yaw=initial_yaw,
            acceleration_limit_scale=task.get("pid_acceleration_limit_scale", 1.0),
            sim=plant_sim(),
        )
        traces["PID"] = (states, applied, commands, {
            "mean_solve_ms": 0.0, "p95_solve_ms": 0.0,
            "failed_solves": 0, "allocator_altered_steps": 0,
        })

    if "Linear MPC" in requested_controllers:
        linear_sim = plant_sim()
        states, applied, commands, diagnostics = run_linear_mpc(
        predictive_reference, steps,
        linear["horizon_seconds"], linear["control_horizon_seconds"],
        linear["r_scale"], task["linearization_mode"],
        linear["thrust_trust_n"], linear["attitude_trust_rad"], 0.0,
        sim=linear_sim, model_sim=quad_sim(),
        q_position_scale=linear["q_position_scale"],
        q_velocity_scale=linear["q_velocity_scale"],
        q_attitude_scale=0.0, q_yaw_scale=0.0, q_rate_scale=0.0,
        terminal_scale=linear["terminal_scale"], yaw_feedforward_only=True,
        reference_mode="kinematic_hover_yaw",
        use_reference_defect=task["linear_use_reference_defect"],
        preview_reference=task["linear_preview_reference"],
        initial_yaw=initial_yaw,
        attitude_error_max_rad=linear.get("attitude_error_max_rad"),
    )
        traces["Linear MPC"] = (states, applied, commands, diagnostics)

    if "EDMDc-MPC" in requested_controllers:
        edmd_sim = plant_sim()
        hover = float(model.get(
        "identified_hover_thrust_n",
        edmd_sim.controller_PX4.quad.m * edmd_sim.controller_PX4.quad.g,
    ))
        states, applied, commands, diagnostics = run_outer_mpc(
        model, edmd_sim, predictive_reference, steps,
        horizon_seconds=edmd["horizon_seconds"],
        control_horizon_seconds=edmd["control_horizon_seconds"],
        offboard_period=0.01, iterations=1, r_scale=edmd["r_scale"],
        rd_scale=edmd.get("rd_scale", 1.0),
        first_move_rd_scale=edmd.get("first_move_rd_scale", 0.0),
        command_slew_max_raw=edmd.get("command_slew_max_raw"),
        thrust_trust_n=edmd["thrust_trust_n"],
        attitude_trust_rad=edmd["attitude_trust_rad"], yaw_trust_rad=0.0,
        thrust_feedforward_offset_n=(
            hover
            - edmd_sim.controller_PX4.quad.m * edmd_sim.controller_PX4.quad.g
        ),
        reference_mode="kinematic_hover_yaw", yaw_feedforward_only=True,
        q_position_scale=edmd["q_position_scale"],
        q_velocity_scale=edmd["q_velocity_scale"],
        q_attitude_scale=0.0, q_yaw_scale=0.0, q_rate_scale=0.0,
        terminal_scale=edmd["terminal_scale"],
        attitude_error_max_rad=(
            edmd.get("attitude_error_max_rad")
            if (edmd.get("attitude_error_max_rad") is not None
                and edmd.get("attitude_error_max_rad") > 0) else None
        ),
        initial_yaw=initial_yaw,
    )
        traces["EDMDc-MPC"] = (states, applied, commands, diagnostics)

    rows = []
    for controller in requested_controllers:
        states, applied, commands, diagnostics = traces[controller]
        metrics = interception_metrics(states, target, dt, task["capture_radius"])
        qualified = qualified_capture_metrics(
            states, target_position, target_velocity, dt,
            task["capture_radius"], task["qualified_dwell_s"],
        )
        rows.append({
            "family": task["family"], "replicate": task["replicate"],
            "scenario_seed": task["seed"], "controller": controller,
            "plant_id": task.get("plant_id", "nominal"),
            "tmax_s": task["tmax"], "capture_radius_m": task["capture_radius"],
            "intercept_lead_s": task["lead"], "controller_period_s": dt,
            "initial_yaw_rad": initial_yaw,
            "target_peak_speed_mps": float(np.max(np.linalg.norm(target_velocity, axis=1))),
            "target_peak_acceleration_mps2": float(np.max(np.linalg.norm(target_acceleration, axis=1))),
            **metrics,
            **qualified,
            "failure_aware_capture_time_s": (
                float(metrics["capture_time"]) if metrics["captured"]
                else float(task["tmax"])
            ),
            "mean_solve_ms": float(diagnostics.get("mean_solve_ms", 0.0)),
            "p95_solve_ms": float(diagnostics.get("p95_solve_ms", 0.0)),
            "failed_solves": int(diagnostics.get("failed_solves", 0)),
            "allocator_altered_steps": int(diagnostics.get("allocator_altered_steps", 0)),
        })
        for key in ("candidate_id", "stage"):
            if key in task:
                rows[-1][key] = task[key]

    if task["save_trace"]:
        trace_dir = Path(task["trace_dir"])
        trace_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "time": times, "target_position": target_position,
            "target_velocity": target_velocity,
            "target_reference": np.asarray([
                [*point["pos"], *point["vel"], *point["acc"],
                 point["yaw"], point["yaw_rate"]]
                for point in current_reference
            ], dtype=float),
        }
        for controller in requested_controllers:
            states, applied, _, diagnostics = traces[controller]
            key = TRACE_KEYS[controller]
            payload[f"{key}_states"] = states
            payload[f"{key}_inputs"] = applied
            payload[f"{key}_solve_times"] = np.asarray(
                diagnostics.get("solve_times", ()), dtype=float
            )
        path = trace_dir / f"test_{task['family']}_rep{task['replicate']}.npz"
        np.savez_compressed(path, **payload)
        path.with_suffix(".json").write_text(json.dumps({
            "split": "locked_test", "family": task["family"],
            "replicate": task["replicate"], "scenario_seed": task["seed"],
            "dt": dt, "tmax": task["tmax"],
            "capture_radius": task["capture_radius"],
            "qualified_dwell_s": task["qualified_dwell_s"],
            "qualified_sample_convention": "elapsed_interval",
            "intercept_lead": task["lead"], "controllers": list(requested_controllers),
            "initial_yaw": "current target yaw for every controller",
            "predictive_reference": "target state shifted by intercept_lead",
        }, indent=2) + "\n")
    return rows


def aggregate(rows):
    output = []
    for controller in CONTROLLERS:
        selected = [row for row in rows if row["controller"] == controller]
        captured = [row for row in selected if row["captured"]]
        output.append({
            "controller": controller, "scenarios": len(selected),
            "captures": len(captured),
            "capture_rate": float(np.mean([row["captured"] for row in selected])),
            "successful_capture_mean_s": (
                float(np.mean([row["capture_time"] for row in captured]))
                if captured else np.nan
            ),
            "failure_aware_capture_mean_s": float(np.mean([
                row["failure_aware_capture_time_s"] for row in selected
            ])),
            "minimum_separation_mean_m": float(np.mean([
                row["minimum_separation"] for row in selected
            ])),
            "successful_relative_speed_mean_mps": (
                float(np.mean([row["capture_relative_speed"] for row in captured]))
                if captured else np.nan
            ),
            "qualified_captures_2_mps": int(sum(
                row["qualified_capture_speed_2p0_mps"] for row in selected
            )),
            "qualified_captures_3_mps": int(sum(
                row["qualified_capture_speed_3p0_mps"] for row in selected
            )),
            "qualified_captures_5_mps": int(sum(
                row["qualified_capture_speed_5p0_mps"] for row in selected
            )),
            "mean_solve_ms": float(np.mean([row["mean_solve_ms"] for row in selected])),
            "failed_solves": int(sum(row["failed_solves"] for row in selected)),
            "allocator_altered_steps": int(sum(
                row["allocator_altered_steps"] for row in selected
            )),
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--linear-config", type=Path, required=True)
    parser.add_argument("--edmd-config", type=Path, required=True)
    parser.add_argument("--reactive-config", type=Path, required=True)
    parser.add_argument(
        "--tuned-config", type=Path,
        help="Frozen interception tuning output; overrides the three base configs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs-per-family", type=int, default=10)
    parser.add_argument("--trace-replicates", type=int, default=2)
    parser.add_argument("--tmax", type=float, default=20.0)
    parser.add_argument("--capture-radius", type=float, default=0.75)
    parser.add_argument("--intercept-lead", type=float, default=0.8)
    parser.add_argument("--qualified-dwell", type=float, default=0.2)
    parser.add_argument("--base-seed", type=int, default=610000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--plant-ids", default="nominal",
        help="Comma-separated hidden plant identities, or nominal.",
    )
    args = parser.parse_args()
    linear_saved = json.loads(args.linear_config.read_text())
    edmd_saved = json.loads(args.edmd_config.read_text())
    reactive_saved = json.loads(args.reactive_config.read_text())
    linear_config = linear_saved["selected"]["hover_linear"]
    edmd_config = edmd_saved["selected"]["edmdc"]
    reactive_gains = reactive_saved["gain_multipliers"]
    pid_acceleration_limit_scale = 1.0
    if args.tuned_config is not None:
        tuned = json.loads(args.tuned_config.read_text())
        selected = tuned["selected"]
        linear_config = selected["Linear MPC"]["linear_config"]
        edmd_config = selected["EDMDc-MPC"]["edmd_config"]
        reactive_gains = selected["PID"]["reactive_gains"]
        pid_acceleration_limit_scale = selected["PID"][
            "pid_acceleration_limit_scale"
        ]
    plant_ids = tuple(value.strip() for value in args.plant_ids.split(",") if value.strip())
    tasks = []
    for plant_id in plant_ids:
        for family_index, family in enumerate(FAMILIES):
            for replicate in range(args.runs_per_family):
                tasks.append({
                "family": family, "replicate": replicate,
                "seed": args.base_seed + 1000 * family_index + replicate,
                "plant_id": plant_id,
                "tmax": args.tmax, "capture_radius": args.capture_radius,
                "lead": args.intercept_lead,
                "qualified_dwell_s": args.qualified_dwell,
                "model_path": str(args.model.resolve()),
                "linear_config": linear_config,
                "linearization_mode": linear_saved.get(
                    "linearization_mode", "yaw_scheduled_hover"
                ),
                "linear_use_reference_defect": linear_saved.get(
                    "linear_use_reference_defect", False
                ),
                "linear_preview_reference": linear_saved.get(
                    "linear_preview_reference", True
                ),
                "edmd_config": edmd_config,
                "reactive_gains": reactive_gains,
                "pid_acceleration_limit_scale": pid_acceleration_limit_scale,
                "save_trace": replicate < args.trace_replicates,
                "trace_dir": str((args.output_dir / "traces").resolve()),
                })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with mp.Pool(min(args.workers, len(tasks))) as pool:
        for result in pool.imap_unordered(run_scenario, tasks):
            rows.extend(result)
            print(f"completed {len(rows)//len(CONTROLLERS)}/{len(tasks)} scenarios", flush=True)
    rows.sort(key=lambda row: (
        row["plant_id"], row["family"], row["replicate"], row["controller"]
    ))
    write_csv(args.output_dir / "episodes.csv", rows)
    summary = aggregate(rows)
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "experiment": "locked_attitude_command_interception_comparison",
        "nominal_plant": plant_ids == ("nominal",), "controller_rate_hz": 100.0,
        "families": list(FAMILIES), "runs_per_family": args.runs_per_family,
        "base_seed": args.base_seed, "tmax_s": args.tmax,
        "capture_radius_m": args.capture_radius,
        "intercept_lead_s": args.intercept_lead,
        "qualified_dwell_s": args.qualified_dwell,
        "qualified_time_reference": "dwell_end",
        "qualified_sample_convention": "elapsed_interval",
        "qualified_speed_limits_mps": [2.0, 3.0, 5.0],
        "plant_ids": list(plant_ids),
        "initial_yaw": "current target yaw for every controller",
        "pid_reference": "current target state (reactive)",
        "predictive_controller_reference": "target state shifted by lead",
        "controllers": list(CONTROLLERS),
        "model": str(args.model.resolve()),
        "linear_config": str(args.linear_config.resolve()),
        "edmd_config": str(args.edmd_config.resolve()),
        "reactive_config": str(args.reactive_config.resolve()),
        "tuned_config": (
            str(args.tuned_config.resolve()) if args.tuned_config is not None else None
        ),
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
