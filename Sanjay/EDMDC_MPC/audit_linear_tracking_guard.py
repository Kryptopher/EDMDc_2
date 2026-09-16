"""Audit whether the hover-linear tracking tail is caused by its attitude guard.

The frozen confirmation used a validation-selected 0.12 rad state-to-command
attitude-error bound.  This script keeps the model, weights, horizon, plant,
preview, trajectories, and seeds fixed while varying only that bound.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from outer_command_mpc import trajectory_metrics
from outer_ltv_mpc import run as run_linear_mpc
from publication_scenarios import tracking_reference
from Simulation import quad_sim


FAMILIES = ("helix", "figure8", "lissajous", "waypoint")
GUARDS = {
    "frozen_0p12": 0.12,
    "guard_0p25": 0.25,
    "guard_0p40": 0.40,
    "unguarded": None,
}


def run_case(task):
    family, replicate, run_index, duration, speed_scale, config = task
    reference = tracking_reference(
        family, run_index, duration, speed_scale=speed_scale
    )
    rows = []
    for label, guard in GUARDS.items():
        sim = quad_sim()
        states, _, commands, diagnostics = run_linear_mpc(
            reference, len(reference), config["horizon_seconds"],
            config["control_horizon_seconds"], config["r_scale"],
            "yaw_scheduled_hover", config["thrust_trust_n"],
            config["attitude_trust_rad"], 0.0, sim=sim,
            model_sim=quad_sim(),
            q_position_scale=config["q_position_scale"],
            q_velocity_scale=config["q_velocity_scale"],
            q_attitude_scale=0.0, q_yaw_scale=0.0, q_rate_scale=0.0,
            terminal_scale=config["terminal_scale"],
            yaw_feedforward_only=True,
            reference_mode="kinematic_hover_yaw",
            use_reference_defect=False, preview_reference=True,
            attitude_error_max_rad=guard,
        )
        metrics = trajectory_metrics(states, reference, sim.dt)
        physical_tilt = np.max(np.abs(states[:, 6:8]), axis=1)
        command_tilt = np.max(np.abs(commands[:, 1:3]), axis=1)
        attitude_error = np.max(
            np.abs(commands[:, 1:3] - states[:, 6:8]), axis=1
        )
        rows.append({
            "family": family,
            "replicate": replicate,
            "run_index": run_index,
            "guard_label": label,
            "attitude_error_max_rad": "" if guard is None else guard,
            **metrics,
            "peak_physical_tilt_deg": float(np.degrees(np.max(physical_tilt))),
            "peak_command_tilt_deg": float(np.degrees(np.max(command_tilt))),
            "peak_attitude_error_deg": float(np.degrees(np.max(attitude_error))),
            "failed_solves": int(diagnostics.get("failed_solves", 0)),
            "allocator_altered_steps": int(
                diagnostics.get("allocator_altered_steps", 0)
            ),
            "mean_solve_ms": float(diagnostics.get("mean_solve_ms", np.nan)),
            "p95_solve_ms": float(diagnostics.get("p95_solve_ms", np.nan)),
        })
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = []
    for label, guard in GUARDS.items():
        selected = [row for row in rows if row["guard_label"] == label]
        values = np.asarray([row["position_rmse_m"] for row in selected])
        summary.append({
            "guard_label": label,
            "attitude_error_max_rad": guard,
            "episodes": len(selected),
            "mean_position_rmse_m": float(np.mean(values)),
            "median_position_rmse_m": float(np.median(values)),
            "p90_position_rmse_m": float(np.percentile(values, 90)),
            "worst_position_rmse_m": float(np.max(values)),
            "runs_over_1m": int(np.sum(values > 1.0)),
            "failed_solves": int(sum(row["failed_solves"] for row in selected)),
            "allocator_altered_steps": int(sum(
                row["allocator_altered_steps"] for row in selected
            )),
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs-per-family", type=int, default=10)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--speed-scale", type=float, default=1.75)
    parser.add_argument("--run-index-start", type=int, default=750000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    saved = json.loads(args.linear_config.read_text())
    config = saved["selected"]["hover_linear"]
    tasks = []
    for family_index, family in enumerate(FAMILIES):
        for replicate in range(args.runs_per_family):
            tasks.append((
                family, replicate,
                args.run_index_start + 1000 * family_index + replicate,
                args.duration_seconds, args.speed_scale, config,
            ))

    rows = []
    with mp.Pool(processes=min(args.workers, len(tasks))) as pool:
        for index, result in enumerate(pool.imap_unordered(run_case, tasks), 1):
            rows.extend(result)
            print(f"completed {index}/{len(tasks)} paired cases", flush=True)
    rows.sort(key=lambda row: (
        row["family"], row["replicate"], row["guard_label"]
    ))
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "episodes.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    manifest = {
        "purpose": __doc__,
        "linear_config": str(args.linear_config.resolve()),
        "configuration": config,
        "families": list(FAMILIES),
        "runs_per_family": args.runs_per_family,
        "duration_seconds": args.duration_seconds,
        "speed_scale": args.speed_scale,
        "run_index_start": args.run_index_start,
        "single_changed_factor": "attitude_error_max_rad",
        "summary": summary,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
