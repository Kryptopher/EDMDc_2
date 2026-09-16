"""Evaluate and plot the attitude-command capacity and learning-curve study."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--horizon-seconds", type=float, default=2.0)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((args.study_dir / "manifest.json").read_text())
    evaluation_dir = args.study_dir / "heldout_evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_evaluation:
        command = [
            sys.executable,
            str((Path(__file__).parent / "evaluate_edmdc_training.py").resolve()),
            "--data", str(args.data.resolve()),
            "--output", str(evaluation_dir.resolve()),
            "--horizons", "0.1,0.5,1,2",
            "--continuous-prefixes", "1,2",
            "--validation-indices", ",".join(map(str, manifest["validation_indices"])),
            "--test-indices", ",".join(map(str, manifest["test_indices"])),
        ]
        for result in manifest["results"]:
            command.extend(["--model", f"{result['name']}={result['model']}"])
        environment = os.environ.copy()
        environment["MPLBACKEND"] = "Agg"
        subprocess.run(command, cwd=Path(__file__).parent, env=environment, check=True)

    aggregate = read_csv(evaluation_dir / "rolling_aggregate.csv")
    lookup = {
        row["model"]: row for row in aggregate
        if row["split"] == "test"
        and np.isclose(float(row["horizon_seconds"]), args.horizon_seconds)
    }
    rows = []
    for result in manifest["results"]:
        metrics = lookup[result["name"]]
        rows.append({
            **result,
            "heldout_horizon_seconds": args.horizon_seconds,
            "heldout_position_rmse_m": float(metrics["rmse_position"]),
            "heldout_velocity_rmse_mps": float(metrics["rmse_velocity"]),
            "heldout_roll_pitch_rmse_rad": float(metrics["rmse_roll_pitch"]),
            "heldout_roll_pitch_rate_rmse_radps": float(
                metrics["rmse_roll_pitch_rates"]
            ),
            "heldout_yaw_rmse_rad": float(metrics["rmse_yaw"]),
            "heldout_yaw_rate_rmse_radps": float(metrics["rmse_yaw_rate"]),
            "heldout_worst_window_position_rmse_m": float(
                metrics["worst_window_position_rmse"]
            ),
            "heldout_divergent_windows": int(float(metrics["divergent_windows"])),
        })
    write_csv(args.study_dir / "heldout_summary.csv", rows)

    capacity = sorted(
        [row for row in rows if row["train_fraction"] == 1.0],
        key=lambda row: row["observable_count"],
    )
    learning = sorted(
        [row for row in rows if row["observable_set"] == "full56"],
        key=lambda row: row["training_transitions"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].plot(
        [row["observable_count"] for row in capacity],
        [row["heldout_position_rmse_m"] for row in capacity],
        "o-", color="#0072B2",
    )
    for row in capacity:
        axes[0].annotate(
            row["observable_set"],
            (row["observable_count"], row["heldout_position_rmse_m"]),
            xytext=(4, 5), textcoords="offset points", fontsize=8,
        )
    axes[0].set_xlabel("Observable dimension")
    axes[0].set_ylabel(f"Held-out {args.horizon_seconds:g} s position RMSE [m]")
    axes[0].set_title("Dictionary-capacity ablation")

    axes[1].plot(
        [row["training_transitions"] for row in learning],
        [row["heldout_position_rmse_m"] for row in learning],
        "o-", color="#D55E00",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Training transitions")
    axes[1].set_ylabel(f"Held-out {args.horizon_seconds:g} s position RMSE [m]")
    axes[1].set_title("Learning curve (fixed 56 observables)")
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.study_dir / "capacity_learning_curve.png", dpi=240)
    fig.savefig(args.study_dir / "capacity_learning_curve.pdf")
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
