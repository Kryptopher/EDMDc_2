"""Plot the paired aggressive PID/hover-linear/EDMDc tracking comparison."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aggressiveness_crossover import LABELS, COLORS


ORDER = ("reactive_pid", "yaw_scheduled_hover", "edmdc")
FAMILIES = ("helix", "figure8", "lissajous", "waypoint")


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.episodes.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    grouped = {
        controller: [row for row in rows if row["controller"] == controller]
        for controller in ORDER
    }
    summary = []
    for controller in ORDER:
        selected = grouped[controller]
        position = np.asarray([float(row["position_rmse_m"]) for row in selected])
        velocity = np.asarray([float(row["velocity_rmse_mps"]) for row in selected])
        yaw = np.asarray([float(row["yaw_rmse_rad"]) for row in selected])
        summary.append({
            "controller": controller,
            "episodes": len(selected),
            "position_mean_m": float(np.mean(position)),
            "position_median_m": float(np.median(position)),
            "position_p90_m": percentile(position, 90),
            "position_worst_m": float(np.max(position)),
            "velocity_mean_mps": float(np.mean(velocity)),
            "yaw_mean_rad": float(np.mean(yaw)),
            "mean_solve_ms": float(np.mean([
                float(row["mean_solve_ms"]) for row in selected
            ])),
            "worst_episode_p95_solve_ms": float(np.max([
                float(row.get("p95_solve_ms", 0.0)) for row in selected
            ])),
            "worst_episode_p99_solve_ms": float(np.max([
                float(row.get("p99_solve_ms", 0.0)) for row in selected
            ])),
            "worst_sample_solve_ms": float(np.max([
                float(row.get("max_solve_ms", 0.0)) for row in selected
            ])),
        })
    with (args.output_dir / "tracking_comparison_statistics.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    data = [[float(row["position_rmse_m"]) for row in grouped[c]] for c in ORDER]
    short_labels = ("Reactive PID", "Hover-linear MPC", "EDMDc-MPC")
    boxes = axes[0, 0].boxplot(data, tick_labels=short_labels,
                              showfliers=True, patch_artist=True)
    axes[0, 0].tick_params(axis="x", labelsize=9)
    for patch, controller in zip(boxes["boxes"], ORDER):
        patch.set_facecolor(COLORS[controller]); patch.set_alpha(0.7)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Position RMSE [m]")
    axes[0, 0].set_title("Distribution over 40 paired trajectories")

    for controller in ORDER:
        values = np.sort([float(row["position_rmse_m"]) for row in grouped[controller]])
        probability = np.arange(1, len(values) + 1) / len(values)
        axes[0, 1].step(values, probability, where="post", color=COLORS[controller],
                        label=LABELS[controller], linewidth=2)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Position RMSE [m]")
    axes[0, 1].set_ylabel("Empirical cumulative probability")
    axes[0, 1].set_title("Tracking-error empirical CDF")
    axes[0, 1].legend(fontsize=8)

    width = 0.24
    x = np.arange(len(FAMILIES))
    for offset, controller in enumerate(ORDER):
        means = []
        for family in FAMILIES:
            values = [float(row["position_rmse_m"]) for row in grouped[controller]
                      if row["family"] == family]
            means.append(np.mean(values))
        axes[1, 0].bar(x + (offset - 1) * width, means, width,
                       color=COLORS[controller], label=LABELS[controller])
    axes[1, 0].set_xticks(x, FAMILIES)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("Mean position RMSE [m]")
    axes[1, 0].set_title("Trajectory-family breakdown")
    axes[1, 0].legend(fontsize=8)

    metric_names = ("Mean", "Median", "P90", "Worst")
    metric_keys = ("position_mean_m", "position_median_m",
                   "position_p90_m", "position_worst_m")
    x = np.arange(len(metric_names))
    for offset, (controller, item) in enumerate(zip(ORDER, summary)):
        axes[1, 1].bar(
            x + (offset - 1) * width, [item[key] for key in metric_keys], width,
            color=COLORS[controller], label=LABELS[controller],
        )
    axes[1, 1].set_xticks(x, metric_names)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("Position RMSE [m]")
    axes[1, 1].set_title("Central and tail performance")

    for axis in axes.flat:
        axis.grid(alpha=0.25, which="both", axis="y")
    fig.suptitle("Nominal aggressive tracking at 100 Hz (1.75x demand)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output_dir / "tracking_comparison_summary.png", dpi=220)
    fig.savefig(args.output_dir / "tracking_comparison_summary.pdf")
    plt.close(fig)

    paired = {}
    for row in rows:
        key = (row["family"], int(row["run_index"]))
        paired.setdefault(key, {})[row["controller"]] = float(row["position_rmse_m"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, baseline in zip(axes, ("reactive_pid", "yaw_scheduled_hover")):
        x_values = [item[baseline] for item in paired.values()]
        y_values = [item["edmdc"] for item in paired.values()]
        axis.scatter(x_values, y_values, color=COLORS["edmdc"], alpha=0.8)
        limits = [min(x_values + y_values) * 0.75, max(x_values + y_values) * 1.25]
        axis.plot(limits, limits, "k--", linewidth=1.5)
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.set_xlim(limits); axis.set_ylim(limits)
        axis.set_xlabel(f"{LABELS[baseline]} RMSE [m]")
        axis.set_ylabel("EDMDc-MPC RMSE [m]")
        wins = sum(y < x for x, y in zip(x_values, y_values))
        axis.set_title(f"EDMDc better on {wins}/{len(y_values)} paired runs")
        axis.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(args.output_dir / "paired_position_rmse.png", dpi=220)
    fig.savefig(args.output_dir / "paired_position_rmse.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
