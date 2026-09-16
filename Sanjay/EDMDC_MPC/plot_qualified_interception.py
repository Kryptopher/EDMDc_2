"""Plot physically qualified interception metrics from an episode CSV."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORDER = ("PID", "Linear MPC", "EDMDc-MPC")
FAMILIES = ("straight", "accelerating", "helix", "weaving")
COLORS = {"PID": "#6B7280", "Linear MPC": "#1F77B4", "EDMDc-MPC": "#D55E00"}


def failure_aware(row, speed, tmax):
    key = f"qualified_capture_time_speed_{speed}p0_mps_s"
    value = float(row[key])
    if not np.isfinite(value):
        return tmax
    if row.get("qualified_time_reference") == "dwell_end":
        return value
    # Saved confirmation CSVs timestamped the beginning of the dwell window.
    # Display the physical completion time without overwriting frozen raw data.
    dt = float(row["controller_period_s"])
    dwell_steps = max(1, int(np.ceil(float(row["qualified_dwell_s"]) / dt)))
    return value + (dwell_steps - 1) * dt


def plot_trace_qualification(trace_dir, output_dir, replicate):
    """Show first sphere entry and completed low-speed dwell separately."""
    fig, axes = plt.subplots(len(FAMILIES), 2, figsize=(12.5, 10), sharex="row")
    plotted = 0
    for family, (distance_axis, speed_axis) in zip(FAMILIES, axes):
        path = trace_dir / f"test_{family}_rep{replicate}.npz"
        if not path.is_file():
            continue
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(path) as stored:
            data = {key: stored[key] for key in stored.files}
        time = data["time"]
        dt = float(metadata["dt"])
        radius = float(metadata["capture_radius"])
        dwell = float(metadata.get("qualified_dwell_s", 0.2))
        dwell_steps = max(1, int(np.ceil(dwell / dt)))
        target_position = data["target_position"]
        target_velocity = data["target_velocity"]
        plotted += 1
        for controller, key in (("PID", "pid"), ("Linear MPC", "linear_mpc"),
                                ("EDMDc-MPC", "edmd_mpc")):
            state_key = f"{key}_states"
            if state_key not in data:
                continue
            state = data[state_key]
            separation = np.linalg.norm(state[:, :3] - target_position, axis=1)
            relative_speed = np.linalg.norm(state[:, 3:6] - target_velocity, axis=1)
            first_hits = np.flatnonzero(separation <= radius)
            first_index = int(first_hits[0]) if first_hits.size else None
            eligible = (separation <= radius) & (relative_speed <= 3.0)
            qualified_index = None
            if len(eligible) >= dwell_steps:
                windows = np.convolve(
                    eligible.astype(int), np.ones(dwell_steps, dtype=int), mode="valid"
                )
                starts = np.flatnonzero(windows == dwell_steps)
                if starts.size:
                    qualified_index = int(starts[0] + dwell_steps - 1)
            color = COLORS[controller]
            label = (f"{controller}: first {time[first_index]:.2f}s, "
                     f"qualified {time[qualified_index]:.2f}s"
                     if qualified_index is not None else
                     f"{controller}: first {time[first_index]:.2f}s, no qualification"
                     if first_index is not None else f"{controller}: no entry")
            distance_axis.plot(time, separation, color=color, label=label)
            speed_axis.plot(time, relative_speed, color=color)
            if first_index is not None:
                for axis, values in ((distance_axis, separation), (speed_axis, relative_speed)):
                    axis.scatter(time[first_index], values[first_index], marker="^",
                                 facecolors="none", edgecolors=color, s=48, zorder=5)
            if qualified_index is not None:
                for axis, values in ((distance_axis, separation), (speed_axis, relative_speed)):
                    axis.scatter(time[qualified_index], values[qualified_index],
                                 marker="o", color=color, s=30, zorder=6)
        distance_axis.axhline(radius, color="black", linestyle="--", linewidth=0.9)
        speed_axis.axhline(3.0, color="black", linestyle="--", linewidth=0.9)
        distance_axis.set_ylabel(f"{family.title()}\nSeparation [m]")
        speed_axis.set_ylabel("Relative speed [m/s]")
        distance_axis.legend(fontsize=6.8, loc="upper right")
        distance_axis.grid(alpha=0.25)
        speed_axis.grid(alpha=0.25)
    if not plotted:
        plt.close(fig)
        raise FileNotFoundError(f"No replicate-{replicate} traces in {trace_dir}")
    for axis in axes[-1]:
        axis.set_xlabel("Time [s]")
    fig.suptitle(
        "First entry (open triangle) vs low-speed dwell completion (filled circle)\n"
        "Qualified capture: separation ≤0.75 m, relative speed ≤3 m/s for 0.2 s"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / f"qualified_interception_traces_rep{replicate}.png", dpi=250)
    fig.savefig(output_dir / f"qualified_interception_traces_rep{replicate}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path,
                        help="Optional saved traces for first-entry versus qualified-capture plots")
    parser.add_argument("--replicate", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.episodes.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    tmax = float(rows[0]["tmax_s"])

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    qualified_times = [
        [failure_aware(row, 3, tmax) for row in rows if row["controller"] == name]
        for name in ORDER
    ]
    boxes = axes[0].boxplot(qualified_times, tick_labels=ORDER, patch_artist=True)
    for patch, name in zip(boxes["boxes"], ORDER):
        patch.set_facecolor(COLORS[name]); patch.set_alpha(0.8)
    axes[0].set_ylabel("Failure-aware time [s]")
    axes[0].set_title("Qualified capture: 3 m/s, 0.2 s dwell")

    x = np.arange(len(ORDER)); width = 0.23
    for offset, speed in enumerate((2, 3, 5)):
        rates = [
            100 * np.mean([
                int(row[f"qualified_capture_speed_{speed}p0_mps"])
                for row in rows if row["controller"] == name
            ]) for name in ORDER
        ]
        axes[1].bar(x + (offset - 1) * width, rates, width, label=f"{speed} m/s")
    axes[1].set_xticks(x, ORDER); axes[1].set_ylim(0, 110)
    axes[1].set_ylabel("Qualified capture rate [%]")
    axes[1].set_title("Speed-limit sensitivity")
    axes[1].legend(title="Relative-speed limit")

    contact = [
        [float(row["capture_relative_speed"]) for row in rows
         if row["controller"] == name and int(row["captured"])]
        for name in ORDER
    ]
    boxes = axes[2].boxplot(contact, tick_labels=ORDER, patch_artist=True)
    for patch, name in zip(boxes["boxes"], ORDER):
        patch.set_facecolor(COLORS[name]); patch.set_alpha(0.8)
    axes[2].axhline(3, color="black", linestyle="--", linewidth=1, label="3 m/s limit")
    axes[2].set_ylabel("Relative speed [m/s]")
    axes[2].set_title("Speed at first sphere entry")
    axes[2].legend()
    for axis in axes:
        axis.tick_params(axis="x", rotation=15); axis.grid(alpha=0.25, axis="y")
    fig.suptitle("Moving-target interception confirmation (40 targets, 100 Hz; dwell-end times)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.output_dir / "qualified_interception_overall.png", dpi=250)
    fig.savefig(args.output_dir / "qualified_interception_overall.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(FAMILIES)); width = 0.24
    for offset, name in enumerate(ORDER):
        means, rates = [], []
        for family in FAMILIES:
            group = [r for r in rows if r["controller"] == name and r["family"] == family]
            means.append(np.mean([failure_aware(r, 3, tmax) for r in group]))
            rates.append(100 * np.mean([int(r["qualified_capture_speed_3p0_mps"]) for r in group]))
        position = x + (offset - 1) * width
        axes[0].bar(position, means, width, color=COLORS[name], label=name)
        axes[1].bar(position, rates, width, color=COLORS[name], label=name)
    axes[0].set_ylabel("Failure-aware qualified time [s]")
    axes[0].set_title("Qualified time by target family")
    axes[1].set_ylabel("Qualified capture rate [%]")
    axes[1].set_ylim(0, 110); axes[1].set_title("Qualified rate by target family")
    for axis in axes:
        axis.set_xticks(x, FAMILIES); axis.grid(alpha=0.25, axis="y")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "qualified_interception_by_family.png", dpi=250)
    fig.savefig(args.output_dir / "qualified_interception_by_family.pdf")
    plt.close(fig)
    if args.trace_dir is not None:
        plot_trace_qualification(args.trace_dir, args.output_dir, args.replicate)


if __name__ == "__main__":
    main()
