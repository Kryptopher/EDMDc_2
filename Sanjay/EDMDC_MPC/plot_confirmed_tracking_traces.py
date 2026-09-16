"""Reproduce and plot representative cases from the fresh tracking confirmation."""

import argparse
import csv
import json
import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aggressiveness_crossover import FAMILIES, run_one


CONTROLLERS = ("reactive_pid", "yaw_scheduled_hover", "edmdc")
LABELS = {
    "reactive_pid": "Reactive PID",
    "yaw_scheduled_hover": "Hover-linear MPC",
    "edmdc": "EDMDc-MPC",
}
COLORS = {
    "reactive_pid": "#009E73",
    "yaw_scheduled_hover": "#0072B2",
    "edmdc": "#D55E00",
}
STATE_LABELS = (
    "x [m]", "y [m]", "z [m]", "vx [m/s]", "vy [m/s]", "vz [m/s]",
    "roll [rad]", "pitch [rad]", "yaw [rad]",
    "p [rad/s]", "q [rad/s]", "r [rad/s]",
)


def plot_case(family, run_index, output_dir, metrics):
    traces = {}
    for controller in CONTROLLERS:
        with np.load(output_dir / family / f"{controller}.npz") as data:
            traces[controller] = {name: data[name].copy() for name in data.files}
    reference = traces["edmdc"]["reference"]
    times = np.arange(len(reference)) * float(traces["edmdc"]["dt"])

    fig = plt.figure(figsize=(9, 7))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(reference[:, 0], reference[:, 1], reference[:, 2],
              "k--", linewidth=2, label="Reference")
    for controller in CONTROLLERS:
        states = traces[controller]["states"]
        axis.plot(states[:, 0], states[:, 1], states[:, 2],
                  color=COLORS[controller], linewidth=1.5,
                  label=f"{LABELS[controller]} ({metrics[controller]:.3f} m)")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title(f"Fresh confirmation: {family}, run {run_index}")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{family}_trajectory.png", dpi=220)
    fig.savefig(output_dir / f"{family}_trajectory.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex=True)
    for index, axis in enumerate(axes.flat):
        if index in (0, 1, 2, 3, 4, 5, 8):
            axis.plot(times, reference[:, index], "k--", linewidth=1.6,
                      label="Reference")
        for controller in CONTROLLERS:
            axis.plot(times, traces[controller]["states"][:, index],
                      color=COLORS[controller], linewidth=1.0,
                      label=LABELS[controller])
        axis.set_ylabel(STATE_LABELS[index])
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Time [s]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.975))
    fig.suptitle(f"Fresh confirmation: all 12 states, {family}, run {run_index}",
                 y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / f"{family}_all_states.png", dpi=220)
    fig.savefig(output_dir / f"{family}_all_states.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4))
    for controller in CONTROLLERS:
        error = np.linalg.norm(
            traces[controller]["states"][:, :3] - reference[:, :3], axis=1)
        axis.plot(times, error, color=COLORS[controller], linewidth=1.5,
                  label=LABELS[controller])
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Position error norm [m]")
    axis.set_title(f"Fresh confirmation: {family}, run {run_index}")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{family}_position_error.png", dpi=220)
    fig.savefig(output_dir / f"{family}_position_error.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirmation-dir", type=Path,
        default=Path("artifacts/acc_balanced_high_tilt_v1/"
                     "tracking_paper_confirmation_fresh_n40"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/acc_balanced_high_tilt_v1/"
                     "attitude_paper_final/representative_tracking"),
    )
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-traces", action="store_true",
                        help="Only remake figures from previously verified traces.")
    args = parser.parse_args()
    manifest = json.loads((args.confirmation_dir / "manifest.json").read_text())
    assert manifest["experiment_role"] == "confirmation"
    assert 0 <= args.replicate < manifest["runs_per_family"]
    linear = json.loads(Path(manifest["linear_config_path"]).read_text())[
        "selected"]["hover_linear"]
    edmd = json.loads(Path(manifest["edmd_config_path"]).read_text())[
        "selected"]["edmdc"]
    gains = json.loads(Path(manifest["reactive_config_path"]).read_text())[
        "gain_multipliers"]
    with (args.confirmation_dir / "episodes.csv").open(newline="") as stream:
        confirmed = list(csv.DictReader(stream))

    tasks = []
    for family_index, family in enumerate(FAMILIES):
        run_index = manifest["run_index_start"] + 1000 * family_index + args.replicate
        for controller in CONTROLLERS:
            tasks.append({
                "controller": controller,
                "family": family,
                "replicate": args.replicate,
                "run_index": run_index,
                "aggressiveness_scale": manifest["scales"][0],
                "duration_seconds": manifest["duration_seconds"],
                "model_path": manifest["model_path"],
                "linear_config": linear,
                "edmd_config": edmd,
                "reactive_gains": gains,
                "trace_path": str(args.output_dir / family / f"{controller}.npz"),
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_traces:
        assert all(Path(task["trace_path"]).is_file() for task in tasks)
        results = [row for row in confirmed if any(
            row["family"] == task["family"] and
            int(row["run_index"]) == task["run_index"] and
            row["controller"] == task["controller"] for task in tasks
        )]
        assert len(results) == len(tasks)
    else:
        with mp.Pool(min(args.workers, len(tasks))) as pool:
            results = list(pool.imap_unordered(run_one, tasks))
        for result in results:
            matched = [row for row in confirmed if (
                row["family"] == result["family"] and
                int(row["run_index"]) == result["run_index"] and
                row["controller"] == result["controller"]
            )]
            assert len(matched) == 1
            assert np.isclose(result["position_rmse_m"],
                              float(matched[0]["position_rmse_m"]),
                              rtol=1e-7, atol=1e-7), (result, matched[0])
    for family_index, family in enumerate(FAMILIES):
        run_index = manifest["run_index_start"] + 1000 * family_index + args.replicate
        metrics = {result["controller"]: float(result["position_rmse_m"])
                   for result in results if result["family"] == family}
        plot_case(family, run_index, args.output_dir, metrics)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "source_confirmation": str(args.confirmation_dir.resolve()),
        "replicate": args.replicate,
        "cases": [{"family": family, "run_index":
                   manifest["run_index_start"] + 1000*i + args.replicate}
                  for i, family in enumerate(FAMILIES)],
        "reproduced_episode_rmse": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    mp.freeze_support()
    main()
