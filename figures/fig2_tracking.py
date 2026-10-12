# Release copy of paper_acc_revision/final_figures/plot_tracking_matched_final.py; only input/output paths and
# writes into the research repository were changed. Plotting code is unchanged.
"""Final matched nominal tracking figure from frozen blind-test traces only."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
INDEX = ROOT / "results" / "tracking" / "blind_test_episodes.json"
TRACES = ROOT / "results" / "figure_inputs" / "tracking"
SELECTION_CSV = OUT / "fig2_selected_cases.csv"
FAMILIES = (("helix", "Helix"), ("figure8", "Figure-eight"), ("lissajous", "Lissajous"))
COLORS = {"Reference": "#202020", "Hover-linear MPC": "#0064A7", "EDMDc-MPC": "#00802B"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_trace(row: dict) -> tuple[Path, dict[str, np.ndarray]]:
    path = TRACES / row["arm"] / f"{row['case_id']}.npz"
    require(path.is_file(), f"missing frozen trace: {path}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in ("time", "states", "reference")}
    return path, arrays


def main() -> None:
    controllers = json.loads((ROOT / "configs" / "tracking_controllers.json").read_text(encoding="utf-8"))["arms"]
    for arm in ("W1", "W2a"):
        config = controllers[arm]["config"]
        require(controllers[arm]["candidate_id"] == "c23" and config["dt"] == 0.01 and config["horizon_steps"] == 75
                and config["block_steps"] == 5, f"{arm} matched configuration")

    episodes = json.loads(INDEX.read_text(encoding="utf-8"))
    paired: dict[str, dict[str, dict]] = {}
    for row in episodes:
        if row["arm"] in ("W1", "W2a") and row["family"] in dict(FAMILIES):
            require(row["candidate_id"] == "c23" and row["completed"],
                    f"incomplete or unmatched case {row['case_id']}")
            paired.setdefault(row["case_id"], {})[row["arm"]] = row

    selected: list[dict] = []
    traces = []
    source_paths = [INDEX]
    maximum_error_mm = 0.0
    for family, display in FAMILIES:
        cases = []
        for case_id, arms in paired.items():
            if arms["W1"]["family"] != family:
                continue
            require(set(arms) == {"W1", "W2a"}, f"unpaired case {case_id}")
            a, b = arms["W1"], arms["W2a"]
            require(a["seed"] == b["seed"] and a["reference_sha256"] == b["reference_sha256"],
                    f"different target or seed: {case_id}")
            cases.append((case_id, b["position_rmse_m"] - a["position_rmse_m"], a, b))
        require(len(cases) == 50, f"{family}: expected 50 paired held-out cases")
        median = float(np.median([case[1] for case in cases]))
        nearest = min(abs(case[1] - median) for case in cases)
        candidates = [case for case in cases if abs(abs(case[1] - median) - nearest) < 1e-12]
        if family == "lissajous":
            require(len(candidates) == 2 and
                    {case[0] for case in candidates} == {
                        "lissajous_92002005_scale1.75_fixed",
                        "lissajous_92002049_scale1.75_path"},
                    "Lissajous median tie changed")
            case_id, difference, a, b = min(candidates,
                                            key=lambda case: ("_path" not in case[0], case[0]))
        else:
            case_id, difference, a, b = min(candidates, key=lambda case: case[0])
        path_a, trace_a = read_trace(a)
        path_b, trace_b = read_trace(b)
        source_paths.extend((path_a, path_b))
        require(np.array_equal(trace_a["time"], trace_b["time"]) and
                np.array_equal(trace_a["reference"], trace_b["reference"]),
                f"{family}: raw target/time mismatch")
        require(len(trace_a["time"]) == 3001 and abs(float(trace_a["time"][-1]) - 30.0) < 1e-10,
                f"{family}: not the full saved 30-s trace")
        for row, trace in ((a, trace_a), (b, trace_b)):
            residual = trace["states"][:, :3] - trace["reference"][:, :3]
            rmse = float(np.sqrt(np.mean(residual ** 2)))
            require(abs(rmse - row["position_rmse_m"]) < 1e-10,
                    f"{family}: raw RMSE disagrees with episode")
            maximum_error_mm = max(maximum_error_mm, float(np.linalg.norm(residual, axis=1).max() * 1000.0))
        record = {"family": family, "family_display": display, "scenario_id": case_id,
                  "seed": a["seed"], "W1_RMSE_m": a["position_rmse_m"],
                  "W2a_RMSE_m": b["position_rmse_m"],
                  "W2a_minus_W1_RMSE_m": difference,
                  "family_median_difference_m": median,
                  "selection_rule": ("paired EDMDc-minus-hover-linear RMSE nearest family median; "
                                     "Lissajous median tie: path-type case for non-ellipse presentation"
                                     if family == "lissajous" else
                                     "paired EDMDc-minus-hover-linear RMSE nearest family median; "
                                     "tie: lexicographic scenario ID")}
        selected.append(record)
        traces.append((display, trace_a, trace_b))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.labelsize": 8.8, "axes.titlesize": 9.4,
                         "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
                         "legend.fontsize": 8.6, "axes.linewidth": 0.7,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none"})
    fig = plt.figure(figsize=(7.12, 4.65))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.22, 0.86],
                           left=0.065, right=0.95, bottom=0.115, top=0.88,
                           wspace=0.25, hspace=0.16)
    axes = []
    upper_mm = float(np.ceil(maximum_error_mm / 5.0) * 5.0)
    for column, (display, linear, edmdc) in enumerate(traces):
        ax3 = fig.add_subplot(grid[0, column], projection="3d")
        ax2 = fig.add_subplot(grid[1, column])
        axes.append((ax3, ax2))
        reference = linear["reference"][:, :3]
        state_linear = linear["states"][:, :3]
        state_edmdc = edmdc["states"][:, :3]
        ax3.plot(*state_linear.T, color=COLORS["Hover-linear MPC"], lw=1.65)
        ax3.plot(*state_edmdc.T, color=COLORS["EDMDc-MPC"], lw=1.35)
        ax3.plot(*reference.T, color=COLORS["Reference"], lw=1.1,
                 ls=(0, (3, 2)))
        ax3.scatter(*reference[0], color="#D7191C", s=22, zorder=10)
        coordinates = np.concatenate((reference, state_linear, state_edmdc))
        low = coordinates.min(axis=0)
        high = coordinates.max(axis=0)
        padding = np.maximum((high - low) * 0.05, 0.05)
        ax3.set_xlim(low[0] - padding[0], high[0] + padding[0])
        ax3.set_ylim(low[1] - padding[1], high[1] + padding[1])
        ax3.set_zlim(low[2] - padding[2], high[2] + padding[2])
        ax3.set_box_aspect((1, 1, 0.78), zoom=1.03)
        ax3.view_init(elev=21, azim=-62)
        ax3.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        ax3.set_xlabel("x [m]", labelpad=-7)
        ax3.set_ylabel("y [m]", labelpad=-7)
        ax3.set_zlabel("z [m]", labelpad=-8)
        ax3.tick_params(labelsize=6.7, pad=-3)
        for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
            axis.pane.fill = False
            axis._axinfo["grid"].update({"color": (0.7, 0.7, 0.7, 0.65),
                                          "linewidth": 0.5})
        ax3.set_title(display, pad=0)

        error_linear = np.linalg.norm(state_linear - reference, axis=1) * 1000.0
        error_edmdc = np.linalg.norm(state_edmdc - reference, axis=1) * 1000.0
        ax2.plot(linear["time"], error_linear,
                 color=COLORS["Hover-linear MPC"], lw=1.8)
        ax2.plot(edmdc["time"], error_edmdc,
                 color=COLORS["EDMDc-MPC"], lw=1.35)
        ax2.set(xlim=(0, 30), ylim=(0, upper_mm), xlabel="Time [s]")
        if column == 0:
            ax2.set_ylabel("Position error [mm]")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(color="#dddddd", lw=0.45)
        ax2.tick_params(direction="out", length=2.5)

    fig.canvas.draw()
    widths = [ax.get_position().width for pair in axes for ax in pair]
    require(max(widths) - min(widths) < 1e-10, "unequal subplot widths")
    for pair in axes:
        centers = [(ax.get_position().x0 + ax.get_position().x1) / 2 for ax in pair]
        require(abs(centers[0] - centers[1]) < 1e-10, "3-D and error panel centers differ")
    centers = [(pair[0].get_position().x0 + pair[0].get_position().x1) / 2 for pair in axes]
    require(abs((centers[1] - centers[0]) - (centers[2] - centers[1])) < 1e-10,
            "unequal column spacing")
    handles = [Line2D([0], [0], color=COLORS["Reference"], lw=1.2,
                      ls=(0, (3, 2)), label="Reference"),
               Line2D([0], [0], color="#D7191C", marker="o", lw=0,
                      markersize=4.2, label="Start"),
               Line2D([0], [0], color=COLORS["Hover-linear MPC"], lw=1.8,
                      label="Hover-linear MPC"),
               Line2D([0], [0], color=COLORS["EDMDc-MPC"], lw=1.35,
                      label="EDMDc-MPC")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=4, frameon=False, columnspacing=1.25)
    for suffix in ("pdf", "eps", "svg", "png"):
        fig.savefig(OUT / f"fig2_tracking.{suffix}",
                    dpi=600, facecolor="white")
    plt.close(fig)

    with SELECTION_CSV.open("w", newline="", encoding="utf-8") as stream:
        columns = ["family", "scenario_id", "seed", "W1_RMSE_m", "W2a_RMSE_m",
                   "W2a_minus_W1_RMSE_m", "family_median_difference_m", "selection_rule"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in columns} for row in selected])
    for row in selected:
        print(f"{row['family_display']}: {row['scenario_id']}, seed {row['seed']}; "
              f"median {row['family_median_difference_m']:.12g} m, "
              f"selected difference {row['W2a_minus_W1_RMSE_m']:.12g} m")
    print("Matched c23 traces and paired references verified; no simulation rerun.")
    for suffix in ("pdf", "eps", "svg", "png"):
        print(OUT / f"fig2_tracking.{suffix}")


if __name__ == "__main__":
    main()
