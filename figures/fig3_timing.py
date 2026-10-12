# Release copy of paper_acc_revision/final_figures/plot_3_controller_update_timing_ecdf.py; only input/output paths and
# writes into the research repository were changed. Plotting code is unchanged.
"""ECDF of four completed one-worker interception predictor/configuration cells."""
from __future__ import annotations

import hashlib
import json
import shutil
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
RES = ROOT / "results" / "timing"
STEM = "fig3_timing_ecdf"
STYLE = (("Linear-A", "#0064A7", "-"),
         ("EDMDc-A", "#58AC71", "-"),
         ("Linear-B", "#5396C4", "--"),
         ("EDMDc-B", "#00802B", "--"),
         ("Trajectory-linear MPC (Np=75)", "#7B3294", "-"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matched_ltv() -> tuple[np.ndarray, list[dict]]:
    with np.load(RES / "Trajectory-linear_matched_raw_update_times_ms.npz", allow_pickle=False) as raw:
        all_values = np.asarray(raw["update_times_ms"], dtype=np.float64)
    assert all_values.shape == (120000,)
    assert abs(float(np.mean(all_values)) - 11.63954765428789) < 1e-7
    assert abs(float(np.percentile(all_values, 95)) - 12.504614988574755) < 1e-7
    return all_values, []


def main() -> None:
    results_path = RES / "four_cell_timing_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert results["no_retuning"] and len(results["cells"]) == 4
    saved = {item["cell"]: item for item in results["cells"]}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.3,
                         "axes.labelsize": 8.8, "xtick.labelsize": 7.8,
                         "ytick.labelsize": 7.8, "legend.fontsize": 7.4,
                         "axes.linewidth": .7, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none"})
    ltv_values, ltv_manifest = matched_ltv()
    curves = []
    for label, color, line in STYLE:
        if label.startswith("Trajectory-linear"):
            values = ltv_values
        else:
            record = saved[label]
            path = RES / f"{label}_raw_update_times_ms.npz"
            assert sha256(path) == record["raw_sha256"]
            with np.load(path, allow_pickle=False) as raw:
                values = np.asarray(raw["update_times_ms"], dtype=np.float64)
            assert values.shape == (120000,)
            assert abs(float(np.mean(values)) - record["mean_ms"]) < 1e-10
            assert int(np.count_nonzero(values > 10)) == record["updates_over_10ms"]
        curves.append((label, color, line, np.sort(values)))
    full_stem = f"{STEM}_full"
    for cropped, output_stem in ((True, STEM), (False, full_stem)):
        fig, ax = plt.subplots(figsize=(3.45, 3.18))
        for label, color, line, ordered in curves:
            shown = int((.95 if cropped else 1.0) * len(ordered))
            ax.plot(ordered[:shown], np.arange(1, shown + 1) / len(ordered),
                    color=color, ls=line, lw=1.5, label=label)
            if cropped:
                ax.plot(ordered[shown - 1], shown / len(ordered), marker="o", ms=2.5,
                        color=color, ls="none")
        ax.axvline(10, color="#666666", lw=.85, ls=(0, (4, 2)), zorder=0)
        ax.text(10.15, .04, "10 ms (100 Hz)", rotation=90, color="#555555",
                fontsize=7.2, ha="left", va="bottom")
        ax.set(xlim=(0, 13.2 if cropped else 21),
               ylim=(0, .97 if cropped else 1.01),
               xlabel="Controller-update time [ms]",
               ylabel="Empirical CDF (to P95)" if cropped else "Empirical CDF")
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12] if cropped else [0, 5, 10, 15, 20])
        ax.set_yticks([0, .25, .5, .75, .95] if cropped else [0, .25, .5, .75, 1])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.legend(loc="upper center", bbox_to_anchor=(.5, .99), ncol=2,
                   frameon=False, handlelength=1.7, columnspacing=1.25)
        fig.subplots_adjust(left=.18, right=.97, bottom=.17, top=.73)
        for ext in ("pdf", "eps", "svg", "png"):
            fig.savefig(OUT / f"{output_stem}.{ext}", dpi=600, facecolor="white")
        plt.close(fig)
    print(f"Saved P95 and full ECDFs in four formats; verified 120,000 updates per curve, including frozen matched Np=75 trajectory-linear timing.")


if __name__ == "__main__":
    main()
