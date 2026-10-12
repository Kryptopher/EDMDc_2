"""Recompute Table III and the paired tracking statistics from episode records.

    python experiments/analyze_tracking.py
    python experiments/analyze_tracking.py --episodes results_local/tracking/episodes.json
"""
import argparse
from pathlib import Path

import numpy as np

from _common import RESULTS, read_json, write_json

LABELS = {"W1": "Hover-linear MPC", "W2a": "EDMDc-MPC", "W3": "Trajectory-linear MPC"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=Path, default=RESULTS / "tracking" / "blind_test_episodes.json")
    parser.add_argument("--output", type=Path, default=None, help="optional JSON output")
    args = parser.parse_args()
    rows = read_json(args.episodes)
    out = {"table_III": {}}
    print("Table III  Nominal tracking (mean over episodes)")
    print(f"{'Controller':22s} {'n':>4s} {'Pos [mm]':>9s} {'Att [mrad]':>11s} {'Yaw [mrad]':>11s}")
    for arm, label in LABELS.items():
        r = [x for x in rows if x["arm"] == arm and x["completed"]]
        if not r:
            continue
        pos = 1e3 * np.mean([x["position_rmse_m"] for x in r])
        # Att. = per-episode RMS over roll, pitch and yaw errors, averaged over episodes.
        att = 1e3 * np.mean([np.sqrt((x["roll_rmse_rad"] ** 2 + x["pitch_rmse_rad"] ** 2 + x["yaw_rmse_rad"] ** 2) / 3) for x in r])
        yaw = 1e3 * np.mean([x["yaw_rmse_rad"] for x in r])
        out["table_III"][label] = {"n": len(r), "position_mm": float(pos), "attitude_mrad": float(att), "yaw_mrad": float(yaw)}
        print(f"{label:22s} {len(r):4d} {pos:9.3f} {att:11.3f} {yaw:11.3f}")

    lookup = {(x["case_id"], x["arm"]): x for x in rows}
    cases = sorted({x["case_id"] for x in rows})
    pairs = [(lookup[(c, "W2a")], lookup[(c, "W1")]) for c in cases
             if (c, "W2a") in lookup and (c, "W1") in lookup and lookup[(c, "W2a")]["completed"] and lookup[(c, "W1")]["completed"]]
    if pairs:
        d = np.array([a["position_rmse_m"] - b["position_rmse_m"] for a, b in pairs])
        # Same bootstrap as run_wrench_controller_tuning.paired_statistics (seed 2026092202, 20,000 resamples).
        rng = np.random.default_rng(2026092202)
        boot = d[rng.integers(0, len(d), (20000, len(d)))].mean(axis=1)
        print(f"\nEDMDc minus hover-linear position RMSE: {1e3 * d.mean():+.3f} mm, "
              f"95% CI [{1e3 * np.percentile(boot, 2.5):.3f}, {1e3 * np.percentile(boot, 97.5):.3f}] mm; "
              f"EDMDc lower in {int(np.sum(d < -1e-15))}/{len(d)}")
        out["paired"] = {"mean_difference_mm": float(1e3 * d.mean()),
                         "ci95_mm": [float(1e3 * np.percentile(boot, 2.5)), float(1e3 * np.percentile(boot, 97.5))],
                         "edmdc_lower": int(np.sum(d < -1e-15)), "n": len(d)}
    if args.output:
        write_json(args.output, out)


if __name__ == "__main__":
    main()
