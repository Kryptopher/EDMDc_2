"""Controller-update timing summary (Section V-B, Fig. 3) from the frozen raw update times.

    python experiments/analyze_timing.py
"""
import argparse
from pathlib import Path

import numpy as np

from _common import RESULTS, read_json, write_json

CELLS = ("Linear-A", "EDMDc-A", "Linear-B", "EDMDc-B")


def stats(values):
    return {"updates": int(values.size), "mean_ms": float(values.mean()), "P95_ms": float(np.percentile(values, 95)),
            "max_ms": float(values.max()), "updates_over_10ms": int(np.count_nonzero(values > 10))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=RESULTS / "timing")
    parser.add_argument("--output", type=Path, default=None, help="optional JSON output")
    args = parser.parse_args()
    out = {"matched_tracking": {}, "four_cell": {}}
    matched = read_json(args.results / "matched_tracking_timing_summary.json")
    print("Matched tracking configuration (Np/Nc/b = 75/15/5), one worker, 120,000 updates each")
    for arm, label in (("W1", "Hover-linear MPC"), ("W2a", "EDMDc-MPC"), ("W3", "Trajectory-linear MPC")):
        row = next(r for r in matched if r["arm"] == arm)
        out["matched_tracking"][label] = {"updates": row["samples"], "mean_ms": row["mean_ms"]}
        print(f"  {label:22s} mean {row['mean_ms']:.3f} ms")
    print("Four frozen interception bundles, one worker, same 40 timing cases")
    for cell in CELLS:
        with np.load(args.results / f"{cell}_raw_update_times_ms.npz", allow_pickle=False) as raw:
            s = stats(np.asarray(raw["update_times_ms"], dtype=np.float64))
        out["four_cell"][cell] = s
        print(f"  {cell:9s} n={s['updates']} mean {s['mean_ms']:.3f} ms, P95 {s['P95_ms']:.3f} ms, "
              f"max {s['max_ms']:.3f} ms, over 10 ms: {s['updates_over_10ms']}")
    if args.output:
        write_json(args.output, out)


if __name__ == "__main__":
    main()
