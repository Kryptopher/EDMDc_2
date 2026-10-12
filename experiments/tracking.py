"""Matched nominal trajectory tracking on the 200 held-out scenarios (Table III, Fig. 2).

Hover-linear (W1) and EDMDc (W2a) use the same configuration c23 (Np/Nc/b = 75/15/5); the
trajectory-linear comparator W3 uses c11. Episodes are executed by the unmodified
src/run_wrench_controller_tuning.py trial() / src/run_wrench_revisit.py run_case().

    python experiments/tracking.py --output results_local/tracking
    python experiments/tracking.py --arms W1 W2a --families helix --replicates 0 1 --output out
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from _common import CONFIGS, MODEL_PATH, check_model, read_json, write_json
import run_wrench_controller_tuning as tracking


def main():
    controllers = read_json(CONFIGS / "tracking_controllers.json")["arms"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", default=list(controllers), choices=list(controllers))
    parser.add_argument("--families", nargs="+", default=list(tracking.FAMILIES), choices=list(tracking.FAMILIES))
    parser.add_argument("--replicates", nargs="+", type=int, default=None, help="replicate indices 0-49 (default all)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_model()
    # blind_cases(): seeds 92,000,000 + 1000*family + replicate; replicates 0-24 fixed heading, 25-49 path heading.
    cases = [c for c in tracking.blind_cases() if c["family"] in args.families
             and (args.replicates is None or c["replicate"] in args.replicates)]
    tasks = []
    for case in cases:
        for arm in args.arms:
            spec = controllers[arm]
            tasks.append({**case, "arm": arm, "controller": spec["predictor"], "candidate_id": spec["candidate_id"],
                          "duration": 30.0, "config": spec["config"],
                          "model": str(MODEL_PATH) if spec["predictor"] == "edmdc" else None,
                          "stage": "blind_test",
                          "output": str((args.output / "traces" / arm / spec["candidate_id"]).resolve())})
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(tracking.trial, t) for t in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            rmse = row["position_rmse_m"]
            print(f"[{i}/{len(tasks)}] {row['arm']} {row['case_id']}: "
                  f"pos RMSE {'n/a' if rmse is None else f'{1e3 * rmse:.3f} mm'} stop={row['stop_reason']}", flush=True)
    rows.sort(key=lambda r: (r["case_id"], r["arm"]))
    write_json(args.output / "episodes.json", rows)
    print(f"Wrote {len(rows)} episodes to {args.output / 'episodes.json'}")


if __name__ == "__main__":
    main()
