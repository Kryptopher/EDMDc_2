"""Controller-update timing (Section V-B, Fig. 3).

Times the complete WrenchMPC.compute() call (prediction/linearization, QP update and OSQP
solve; not allocation or plant stepping) with ONE worker over the 40 stratified 30-s tracking
cases in configs/timing_cases.json. Cells: the four interception predictor/configuration
bundles, plus the trajectory-linear MPC at the matched tracking configuration.
Run on an otherwise idle machine; absolute times depend on hardware.

    python experiments/timing.py --output results_local/timing
    python experiments/timing.py --cells EDMDc-B --cases 2 --output out
"""
import argparse
from pathlib import Path

import numpy as np

from _common import CONFIGS, MODEL_PATH, check_model, read_json, write_json
import run_wrench_controller_tuning as runner


def main():
    inter = read_json(CONFIGS / "interception_controllers.json")
    track = read_json(CONFIGS / "tracking_controllers.json")["arms"]
    cells = {name: (spec["predictor"], inter["configurations"][spec["configuration"]])
             for name, spec in inter["cells"].items()}
    cells["Trajectory-linear-matched"] = ("ltv", {"candidate_id": track["W3"]["candidate_id"], "config": track["W3"]["config"]})
    timing = read_json(CONFIGS / "timing_cases.json")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", nargs="+", default=list(cells), choices=list(cells))
    parser.add_argument("--cases", type=int, default=len(timing["cases"]), help="use only the first N cases")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_model()
    arm_of = {"yaw_linear": "W1", "edmdc": "W2a", "ltv": "W3"}
    summary = {}
    for label in args.cells:
        predictor, bundle = cells[label]
        tasks = [{**case, "arm": arm_of[predictor], "controller": predictor, "candidate_id": bundle["candidate_id"],
                  "duration": timing["duration_seconds"], "config": bundle["config"],
                  "model": str(MODEL_PATH) if predictor == "edmdc" else None, "stage": "timing_pass",
                  "output": str((args.output / "traces" / label).resolve())}
                 for case in timing["cases"][:args.cases]]
        rows = runner.run_stage(args.output / label, "timing_pass", tasks, 1)
        values = []
        for row in sorted(rows, key=lambda r: r["case_id"]):
            with np.load(row["trace"], allow_pickle=False) as trace:
                values.append(np.asarray(trace["solve_seconds"], dtype=np.float64) * 1000.0)
        values = np.concatenate(values)
        np.savez_compressed(args.output / f"{label}_raw_update_times_ms.npz", update_times_ms=values)
        summary[label] = {"updates": int(values.size), "mean_ms": float(values.mean()),
                          "P95_ms": float(np.percentile(values, 95)), "P99_ms": float(np.percentile(values, 99)),
                          "max_ms": float(values.max()), "updates_over_10ms": int(np.count_nonzero(values > 10))}
        s = summary[label]
        print(f"{label:26s} n={s['updates']:6d} mean {s['mean_ms']:.3f} ms  P95 {s['P95_ms']:.3f}  max {s['max_ms']:.3f}  >10 ms: {s['updates_over_10ms']}", flush=True)
    write_json(args.output / "timing_summary.json", summary)


if __name__ == "__main__":
    main()
