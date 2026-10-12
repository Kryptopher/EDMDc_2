"""Interception controller selection: 25 candidates -> screening -> 4 finalists -> validation -> selection.

Per predictor (hover-linear W1, EDMDc W2a), candidates are ranked lexicographically by
(1) incomplete episodes, (2) solver failures + rotor-saturated steps, (3) mean position RMSE
over completed episodes, (4) candidate index. Capture metrics are NOT used for selection.
Episodes run through the unmodified src/wrench_interception_v1/core.py run_episode()
(known-future target reference, as in the original selection campaign).

    python experiments/selection.py --output results_local/selection
    python experiments/selection.py --stage screening --arms W2a --candidates c22 --output out
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from _common import CONFIGS, MODEL_PATH, check_model, read_json, write_json
import core

ARMS = ("W1", "W2a")


def rank(rows):
    ranked = {}
    for arm in sorted({r["arm"] for r in rows}):
        out = []
        for cid in sorted({r["candidate_id"] for r in rows if r["arm"] == arm}):
            group = [r for r in rows if r["arm"] == arm and r["candidate_id"] == cid]
            complete = [r for r in group if r["completed"]]
            failures = sum((r.get("solver_failures") or 0) + (r.get("rotor_saturated_steps") or 0) for r in group)
            mean = float(np.mean([r["position_rmse_m"] for r in complete])) if complete else None
            out.append(dict(candidate_id=cid, incomplete=len(group) - len(complete), failure_steps=failures,
                            mean_position_rmse_m=mean, episodes=len(group)))
        ranked[arm] = sorted(out, key=lambda r: (r["incomplete"], r["failure_steps"],
                                                 r["mean_position_rmse_m"] if r["mean_position_rmse_m"] is not None else float("inf"),
                                                 r["candidate_id"]))
    return ranked


def run_stage(stage, choices, scenarios, duration, output, workers):
    tasks = [dict(**case, arm=arm, candidate_id=c["candidate_id"], config=c["config"], stage=stage,
                  model=str(MODEL_PATH), duration=duration, output=str(output / stage / "runs"))
             for arm in choices for c in choices[arm] for case in scenarios]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(core.safe_episode, t) for t in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            if i % 20 == 0 or i == len(tasks):
                print(f"{stage}: {i}/{len(tasks)}", flush=True)
    rows.sort(key=lambda r: (r["arm"], r["candidate_id"], r["case_id"]))
    write_json(output / stage / "episodes.json", rows)
    return rows


def main():
    candidates = read_json(CONFIGS / "interception_candidates.json")["candidates"]
    scenarios = read_json(CONFIGS / "selection_scenarios.json")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("all", "screening", "validation"), default="all")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--candidates", nargs="+", default=None, help="restrict to candidate IDs (smoke tests)")
    parser.add_argument("--finalists", type=Path, default=None,
                        help="finalists JSON for --stage validation (default: from screening in --output)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_model()
    pool = [c for c in candidates if args.candidates is None or c["candidate_id"] in args.candidates]
    by_id = {c["candidate_id"]: c for c in candidates}
    if args.stage in ("all", "screening"):
        rows = run_stage("screening", {a: pool for a in args.arms}, scenarios["screening"],
                         scenarios["screening_seconds"], args.output, args.workers)
        ranking = rank(rows)
        write_json(args.output / "screening" / "ranking.json", ranking)
        finalists = {a: [by_id[r["candidate_id"]] for r in ranking[a][:scenarios["finalists"]]] for a in ranking}
        write_json(args.output / "validation" / "finalists.json", finalists)
        for a in ranking:
            print(f"{a}: candidates completing every screening episode: "
                  f"{sum(r['incomplete'] == 0 for r in ranking[a])}/{len(ranking[a])}; "
                  f"finalists {[r['candidate_id'] for r in ranking[a][:scenarios['finalists']]]}")
    if args.stage in ("all", "validation"):
        finalists = read_json(args.finalists or args.output / "validation" / "finalists.json")
        finalists = {a: v for a, v in finalists.items() if a in args.arms}
        rows = run_stage("validation", finalists, scenarios["validation"], scenarios["validation_seconds"],
                         args.output, args.workers)
        ranking = rank(rows)
        write_json(args.output / "validation" / "ranking.json", ranking)
        selected = {a: by_id[ranking[a][0]["candidate_id"]] for a in ranking}
        write_json(args.output / "selection.json", selected)
        for a, c in selected.items():
            cfg = c["config"]
            print(f"Selected for {a}: {c['candidate_id']} (Np={cfg['horizon_steps']}, "
                  f"Nc={cfg['horizon_steps'] // cfg['block_steps']}, b={cfg['block_steps']})")


if __name__ == "__main__":
    main()
