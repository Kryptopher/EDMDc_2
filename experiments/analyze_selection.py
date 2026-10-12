"""Summarize the interception controller-selection results (Section IV-C, Table II).

    python experiments/analyze_selection.py
    python experiments/analyze_selection.py --results results_local/selection
"""
import argparse
from pathlib import Path

from _common import CONFIGS, RESULTS, read_json, write_json
from selection import rank

NAMES = {"W1": "hover-linear", "W2a": "EDMDc"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=RESULTS / "selection")
    parser.add_argument("--output", type=Path, default=None, help="optional JSON output")
    args = parser.parse_args()
    out = {}
    frozen = args.results == RESULTS / "selection"
    screening = read_json(args.results / ("screening_episodes.json" if frozen else "screening/episodes.json"))
    validation = read_json(args.results / ("validation_episodes.json" if frozen else "validation/episodes.json"))
    s_rank, v_rank = rank(screening), rank(validation)
    candidates = {c["candidate_id"]: c["config"] for c in read_json(CONFIGS / "interception_candidates.json")["candidates"]}
    for arm in s_rank:
        clean = [r["candidate_id"] for r in s_rank[arm] if r["incomplete"] == 0]
        arm_rows = [r for r in screening if r["arm"] == arm]
        aborts = sum(r["stop_reason"] == "euler_attitude_envelope_80deg" for r in arm_rows)
        print(f"{NAMES[arm]:12s} screening: {len(clean)}/{len(s_rank[arm])} candidates completed all 12 episodes "
              f"({', '.join(sorted(clean))}); attitude aborts {aborts}/{len(arm_rows)}")
        print(f"{'':12s} finalists: {[r['candidate_id'] for r in s_rank[arm][:4]]}")
        best = v_rank[arm][0]["candidate_id"]
        cfg = candidates[best]
        print(f"{'':12s} selected after validation: {best} "
              f"(Np/Nc/b = {cfg['horizon_steps']}/{cfg['horizon_steps'] // cfg['block_steps']}/{cfg['block_steps']})")
        out[NAMES[arm]] = {"clean_candidates": len(clean), "candidates": len(s_rank[arm]),
                           "finalists": [r["candidate_id"] for r in s_rank[arm][:4]], "selected": best,
                           "Np_Nc_b": [cfg["horizon_steps"], cfg["horizon_steps"] // cfg["block_steps"], cfg["block_steps"]]}
    if args.output:
        write_json(args.output, out)


if __name__ == "__main__":
    main()
