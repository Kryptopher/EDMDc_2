"""Recompute the paper's interception numbers (Tables IV, V and Section V text) from episode records.

Default input is the frozen results/interception/episodes.json; pass --episodes to analyze
records produced by experiments/interception.py instead.

    python experiments/analyze_interception.py
"""
import argparse
import collections
from pathlib import Path

import numpy as np

from _common import CONFIGS, RESULTS, read_json, write_json

MODES = ["oracle", "cv", "delayed_cv_100ms", "delayed_cv_250ms"]
LABEL = {"oracle": "Oracle", "cv": "Current/CV", "delayed_cv_100ms": "100-ms delay/CV", "delayed_cv_250ms": "250-ms delay/CV"}
FAMILIES = ["straight", "accelerating", "helix", "weaving"]
ABORT = "euler_attitude_envelope_80deg"


def bootstrap(d, seed, chunked=False):
    """Percentile 95% CI of the mean over 20,000 paired resamples."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    if chunked:  # draw pattern of the crossed-oracle analysis (paper_gap_audit_20261001/analyze.py)
        means = np.empty(20000)
        for start in range(0, 20000, 1000):
            means[start:start + 1000] = d[rng.integers(len(d), size=(1000, len(d)))].mean(axis=1)
    else:  # draw pattern of the target-information analysis
        means = d[rng.integers(0, len(d), size=(20000, len(d)))].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def paired(a, b, key, seed, joint_only, chunked=False, order="seed"):
    """b minus a over scenarios (optionally only jointly captured).

    The bootstrap depends on pair order: the target-information analysis orders by seed,
    the crossed-oracle analysis by case_id; both are reproduced exactly.
    """
    aa = {r[order]: r for r in a}
    bb = {r[order]: r for r in b}
    assert aa.keys() == bb.keys()
    seeds = [s for s in sorted(aa) if not joint_only or (aa[s]["captured"] and bb[s]["captured"])]
    d = np.array([bb[s][key] - aa[s][key] for s in seeds])
    if not len(d):
        return {"n": 0}
    return {"n": len(d), "mean_difference_s": float(d.mean()), "ci95_s": bootstrap(d, seed, chunked),
            "second_earlier": int(np.sum(d < -1e-12)), "second_later": int(np.sum(d > 1e-12)),
            "ties": int(np.sum(np.abs(d) <= 1e-12))}


def summary(rows):
    q = [r["qualified_time_s"] for r in rows if r["captured"]]
    return {"captures": sum(r["captured"] for r in rows), "n": len(rows),
            "attitude_aborts": sum(r["stop_reason"] == ABORT for r in rows),
            "solver_failures": sum(r.get("solver_failures") or 0 for r in rows),
            "allocator_events": sum(r.get("allocator_events") or 0 for r in rows),
            "success_mean_s": float(np.mean(q)) if q else None,
            "penalized_mean_s": float(np.mean([r["scored_time_s"] for r in rows]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=Path, default=RESULTS / "interception" / "episodes.json")
    parser.add_argument("--output", type=Path, default=None, help="optional JSON output")
    args = parser.parse_args()
    rows = read_json(args.episodes)
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["cell"], r["mode"])].append(r)
    out = {"table_IV": {}, "table_V_oracle_crossed": {}, "causal_crossed": {}, "paired": {}, "family_captures": {}}

    print("Table IV  Selected-controller interception versus target information")
    print(f"{'Controller':10s} {'Information':16s} {'Captures':>8s} {'Att.ab':>6s} {'Success':>9s} {'Penalized':>9s}")
    for cell in ("Linear-A", "EDMDc-B"):
        for mode in MODES:
            if (cell, mode) not in g:
                continue
            s = summary(g[(cell, mode)])
            out["table_IV"][f"{cell} {mode}"] = s
            fmt = lambda v: "   n/a" if v is None else f"{v:9.4f}"
            print(f"{cell:10s} {LABEL[mode]:16s} {s['captures']:>4d}/{s['n']:<3d} {s['attitude_aborts']:>6d} "
                  f"{fmt(s['success_mean_s'])} {fmt(s['penalized_mean_s'])}")

    print("\nEDMDc-B minus Linear-A (seed 2026100201)")
    for mode in MODES:
        if ("Linear-A", mode) in g and ("EDMDc-B", mode) in g:
            j = paired(g[("Linear-A", mode)], g[("EDMDc-B", mode)], "qualified_time_s", 2026100201, True)
            p = paired(g[("Linear-A", mode)], g[("EDMDc-B", mode)], "scored_time_s", 2026100201, False)
            out["paired"][f"EDMDc-B minus Linear-A {mode}"] = {"joint_captures": j, "failure_penalized_all": p}
            if j["n"]:
                print(f"  {LABEL[mode]:16s} joint n={j['n']:2d} mean {j['mean_difference_s']:+.4f} s "
                      f"CI [{j['ci95_s'][0]:.4f}, {j['ci95_s'][1]:.4f}] EDMDc earlier in {j['second_earlier']}; "
                      f"penalized {p['mean_difference_s']:+.4f} s")

    if ("EDMDc-B", "oracle") in g and ("EDMDc-B", "cv") in g:
        c = paired(g[("EDMDc-B", "oracle")], g[("EDMDc-B", "cv")], "qualified_time_s", 2026100201, True)
        out["paired"]["EDMDc-B Current/CV minus oracle"] = c
        print(f"\nEDMDc-B Current/CV minus oracle: n={c['n']} mean {c['mean_difference_s']:+.4f} s "
              f"CI [{c['ci95_s'][0]:.4f}, {c['ci95_s'][1]:.4f}]")

    print("\nCurrent/CV captures by family")
    for cell in ("Linear-A", "EDMDc-B"):
        if (cell, "cv") in g:
            fam = {f: sum(r["captured"] for r in g[(cell, "cv")] if r["family"] == f) for f in FAMILIES}
            out["family_captures"][cell] = fam
            print(f"  {cell:9s} " + ", ".join(f"{f} {n}/10" for f, n in fam.items()))

    print("\nTable V  Crossed oracle predictor-configuration evaluation")
    for cell, pred, conf in (("Linear-A", "Hover-linear", "A"), ("EDMDc-A", "EDMDc", "A"),
                             ("Linear-B", "Hover-linear", "B"), ("EDMDc-B", "EDMDc", "B")):
        if (cell, "oracle") in g:
            s = summary(g[(cell, "oracle")])
            out["table_V_oracle_crossed"][cell] = s
            print(f"  {pred:12s} {conf}  {s['captures']:>2d}/40  aborts {s['attitude_aborts']:>2d}  mean score {s['penalized_mean_s']:.4f} s")
    if ("Linear-A", "oracle") in g and ("EDMDc-A", "oracle") in g:
        c = paired(g[("Linear-A", "oracle")], g[("EDMDc-A", "oracle")], "scored_time_s", 2026100101, False, chunked=True, order="case_id")
        out["paired"]["EDMDc-A minus Linear-A oracle"] = c
        print(f"  Config A, EDMDc minus hover-linear: {c['mean_difference_s']:+.5f} s CI [{c['ci95_s'][0]:.5f}, "
              f"{c['ci95_s'][1]:.5f}] (seed 2026100101); EDMDc later in {c['second_later']}/{c['n']}")

    print("\nCausal crossed captures")
    for cell in ("EDMDc-A", "Linear-A", "Linear-B", "EDMDc-B"):
        counts = {m: summary(g[(cell, m)]) for m in MODES[1:] if (cell, m) in g}
        out["causal_crossed"][cell] = counts
        if counts:
            print(f"  {cell:9s} " + ", ".join(f"{LABEL[m]} {s['captures']}/40 (aborts {s['attitude_aborts']})" for m, s in counts.items()))
    causal = [r for r in rows if r["mode"] != "oracle"]
    print(f"  solver failures in causal runs: {sum(r.get('solver_failures') or 0 for r in causal)}, "
          f"allocator events: {sum(r.get('allocator_events') or 0 for r in causal)}")

    if ("EDMDc-B", "cv") in g:
        tilt = np.array([max(r["maximum_abs_roll_deg"], r["maximum_abs_pitch_deg"]) for r in g[("EDMDc-B", "cv")]])
        out["edmdc_b_cv_peak_tilt_deg"] = {"mean": float(tilt.mean()), "max": float(tilt.max()), "over_30": int((tilt > 30).sum())}
        print(f"\nEDMDc-B Current/CV peak tilt: mean {tilt.mean():.2f} deg, max {tilt.max():.2f} deg, {int((tilt > 30).sum())}/40 above 30 deg")
    sc = read_json(CONFIGS / "interception_scenarios.json")["scenarios"]
    try:
        from interception import build_scenario
        from Simulation import quad_sim
        sim = quad_sim()
        d = np.array([np.linalg.norm(build_scenario(c, sim)[0][0, :3]) for c in sc])
        out["initial_separation_m"] = {"min": float(d.min()), "max": float(d.max()), "mean": float(d.mean())}
        print(f"Initial separation: {d.min():.2f}-{d.max():.2f} m (mean {d.mean():.2f} m)")
    except ImportError:
        pass
    if args.output:
        write_json(args.output, out)


if __name__ == "__main__":
    main()
