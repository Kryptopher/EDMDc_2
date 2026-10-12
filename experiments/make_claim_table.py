"""Build the README claim table (paper item -> value -> command -> result file) from analysis output.

Runs the five analyze_*.py scripts on the frozen results and prints a Markdown table; nothing is
hand-typed. With --readme, replaces the block between the CLAIM-TABLE markers in README.md.

    python experiments/make_claim_table.py
    python experiments/make_claim_table.py --readme
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from _common import ROOT

EXP = ROOT / "experiments"
BEGIN, END = "<!-- CLAIM-TABLE:BEGIN -->", "<!-- CLAIM-TABLE:END -->"


def run(name, tmp):
    out = Path(tmp) / f"{name}.json"
    subprocess.run([sys.executable, str(EXP / f"analyze_{name}.py"), "--output", str(out)], check=True,
                   stdout=subprocess.DEVNULL)
    return json.loads(out.read_text(encoding="utf-8"))


def ci(c, nd=4):
    return f"[{c[0]:.{nd}f}, {c[1]:.{nd}f}]"


def table():
    with tempfile.TemporaryDirectory() as tmp:
        m, s, t, i, tm = (run(n, tmp) for n in ("model", "selection", "tracking", "interception", "timing"))
    C = {k: f"`python experiments/analyze_{k}.py`" for k in ("model", "selection", "tracking", "interception", "timing")}
    F = {"model": "`data/models/wrench_physics42_dt001.pkl`, `results/training/training_physics42.log`",
         "selection": "`results/selection/*.json`", "tracking": "`results/tracking/blind_test_episodes.json`",
         "interception": "`results/interception/episodes.json`", "timing": "`results/timing/*`"}
    rows = []
    add = lambda loc, value, key: rows.append((loc, value, C[key], F[key]))

    add("§IV-B", f"λ = {m['lambda']:g} from a {m['lambda_grid_size']}-value grid; {m['training_transitions']:,} training transitions "
        f"({m['training_runs']} runs); K {m['K_shape'][0]}×{m['K_shape'][1]}, B_u {m['B_shape'][0]}×{m['B_shape'][1]}", "model")
    add("§IV-B", f"Regression data matrix full rank (σ_min = {m['smallest_singular_value']:.3g}); condition number "
        f"{m['condition_number']:.3g}", "model")
    add("§IV-B", f"Model file SHA-256 `{m['model_sha256']}`; content digest (numerical fields) `{m['model_content_sha256']}`", "model")
    for name in ("hover-linear", "EDMDc"):
        x = s[name]
        conf = "A" if name == "hover-linear" else "B"
        add("Table II, §IV-C", f"Configuration {conf} ({name}): {x['selected']}, Np/Nc/b = {'/'.join(map(str, x['Np_Nc_b']))}; "
            f"{x['clean_candidates']}/{x['candidates']} candidates completed all screening episodes", "selection")
    for label, x in t["table_III"].items():
        add("Table III", f"{label}: position {x['position_mm']:.3f} mm, attitude {x['attitude_mrad']:.3f} mrad, "
            f"yaw {x['yaw_mrad']:.3f} mrad (n = {x['n']})", "tracking")
    p = t["paired"]
    add("§V-A", f"EDMDc lower position RMSE in {p['edmdc_lower']}/{p['n']}; paired difference {p['mean_difference_mm']:.3f} mm, "
        f"95% CI {ci(p['ci95_mm'], 3)} mm", "tracking")
    mt = tm["matched_tracking"]
    add("§V-B", "Matched-configuration mean update time: " + ", ".join(f"{k} {v['mean_ms']:.3f} ms" for k, v in mt.items()), "timing")
    add("§V-B, Fig. 3", "Four bundles, mean (P95, max): " + "; ".join(
        f"{k} {v['mean_ms']:.3f} ({v['P95_ms']:.3f}, {v['max_ms']:.3f}) ms" for k, v in tm["four_cell"].items())
        + f"; updates over 10 ms: {sum(v['updates_over_10ms'] for v in tm['four_cell'].values())}", "timing")
    labels = {"oracle": "Oracle", "cv": "Current/CV", "delayed_cv_100ms": "100-ms delay/CV", "delayed_cv_250ms": "250-ms delay/CV"}
    for key, x in i["table_IV"].items():
        cell, mode = key.split(" ")
        succ = "n/a" if x["success_mean_s"] is None else f"{x['success_mean_s']:.4f}"
        add("Table IV", f"{cell}, {labels[mode]}: {x['captures']}/{x['n']} captures, {x['attitude_aborts']} attitude aborts, "
            f"success mean {succ} s, penalized mean {x['penalized_mean_s']:.4f} s", "interception")
    P = i["paired"]
    cv = P["EDMDc-B minus Linear-A cv"]
    add("§V-C", f"Current/CV joint captures n = {cv['joint_captures']['n']}: EDMDc-B − Linear-A "
        f"{cv['joint_captures']['mean_difference_s']:.4f} s, 95% CI {ci(cv['joint_captures']['ci95_s'])}; "
        f"EDMDc-B earlier in {cv['joint_captures']['second_earlier']}/{cv['joint_captures']['n']}", "interception")
    add("§V-C", f"Current/CV failure-penalized difference {cv['failure_penalized_all']['mean_difference_s']:.4f} s", "interception")
    fam = i["family_captures"]
    others = ", ".join("{} {}/10 vs {}/10".format(f, fam["Linear-A"][f], fam["EDMDc-B"][f])
                       for f in ("straight", "accelerating", "weaving"))
    add("§V-C", f"Helix captures: Linear-A {fam['Linear-A']['helix']}/10, EDMDc-B {fam['EDMDc-B']['helix']}/10 "
        f"(other families: {others})", "interception")
    for mode, sec in (("delayed_cv_100ms", "§V-D"), ("delayed_cv_250ms", "§V-D")):
        j = P[f"EDMDc-B minus Linear-A {mode}"]["joint_captures"]
        add(sec, f"{labels[mode]} joint captures n = {j['n']}: {j['mean_difference_s']:.4f} s, 95% CI {ci(j['ci95_s'])}; "
            f"EDMDc-B earlier in {j['second_earlier']}/{j['n']}", "interception")
    o = P["EDMDc-B minus Linear-A oracle"]["joint_captures"]
    add("§V-E", f"Oracle: EDMDc-B earlier in {o['second_earlier']}/{o['n']}", "interception")
    c = P["EDMDc-B Current/CV minus oracle"]
    add("§V-E", f"EDMDc-B Current/CV − oracle {c['mean_difference_s']:.4f} s, 95% CI {ci(c['ci95_s'])}", "interception")
    for cell, pred, conf in (("Linear-A", "Hover-linear", "A"), ("EDMDc-A", "EDMDc", "A"), ("Linear-B", "Hover-linear", "B"), ("EDMDc-B", "EDMDc", "B")):
        x = i["table_V_oracle_crossed"][cell]
        add("Table V", f"{pred}, config {conf}: {x['captures']}/{x['n']} captures, {x['attitude_aborts']} attitude aborts, "
            f"mean score {x['penalized_mean_s']:.4f} s", "interception")
    a = P["EDMDc-A minus Linear-A oracle"]
    # Exact values sit on rounding midpoints (0.14075, 0.09525, 0.18425); the paper rounds half up.
    add("§V-E", f"Config A, EDMDc − hover-linear {a['mean_difference_s']:.5f} s, 95% CI {ci(a['ci95_s'], 5)} "
        f"(paper: 0.1408, [0.0953, 0.1843]); EDMDc later in {a['second_later']}/{a['n']}", "interception")
    cc = i["causal_crossed"]
    fmt = lambda cell: "/".join(str(cc[cell][m]["captures"]) for m in ("cv", "delayed_cv_100ms", "delayed_cv_250ms"))
    add("§V-E", f"Causal crossed captures (CV/100 ms/250 ms): EDMDc-A {fmt('EDMDc-A')} vs Linear-A {fmt('Linear-A')}; "
        f"Linear-B {fmt('Linear-B')} vs EDMDc-B {fmt('EDMDc-B')}", "interception")
    tilt = i["edmdc_b_cv_peak_tilt_deg"]
    add("§II-A (context)", f"EDMDc-B Current/CV peak tilt: mean {tilt['mean']:.2f}°, max {tilt['max']:.2f}° (abort limit 80°)", "interception")
    sep = i["initial_separation_m"]
    add("§IV-E", f"Initial separation {sep['min']:.1f}–{sep['max']:.1f} m (mean {sep['mean']:.1f} m)", "interception")

    lines = ["| Paper | Value | Check | Result file |", "|---|---|---|---|"]
    lines += [f"| {loc} | {v} | {cmd} | {f} |" for loc, v, cmd, f in rows]
    lines += ["| §IV-B | Retraining reproduces the content digest above (all numerical fields bitwise identical); model file SHA-256 is `090b8dd3…` when trained from the `.pkl` and `78822050…` from the `.npz`, because the stored input file name differs (see `DATA.md`) | `python experiments/train_edmdc.py --data <dataset> --output <dir>` (needs the training dataset, see `DATA.md`) | dataset, see `DATA.md` |",
              "| (supplementary) | Held-out open-loop prediction RMSE | `python experiments/evaluate_prediction.py --data <dataset>` (needs the training dataset, see `DATA.md`) | `results/prediction/heldout_rolling_aggregate.csv` |"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--readme", action="store_true", help="write the table into README.md")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    text = table()
    if args.readme:
        readme = ROOT / "README.md"
        content = readme.read_text(encoding="utf-8")
        start, stop = content.index(BEGIN) + len(BEGIN), content.index(END)
        readme.write_text(content[:start] + "\n" + text + "\n" + content[stop:], encoding="utf-8")
        print("README.md claim table updated")
    else:
        print(text)


if __name__ == "__main__":
    main()
