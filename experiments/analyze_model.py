"""EDMDc model facts (Section IV-B) from the frozen model file and its training log.

Shapes, lambda, the 22-value lambda grid and the split come from data/models/wrench_physics42_dt001.pkl.
Rank and condition number of the regression data matrix come from the frozen training log
(recomputing them requires the 894 MB dataset; experiments/train_edmdc.py prints them again).

    python experiments/analyze_model.py
"""
import argparse
import pickle
import re
from pathlib import Path

from _common import MODEL_PATH, ROOT, check_model, sha256, write_json
from model_digest import content_digest

LOG = ROOT / "results" / "training" / "training_physics42.log"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=LOG, help="training log (default: frozen paper log)")
    parser.add_argument("--output", type=Path, default=None, help="optional JSON output")
    args = parser.parse_args()
    check_model()
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    text = args.log.read_text(encoding="utf-8", errors="replace")
    shape = re.search(r"Omega shape: \((\d+), (\d+)\)", text)
    smallest = re.search(r"Omega singular values \(last 10\): \[([^\]]*)\]", text, re.S)
    cond = float(re.search(r"Omega condition number: ([0-9.eE+]+)", text).group(1))
    rows, transitions = int(shape.group(1)), int(shape.group(2))
    sigma_min = float(smallest.group(1).split()[-1])
    out = {"model_sha256": sha256(MODEL_PATH), "model_content_sha256": content_digest(model), "K_shape": list(model["A"].shape), "B_shape": list(model["B"].shape),
           "lambda": float(model["lambda"]), "lambda_grid_size": len(model["regularization_sweep"]),
           "training_runs": len(model["train_indices"]), "training_transitions": transitions,
           "regression_matrix_shape": [rows, transitions], "smallest_singular_value": sigma_min,
           "full_rank": sigma_min > 0, "condition_number": cond}
    print(f"Model file SHA-256:   {out['model_sha256']}")
    print(f"Model content digest: {out['model_content_sha256']} (numerical fields only, experiments/model_digest.py)")
    print(f"K {tuple(out['K_shape'])}, B_u {tuple(out['B_shape'])}; lambda = {out['lambda']:g} "
          f"(selected from a {out['lambda_grid_size']}-value grid)")
    print(f"Training: {out['training_runs']} runs, {transitions:,} transitions")
    print(f"Regression data matrix {rows} x {transitions:,}: smallest singular value {sigma_min:.4g} "
          f"(full rank: {out['full_rank']}), condition number {cond:.6g} ({cond:.3g})")
    if args.output:
        write_json(args.output, out)


if __name__ == "__main__":
    main()
