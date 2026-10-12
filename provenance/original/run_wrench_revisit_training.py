"""Isolated 100 Hz applied-wrench identification; never overwrites paper models."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
VALIDATION = [38, 58, 128, 154, 209, 352, 353, 354, 355]
TEST = [39, 59, 129, 155, 210, 356, 357, 358, 359]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/acc_balanced_high_tilt_v1/runs_mixed_n360_high_tilt.pkl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "training_status.json"
    if (args.output / "selected_model.json").exists():
        raise FileExistsError("Completed identification exists; choose a new output directory")
    with args.data.open("rb") as stream:
        data = pickle.load(stream)
    assert data["input_type"] == "applied_wrench"
    assert data["states"].shape == (360, 6000, 12)
    assert data["U"].shape == (360, 6000, 4)
    assert np.isfinite(data["states"]).all() and np.isfinite(data["U"]).all()
    np.testing.assert_allclose(np.diff(data["t"], axis=1), 0.01, atol=1e-10)
    contract = {
        "experiment": "wrench_revisit_2026_09_21", "status": "training",
        "dataset": str(args.data.resolve()), "dataset_sha256": sha256(args.data),
        "state_labels": ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r"],
        "input_labels": ["thrust", "tau_roll", "tau_pitch", "tau_yaw"],
        "input_key": "U", "alignment": "x_k,U_k,x_k+1", "dt": 0.01,
        "validation_indices": VALIDATION, "test_indices": TEST,
        "training_runs": 342, "validation_selection_only": True,
        "kinematic_row_overwrite": False, "constant_row_preserved": True,
        "early_transient_weight": 1.0, "dictionaries": ["physics42", "full56"],
        "selection_rule": "minimum existing normalized validation rolling position/velocity/yaw/yaw-rate score",
        "training_source_sha256": sha256(ROOT / "EDMDc_training.py"),
        "historical_split_note": "Previously inspected held-out runs are reused for diagnostic continuity, not claimed as a new pristine test set.",
        "started_unix": time.time(), "results": [],
    }
    status_path.write_text(json.dumps(contract, indent=2))
    del data
    gc.collect()
    for dictionary in contract["dictionaries"]:
        model_path = args.output / "models" / f"wrench_{dictionary}_dt001.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if model_path.exists():
            raise FileExistsError(model_path)
        log_path = args.output / f"training_{dictionary}.log"
        environment = {key: value for key, value in os.environ.items() if not key.startswith("EDMDC_")}
        environment.update({
            "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1",
            "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
            "EDMDC_DATA_FILE": str(args.data.resolve()),
            "EDMDC_MODEL_FILE": str(model_path.resolve()),
            "EDMDC_PLOT_DIR": str((args.output / "training_plots" / dictionary).resolve()),
            "EDMDC_INPUT_SOURCE": "applied_wrench", "EDMDC_DT": "0.01",
            "EDMDC_OBSERVABLE_SET": dictionary, "EDMDC_TRAIN_FRACTION": "1.0",
            "EDMDC_SHORT_HORIZON_SECONDS": "2.0", "EDMDC_TILT_WEIGHTING": "none",
            "EDMDC_ENFORCE_KINEMATIC_ROWS": "0", "EDMDC_EARLY_TRANSIENT_WEIGHT": "1.0",
            "EDMDC_BATCH_ROLLOUT": "1",
            "EDMDC_VALIDATION_INDICES": ",".join(map(str, VALIDATION)),
            "EDMDC_TEST_INDICES": ",".join(map(str, TEST)),
        })
        contract["current_dictionary"] = dictionary
        status_path.write_text(json.dumps(contract, indent=2))
        print(f"Training {dictionary}; log: {log_path}", flush=True)
        try:
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-u", str(ROOT / "EDMDc_training.py")], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        except Exception as error:
            contract.update(status="failed", error=repr(error))
            status_path.write_text(json.dumps(contract, indent=2))
            raise
        with model_path.open("rb") as stream:
            model = pickle.load(stream)
        assert model["input_type"] == "applied_wrench" and model["dt"] == 0.01
        assert not model["enforce_kinematic_rows"]
        row = {"dictionary": dictionary, "model": str(model_path.resolve()), "sha256": sha256(model_path)}
        for key in ("lambda", "lambda_selection_score", "lambda_selection_rolling_pos", "lambda_selection_rolling_vel", "lambda_selection_yaw", "lambda_selection_r"):
            row[key] = float(model[key])
        contract["results"].append(row)
        status_path.write_text(json.dumps(contract, indent=2))
        print(json.dumps(row), flush=True)
        del model
        gc.collect()
    selected = min(contract["results"], key=lambda item: item["lambda_selection_score"])
    contract.update(status="complete", completed_unix=time.time(), selected=selected)
    status_path.write_text(json.dumps(contract, indent=2))
    (args.output / "selected_model.json").write_text(json.dumps(selected, indent=2))
    print("Training complete; selected: " + json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
