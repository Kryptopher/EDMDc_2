"""Run the compact attitude-command observable and learning-curve study.

Dictionary capacity is compared at 100% of the fixed training split.  The
learning curve then keeps the selected full dictionary fixed and varies only
the deterministic fraction of training runs.  Validation and test indices are
identical in every experiment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path


ORIGINAL_VALIDATION = [38, 58, 128, 154, 209]
ORIGINAL_TEST = [39, 59, 129, 155, 210]
DICTIONARIES = ("state13", "trig19", "physics42")
LEARNING_FRACTIONS = (0.10, 0.25, 0.50)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train(script, dataset, output_dir, name, observable_set, fraction,
          validation, test, skip_existing):
    model = output_dir / "models" / f"{name}.pkl"
    plot_dir = output_dir / "training_plots" / name
    log_path = output_dir / "logs" / f"{name}.log"
    model.parent.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and model.is_file():
        print(f"reusing {model}", flush=True)
    else:
        environment = os.environ.copy()
        environment.update({
            "MPLBACKEND": "Agg",
            "EDMDC_DATA_FILE": str(dataset.resolve()),
            "EDMDC_MODEL_FILE": str(model.resolve()),
            "EDMDC_PLOT_DIR": str(plot_dir.resolve()),
            "EDMDC_DT": "0.01",
            "EDMDC_SHORT_HORIZON_SECONDS": "2.0",
            "EDMDC_INPUT_SOURCE": "outer_command",
            "EDMDC_OUTER_INPUT_LIFT": "attitude_error",
            "EDMDC_VALIDATION_INDICES": ",".join(map(str, validation)),
            "EDMDC_TEST_INDICES": ",".join(map(str, test)),
            "EDMDC_TILT_WEIGHTING": "none",
            "EDMDC_OBSERVABLE_SET": observable_set,
            "EDMDC_TRAIN_FRACTION": f"{fraction:.8g}",
        })
        command = [sys.executable, str(script.resolve())]
        print(f"training {name}: {observable_set}, fraction={fraction}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=script.parent, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    with model.open("rb") as stream:
        saved = pickle.load(stream)
    return {
        "name": name,
        "role": "dictionary" if fraction == 1.0 else "learning_curve",
        "observable_set": observable_set,
        "observable_count": int(saved["n_obs"]),
        "train_fraction": float(saved.get("train_fraction", fraction)),
        "training_runs": len(saved["train_indices"]),
        "training_transitions": len(saved["train_indices"]) * 5999,
        "lambda": float(saved["lambda"]),
        "validation_rolling_position_rmse_m": float(
            saved["lambda_selection_rolling_pos"]
        ),
        "validation_rolling_velocity_rmse_mps": float(
            saved["lambda_selection_rolling_vel"]
        ),
        "validation_yaw_rmse_rad": float(saved["lambda_selection_yaw"]),
        "validation_yaw_rate_rmse_radps": float(saved["lambda_selection_r"]),
        "model": str(model.resolve()),
        "model_sha256": sha256(model),
        "log": str(log_path.resolve()),
        "plot_dir": str(plot_dir.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--accepted-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.dataset.open("rb") as stream:
        data = pickle.load(stream)
    augmentation = data.get("augmentation")
    if not augmentation:
        raise ValueError("Dataset lacks high-tilt augmentation metadata")
    validation = ORIGINAL_VALIDATION + list(augmentation["validation_indices"])
    test = ORIGINAL_TEST + list(augmentation["test_indices"])
    del data
    gc.collect()

    script = Path(__file__).resolve().parent / "EDMDc_training.py"
    results = []
    for dictionary in DICTIONARIES:
        results.append(train(
            script, args.dataset, args.output_dir,
            f"dictionary_{dictionary}_f100", dictionary, 1.0,
            validation, test, args.skip_existing,
        ))
        gc.collect()
    for fraction in LEARNING_FRACTIONS:
        results.append(train(
            script, args.dataset, args.output_dir,
            f"learning_full56_f{int(round(100*fraction)):03d}",
            "full56", fraction, validation, test, args.skip_existing,
        ))
        gc.collect()

    with args.accepted_model.open("rb") as stream:
        accepted = pickle.load(stream)
    results.append({
        "name": "accepted_full56_f100",
        "role": "dictionary_and_learning_curve",
        "observable_set": accepted.get("observable_set", "full56"),
        "observable_count": int(accepted["n_obs"]),
        "train_fraction": 1.0,
        "training_runs": len(accepted["train_indices"]),
        "training_transitions": len(accepted["train_indices"]) * 5999,
        "lambda": float(accepted["lambda"]),
        "validation_rolling_position_rmse_m": float(
            accepted["lambda_selection_rolling_pos"]
        ),
        "validation_rolling_velocity_rmse_mps": float(
            accepted["lambda_selection_rolling_vel"]
        ),
        "validation_yaw_rmse_rad": float(accepted["lambda_selection_yaw"]),
        "validation_yaw_rate_rmse_radps": float(accepted["lambda_selection_r"]),
        "model": str(args.accepted_model.resolve()),
        "model_sha256": sha256(args.accepted_model),
        "log": None,
        "plot_dir": None,
    })

    manifest = {
        "experiment": "attitude_command_capacity_and_learning_curve",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "dt_seconds": 0.01,
        "prediction_horizon_seconds": 2.0,
        "input_source": "outer_command",
        "input_lift": "outer_attitude_error_thrust_vector",
        "validation_indices": validation,
        "test_indices": test,
        "selection_rule": "ridge penalty selected on fixed validation split",
        "test_split_role": "final comparison only",
        "results": results,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
