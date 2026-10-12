"""Train the 42-observable wrench-input EDMDc model (Section IV-B).

Runs the unmodified src/EDMDc_training.py with exactly the settings used for the paper model
(configs/edmdc_training.json). Requires the 360-run training dataset (see DATA.md): the published
.npz archive (preferred) or the original .pkl. The .npz is fed to the training script through
experiments/_train_launcher.py, which rebuilds the identical in-memory dataset.

    python experiments/train_edmdc.py --data data/runs_mixed_n360_high_tilt.npz --output results_local/training
"""
import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

from _common import CONFIGS, SRC, read_json, sha256
from model_digest import content_digest


def main():
    settings = read_json(CONFIGS / "edmdc_training.json")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-hash-check", action="store_true")
    args = parser.parse_args()
    expected = settings["dataset_npz_sha256"] if args.data.suffix == ".npz" else settings["dataset_sha256"]
    if not args.skip_hash_check and sha256(args.data) != expected:
        raise SystemExit("Dataset SHA-256 does not match the paper dataset (use --skip-hash-check to override)")
    model = (args.output / "models" / "wrench_physics42_dt001.pkl").resolve()
    model.parent.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("EDMDC_")}
    env.update(settings["environment"])
    env.update({
        "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "EDMDC_DATA_FILE": str(args.data.resolve()),
        "EDMDC_MODEL_FILE": str(model),
        "EDMDC_PLOT_DIR": str((args.output / "training_plots").resolve()),
        "EDMDC_VALIDATION_INDICES": ",".join(map(str, settings["validation_indices"])),
        "EDMDC_TEST_INDICES": ",".join(map(str, settings["test_indices"])),
    })
    log = args.output / "training_physics42.log"
    print(f"Training physics42 EDMDc; log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        command = [sys.executable, "-u", str(SRC / "EDMDc_training.py")]
        if args.data.suffix == ".npz":
            command = [sys.executable, "-u", str(Path(__file__).with_name("_train_launcher.py")), str(SRC / "EDMDc_training.py")]
        subprocess.run(command, cwd=SRC, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    with model.open("rb") as stream:
        content = content_digest(pickle.load(stream))
    kind = "npz" if args.data.suffix == ".npz" else "pkl"
    file_hash = sha256(model)
    expected_file = settings["expected_model_file_sha256_by_dataset"][kind]
    passed = content == settings["expected_model_content_sha256"]
    print(f"Model: {model}")
    print(f"Content digest (numerical fields): {content} -> "
          f"{'PASS: identical to the paper model' if passed else 'FAIL: differs from the paper model'}")
    print(f"File SHA-256: {file_hash} ({'as expected' if file_hash == expected_file else 'unexpected'} for a model "
          f"trained from the .{kind} dataset; the paper model file is {settings['expected_model_sha256']})")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
