"""Convert the training dataset from the original pickle to the published, lossless .npz archive.

The NPZ layout and the loader live in experiments/dataset_io.py (single definition used for both
writing and reading). After writing, the archive is reloaded and compared with the pickle bitwise:
every array (with dtype and shape), dict key, list element and scalar must be identical.

    python tools/pkl_to_npz.py --pickle runs_mixed_n360_high_tilt.pkl --output data/runs_mixed_n360_high_tilt.npz

Optionally compare the array contents with an existing archive (e.g. the published one):

    python tools/pkl_to_npz.py --pickle ... --output new.npz --compare data/runs_mixed_n360_high_tilt.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from _common import CONFIGS, read_json, sha256  # noqa: E402
from dataset_io import convert, verify  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pickle", type=Path, required=True, help="original runs_mixed_n360_high_tilt.pkl")
    parser.add_argument("--output", type=Path, required=True, help="output .npz path")
    parser.add_argument("--compare", type=Path, default=None, help="existing .npz to compare array contents with")
    args = parser.parse_args()
    settings = read_json(CONFIGS / "edmdc_training.json")
    if sha256(args.pickle) != settings["dataset_sha256"]:
        raise SystemExit("input pickle is not the paper dataset (SHA-256 mismatch)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    convert(args.pickle, args.output)
    data = verify(args.pickle, args.output)
    held = set(settings["validation_indices"]) | set(settings["test_indices"])
    train = [i for i in range(data["states"].shape[0]) if i not in held]
    transitions = (data["states"].shape[1] - 1) * len(train)
    assert len(train) == settings["training_runs"] == 342 and transitions == 2051658, (len(train), transitions)
    print("Round trip bitwise identical (every array, dtype, key and scalar).")
    print(f"Runs: {data['states'].shape[0]} (train {len(train)}, validation {len(settings['validation_indices'])}, "
          f"test {len(settings['test_indices'])}); training transitions {transitions:,}")
    print(f"{args.output}: {args.output.stat().st_size:,} bytes, SHA-256 {sha256(args.output)}")
    if args.compare:
        with np.load(args.output, allow_pickle=False) as a, np.load(args.compare, allow_pickle=False) as b:
            assert sorted(a.files) == sorted(b.files), "different archive members"
            for key in a.files:
                x, y = a[key], b[key]
                assert x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y), key
        same_file = sha256(args.output) == sha256(args.compare)
        print(f"Array contents identical to {args.compare} ({len(a.files)} members); "
              f"archive files {'byte-identical' if same_file else 'differ in container bytes only'}.")


if __name__ == "__main__":
    main()
