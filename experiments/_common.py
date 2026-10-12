"""Shared paths and helpers for the experiment entry points."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (SRC, SRC / "wrench_interception_v1"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
MODEL_PATH = ROOT / "data" / "models" / "wrench_physics42_dt001.pkl"
MODEL_SHA256 = "090b8dd3106dcf3646dcf991fe23c030d3fe27435443a5f8fae19c1c377b2db4"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def check_model(path=MODEL_PATH):
    digest = sha256(path)
    if digest != MODEL_SHA256:
        raise RuntimeError(f"EDMDc model {path} has SHA-256 {digest}, expected {MODEL_SHA256}")
    return Path(path)
