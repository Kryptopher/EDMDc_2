"""Content digest of an EDMDc model: SHA-256 over its numerical data only.

The model file also stores text metadata, e.g. the name of the training data file (`source_file`),
so two numerically identical models trained from the .pkl and the .npz copies of the dataset have
different file hashes. This digest covers every number in the model and ignores all strings:

- numpy arrays: dtype, shape and raw bytes (K = model["A"], B_u = model["B"], ...)
- int / float / bool scalars (lambda, dt, n_obs, selection scores, ...), as int64 / float64 / bool bytes
- the scikit-learn scalers' numerical attributes (mean_, scale_, var_, n_samples_seen_, ...)
- lists and dicts recursively (lambda sweep, run indices), dict keys in sorted order
"""
import hashlib

import numpy as np


def _feed(h, value, path):
    if isinstance(value, str) or value is None:
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "OUS":
            return
        h.update(f"{path}:a:{value.dtype.str}:{value.shape};".encode())
        h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, (bool, np.bool_)):
        h.update(f"{path}:b:{int(value)};".encode())
    elif isinstance(value, (int, np.integer)):
        h.update(f"{path}:i;".encode() + np.int64(value).tobytes())
    elif isinstance(value, (float, np.floating)):
        h.update(f"{path}:f;".encode() + np.float64(value).tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            _feed(h, value[key], f"{path}/{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _feed(h, item, f"{path}[{i}]")
    elif hasattr(value, "__dict__"):  # e.g. sklearn StandardScaler
        _feed(h, vars(value), f"{path}<{type(value).__name__}>")
    else:
        raise TypeError(f"{path}: unsupported type {type(value).__name__}")


def content_digest(model):
    h = hashlib.sha256()
    _feed(h, model, "")
    return h.hexdigest()
