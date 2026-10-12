"""Training-dataset I/O: lossless pickle -> NPZ conversion and a loader for either format.

NPZ layout (all numeric arrays keep their original dtype; axis 0 is always the run index, so run
boundaries and the run-level train/validation/test split are preserved):

    time            (6000,)          float64   common time axis
    t               (360, 6000)      float64   per-run time
    states          (360, 6000, 12)  float64   [p, v, phi/theta/psi, p/q/r]
    U               (360, 6000, 4)   float64   applied wrench (training input)
    U_requested     (360, 6000, 4)   float64   requested wrench before allocation
    U_outer         (360, 6000, 4)   float64   outer-loop desired attitude command
    ref_pos/vel/acc (360, 6000, 3)   float64   reference trajectory (from ref_traj_list)
    ref_yaw         (360, 6000)      float64
    ref_yaw_rate    (360, 6000)      float64
    ref_outer_command (360, 6000, 4) float64   'outer_command' reference entry (high-tilt runs only;
                                               zeros elsewhere, see has_outer_command)
    has_outer_command (360,)         bool      runs whose reference points carry 'outer_command'
    family_labels   (360,)           str
    run_seeds       (360,)           int64
    metadata_json   ()               str       every remaining (non-array) field, type-preserving JSON

load_dataset() rebuilds exactly the dictionary that the original pickle contains, including
ref_traj_list as a list (runs) of lists (time steps) of {'pos','vel','acc','yaw','yaw_rate'} dicts.
"""
import json
import pickle
from pathlib import Path

import numpy as np

ARRAY_KEYS = ("time", "t", "states", "U", "U_requested", "U_outer")
REF_VECTORS = ("pos", "vel", "acc")
REF_SCALARS = ("yaw", "yaw_rate")
SPECIAL_KEYS = set(ARRAY_KEYS) | {"ref_traj_list", "family_labels", "run_seeds"}


def _encode(o):
    """JSON-safe, type-preserving encoding (dict keys of any scalar type, tuples, floats exact)."""
    if isinstance(o, dict):
        return {"__dict__": [[_encode(k), _encode(v)] for k, v in o.items()]}
    if isinstance(o, tuple):
        return {"__tuple__": [_encode(v) for v in o]}
    if isinstance(o, list):
        return [_encode(v) for v in o]
    if o is None or type(o) in (str, int, float, bool):
        return o
    raise TypeError(f"metadata value of type {type(o).__name__} is not supported")


def _decode(o):
    if isinstance(o, dict):
        if "__dict__" in o:
            return {_decode(k): _decode(v) for k, v in o["__dict__"]}
        return tuple(_decode(v) for v in o["__tuple__"])
    if isinstance(o, list):
        return [_decode(v) for v in o]
    return o


def convert(pickle_path, npz_path):
    with Path(pickle_path).open("rb") as f:
        data = pickle.load(f)
    unknown = [k for k, v in data.items() if k not in SPECIAL_KEYS and isinstance(v, np.ndarray)]
    if unknown:
        raise ValueError(f"unexpected array fields {unknown}")
    arrays = {k: data[k] for k in ARRAY_KEYS}
    refs = data["ref_traj_list"]
    runs, steps = len(refs), len(refs[0])
    for name in REF_VECTORS:
        out = np.empty((runs, steps, 3), dtype=np.float64)
        for r, run in enumerate(refs):
            assert len(run) == steps
            for k, point in enumerate(run):
                value = point[name]
                assert isinstance(value, np.ndarray) and value.dtype == np.float64 and value.shape == (3,)
                out[r, k] = value
        arrays[f"ref_{name}"] = out
    for name in REF_SCALARS:
        out = np.empty((runs, steps), dtype=np.float64)
        for r, run in enumerate(refs):
            for k, point in enumerate(run):
                value = point[name]
                assert type(value) is float, type(value)
                out[r, k] = value
        arrays[f"ref_{name}"] = out
    base_keys = list(REF_VECTORS + REF_SCALARS)
    has_outer = np.zeros(runs, dtype=bool)
    outer = np.zeros((runs, steps, 4), dtype=np.float64)
    for r, run in enumerate(refs):
        keys = [list(point) for point in run]
        if all(k == base_keys for k in keys):
            continue
        assert all(k == base_keys + ["outer_command"] for k in keys), f"run {r}: unexpected reference keys"
        has_outer[r] = True
        for k, point in enumerate(run):
            value = point["outer_command"]
            assert isinstance(value, np.ndarray) and value.dtype == np.float64 and value.shape == (4,)
            outer[r, k] = value
    arrays["ref_outer_command"] = outer
    arrays["has_outer_command"] = has_outer
    assert all(type(x) is str for x in data["family_labels"]) and all(type(x) is int for x in data["run_seeds"])
    arrays["family_labels"] = np.asarray(data["family_labels"], dtype=str)
    arrays["run_seeds"] = np.asarray(data["run_seeds"], dtype=np.int64)
    meta = {k: v for k, v in data.items() if k not in SPECIAL_KEYS}
    arrays["metadata_json"] = np.asarray(json.dumps(_encode(meta), allow_nan=True))
    arrays["key_order_json"] = np.asarray(json.dumps(list(data.keys())))
    np.savez_compressed(npz_path, **arrays)


def load_dataset(path):
    """Return the dataset dictionary from an .npz archive (preferred) or the original .pkl."""
    path = Path(path)
    if path.suffix != ".npz":
        with path.open("rb") as f:
            return pickle.load(f)
    with np.load(path, allow_pickle=False) as z:
        meta = _decode(json.loads(str(z["metadata_json"])))
        order = json.loads(str(z["key_order_json"]))
        vec = {n: z[f"ref_{n}"] for n in REF_VECTORS}
        sca = {n: z[f"ref_{n}"] for n in REF_SCALARS}
        outer, has_outer = z["ref_outer_command"], z["has_outer_command"]
        data = {k: z[k] for k in ARRAY_KEYS}
        data["family_labels"] = [str(x) for x in z["family_labels"]]
        data["run_seeds"] = [int(x) for x in z["run_seeds"]]
    runs, steps = sca["yaw"].shape
    def point(r, k):
        p = {"pos": vec["pos"][r, k].copy(), "vel": vec["vel"][r, k].copy(), "acc": vec["acc"][r, k].copy(),
             "yaw": float(sca["yaw"][r, k]), "yaw_rate": float(sca["yaw_rate"][r, k])}
        if has_outer[r]:
            p["outer_command"] = outer[r, k].copy()
        return p
    data["ref_traj_list"] = [[point(r, k) for k in range(steps)] for r in range(runs)]
    data.update(meta)
    return {k: data[k] for k in order}


def _equal(a, b, path="root"):
    if type(a) is not type(b):
        raise AssertionError(f"{path}: type {type(a).__name__} != {type(b).__name__}")
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
            raise AssertionError(f"{path}: array differs ({a.dtype}{a.shape} vs {b.dtype}{b.shape})")
    elif isinstance(a, dict):
        if list(a) != list(b):
            raise AssertionError(f"{path}: keys differ")
        for k in a:
            _equal(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _equal(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        if not (a == b or (a != a and b != b)) or np.float64(a).tobytes() != np.float64(b).tobytes():
            raise AssertionError(f"{path}: {a!r} != {b!r}")
    elif a != b:
        raise AssertionError(f"{path}: {a!r} != {b!r}")


def verify(pickle_path, npz_path):
    """Deep, bitwise comparison of the pickle and the NPZ reconstruction (raises on any difference)."""
    with Path(pickle_path).open("rb") as f:
        original = pickle.load(f)
    _equal(original, load_dataset(npz_path))
    return original
