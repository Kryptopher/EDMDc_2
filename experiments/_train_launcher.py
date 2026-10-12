"""Run the unmodified src/EDMDc_training.py on an .npz dataset.

EDMDc_training.py opens EDMDC_DATA_FILE and calls pickle.load on it. For an .npz file this launcher
answers that single call with dataset_io.load_dataset(), which rebuilds exactly the dictionary
stored in the original pickle, and then executes the training script unchanged.
"""
import os
import pickle
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dataset_io import load_dataset  # noqa: E402

DATA = Path(os.environ["EDMDC_DATA_FILE"]).resolve()
_pickle_load = pickle.load


def _load(stream, *args, **kwargs):
    name = getattr(stream, "name", None)
    if name is not None and Path(name).resolve() == DATA and DATA.suffix == ".npz":
        return load_dataset(DATA)
    return _pickle_load(stream, *args, **kwargs)


pickle.load = _load
script = Path(sys.argv[1]).resolve()
sys.argv = [str(script)]
# Same import path as running the script directly: its own directory first.
sys.path.remove(str(HERE))
sys.path[0] = str(script.parent)
runpy.run_path(str(script), run_name="__main__")
