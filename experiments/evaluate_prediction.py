"""Held-out open-loop prediction RMSE of the EDMDc and yaw-scheduled hover-linear predictors.

Rollouts start every 1 s on each of the nine held-out test runs, use the recorded applied
wrench, and are never reset inside a window. Metrics are averaged over windows, then equally
over runs (same procedure as the original evaluation script); frozen output:
results/prediction/heldout_rolling_aggregate.csv. Requires the training dataset (see DATA.md),
as the published .npz archive or the original .pkl.

    python experiments/evaluate_prediction.py --data data/runs_mixed_n360_high_tilt.npz
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

from _common import CONFIGS, check_model, read_json
from dataset_io import load_dataset
from Simulation import quad_sim
from wrench_revisit_mpc import LiftedWrenchDynamics, NominalWrenchDynamics

HORIZONS = (0.5, 1.0, 2.0)


def reference_yaw(data, index):
    if "reference_array" in data:
        array = np.asarray(data["reference_array"][index])
        if array.ndim == 2 and array.shape[1] >= 10:
            return array[:, 9]
    return np.asarray([point["yaw"] for point in data["ref_traj_list"][index]], dtype=float)


def starts_for(n, horizon, stride=100):
    count = n - horizon
    starts = list(range(0, count, stride))
    if starts[-1] != count - 1:
        starts.append(count - 1)
    return np.asarray(starts)


def wrap(errors):
    errors[:, :, 6:9] = np.arctan2(np.sin(errors[:, :, 6:9]), np.cos(errors[:, :, 6:9]))
    return errors


def edmdc_errors(states, inputs, starts, horizon, dynamics):
    z = np.stack([dynamics.initialize(states[s]) for s in starts])
    pred = np.empty((len(starts), horizon + 1, 12))
    pred[:, 0] = dynamics.decode(z)
    for step in range(horizon):
        z = dynamics.step(z, inputs[starts + step])
        pred[:, step + 1] = dynamics.decode(z)
    truth = np.stack([states[s:s + horizon + 1] for s in starts])
    return wrap(truth - pred)


def yaw_linear_errors(states, inputs, yaws, starts, horizon, dynamics, hover):
    x = states[starts].copy()
    pred = np.empty((len(starts), horizon + 1, 12))
    pred[:, 0] = x
    for step in range(horizon):
        eq = np.zeros_like(x)
        eq[:, 8] = yaws[starts + step + 1]  # prescribed reference yaw, never measured future yaw
        F, G = dynamics.jacobians(eq, np.broadcast_to(hover, (len(starts), 4)))
        x = eq + np.einsum("nij,nj->ni", F, x - eq) + np.einsum("nij,nj->ni", G, inputs[starts + step] - hover)
        pred[:, step + 1] = x
    truth = np.stack([states[s:s + horizon + 1] for s in starts])
    return wrap(truth - pred)


def grouped(errors, sl):
    return float(np.mean(np.sqrt(np.mean(errors[:, :, sl] ** 2, axis=(1, 2)))))


def main():
    settings = read_json(CONFIGS / "edmdc_training.json")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", type=int, default=settings["test_indices"])
    args = parser.parse_args()
    with check_model().open("rb") as f:
        edmdc = LiftedWrenchDynamics(pickle.load(f))
    data = load_dataset(args.data)
    sim = quad_sim()
    linear = NominalWrenchDynamics(sim.quad, 0.01)
    hover = np.array([sim.quad.m * sim.quad.g, 0.0, 0.0, 0.0])
    table = {}
    for index in args.runs:
        states = np.asarray(data["states"][index], dtype=float)
        inputs = np.asarray(data["U"][index], dtype=float)
        yaws = reference_yaw(data, index)
        for seconds in HORIZONS:
            h = int(round(seconds / 0.01))
            s = starts_for(len(states), h)
            for name, e in (("EDMDc", edmdc_errors(states, inputs, s, h, edmdc)),
                            ("Hover-linear", yaw_linear_errors(states, inputs, yaws, s, h, linear, hover))):
                row = [grouped(e, slice(0, 3)) * 1e3, grouped(e, slice(3, 6)) * 1e3, grouped(e, slice(6, 7)) * 1e3,
                       grouped(e, slice(7, 8)) * 1e3, grouped(e, slice(8, 9)) * 1e3]
                table.setdefault((name, seconds), []).append(row)
    print(f"Held-out open-loop RMSE over runs {args.runs}")
    print(f"{'Model':13s} {'H [s]':>5s} {'pos [mm]':>9s} {'vel [mm/s]':>11s} {'roll [mrad]':>12s} {'pitch [mrad]':>13s} {'yaw [mrad]':>11s}")
    for (name, seconds), rows in table.items():
        m = np.mean(rows, axis=0)
        print(f"{name:13s} {seconds:5.1f} {m[0]:9.2f} {m[1]:11.2f} {m[2]:12.2f} {m[3]:13.2f} {m[4]:11.2f}")


if __name__ == "__main__":
    main()
