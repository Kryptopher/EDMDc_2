"""Numeric held-out rolling prediction audit of an explicitly selected wrench model."""
import argparse
import csv
import gc
import json
from pathlib import Path
import pickle

import numpy as np

from edmdc_mpc import observables
from edmdc_rollout_batch import rolling_horizon_rmse_batch
from run_wrench_revisit_training import VALIDATION, TEST, sha256
from wrench_revisit_mpc import wrench_features, LiftedWrenchDynamics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.model.open("rb") as stream:
        model = pickle.load(stream)
    dynamics = LiftedWrenchDynamics(model)
    with args.data.open("rb") as stream:
        data = pickle.load(stream)
    indices = VALIDATION + TEST
    states = {index: data["states"][index].copy() for index in indices}
    inputs = {index: data["U"][index].copy() for index in indices}
    del data
    gc.collect()
    rows = []
    for split, run_indices in (("validation", VALIDATION), ("test", TEST)):
        for index in run_indices:
            for horizon in (.1, .5, 1., 2.):
                position, velocity, total, by_state = rolling_horizon_rmse_batch(
                    states[index], inputs[index], model["A"], model["B"], model["scaler"], model["u_scaler"],
                    int(round(horizon/model["dt"])),
                    lambda x, scaler: observables(x, scaler)[dynamics.indices],
                    wrench_features, stride=100,
                )
                rows.append(dict(split=split, run_index=index, horizon_seconds=horizon,
                                 position_rmse_m=position, velocity_rmse_mps=velocity,
                                 yaw_rmse_rad=float(by_state[8]), yaw_rate_rmse_radps=float(by_state[11]),
                                 **{f"rmse_{name}": float(value) for name, value in zip(model["state_labels"], by_state)}))
    with (args.output / "rolling_by_run.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    aggregate = []
    for split in ("validation", "test"):
        for horizon in (.1, .5, 1., 2.):
            chosen = [r for r in rows if r["split"] == split and r["horizon_seconds"] == horizon]
            aggregate.append(dict(split=split, horizon_seconds=horizon, runs=len(chosen),
                **{key: float(np.mean([r[key] for r in chosen])) for key in ("position_rmse_m", "velocity_rmse_mps", "yaw_rmse_rad", "yaw_rate_rmse_radps")}))
    (args.output / "rolling_aggregate.json").write_text(json.dumps(aggregate, indent=2))
    with (args.output / "rolling_aggregate.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0])); writer.writeheader(); writer.writerows(aggregate)
    contract = dict(model=str(args.model.resolve()), model_sha256=sha256(args.model), dataset=str(args.data.resolve()),
                    dataset_sha256=sha256(args.data), dt=model["dt"], aggregation="Arithmetic mean of per-window component RMS errors, then equal mean over nine whole runs.",
                    windows="Complete windows starting every 1 s, plus the final complete window; initial state error is included and is effectively zero.",
                    reset="One measurement initialization per window; no resets within a rollout.",
                    input="Recorded applied wrench U; no reference/desired attitude input.",
                    historical_test_note="These split indices were inspected in earlier research; this is not a new independent untouched test dataset.")
    (args.output / "evaluation_contract.json").write_text(json.dumps(contract, indent=2))
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
