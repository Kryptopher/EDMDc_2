"""Reproducible open-loop audit of the yaw-aware EDMDc training products.

This script deliberately evaluates identification only. Every prediction uses
the recorded input source declared by its checkpoint (motor-feasible wrench or
outer desired-attitude command) and never invokes an MPC or PID controller. It
reports the legacy 10-state subset separately from yaw and yaw rate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edmdc_mpc import observables, scaled_lifted_input_from_phys


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "artifacts" / "acc_paper_data" / "runs_mixed_n300.pkl"
DEFAULT_OUTPUT = SCRIPT_DIR / "artifacts" / "edmdc_training_evaluation_2026-08-18"
VALIDATION_INDICES = [38, 58, 128, 154, 209]
TEST_INDICES = [39, 59, 129, 155, 210]
STATE_LABELS = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r"]
LEGACY_10_INDICES = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10], dtype=int)
GROUPS = {
    "position": np.array([0, 1, 2]),
    "velocity": np.array([3, 4, 5]),
    "roll_pitch": np.array([6, 7]),
    "roll_pitch_rates": np.array([9, 10]),
    "legacy_10_state": LEGACY_10_INDICES,
    "yaw": np.array([8]),
    "yaw_rate": np.array([11]),
    "full_12_state": np.arange(12),
}
MODEL_PATHS = {
    "selected23_dt001": SCRIPT_DIR / "artifacts" / "overnight_round2_2026-08-05" / "models" / "closed_loop_selected_dt001.pkl",
    "compact21_dt001": SCRIPT_DIR / "artifacts" / "overnight_search_2026-08-04" / "models" / "compact_robust_dt001.pkl",
    "compact33_dt001": SCRIPT_DIR / "artifacts" / "overnight_search_2026-08-04" / "models" / "compact_best_dt001.pkl",
    "full56_dt001": SCRIPT_DIR / "artifacts" / "acc_paper_data" / "edmdc_model_dt001_candidate.pkl",
    "full56_dt010_nominal": SCRIPT_DIR / "artifacts" / "acc_paper_data" / "edmdc_model_yaw_wrench.pkl",
    "full56_dt010_stability": SCRIPT_DIR / "artifacts" / "acc_paper_data" / "edmdc_model_stability_candidate.pkl",
}


def wrap_angle(error: np.ndarray) -> np.ndarray:
    return (error + np.pi) % (2.0 * np.pi) - np.pi


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def parse_indices(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


@dataclass
class PreparedModel:
    name: str
    path: Path
    raw: dict

    def __post_init__(self):
        full_a = np.asarray(self.raw["A"], dtype=float)
        full_b = np.asarray(self.raw["B"], dtype=float)
        active = np.asarray(
            self.raw.get("active_observable_indices", np.arange(full_a.shape[0])),
            dtype=int,
        )
        # Compact search artifacts are stored in padded 56-observable matrices.
        if full_a.shape[0] != len(active):
            self.a = full_a[np.ix_(active, active)]
            self.b = full_b[active]
        else:
            # Natively compact models store an n-by-n A matrix while
            # ``active`` still names the selected coordinates in the full
            # 56-term lifting map (for example, state13 includes bias index
            # 55).  Those source indices need not be 0..n-1.
            self.a = full_a
            self.b = full_b
        self.active = active
        self.scaler = self.raw["scaler"]
        self.u_scaler = self.raw["u_scaler"]
        self.state_mean = np.asarray(self.scaler.mean_, dtype=float)
        self.state_scale = np.asarray(self.scaler.scale_, dtype=float)
        self.state_dimension = len(self.state_mean)
        self.input_mean = np.asarray(self.u_scaler.mean_, dtype=float)
        self.input_scale = np.asarray(self.u_scaler.scale_, dtype=float)
        self.dt = float(self.raw.get("dt", 0.1))
        downsampling = self.raw.get("downsampling", {})
        input_method = str(downsampling.get("input_method", "")).lower()
        if input_method in ("decimate", "interval_mean"):
            self.input_downsampling = input_method
        else:
            self.input_downsampling = (
                "decimate" if "take every" in input_method else "interval_mean"
            )
        self.input_source = str(
            self.raw.get("input_source", "applied_wrench")
        ).strip().lower()
        if self.input_source not in (
            "applied_wrench", "requested_wrench", "outer_command"
        ):
            raise ValueError(
                f"{self.name}: unsupported input_source={self.input_source!r}"
            )
        if self.state_dimension == 10:
            # The historical no-yaw model stores its ten reduced physical
            # states as the first ten observables.  Yaw and yaw rate are
            # decoded as zero, making their unavailable prediction explicit.
            self.physical_local = np.arange(10, dtype=int)
            self.physical_indices = LEGACY_10_INDICES
        elif self.state_dimension == 12:
            lookup = {int(raw_index): local for local, raw_index in enumerate(active)}
            missing = [index for index in range(12) if index not in lookup]
            if missing:
                raise ValueError(f"{self.name}: physical observables missing: {missing}")
            self.physical_local = np.array([lookup[index] for index in range(12)], dtype=int)
            self.physical_indices = np.arange(12, dtype=int)
        else:
            raise ValueError(
                f"{self.name}: unsupported state scaler dimension {self.state_dimension}"
            )

    def lift(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        physical = state[self.physical_indices]
        state_std = (physical - self.state_mean) / self.state_scale
        full = observables(state_std, self.scaler)
        return np.asarray(full, dtype=float)[self.active]

    def decode(self, lifted: np.ndarray) -> np.ndarray:
        state_std = np.asarray(lifted, dtype=float)[self.physical_local]
        physical = state_std * self.state_scale + self.state_mean
        if self.state_dimension == 12:
            return physical
        state = np.zeros(12, dtype=float)
        state[self.physical_indices] = physical
        return state

    def scaled_input(self, state: np.ndarray, raw_input: np.ndarray) -> np.ndarray:
        """Apply the input map declared by the trained model."""
        state = np.asarray(state, dtype=float)
        raw = np.asarray(raw_input, dtype=float)[:4]
        if self.input_source == "outer_command":
            return scaled_lifted_input_from_phys(
                state,
                raw,
                self.u_scaler,
                input_lift_type=self.raw.get("input_lift_type", "raw_outer_command"),
            )

        # Exact applied-wrench input lift without per-sample sklearn dispatch.
        phi, theta, psi = state[6:9]
        s_phi, c_phi = np.sin(phi), np.cos(phi)
        s_theta, c_theta = np.sin(theta), np.cos(theta)
        s_psi, c_psi = np.sin(psi), np.cos(psi)
        thrust_direction = np.array([
            c_psi * s_theta * c_phi + s_psi * s_phi,
            s_psi * s_theta * c_phi - c_psi * s_phi,
            c_theta * c_phi,
        ])
        lifted_input = np.concatenate([
            raw,
            raw[0] * thrust_direction,
            raw[1:4] * state[9:12],
        ])
        expected = len(self.input_mean)
        return (lifted_input[:expected] - self.input_mean) / self.input_scale

    def next_lifted(self, lifted: np.ndarray, raw_input: np.ndarray) -> np.ndarray:
        state = self.decode(lifted)
        input_std = self.scaled_input(state, raw_input)
        return self.a @ lifted + self.b @ input_std

    @property
    def spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.a))))


def downsample_series(
    states: np.ndarray,
    inputs: np.ndarray,
    sim_dt: float,
    model_dt: float,
    input_downsampling: str = "interval_mean",
):
    ratio = model_dt / sim_dt
    step = int(round(ratio))
    if step < 1 or not np.isclose(ratio, step, rtol=1e-7, atol=1e-9):
        raise ValueError(f"model dt {model_dt} is not an integer multiple of simulation dt {sim_dt}")
    indices = np.arange(0, len(states), step, dtype=int)
    states_ds = states[indices]
    if input_downsampling == "decimate":
        return states_ds, np.asarray(inputs[indices], dtype=float).copy()
    if input_downsampling != "interval_mean":
        raise ValueError(f"unsupported input downsampling: {input_downsampling}")
    inputs_ds = np.empty((len(indices), inputs.shape[1]), dtype=float)
    if step == 1:
        inputs_ds[:] = inputs[indices]
    else:
        for local in range(len(indices) - 1):
            inputs_ds[local] = np.mean(inputs[indices[local]:indices[local + 1]], axis=0)
        inputs_ds[-1] = inputs_ds[-2]
    return states_ds, inputs_ds


def error_array(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    error = np.asarray(predicted) - np.asarray(truth)
    error[..., 8] = wrap_angle(error[..., 8])
    return error


def metric_columns(error: np.ndarray) -> dict[str, float]:
    error = np.asarray(error, dtype=float)
    output = {}
    for label, indices in GROUPS.items():
        output[f"rmse_{label}"] = float(np.sqrt(np.mean(np.square(error[..., indices]))))
    for index, label in enumerate(STATE_LABELS):
        output[f"rmse_{label}"] = float(np.sqrt(np.mean(np.square(error[..., index]))))
    return output


def rollout(model: PreparedModel, x0: np.ndarray, inputs: np.ndarray, steps: int,
            divergence_limit: float = 1e6):
    predicted = np.full((steps, 12), np.nan, dtype=float)
    predicted[0] = x0
    lifted = model.lift(x0)
    valid_steps = 1
    for index in range(steps - 1):
        lifted = model.next_lifted(lifted, inputs[index])
        state = model.decode(lifted)
        if not np.all(np.isfinite(state)) or np.max(np.abs(state)) > divergence_limit:
            break
        predicted[index + 1] = state
        valid_steps = index + 2
    return predicted, valid_steps


def one_step_metrics(model: PreparedModel, truth: np.ndarray, inputs: np.ndarray):
    predicted = np.empty_like(truth)
    predicted[0] = truth[0]
    for index in range(len(truth) - 1):
        lifted = model.lift(truth[index])
        predicted[index + 1] = model.decode(model.next_lifted(lifted, inputs[index]))
    return metric_columns(error_array(predicted[1:], truth[1:]))


def rolling_metrics(model: PreparedModel, truth: np.ndarray, inputs: np.ndarray,
                    horizon_seconds: float, stride_seconds: float = 1.0):
    horizon_steps = max(1, int(round(horizon_seconds / model.dt)))
    stride_steps = max(1, int(round(stride_seconds / model.dt)))
    squared_error = np.zeros(12, dtype=float)
    samples = 0
    windows = 0
    divergent_windows = 0
    worst_position_rmse = 0.0
    for start in range(0, len(truth) - 1, stride_steps):
        stop = min(start + horizon_steps + 1, len(truth))
        predicted, valid = rollout(model, truth[start], inputs[start:stop], stop - start)
        windows += 1
        if valid < stop - start:
            divergent_windows += 1
            continue
        error = error_array(predicted[1:], truth[start + 1:stop])
        squared_error += np.sum(np.square(error), axis=0)
        samples += len(error)
        worst_position_rmse = max(
            worst_position_rmse,
            float(np.sqrt(np.mean(np.square(error[:, :3])))),
        )
    if samples == 0:
        metrics = {f"rmse_{label}": float("nan") for label in GROUPS}
        metrics.update({f"rmse_{label}": float("nan") for label in STATE_LABELS})
    else:
        state_rmse = np.sqrt(squared_error / samples)
        metrics = {}
        for label, indices in GROUPS.items():
            metrics[f"rmse_{label}"] = float(np.sqrt(np.mean(state_rmse[indices] ** 2)))
        for index, label in enumerate(STATE_LABELS):
            metrics[f"rmse_{label}"] = float(state_rmse[index])
    metrics.update({
        "samples": samples,
        "windows": windows,
        "divergent_windows": divergent_windows,
        "worst_window_position_rmse": worst_position_rmse,
    })
    return metrics


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def aggregate_rows(rows: list[dict], keys: list[str], metric_prefix: str = "rmse_"):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, members in groups.items():
        aggregate = dict(zip(keys, key))
        aggregate["runs"] = len(members)
        for field in members[0]:
            if field.startswith(metric_prefix):
                values = np.asarray([float(member[field]) for member in members])
                aggregate[field] = float(np.sqrt(np.nanmean(values ** 2)))
        if "worst_window_position_rmse" in members[0]:
            aggregate["worst_window_position_rmse"] = float(
                np.nanmax([
                    float(member["worst_window_position_rmse"])
                    for member in members
                ])
            )
            aggregate["divergent_windows"] = int(sum(
                int(float(member["divergent_windows"])) for member in members
            ))
            aggregate["windows"] = int(sum(
                int(float(member["windows"])) for member in members
            ))
        output.append(aggregate)
    return output


def family_name(data: dict, run: int) -> str:
    labels = data.get("family_labels")
    if labels is not None:
        return str(labels[run])
    if run < 50:
        return "helix"
    if run < 100:
        return "fig8"
    if run < 150:
        return "lissajous"
    if run < 200:
        return "waypoint"
    if run < 230:
        return "hover_excitation"
    return "yaw_prbs"


def yaw_profile(data: dict):
    rows = []
    for run, states in enumerate(data["states"]):
        yaw = states[:, 8]
        delta = np.diff(yaw)
        rows.append({
            "run": run,
            "family": family_name(data, run),
            "yaw_min_rad": float(np.min(yaw)),
            "yaw_max_rad": float(np.max(yaw)),
            "yaw_span_rad": float(np.ptp(yaw)),
            "max_abs_yaw_step_rad": float(np.max(np.abs(delta))),
            "pi_boundary_jumps": int(np.count_nonzero(np.abs(delta) > np.pi)),
            "yaw_rate_min_rad_s": float(np.min(states[:, 11])),
            "yaw_rate_max_rad_s": float(np.max(states[:, 11])),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizons", default="0.1,0.5,1,2,5,10")
    parser.add_argument("--continuous-prefixes", default="1,2,5,10,20,50,99")
    parser.add_argument(
        "--validation-indices",
        default=",".join(str(value) for value in VALIDATION_INDICES),
    )
    parser.add_argument(
        "--test-indices",
        default=",".join(str(value) for value in TEST_INDICES),
    )
    parser.add_argument(
        "--model", action="append", default=[], metavar="NAME=PATH",
        help=(
            "Model to audit. Repeat for multiple models. When supplied, these "
            "replace the script's legacy default model list."
        ),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print(f"Loading {args.data}", flush=True)
    with args.data.open("rb") as stream:
        data = pickle.load(stream)
    states_all = np.asarray(data["states"])
    sim_dt = float(data.get("sim_dt", data["t"][0, 1] - data["t"][0, 0]))
    horizons = [float(item) for item in args.horizons.split(",")]
    prefixes = [float(item) for item in args.continuous_prefixes.split(",")]
    validation_indices = parse_indices(args.validation_indices)
    test_indices = parse_indices(args.test_indices)
    if not test_indices:
        parser.error("test indices must be non-empty")
    if set(validation_indices) & set(test_indices):
        parser.error("validation and test indices must be disjoint")
    invalid = [
        index for index in validation_indices + test_indices
        if index < 0 or index >= len(states_all)
    ]
    if invalid:
        parser.error(f"indices outside dataset: {invalid}")

    models = []
    diagnostics = {
        "generated_at_unix": time.time(),
        "data_file": str(args.data.resolve()),
        "data_sha256": file_sha256(args.data),
        "sim_dt": sim_dt,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "yaw_error_definition": "wrapped to [-pi, pi)",
        "rolling_stride_seconds": 1.0,
        "models": {},
    }
    model_paths = MODEL_PATHS
    if args.model:
        model_paths = {}
        for specification in args.model:
            if "=" not in specification:
                parser.error(f"--model must be NAME=PATH, got {specification!r}")
            name, raw_path = specification.split("=", 1)
            if not name.strip() or not raw_path.strip():
                parser.error(f"--model must be NAME=PATH, got {specification!r}")
            model_paths[name.strip()] = Path(raw_path.strip())

    for name, path in model_paths.items():
        if not path.exists():
            print(f"Skipping absent model: {path}", flush=True)
            continue
        with path.open("rb") as stream:
            raw = pickle.load(stream)
        model = PreparedModel(name, path, raw)
        models.append(model)
        diagnostics["models"][name] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "dt": model.dt,
            "active_observables": len(model.active),
            "active_observable_indices": model.active,
            "active_observable_labels": raw.get("active_observable_labels", raw.get("observable_labels", [])),
            "lambda": raw.get("lambda"),
            "input_source": model.input_source,
            "input_downsampling": model.input_downsampling,
            "spectral_radius_active_A": model.spectral_radius,
            "stored_validation_metrics": raw.get("validation_metrics"),
        }
        print(
            f"Loaded {name}: dt={model.dt:g}, n={len(model.active)}, "
            f"input={model.input_source}, rho={model.spectral_radius:.6f}",
            flush=True,
        )

    split_map = {"test": test_indices}
    if validation_indices:
        # Keep validation first in historical audits while allowing a newly
        # generated confirmatory dataset to contain test runs only.
        split_map = {"validation": validation_indices, **split_map}
    contract = {
        "format": "edmdc_prediction_evaluation_v1",
        "data_file": str(args.data.resolve()),
        "data_sha256": file_sha256(args.data.resolve()),
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "horizons_seconds": horizons,
        "continuous_prefixes_seconds": prefixes,
        "models": {
            model.name: {
                "path": str(model.path.resolve()),
                "sha256": file_sha256(model.path.resolve()),
                "dt": model.dt,
                "input_source": model.input_source,
                "input_downsampling": model.input_downsampling,
            }
            for model in models
        },
    }
    contract_path = args.output / "evaluation_contract.json"
    partial_paths = {
        "one": args.output / "one_step_by_run.partial.csv",
        "rolling": args.output / "rolling_by_run.partial.csv",
        "continuous": args.output / "continuous_by_run.partial.csv",
    }
    present_partial = [path.exists() for path in partial_paths.values()]
    if any(present_partial):
        if not all(present_partial) or not contract_path.exists():
            raise RuntimeError(
                "Prediction evaluation has an incomplete checkpoint set"
            )
        recorded = json.loads(contract_path.read_text(encoding="utf-8"))
        if recorded != json_safe(contract):
            raise RuntimeError(
                "Prediction-evaluation checkpoint contract mismatch; refusing reuse"
            )
        restored_one = read_csv(partial_paths["one"])
        restored_rolling = read_csv(partial_paths["rolling"])
        restored_continuous = read_csv(partial_paths["continuous"])
    else:
        atomic_json(contract_path, contract)
        restored_one, restored_rolling, restored_continuous = [], [], []

    total_runs = sum(len(runs) for runs in split_map.values())
    completed_models = set()
    for model in models:
        name = model.name
        counts = (
            sum(row.get("model") == name for row in restored_one),
            sum(row.get("model") == name for row in restored_rolling),
            sum(row.get("model") == name for row in restored_continuous),
        )
        expected_counts = (
            total_runs, total_runs * len(horizons), total_runs * len(prefixes)
        )
        if counts == expected_counts:
            completed_models.add(name)
        elif counts != (0, 0, 0):
            raise RuntimeError(
                f"Partial prediction rows for {name}: {counts}; "
                f"expected {expected_counts}"
            )
    print(
        f"Prediction checkpoint restored {len(completed_models)}/{len(models)} models",
        flush=True,
    )

    yaw_rows = yaw_profile(data)
    write_csv(args.output / "yaw_data_profile.csv", yaw_rows)
    one_step_rows = [
        row for row in restored_one if row.get("model") in completed_models
    ]
    rolling_rows = [
        row for row in restored_rolling if row.get("model") in completed_models
    ]
    continuous_rows = [
        row for row in restored_continuous if row.get("model") in completed_models
    ]
    for model in models:
        if model.name in completed_models:
            print(f"Restored {model.name}", flush=True)
            continue
        print(f"Evaluating {model.name}", flush=True)
        input_key = {
            "outer_command": "U_outer",
            "requested_wrench": "U_requested",
            "applied_wrench": "U",
        }[model.input_source]
        if input_key not in data:
            raise ValueError(
                f"Dataset lacks {input_key} required by model {model.name}"
            )
        inputs_all = np.asarray(data[input_key])
        for split, runs in split_map.items():
            for run in runs:
                truth, inputs = downsample_series(
                    states_all[run], inputs_all[run], sim_dt, model.dt,
                    model.input_downsampling,
                )
                base = {
                    "model": model.name,
                    "input_source": model.input_source,
                    "input_downsampling": model.input_downsampling,
                    "split": split,
                    "run": run,
                    "family": family_name(data, run),
                    "dt": model.dt,
                }
                one_step_rows.append({**base, **one_step_metrics(model, truth, inputs)})
                for horizon in horizons:
                    metrics = rolling_metrics(model, truth, inputs, horizon)
                    rolling_rows.append({**base, "horizon_seconds": horizon, **metrics})

                predicted, valid_steps = rollout(model, truth[0], inputs, len(truth))
                for prefix in prefixes:
                    requested_steps = min(int(round(prefix / model.dt)) + 1, len(truth))
                    valid = min(valid_steps, requested_steps)
                    row = {
                        **base,
                        "prefix_seconds": prefix,
                        "requested_steps": requested_steps,
                        "valid_steps": valid,
                        "diverged_before_prefix": int(valid < requested_steps),
                    }
                    if valid > 1:
                        row.update(metric_columns(error_array(predicted[1:valid], truth[1:valid])))
                    continuous_rows.append(row)

        # Checkpoint after each model so a long audit remains inspectable and resumable.
        write_csv(partial_paths["one"], one_step_rows)
        write_csv(partial_paths["rolling"], rolling_rows)
        write_csv(partial_paths["continuous"], continuous_rows)
        print(f"Checkpointed {model.name}", flush=True)

    one_step_aggregate = aggregate_rows(one_step_rows, ["model", "split", "dt"])
    rolling_aggregate = aggregate_rows(rolling_rows, ["model", "split", "dt", "horizon_seconds"])
    continuous_aggregate = aggregate_rows(continuous_rows, ["model", "split", "dt", "prefix_seconds"])

    write_csv(args.output / "one_step_by_run.csv", one_step_rows)
    write_csv(args.output / "one_step_aggregate.csv", one_step_aggregate)
    write_csv(args.output / "rolling_by_run.csv", rolling_rows)
    write_csv(args.output / "rolling_aggregate.csv", rolling_aggregate)
    write_csv(args.output / "continuous_by_run.csv", continuous_rows)
    write_csv(args.output / "continuous_aggregate.csv", continuous_aggregate)
    diagnostics["elapsed_seconds"] = time.time() - started
    atomic_json(args.output / "model_diagnostics.json", diagnostics)
    print(f"Evaluation complete in {diagnostics['elapsed_seconds']:.1f} s: {args.output}", flush=True)


if __name__ == "__main__":
    main()
