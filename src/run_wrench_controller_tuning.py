"""Preregister, tune, freeze, blindly test, and report direct-wrench MPCs.

Tracking only.  The four arms differ only in their prediction model.  The
protocol and source/model hashes are frozen before the first tuning episode,
and the blind-test command refuses to run before selected configurations are
independently frozen and hashed.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np
from scipy.stats import binomtest

from PID_Mixer import pid_mixer
from Simulation import quad_sim
from publication_scenarios import latin_hypercube, tracking_reference
from run_wrench_revisit import ROOT, run_case, sha256, write_csv
from wrench_revisit_mpc import WrenchConfig, physical_reference


ARMS = {
    "W1": {
        "label": "Yaw-scheduled hover-linear MPC",
        "controller": "yaw_linear",
        "model_key": None,
    },
    "W2a": {
        "label": "EDMDc-MPC (training_v2)",
        "controller": "edmdc",
        "model_key": "training_v2",
    },
    "W2b": {
        "label": "EDMDc-MPC (new_physics42)",
        "controller": "edmdc",
        "model_key": "new_physics42",
    },
    "W3": {
        "label": "Nominal-model LTV MPC (accuracy oracle)",
        "controller": "ltv",
        "model_key": None,
    },
}
FAMILIES = ("helix", "figure8", "lissajous", "waypoint")
MODEL_PATHS = {
    "training_v2": ROOT / "artifacts/wrench_revisit_2026-09-21/training_v2/models/wrench_physics42_dt001.pkl",
    "new_physics42": ROOT / "artifacts/wrench_reidentification_2026-09-21/dataset_v1/training/wrench_physics42_reidentified_dt001.pkl",
}
MODEL_HASHES = {
    "training_v2": "090b8dd3106dcf3646dcf991fe23c030d3fe27435443a5f8fae19c1c377b2db4",
    "new_physics42": "e36916ea6cb571b136e64f1755718235774b26770931dc48e7881bb90b415fa4",
}
SOURCES = (
    "run_wrench_controller_tuning.py", "run_wrench_revisit.py",
    "wrench_revisit_mpc.py", "publication_scenarios.py", "Simulation.py",
    "quadcopter.py", "PID_Mixer.py", "edmdc_mpc.py",
)
PROTOCOL_NAME = "wrench_tuning_protocol.json"
PROTOCOL_HASH_NAME = "wrench_tuning_protocol.sha256.json"
LHS_SEED = 2026092201
BOOTSTRAP_SEED = 2026092202
BOOTSTRAP_RESAMPLES = 20_000
TUNING_BASE_SEED = 91_000_000
BLIND_BASE_SEED = 92_000_000


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for attempt in range(9):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 8:
                raise
            time.sleep(min(0.025 * 2**attempt, 0.25))


def stable_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_once(path, value):
    path = Path(path)
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != value:
            raise RuntimeError(f"Frozen file differs: {path}")
        return
    dump(path, value)


def log_map(unit, lower, upper):
    return float(np.exp(np.log(lower) + float(unit) * np.log(upper / lower)))


def choose(unit, values):
    return values[min(int(float(unit) * len(values)), len(values) - 1)]


def candidate_design():
    """Current configuration plus 24 identical LHS candidates for every arm."""
    base = asdict(WrenchConfig())
    candidates = [{
        "candidate_id": "c00",
        "label": "current",
        "config": base,
        "design": {
            "horizon_seconds": base["horizon_steps"] * base["dt"],
            "move_blocks": base["horizon_steps"] // base["block_steps"],
            "stage_input_penalty": True,
        },
    }]
    design = latin_hypercube(24, 12, LHS_SEED)
    horizon_steps = (30, 50, 75, 100)
    move_blocks = {
        30: (5, 6, 10, 15),
        50: (5, 10, 25),
        75: (5, 15, 25),
        100: (5, 10, 20, 25),
    }
    factor_bounds = {
        "position": (0.25, 8.0),
        "velocity": (0.25, 8.0),
        "attitude": (0.25, 8.0),
        "body_rates": (0.10, 4.0),
        "terminal": (0.40, 4.0),
        "input": (0.05, 5.0),
        "increment": (0.10, 10.0),
    }
    for number, unit in enumerate(design, 1):
        steps = choose(unit[0], horizon_steps)
        blocks = choose(unit[1], move_blocks[steps])
        factors = {
            name: log_map(unit[index + 2], *bounds)
            for index, (name, bounds) in enumerate(factor_bounds.items())
        }
        config = json.loads(json.dumps(base))
        config["horizon_steps"] = steps
        config["block_steps"] = steps // blocks
        config["q"] = (
            np.asarray(base["q"], dtype=float)
            * np.repeat([
                factors["position"], factors["velocity"],
                factors["attitude"], factors["body_rates"],
            ], 3)
        ).tolist()
        config["terminal_scale"] = float(
            base["terminal_scale"] * factors["terminal"]
        )
        # WrenchMPC tiles R over prediction stages, not move blocks. Thus this
        # multiplier is a per-stage cost and is invariant to block count.
        config["r"] = (
            np.asarray(base["r"], dtype=float) * factors["input"]
        ).tolist()
        config["rd"] = (
            np.asarray(base["rd"], dtype=float) * factors["increment"]
        ).tolist()
        config["trust"] = [
            float(5.0 + 15.0 * unit[9]),
            float(0.20 + 0.90 * unit[10]),
            float(0.20 + 0.90 * unit[10]),
            float(0.05 + 0.30 * unit[11]),
        ]
        candidates.append({
            "candidate_id": f"c{number:02d}",
            "label": f"lhs_{number:02d}",
            "config": config,
            "design": {
                "horizon_seconds": steps * base["dt"],
                "move_blocks": blocks,
                "factors": factors,
                "trust": {
                    "thrust_n": config["trust"][0],
                    "roll_pitch_torque_nm": config["trust"][1],
                    "yaw_torque_nm": config["trust"][3],
                },
                "stage_input_penalty": True,
            },
        })
    assert len(candidates) == 25
    return candidates


def tuning_cases():
    cases = []
    for family_index, family in enumerate(FAMILIES):
        for replicate in range(3):
            seed = TUNING_BASE_SEED + 1000 * family_index + replicate
            for scale in (1.0, 1.75):
                for yaw_mode in ("fixed", "path"):
                    cases.append({
                        "case_id": (
                            f"{family}_{seed}_scale{scale:g}_{yaw_mode}"
                        ),
                        "family": family, "seed": seed, "scale": scale,
                        "yaw_mode": yaw_mode, "replicate": replicate,
                    })
    assert len(cases) == 48
    return cases


def screening_cases(all_cases):
    """Twelve cases: 3/family, 6/heading, 6/speed scale."""
    selected = []
    for family_index, family in enumerate(FAMILIES):
        family_rows = [row for row in all_cases if row["family"] == family]
        patterns = [(0, 1.0, "fixed"), (1, 1.75, "path")]
        patterns.append(
            (2, 1.75, "fixed") if family_index % 2 == 0
            else (2, 1.0, "path")
        )
        for replicate, scale, yaw_mode in patterns:
            selected.append(next(
                row for row in family_rows
                if row["replicate"] == replicate
                and row["scale"] == scale
                and row["yaw_mode"] == yaw_mode
            ))
    assert len(selected) == 12
    assert {family: sum(r["family"] == family for r in selected)
            for family in FAMILIES} == {family: 3 for family in FAMILIES}
    assert sum(r["yaw_mode"] == "fixed" for r in selected) == 6
    assert sum(r["scale"] == 1.0 for r in selected) == 6
    return selected


def blind_cases():
    cases = []
    for family_index, family in enumerate(FAMILIES):
        for replicate in range(50):
            seed = BLIND_BASE_SEED + 1000 * family_index + replicate
            yaw_mode = "fixed" if replicate < 25 else "path"
            cases.append({
                "case_id": f"{family}_{seed}_scale1.75_{yaw_mode}",
                "family": family, "seed": seed, "scale": 1.75,
                "yaw_mode": yaw_mode, "replicate": replicate,
            })
    assert len(cases) == 200
    return cases


def environment_record():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "osqp")
        },
    }


def build_protocol(workers):
    all_tuning = tuning_cases()
    screen = screening_cases(all_tuning)
    test = blind_cases()
    return {
        "experiment": "wrench_controller_tuning_and_blind_test_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "tracking only; no interception; no unrequested tail metrics",
        "rate_hz": 100,
        "arms": ARMS,
        "shared_contract": {
            "plant": "same nonlinear quadrotor plant",
            "allocator_and_motor_limits": "shared",
            "reference_preview": "same prescribed reference",
            "qp_structure_solver_settings_warm_start": "shared",
            "difference_between_arms": "prediction model only",
            "W3_label": "accuracy oracle",
            "W3_timing_note": (
                "unoptimized Python with complex-step Jacobians; no cost "
                "ordering is claimed"
            ),
        },
        "models": {
            key: {"path": str(path.resolve()), "sha256": MODEL_HASHES[key]}
            for key, path in MODEL_PATHS.items()
        },
        "candidate_design": {
            "common_seed": LHS_SEED,
            "current_plus_lhs": [1, 24],
            "same_candidate_list_for_every_arm": True,
            "horizon_seconds": [0.3, 0.5, 0.75, 1.0],
            "move_blocks_by_horizon_steps": {
                "30": [5, 6, 10, 15], "50": [5, 10, 25],
                "75": [5, 15, 25], "100": [5, 10, 20, 25],
            },
            "multiplicative_ranges": {
                "position": [0.25, 8.0], "velocity": [0.25, 8.0],
                "attitude": [0.25, 8.0], "body_rates": [0.10, 4.0],
                "terminal": [0.40, 4.0], "input": [0.05, 5.0],
                "increment": [0.10, 10.0],
            },
            "trust_ranges": {
                "thrust_n": [5.0, 20.0],
                "roll_pitch_torque_nm": [0.20, 1.10],
                "yaw_torque_nm": [0.05, 0.35],
            },
            "input_penalty_semantics": (
                "R is applied once per prediction stage in WrenchMPC; it is "
                "not multiplied by the number of move blocks"
            ),
            "candidates": candidate_design(),
        },
        "tuning": {
            "seed_range_inclusive": [TUNING_BASE_SEED, TUNING_BASE_SEED + 3002],
            "all_48_cases": all_tuning,
            "screen_12_cases": screen,
            "screen_duration_seconds": 10,
            "validation_duration_seconds": 30,
            "finalists_per_arm": "three best non-current candidates plus current",
            "rank": [
                "incomplete episodes", "allocator-altered steps",
                "solver failures", "mean tracking score", "candidate ID",
            ],
            "score": (
                "position_rmse/0.1m + 0.25*velocity_rmse/0.25mps + "
                "0.1*yaw_rmse/0.1rad + "
                "0.01*normalized_input_increment_rms"
            ),
            "solve_time_used_for_ranking": False,
        },
        "selection": {
            "separate_config_per_arm": True,
            "freeze_before_blind_test": True,
            "W2_forward_rule": (
                "If W2a and W2b selected configurations have equal reliability "
                "counts, the lower selected full-validation tracking score goes "
                "forward. Otherwise the same preregistered lexicographic "
                "reliability ordering applies. Both remain in the blind test."
            ),
        },
        "blind_test": {
            "seed_range_inclusive": [BLIND_BASE_SEED, BLIND_BASE_SEED + 3049],
            "cases": test,
            "scenarios": 200, "scenarios_per_family": 50,
            "scenarios_per_heading_within_family": 25,
            "speed_scale": 1.75, "duration_seconds": 30,
            "seed_paired_across_arms": True,
            "generated_only_after_selection_freeze": True,
            "reported_metrics": [
                "completion count",
                "position velocity yaw RMSE mean median P95 max on completed",
                "roll and pitch RMSE",
                "allocator-altered steps and constraint violations",
                "solver failures",
                "solve time mean P95 max count over 10ms with throughput probe",
            ],
            "paired_comparisons": {
                "pairs": ["W2a-W1", "W2a-W3", "W2b-W1", "W2b-W3"],
                "metric": "per-scenario position RMSE among mutually completed pairs",
                "win_test": "two-sided exact sign test, ties excluded",
                "bootstrap": {
                    "resamples": BOOTSTRAP_RESAMPLES,
                    "seed": BOOTSTRAP_SEED,
                    "statistic": "mean paired first-minus-second difference",
                },
            },
        },
        "excluded_data": [
            "12-run held-out identification set", "32 sanity cases",
            "all earlier confirmation seeds",
        ],
        "failure_policy": "retain and label every failed run",
        "workers": workers,
        "sources": {name: sha256(ROOT / name) for name in SOURCES},
        "environment": environment_record(),
        "hardware_actions": False,
        "manuscript_edits": False,
    }


def verify_protocol(output):
    protocol_path = output / PROTOCOL_NAME
    hash_path = output / PROTOCOL_HASH_NAME
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    recorded = json.loads(hash_path.read_text(encoding="utf-8"))["sha256"]
    if sha256(protocol_path) != recorded:
        raise RuntimeError("Protocol hash mismatch")
    for name, expected in protocol["sources"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"Frozen source changed: {name}")
    for key, record in protocol["models"].items():
        if sha256(Path(record["path"])) != record["sha256"]:
            raise RuntimeError(f"Frozen model changed: {key}")
    return protocol


def command_freeze(output, workers):
    output.mkdir(parents=True, exist_ok=False)
    protocol = build_protocol(workers)
    path = output / PROTOCOL_NAME
    dump(path, protocol)
    dump(output / PROTOCOL_HASH_NAME, {"sha256": sha256(path)})
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in SOURCES:
        (snapshot / name).write_bytes((ROOT / name).read_bytes())
    print(json.dumps({
        "protocol": str(path), "sha256": sha256(path),
        "candidates": len(protocol["candidate_design"]["candidates"]),
        "screen_cases": len(protocol["tuning"]["screen_12_cases"]),
        "validation_cases": len(protocol["tuning"]["all_48_cases"]),
        "blind_cases": len(protocol["blind_test"]["cases"]),
    }, indent=2), flush=True)


def episode_score(row):
    if not row["completed"]:
        return None
    return float(
        row["position_rmse_m"] / 0.1
        + 0.25 * row["velocity_rmse_mps"] / 0.25
        + 0.1 * row["yaw_rmse_rad"] / 0.1
        + 0.01 * row["normalized_input_increment_rms"]
    )


def enrich_trial(task, row):
    with np.load(row["trace"]) as trace:
        inputs = np.asarray(trace["U"], dtype=float)
        requested = np.asarray(trace["U_requested"], dtype=float)
        forces = np.asarray(trace["motor_forces"], dtype=float)
        increments = np.diff(inputs, axis=0) / np.array([10.0, 1.0, 1.0, 0.3])
        row["normalized_input_increment_rms"] = float(
            np.sqrt(np.mean(increments ** 2)) if len(increments) else 0.0
        )
        states = np.asarray(trace["states"], dtype=float)
        reference = np.asarray(trace["reference"], dtype=float)
        error = states - reference
        error[:, 6:9] = np.arctan2(
            np.sin(error[:, 6:9]), np.cos(error[:, 6:9])
        )
        state_rmse = np.sqrt(np.mean(error ** 2, axis=0))
        row["roll_rmse_rad"] = float(state_rmse[6]) if row["completed"] else None
        row["pitch_rmse_rad"] = float(state_rmse[7]) if row["completed"] else None
        row["updates"] = len(inputs)
        sim = quad_sim()
        hover = sim.quad.m * sim.quad.g
        lower = np.array([0.05 * hover, -1.1, -1.1, -0.35])
        upper = np.array([sim.quad.m * (sim.quad.g + 10), 1.1, 1.1, 0.35])
        force_max = pid_mixer.fct_max_motor_forces(
            sim.quad.kT, sim.max_speed, sim.quad.prop_efficiency
        )
        input_bad = np.any(
            (requested < lower - 1e-5) | (requested > upper + 1e-5), axis=1
        )
        motor_bad = np.any(
            (forces < 1e-4 - 1e-5)
            | (forces > force_max - 1e-4 + 1e-5), axis=1
        )
        row["constraint_violations"] = int(np.count_nonzero(input_bad | motor_bad))
        # run_case extends a reference by the candidate-specific horizon. Hash
        # a canonical duration+1 s prefix so different horizons and incomplete
        # episodes are still auditable against exactly the same reference.
        canonical_reference = tracking_reference(
            task["family"], task["seed"], task["duration"] + 1.01,
            speed_scale=task["scale"], yaw_rate_scale=1.0,
        )
        if task["yaw_mode"] == "fixed":
            heading = float(canonical_reference[0]["yaw"])
            for point in canonical_reference:
                point["yaw"], point["yaw_rate"] = heading, 0.0
        canonical_desired = physical_reference(canonical_reference, sim, 0.01)
        canonical_count = int(round(task["duration"] / 0.01)) + 1
        row["reference_sha256"] = hashlib.sha256(
            canonical_desired[:canonical_count].tobytes()
        ).hexdigest()
    row.update(
        arm=task["arm"], candidate_id=task["candidate_id"],
        stage=task["stage"], task_sha256=stable_digest(task),
    )
    row["selection_score"] = episode_score(row)
    dump(Path(row["trace"]).with_suffix(".json"), row)
    return row


def trial(task):
    return enrich_trial(task, run_case(task))


def make_tasks(output, stage, cases, choices, duration):
    tasks = []
    max_choices = max(len(value) for value in choices.values())
    for case in cases:
        for slot in range(max_choices):
            for arm, contract in ARMS.items():
                if slot >= len(choices[arm]):
                    continue
                candidate = choices[arm][slot]
                model_key = contract["model_key"]
                tasks.append({
                    **case, "arm": arm, "controller": contract["controller"],
                    "candidate_id": candidate["candidate_id"],
                    "duration": duration, "config": candidate["config"],
                    "model": (
                        str(MODEL_PATHS[model_key].resolve())
                        if model_key else None
                    ),
                    "stage": stage,
                    "output": str(
                        (output / stage / arm / candidate["candidate_id"]).resolve()
                    ),
                })
    return tasks


def run_stage(output, stage, tasks, workers):
    folder = output / stage
    folder.mkdir(parents=True, exist_ok=True)
    write_once(folder / "tasks.json", tasks)
    rows, pending = [], []
    for task in tasks:
        path = (Path(task["output"]) / "traces" / task["case_id"]
                / f"{task['controller']}.json")
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("task_sha256") != stable_digest(task):
                raise RuntimeError(f"Cached task mismatch: {path}")
            if not Path(cached["trace"]).is_file():
                raise RuntimeError(f"Cached trace missing: {path}")
            rows.append(cached)
        else:
            pending.append(task)
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(trial, task): task for task in pending}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_csv(folder / "episodes.csv", rows)
            dump(folder / "episodes.json", rows)
            progress = {
                "status": "running", "stage": stage,
                "completed": len(rows), "total": len(tasks),
                "elapsed_seconds": time.time() - started,
                "last": {key: row[key] for key in (
                    "arm", "candidate_id", "case_id", "completed",
                    "position_rmse_m", "stop_reason",
                )},
            }
            dump(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    for case_id in {row["case_id"] for row in rows}:
        hashes = {row["reference_sha256"] for row in rows
                  if row["case_id"] == case_id}
        if len(hashes) != 1:
            raise RuntimeError(f"Reference mismatch for {case_id}")
    write_csv(folder / "episodes.csv", rows)
    dump(folder / "episodes.json", rows)
    return rows


def rank_candidates(rows, candidates):
    ranked = []
    for candidate in candidates:
        group = [row for row in rows
                 if row["candidate_id"] == candidate["candidate_id"]]
        if not group:
            raise RuntimeError(f"Unevaluated candidate {candidate['candidate_id']}")
        completed = [row for row in group if row["completed"]]
        ranked.append({
            "candidate_id": candidate["candidate_id"],
            "label": candidate["label"], "episodes": len(group),
            "incomplete_episodes": len(group) - len(completed),
            "allocator_altered_steps": sum(
                row["allocator_altered_steps"] for row in group
            ),
            "constraint_violations": sum(
                row["constraint_violations"] for row in group
            ),
            "solver_failures": sum(row["solver_failures"] for row in group),
            "mean_tracking_score": (
                float(np.mean([row["selection_score"] for row in completed]))
                if completed else None
            ),
            "mean_position_rmse_completed_m": (
                float(np.mean([row["position_rmse_m"] for row in completed]))
                if completed else None
            ),
        })
    return sorted(ranked, key=lambda row: (
        row["incomplete_episodes"], row["allocator_altered_steps"],
        row["solver_failures"],
        row["mean_tracking_score"]
        if row["mean_tracking_score"] is not None else float("inf"),
        row["candidate_id"],
    ))


def verify_selection_freeze(output, protocol):
    path = output / "selected_configs" / "selection_freeze.json"
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze["protocol_sha256"] != sha256(output / PROTOCOL_NAME):
        raise RuntimeError("Selection freeze refers to the wrong protocol")
    for arm, record in freeze["configs"].items():
        config_path = Path(record["path"])
        if sha256(config_path) != record["sha256"]:
            raise RuntimeError(f"Selected config hash mismatch: {arm}")
    if set(freeze["configs"]) != set(ARMS):
        raise RuntimeError("Selection freeze does not contain all four arms")
    if protocol["blind_test"]["generated_only_after_selection_freeze"] is not True:
        raise RuntimeError("Invalid blind-test ordering contract")
    return freeze


def command_tune(output, workers):
    protocol = verify_protocol(output)
    if workers != protocol["workers"]:
        raise RuntimeError("Worker count must match the frozen protocol")
    candidates = protocol["candidate_design"]["candidates"]
    choices = {arm: candidates for arm in ARMS}
    screen = run_stage(
        output, "screening", make_tasks(
            output, "screening", protocol["tuning"]["screen_12_cases"],
            choices, protocol["tuning"]["screen_duration_seconds"],
        ), workers,
    )
    finalists, screen_rankings = {}, {}
    for arm in ARMS:
        ranking = rank_candidates(
            [row for row in screen if row["arm"] == arm], candidates
        )
        screen_rankings[arm] = ranking
        best_noncurrent = [row["candidate_id"] for row in ranking
                           if row["candidate_id"] != "c00"][:3]
        ids = ["c00", *best_noncurrent]
        finalists[arm] = [next(c for c in candidates
                               if c["candidate_id"] == identity)
                           for identity in ids]
    write_once(output / "screening" / "rankings.json", screen_rankings)
    validation = run_stage(
        output, "validation", make_tasks(
            output, "validation", protocol["tuning"]["all_48_cases"],
            finalists, protocol["tuning"]["validation_duration_seconds"],
        ), workers,
    )
    validation_rankings, selected = {}, {}
    for arm in ARMS:
        ranking = rank_candidates(
            [row for row in validation if row["arm"] == arm], finalists[arm]
        )
        validation_rankings[arm] = ranking
        winner = ranking[0]
        candidate = next(c for c in candidates
                         if c["candidate_id"] == winner["candidate_id"])
        selected[arm] = {"candidate": candidate, "validation": winner}
    write_once(output / "validation" / "rankings.json", validation_rankings)
    for arm in ARMS:
        write_csv(output / "screening" / f"ranking_{arm}.csv",
                  screen_rankings[arm])
        write_csv(output / "validation" / f"ranking_{arm}.csv",
                  validation_rankings[arm])

    def model_choice_tuple(arm):
        row = selected[arm]["validation"]
        return (
            row["incomplete_episodes"], row["allocator_altered_steps"],
            row["solver_failures"],
            row["mean_tracking_score"]
            if row["mean_tracking_score"] is not None else float("inf"),
            arm,
        )

    forward = min(("W2a", "W2b"), key=model_choice_tuple)
    selected_dir = output / "selected_configs"
    selected_dir.mkdir(exist_ok=True)
    records = {}
    for arm in ARMS:
        record = {
            "arm": arm, "label": ARMS[arm]["label"],
            "prediction_model": ARMS[arm]["model_key"],
            "prediction_model_sha256": (
                MODEL_HASHES[ARMS[arm]["model_key"]]
                if ARMS[arm]["model_key"] else None
            ),
            "selected_candidate": selected[arm]["candidate"],
            "selected_validation": selected[arm]["validation"],
            "full_candidate_table": candidates,
            "screening_ranking": screen_rankings[arm],
            "validation_ranking": validation_rankings[arm],
            "protocol_sha256": sha256(output / PROTOCOL_NAME),
        }
        path = selected_dir / f"{arm}.json"
        write_once(path, record)
        digest = sha256(path)
        write_once(path.with_suffix(".sha256.json"), {"sha256": digest})
        records[arm] = {"path": str(path.resolve()), "sha256": digest}
    freeze = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(output / PROTOCOL_NAME),
        "configs": records,
        "W2_model_forward": forward,
        "W2_model_forward_reason": (
            "predeclared lexicographic reliability ordering followed by "
            "selected full-validation tracking score"
        ),
        "blind_test_started": False,
    }
    write_once(selected_dir / "selection_freeze.json", freeze)
    dump(output / "progress.json", {
        "status": "selection_frozen", "stage": "selection_frozen",
        "screen_episodes": len(screen), "validation_episodes": len(validation),
        "W2_model_forward": forward,
    })
    print(json.dumps(freeze, indent=2), flush=True)


def idle_probe(seconds=1.0):
    started = time.perf_counter()
    iterations, value = 0, 0.0
    while time.perf_counter() - started < seconds:
        value += np.sqrt(iterations + 1.0)
        iterations += 1
    return {
        "iterations_per_second": float(iterations / seconds),
        "probe_seconds": seconds, "guard_value": float(value),
    }


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {key: None for key in ("mean", "median", "p95", "max")}
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)), "max": float(np.max(values)),
    }


def summarize_blind(rows):
    summaries = []
    for arm in ARMS:
        group = [row for row in rows if row["arm"] == arm]
        completed = [row for row in group if row["completed"]]
        solve_samples = []
        for row in group:
            with np.load(row["trace"]) as trace:
                solve_samples.append(np.asarray(trace["solve_seconds"]) * 1000.0)
        solve = np.concatenate(solve_samples)
        summary = {
            "arm": arm, "label": ARMS[arm]["label"],
            "accuracy_oracle": arm == "W3", "episodes": len(group),
            "completed": len(completed),
            "allocator_altered_steps": sum(
                row["allocator_altered_steps"] for row in group
            ),
            "constraint_violations": sum(
                row["constraint_violations"] for row in group
            ),
            "solver_failures": sum(row["solver_failures"] for row in group),
            "position_rmse_m": distribution(
                [row["position_rmse_m"] for row in completed]
            ),
            "velocity_rmse_mps": distribution(
                [row["velocity_rmse_mps"] for row in completed]
            ),
            "yaw_rmse_rad": distribution(
                [row["yaw_rmse_rad"] for row in completed]
            ),
            "roll_rmse_rad": distribution(
                [row["roll_rmse_rad"] for row in completed]
            ),
            "pitch_rmse_rad": distribution(
                [row["pitch_rmse_rad"] for row in completed]
            ),
            "solve_time_ms": {
                "mean": float(np.mean(solve)),
                "p95": float(np.percentile(solve, 95)),
                "max": float(np.max(solve)),
                "samples_over_10ms": int(np.count_nonzero(solve > 10.0)),
                "samples": int(len(solve)),
            },
        }
        summaries.append(summary)
    return summaries


def paired_statistics(rows):
    lookup = {(row["case_id"], row["arm"]): row for row in rows}
    results = []
    for comparison_index, (first, second) in enumerate((
        ("W2a", "W1"), ("W2a", "W3"),
        ("W2b", "W1"), ("W2b", "W3"),
    )):
        case_ids = sorted({row["case_id"] for row in rows})
        paired = [
            (lookup[(case, first)], lookup[(case, second)])
            for case in case_ids
            if lookup[(case, first)]["completed"]
            and lookup[(case, second)]["completed"]
        ]
        differences = np.asarray([
            left["position_rmse_m"] - right["position_rmse_m"]
            for left, right in paired
        ])
        wins = int(np.count_nonzero(differences < -1e-15))
        losses = int(np.count_nonzero(differences > 1e-15))
        ties = int(len(differences) - wins - losses)
        p_value = float(binomtest(wins, wins + losses, 0.5).pvalue) \
            if wins + losses else 1.0
        rng = np.random.default_rng(BOOTSTRAP_SEED + comparison_index)
        indices = rng.integers(
            0, len(differences), (BOOTSTRAP_RESAMPLES, len(differences))
        ) if len(differences) else np.empty((0, 0), dtype=int)
        bootstrap = np.mean(differences[indices], axis=1) \
            if len(differences) else np.asarray([])
        results.append({
            "first": first, "second": second,
            "first_label": ARMS[first]["label"],
            "second_label": ARMS[second]["label"],
            "second_is_accuracy_oracle": second == "W3",
            "total_scenarios": len(case_ids), "mutually_completed": len(paired),
            "excluded_for_incompletion": len(case_ids) - len(paired),
            "first_wins": wins, "second_wins": losses, "ties": ties,
            "sign_test_two_sided_p": p_value,
            "mean_first_minus_second_m": (
                float(np.mean(differences)) if len(differences) else None
            ),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED + comparison_index,
            "bootstrap_ci95_m": (
                [float(np.percentile(bootstrap, 2.5)),
                 float(np.percentile(bootstrap, 97.5))]
                if len(bootstrap) else [None, None]
            ),
        })
    return results


def make_report(output, summaries, comparisons, throughput, freeze):
    lines = [
        "# Wrench controller tuning and blind tracking test", "",
        f"Protocol SHA-256: `{sha256(output / PROTOCOL_NAME)}`", "",
        "The test is tracking-only. W3 is a nominal-model accuracy oracle. "
        "Its unoptimized Python/complex-step timing is reported without any "
        "claim about implementation cost ordering.", "",
        "## Frozen tuning selections", "",
        "| Arm | Selected candidate | Full-validation score | Incomplete | "
        "Allocator altered | Solver failures |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm, frozen in freeze["configs"].items():
        record = json.loads(Path(frozen["path"]).read_text(encoding="utf-8"))
        selected = record["selected_candidate"]
        validation = record["selected_validation"]
        score = validation["mean_tracking_score"]
        score_text = f"{score:.8g}" if score is not None else "N/A"
        label = record["label"] + (" **(accuracy oracle)**" if arm == "W3" else "")
        lines.append(
            f"| {label} | {selected['candidate_id']}: {selected['label']} | "
            f"{score_text} | {validation['incomplete_episodes']} | "
            f"{validation['allocator_altered_steps']} | "
            f"{validation['solver_failures']} |"
        )
    lines += [
        "", "Complete candidate and ranking tables are in each frozen arm "
        "configuration and in the screening/validation CSV files.", "",
        "## Blind test", "",
        "RMSE distributions contain completed episodes only and are always "
        "shown with completion counts.", "",
        "| Arm | Complete | Position mean/median/P95/max [m] | Velocity "
        "mean/median/P95/max [m/s] | Yaw mean/median/P95/max [rad] |",
        "|---|---:|---|---|---|",
    ]
    for row in summaries:
        label = row["label"]
        if row["accuracy_oracle"]:
            label += " **(accuracy oracle)**"
        def compact(key):
            value = row[key]
            return "/".join(
                f"{value[name]:.6g}" if value[name] is not None else "N/A"
                for name in ("mean", "median", "p95", "max")
            )
        lines.append(
            f"| {label} | {row['completed']}/{row['episodes']} | "
            f"{compact('position_rmse_m')} | {compact('velocity_rmse_mps')} | "
            f"{compact('yaw_rmse_rad')} |"
        )
    lines += [
        "", "| Arm | Roll mean/median/P95/max [rad] | Pitch "
        "mean/median/P95/max [rad] | Allocator altered | Constraint "
        "violations | Solver failures |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        def compact(key):
            value = row[key]
            return "/".join(
                f"{value[name]:.6g}" if value[name] is not None else "N/A"
                for name in ("mean", "median", "p95", "max")
            )
        label = row["label"] + (" **(accuracy oracle)**"
                                if row["accuracy_oracle"] else "")
        lines.append(
            f"| {label} | {compact('roll_rmse_rad')} | "
            f"{compact('pitch_rmse_rad')} | {row['allocator_altered_steps']} | "
            f"{row['constraint_violations']} | {row['solver_failures']} |"
        )
    lines += [
        "", "## Solve time", "",
        "| Arm | Mean [ms] | P95 [ms] | Max [ms] | >10 ms / samples |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        timing = row["solve_time_ms"]
        label = row["label"] + (" **(accuracy oracle)**"
                                if row["accuracy_oracle"] else "")
        lines.append(
            f"| {label} | {timing['mean']:.4f} | {timing['p95']:.4f} | "
            f"{timing['max']:.4f} | {timing['samples_over_10ms']}/"
            f"{timing['samples']} |"
        )
    before = throughput["before"]["iterations_per_second"]
    after = throughput["after"]["iterations_per_second"]
    lines += [
        "", f"Throughput probe: {before:.0f} iterations/s before and "
        f"{after:.0f} iterations/s after (ratio {after/before:.3f}).", "",
        "## Paired position comparisons", "",
        "Differences are first minus second; negative favors the EDMDc arm. "
        "The sign test excludes ties. Bootstrap intervals use 20,000 paired "
        "scenario resamples.", "",
        "| Comparison | Mutual completion | Wins-losses-ties | Sign p | Mean "
        "difference [m] | Bootstrap 95% CI [m] |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in comparisons:
        oracle = " (accuracy oracle)" if row["second_is_accuracy_oracle"] else ""
        ci = row["bootstrap_ci95_m"]
        difference = row["mean_first_minus_second_m"]
        difference_text = f"{difference:.6g}" if difference is not None else "N/A"
        ci_text = (f"[{ci[0]:.6g}, {ci[1]:.6g}]"
                   if ci[0] is not None else "N/A")
        lines.append(
            f"| {row['first']} vs {row['second']}{oracle} | "
            f"{row['mutually_completed']}/{row['total_scenarios']} | "
            f"{row['first_wins']}-{row['second_wins']}-{row['ties']} | "
            f"{row['sign_test_two_sided_p']:.6g} | {difference_text} | "
            f"{ci_text} |"
        )
    lines += [
        "", "## Frozen selection", "",
        f"EDMDc model going forward by the preregistered rule: "
        f"**{freeze['W2_model_forward']}**.", "",
        "Full candidate and ranking tables are embedded in each JSON under "
        "`selected_configs/`.", "", "## Things that need my decision", "",
        "None for completion of this preregistered campaign.", "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def command_blind(output, workers):
    protocol = verify_protocol(output)
    if workers != protocol["workers"]:
        raise RuntimeError("Worker count must match the frozen protocol")
    freeze = verify_selection_freeze(output, protocol)
    before = idle_probe()
    selected = {
        arm: json.loads(Path(record["path"]).read_text(encoding="utf-8"))
        for arm, record in freeze["configs"].items()
    }
    choices = {
        arm: [record["selected_candidate"]]
        for arm, record in selected.items()
    }
    tasks = make_tasks(
        output, "blind_test", protocol["blind_test"]["cases"], choices,
        protocol["blind_test"]["duration_seconds"],
    )
    rows = run_stage(output, "blind_test", tasks, workers)
    after = idle_probe()
    throughput = {"before": before, "after": after,
                  "ratio_after_over_before": after["iterations_per_second"]
                  / before["iterations_per_second"]}
    summaries = summarize_blind(rows)
    comparisons = paired_statistics(rows)
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=True)
    dump(analysis / "blind_summary.json", summaries)
    dump(analysis / "paired_comparisons.json", comparisons)
    dump(analysis / "throughput_probe.json", throughput)
    write_csv(analysis / "blind_summary_flat.csv", [{
        "arm": row["arm"], "label": row["label"],
        "accuracy_oracle": row["accuracy_oracle"],
        "episodes": row["episodes"], "completed": row["completed"],
        "allocator_altered_steps": row["allocator_altered_steps"],
        "constraint_violations": row["constraint_violations"],
        "solver_failures": row["solver_failures"],
        **{f"{metric}_{stat}": row[metric][stat]
           for metric in ("position_rmse_m", "velocity_rmse_mps",
                          "yaw_rmse_rad", "roll_rmse_rad", "pitch_rmse_rad")
           for stat in ("mean", "median", "p95", "max")},
        **{f"solve_time_ms_{key}": value
           for key, value in row["solve_time_ms"].items()},
    } for row in summaries])
    write_csv(analysis / "paired_comparisons.csv", comparisons)
    make_report(output, summaries, comparisons, throughput, freeze)
    artifact_paths = [
        output / PROTOCOL_NAME, output / "selected_configs/selection_freeze.json",
        analysis / "blind_summary.json", analysis / "paired_comparisons.json",
        analysis / "throughput_probe.json", output / "RESULTS.md",
    ] + [Path(record["path"]) for record in freeze["configs"].values()]
    manifest = {
        "status": "complete", "protocol_sha256": sha256(output / PROTOCOL_NAME),
        "artifacts": {str(path.resolve()): sha256(path) for path in artifact_paths},
        "models": protocol["models"], "source_hashes": protocol["sources"],
        "episodes": len(rows), "hardware_actions": False,
        "manuscript_edits": False,
    }
    dump(output / "result_manifest.json", manifest)
    dump(output / "progress.json", {
        "status": "complete", "stage": "complete", "episodes": len(rows),
        "report": str((output / "RESULTS.md").resolve()),
    })
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "tune", "blind"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = args.output.resolve()
    try:
        if args.command == "freeze":
            command_freeze(output, args.workers)
        elif args.command == "tune":
            command_tune(output, args.workers)
        else:
            command_blind(output, args.workers)
    except Exception:
        if output.exists():
            dump(output / "failure.json", {
                "status": "failed", "command": args.command,
                "traceback": traceback.format_exc(),
            })
        raise


if __name__ == "__main__":
    main()
