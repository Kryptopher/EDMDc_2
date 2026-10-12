"""One-worker repeat timing for the five frozen interception predictor/bundle cells."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from run_wrench_controller_tuning import environment_record, idle_probe, run_stage, sha256, stable_digest  # noqa: E402

PARENT = ROOT / "artifacts/wrench_controller_tuning_v1"
SELECTION = ROOT / "overnight_campaign_20260929/continuation_20260930/interception/selection.json"
MODEL = ROOT / "artifacts/wrench_revisit_2026-09-21/training_v2/models/wrench_physics42_dt001.pkl"
MODEL_SHA = "090b8dd3106dcf3646dcf991fe23c030d3fe27435443a5f8fae19c1c377b2db4"
CELLS = (("Linear-A", "W1", "W1"), ("EDMDc-A", "W2a", "W1"),
         ("Linear-B", "W1", "W2a"), ("EDMDc-B", "W2a", "W2a"),
         ("Trajectory-linear-B", "W3", "W3"))
STAGE = "timing_pass"


def save_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare() -> tuple[list[dict], dict]:
    parent_path = PARENT / "wrench_tuning_protocol.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    for relative, expected in parent["sources"].items():
        assert sha256(ROOT / relative) == expected, f"Original runner changed: {relative}"
    assert environment_record() == parent["environment"], "Hardware/software environment changed"
    assert sha256(MODEL) == MODEL_SHA, "EDMDc model changed"
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))["selected"]
    originals = json.loads((PARENT / "timing_pass/tasks.json").read_text(encoding="utf-8"))
    templates = {arm: sorted((t for t in originals if t["arm"] == arm), key=lambda t: t["case_id"])
                 for arm in ("W1", "W2a", "W3")}
    assert all(len(rows) == 40 for rows in templates.values())
    assert len({tuple(t["case_id"] for t in rows) for rows in templates.values()}) == 1
    tasks, cell_specs = [], []
    for label, predictor, bundle_owner in CELLS:
        choice = selection[bundle_owner]
        assert len(choice) == 1
        candidate = choice[0]["candidate_id"]
        config = choice[0]["config"]
        assert candidate == ("c04" if bundle_owner == "W1" else "c22")
        assert config["dt"] == 0.01
        assert (config["horizon_steps"], config["block_steps"]) == (
            (50, 10) if bundle_owner == "W1" else (100, 5))
        for template in templates[predictor]:
            task = dict(template)
            task.update(candidate_id=candidate, config=config,
                        output=str((HERE / STAGE / label).resolve()))
            tasks.append(task)
        cell_specs.append({"label": label, "predictor": predictor, "bundle_owner": bundle_owner,
                           "candidate_id": candidate, "config": config,
                           "config_sha256": stable_digest(config)})
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "repeat one-worker complete MPC compute-call timing; no retuning",
        "cells": cell_specs, "case_count_per_cell": 40,
        "planned_updates_per_cell": 120000, "planned_total_updates": 600000,
        "workers": 1, "concurrent_campaign_jobs": 0,
        "timing_field": "solve_seconds measures entire WrenchMPC.compute() call; excludes allocator and nonlinear plant stepping",
        "scenario_construction": "same 40 original isolated timing-pass tracking cases, each 30 s at dt=0.01 s",
        "scenario_ids": [t["case_id"] for t in templates["W1"]],
        "selection_path": str(SELECTION.resolve()), "selection_sha256": sha256(SELECTION),
        "original_tasks_path": str((PARENT / "timing_pass/tasks.json").resolve()),
        "original_tasks_sha256": sha256(PARENT / "timing_pass/tasks.json"),
        "original_parent_protocol_sha256": sha256(parent_path),
        "runner_source_sha256": parent["sources"],
        "model_path": str(MODEL.resolve()), "model_sha256": MODEL_SHA,
        "script_path": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__)),
        "environment": environment_record(), "no_retuning": True,
    }
    protocol_path = HERE / "TIMING_PROTOCOL.json"
    hash_path = HERE / "TIMING_PROTOCOL.sha256"
    if protocol_path.exists():
        assert hash_path.read_text(encoding="ascii").strip() == sha256(protocol_path)
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        existing.pop("created_utc")
        protocol.pop("created_utc")
        assert existing == protocol, "Frozen repeat protocol changed"
    else:
        save_json(protocol_path, protocol)
        hash_path.write_text(sha256(protocol_path) + "\n", encoding="ascii")
    assert hash_path.read_text(encoding="ascii").strip() == sha256(protocol_path)
    print(f"Frozen protocol SHA-256: {sha256(protocol_path)}", flush=True)
    return tasks, protocol


def main() -> None:
    tasks, _ = prepare()
    before = idle_probe()
    rows = run_stage(HERE, STAGE, tasks, 1)
    after = idle_probe()
    original_rows = {(r["arm"], r["case_id"]): r for r in json.loads(
        (PARENT / "timing_pass/episodes.json").read_text(encoding="utf-8"))}
    summaries, manifest = [], []
    for label, predictor, bundle_owner in CELLS:
        candidate = "c04" if bundle_owner == "W1" else "c22"
        group = sorted((r for r in rows if r["arm"] == predictor and r["candidate_id"] == candidate),
                       key=lambda r: r["case_id"])
        assert len(group) == 40
        chunks = []
        for row in group:
            assert row["reference_sha256"] == original_rows[(predictor, row["case_id"])]["reference_sha256"]
            path = Path(row["trace"])
            with np.load(path, allow_pickle=False) as trace:
                values = np.asarray(trace["solve_seconds"], dtype=np.float64) * 1000.0
            assert len(values) == row["updates"] and np.isfinite(values).all() and (values > 0).all()
            chunks.append(values)
            manifest.append({"cell": label, "case_id": row["case_id"], "seed": row["seed"],
                             "reference_sha256": row["reference_sha256"],
                             "completed": row["completed"], "updates": len(values),
                             "trace": str(path.resolve()), "trace_sha256": sha256(path)})
        values = np.concatenate(chunks)
        np.savez_compressed(HERE / f"{label}_raw_update_times_ms.npz", update_times_ms=values)
        summaries.append({"cell": label, "predictor": predictor, "configuration": candidate,
                          "cases": len(group), "completed_episodes": sum(r["completed"] for r in group),
                          "measured_updates": int(values.size), "mean_ms": float(values.mean()),
                          "median_ms": float(np.median(values)), "P95_ms": float(np.percentile(values, 95)),
                          "P99_ms": float(np.percentile(values, 99)), "maximum_ms": float(values.max()),
                          "updates_over_10ms": int(np.count_nonzero(values > 10)),
                          "percentage_over_10ms": float(100 * np.mean(values > 10)),
                          "raw_sha256": sha256(HERE / f"{label}_raw_update_times_ms.npz")})
    save_json(HERE / "RAW_SOURCE_MANIFEST.json", manifest)
    save_json(HERE / "TIMING_SUMMARY.json", {
        "classification": "complete MPC compute-call timing; not QP-only timing",
        "protocol_sha256": sha256(HERE / "TIMING_PROTOCOL.json"),
        "cells": summaries, "throughput_probe": {"before": before, "after": after},
        "no_retuning": True})
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
