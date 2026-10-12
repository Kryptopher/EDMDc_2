"""Summarize only the four completed cells after the user stopped W3 timing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CELLS = (("Linear-A", "W1", "c04", 50, 5, 10),
         ("EDMDc-A", "W2a", "c04", 50, 5, 10),
         ("Linear-B", "W1", "c22", 100, 20, 5),
         ("EDMDc-B", "W2a", "c22", 100, 20, 5))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = HERE / "TIMING_PROTOCOL.json"
    assert (HERE / "TIMING_PROTOCOL.sha256").read_text(encoding="ascii").strip() == sha256(protocol)
    episodes_path = HERE / "timing_pass/episodes.json"
    rows = json.loads(episodes_path.read_text(encoding="utf-8"))
    results, sources, reference_sets = [], [], []
    for label, arm, candidate, np_horizon, nc, block in CELLS:
        group = sorted((r for r in rows if r["arm"] == arm and r["candidate_id"] == candidate),
                       key=lambda r: r["case_id"])
        assert len(group) == 40 and all(r["completed"] and r["updates"] == 3000 for r in group)
        reference_sets.append({r["case_id"]: (r["seed"], r["reference_sha256"]) for r in group})
        chunks = []
        for row in group:
            path = Path(row["trace"])
            with np.load(path, allow_pickle=False) as trace:
                values = np.asarray(trace["solve_seconds"], dtype=np.float64) * 1000
            assert values.shape == (3000,) and np.isfinite(values).all() and (values > 0).all()
            chunks.append(values)
            sources.append({"cell": label, "case_id": row["case_id"], "seed": row["seed"],
                            "reference_sha256": row["reference_sha256"],
                            "updates": 3000, "trace": str(path), "sha256": sha256(path)})
        samples = np.concatenate(chunks)
        raw_path = HERE / f"{label}_raw_update_times_ms.npz"
        np.savez_compressed(raw_path, update_times_ms=samples)
        results.append({"cell": label, "predictor": arm, "candidate_id": candidate,
                        "Np": np_horizon, "Nc": nc, "block": block,
                        "completed_cases": 40, "measured_updates": len(samples),
                        "mean_ms": float(np.mean(samples)),
                        "median_ms": float(np.median(samples)),
                        "P95_ms": float(np.percentile(samples, 95)),
                        "P99_ms": float(np.percentile(samples, 99)),
                        "maximum_ms": float(np.max(samples)),
                        "updates_over_10ms": int(np.count_nonzero(samples > 10)),
                        "percentage_over_10ms": float(100 * np.mean(samples > 10)),
                        "raw_file": str(raw_path), "raw_sha256": sha256(raw_path)})
    assert reference_sets[0] == reference_sets[1] == reference_sets[2] == reference_sets[3]
    excluded = [r for r in rows if r["arm"] == "W3"]
    output = {"classification": "complete MPC compute-call time, not QP-only time",
              "measurement": "one-worker repeat; same 40 tracking timing scenarios for all four cells",
              "protocol_sha256": sha256(protocol),
              "episode_index": str(episodes_path), "episode_index_sha256": sha256(episodes_path),
              "cells": results,
              "trajectory_linear_status": f"Stopped at user request after {len(excluded)} of 40 cases; excluded from results",
              "no_retuning": True}
    (HERE / "FOUR_CELL_TIMING_RESULTS.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    (HERE / "FOUR_CELL_RAW_SOURCE_MANIFEST.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    lines = ["# Four-cell interception configuration timing repeat", "",
             "The one-worker repeat completed 40 cases and 120,000 compute-call updates for each crossed cell.",
             "All four cells used the same 40 scenario IDs, seeds, and reference hashes.",
             "The timer covers the full MPC `compute()` call, excluding external allocation and nonlinear plant stepping.",
             f"The planned trajectory-linear block was stopped at the user's request after {len(excluded)} cases; its partial data are excluded.",
             "The original five-cell protocol remains unchanged; this report records the user-directed scope change.",
             "No controller was retuned; no prior frozen timing result was modified.", "",
             "| Cell | Updates | Mean [ms] | P95 [ms] | P99 [ms] | Max [ms] | >10 ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['cell']} | {r['measured_updates']:,} | {r['mean_ms']:.4f} | {r['P95_ms']:.4f} | {r['P99_ms']:.4f} | {r['maximum_ms']:.4f} | {r['updates_over_10ms']} |")
    lines += ["", f"Protocol: `{protocol}` (SHA-256 `{sha256(protocol)}`).",
              f"Episode index: `{episodes_path}` (SHA-256 `{sha256(episodes_path)}`).",
              "Individual source files and hashes: `FOUR_CELL_RAW_SOURCE_MANIFEST.json`.",
              "Per-cell concatenated raw samples: `<cell>_raw_update_times_ms.npz`.",
              "Analysis script: `summarize_four_cells.py`.", ""]
    (HERE / "FOUR_CELL_TIMING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
