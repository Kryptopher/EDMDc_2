"""Audit and summarize the locked attitude-command paper comparisons.

Run only after the fresh-seed tracking confirmation has completed. This script
never tunes a model or controller; it reads existing episode-level CSV files.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


TRACKING_CONTROLLERS = ("reactive_pid", "yaw_scheduled_hover", "edmdc")
INTERCEPTION_CONTROLLERS = ("PID", "Linear MPC", "EDMDc-MPC")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grouped_pairs(rows, keys, controllers):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in keys)
        controller = row["controller"]
        assert controller in controllers, (key, controller)
        assert controller not in groups.setdefault(key, {}), (key, controller)
        groups[key][controller] = row
    assert groups and all(set(group) == set(controllers) for group in groups.values())
    return groups


def bootstrap_ci(values, seed=20260913, resamples=20000):
    values = np.asarray(values, dtype=float)
    assert values.size and np.all(np.isfinite(values))
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(resamples, len(values)), replace=True), axis=1)
    return [float(np.quantile(means, q)) for q in (0.025, 0.975)]


def audit(rows, controllers):
    for row in rows:
        assert int(row["failed_solves"]) == 0, row
        assert int(row["allocator_altered_steps"]) == 0, row
        assert row["controller"] in controllers


def summarize_tracking(rows, manifest):
    assert manifest["experiment_role"] == "confirmation"
    assert manifest["screening_only"] is False
    assert manifest["run_index_start"] == 790000
    assert manifest["scales"] == [1.75]
    assert manifest["duration_seconds"] == 30.0
    assert manifest["controllers"] == list(TRACKING_CONTROLLERS)
    for prefix in ("model", "linear_config", "edmd_config", "reactive_config"):
        assert sha256(Path(manifest[f"{prefix}_path"])) == manifest[f"{prefix}_sha256"]
    audit(rows, TRACKING_CONTROLLERS)
    paired = grouped_pairs(rows, ("family", "run_index"), TRACKING_CONTROLLERS)
    assert len(paired) == 40
    assert {key[0] for key in paired} == {"helix", "figure8", "lissajous", "waypoint"}
    assert all(sum(key[0] == family for key in paired) == 10 for family in
               ("helix", "figure8", "lissajous", "waypoint"))
    for group in paired.values():
        assert all(float(row["aggressiveness_scale"]) == 1.75 for row in group.values())
        assert len({row["reference_peak_speed_mps"] for row in group.values()}) == 1
        assert len({row["reference_p95_implied_tilt_deg"] for row in group.values()}) == 1
        assert all(np.isfinite(float(row["position_rmse_m"])) for row in group.values())
    summary = {}
    for controller in TRACKING_CONTROLLERS:
        selected = [group[controller] for group in paired.values()]
        values = np.array([float(row["position_rmse_m"]) for row in selected])
        summary[controller] = {
            "episodes": len(values),
            "mean_position_rmse_m": float(np.mean(values)),
            "mean_ci95_m": bootstrap_ci(values),
            "median_position_rmse_m": float(np.median(values)),
            "p90_position_rmse_m": float(np.quantile(values, 0.9)),
            "worst_position_rmse_m": float(np.max(values)),
            "episodes_over_1m_rmse": int(np.sum(values > 1.0)),
            "mean_velocity_rmse_mps": float(np.mean([
                float(row["velocity_rmse_mps"]) for row in selected])),
            "mean_yaw_rmse_rad": float(np.mean([
                float(row["yaw_rmse_rad"]) for row in selected])),
            "mean_solve_ms": float(np.mean([
                float(row["mean_solve_ms"]) for row in selected])),
            "worst_episode_p95_solve_ms": float(np.max([
                float(row.get("p95_solve_ms", 0.0)) for row in selected])),
            "worst_episode_p99_solve_ms": float(np.max([
                float(row.get("p99_solve_ms", 0.0)) for row in selected])),
            "worst_sample_solve_ms": float(np.max([
                float(row.get("max_solve_ms", 0.0)) for row in selected])),
        }
    comparisons = {}
    for baseline in TRACKING_CONTROLLERS[:2]:
        differences = np.array([
            float(group["edmdc"]["position_rmse_m"]) -
            float(group[baseline]["position_rmse_m"])
            for group in paired.values()
        ])
        comparisons[baseline] = {
            "mean_difference_m": float(np.mean(differences)),
            "mean_difference_ci95_m": bootstrap_ci(differences),
            "edmd_better_scenarios": int(np.sum(differences < 0)),
            "pairs": len(differences),
        }
    by_family = {}
    for family in ("helix", "figure8", "lissajous", "waypoint"):
        groups = [group for key, group in paired.items() if key[0] == family]
        by_family[family] = {
            controller: float(np.mean([
                float(group[controller]["position_rmse_m"]) for group in groups
            ])) for controller in TRACKING_CONTROLLERS
        }
    return {"summary": summary, "paired_comparisons": comparisons,
            "family_mean_position_rmse_m": by_family}


def qualified_time(row):
    value = float(row["qualified_capture_time_speed_3p0_mps_s"])
    if not np.isfinite(value):
        return float(row["tmax_s"])
    if row.get("qualified_time_reference") != "dwell_end":
        dt = float(row["controller_period_s"])
        dwell_steps = max(1, int(np.ceil(float(row["qualified_dwell_s"]) / dt)))
        value += (dwell_steps - 1) * dt
    return value


def summarize_interception(rows, expected_plants, scenarios_per_plant):
    audit(rows, INTERCEPTION_CONTROLLERS)
    assert {row["plant_id"] for row in rows} == set(expected_plants)
    assert all(float(row["controller_period_s"]) == 0.01 for row in rows)
    assert all(float(row["capture_radius_m"]) == 0.75 for row in rows)
    assert all(float(row["qualified_dwell_s"]) == 0.2 for row in rows)
    assert all(float(row["intercept_lead_s"]) == 0.0 for row in rows)
    paired = grouped_pairs(rows, ("plant_id", "family", "scenario_seed"),
                           INTERCEPTION_CONTROLLERS)
    result = {}
    for plant in expected_plants:
        groups = [group for key, group in paired.items() if key[0] == plant]
        assert len(groups) == scenarios_per_plant
        assert {group["PID"]["family"] for group in groups} == {
            "straight", "accelerating", "helix", "weaving"}
        result[plant] = {}
        for controller in INTERCEPTION_CONTROLLERS:
            selected = [group[controller] for group in groups]
            times = [qualified_time(row) for row in selected]
            result[plant][controller] = {
                "episodes": len(selected),
                "geometric_captures": sum(int(row["captured"]) for row in selected),
                "qualified_captures_3_mps": sum(
                    int(row["qualified_capture_speed_3p0_mps"]) for row in selected),
                "mean_geometric_failure_aware_time_s": float(np.mean([
                    float(row["failure_aware_capture_time_s"]) for row in selected])),
                "mean_qualified_failure_aware_time_s": float(np.mean(times)),
                "mean_qualified_time_ci95_s": bootstrap_ci(times),
                "mean_contact_speed_mps": float(np.nanmean([
                    float(row["capture_relative_speed"]) for row in selected])),
                "mean_solve_ms": float(np.mean([
                    float(row["mean_solve_ms"]) for row in selected])),
            }
        result[plant]["paired_qualified_time_differences"] = {}
        for baseline in INTERCEPTION_CONTROLLERS[:2]:
            differences = [qualified_time(group["EDMDc-MPC"]) -
                           qualified_time(group[baseline]) for group in groups]
            result[plant]["paired_qualified_time_differences"][baseline] = {
                "mean_difference_s": float(np.mean(differences)),
                "mean_difference_ci95_s": bootstrap_ci(differences),
                "edmd_faster_scenarios": int(np.sum(np.asarray(differences) < 0)),
                "pairs": len(differences),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("artifacts/acc_balanced_high_tilt_v1")
    parser.add_argument("--tracking", type=Path,
                        default=base / "tracking_paper_confirmation_fresh_n40")
    parser.add_argument("--nominal", type=Path,
                        default=base / "interception_final_confirmation_n40")
    parser.add_argument("--mismatch", type=Path,
                        default=base / "interception_final_mismatch_n60")
    parser.add_argument("--skip-mismatch", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=base / "attitude_paper_final" / "results_audit.json")
    args = parser.parse_args()
    tracking_manifest = json.loads((args.tracking / "manifest.json").read_text())
    nominal_manifest = json.loads((args.nominal / "manifest.json").read_text())
    mismatch_manifest = (
        None if args.skip_mismatch else
        json.loads((args.mismatch / "manifest.json").read_text())
    )
    assert nominal_manifest["base_seed"] == 800000
    assert nominal_manifest["controllers"] == list(INTERCEPTION_CONTROLLERS)
    assert nominal_manifest["controller_rate_hz"] == 100.0
    if mismatch_manifest is not None:
        assert mismatch_manifest["base_seed"] == 680000
        assert mismatch_manifest["controllers"] == list(INTERCEPTION_CONTROLLERS)
        assert nominal_manifest["tuned_config"] == mismatch_manifest["tuned_config"]
        assert mismatch_manifest["controller_rate_hz"] == 100.0
        assert mismatch_manifest["plant_ids"] == [
            "heavy_asymmetric", "light_low_drag", "yaw_inertia_motor"]
    tracking = summarize_tracking(read_csv(args.tracking / "episodes.csv"), tracking_manifest)
    nominal = summarize_interception(read_csv(args.nominal / "episodes.csv"),
                                     ("nominal",), 40)
    mismatch = (
        None if mismatch_manifest is None else
        summarize_interception(read_csv(args.mismatch / "episodes.csv"),
                               ("heavy_asymmetric", "light_low_drag",
                                "yaw_inertia_motor"), 20)
    )
    output = {
        "audit_pass": True,
        "tracking": tracking,
        "nominal_interception": nominal,
        "mismatch_interception": mismatch,
        "provenance": {
            "tracking_manifest_sha256": sha256(args.tracking / "manifest.json"),
            "nominal_manifest_sha256": sha256(args.nominal / "manifest.json"),
            "mismatch_manifest_sha256": (
                None if mismatch_manifest is None else
                sha256(args.mismatch / "manifest.json")
            ),
            "tracking_episodes_sha256": sha256(args.tracking / "episodes.csv"),
            "nominal_episodes_sha256": sha256(args.nominal / "episodes.csv"),
            "mismatch_episodes_sha256": (
                None if mismatch_manifest is None else
                sha256(args.mismatch / "episodes.csv")
            ),
        },
        "metadata_note": (
            None if mismatch_manifest is None else
            "The preexisting mismatch manifest says nominal_plant=true. "
            "Its plant_ids and all episode plant_id values identify the three "
            "mismatched plants; the runner's metadata bug is fixed for future runs."
        ),
        "limitations": [
            "Simulation only; no flight-control or hardware performance claim.",
            "Tracking and interception use different selected MPC configurations.",
            "PID is reactive; predictive controllers receive future target state.",
            "The legacy wrench-level publication campaign is a separate architecture.",
        ] + ([] if mismatch_manifest is None else [
            "Mismatch scenarios reuse target seeds across plant variants; do not pool as independent."
        ]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
