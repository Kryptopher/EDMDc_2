"""Replace only the waypoint block in a mixed yaw-aware EDMDc dataset."""

import argparse
import pickle
from pathlib import Path

import numpy as np


ARRAY_KEYS = ("t", "states", "U", "U_requested", "U_outer")
LIST_KEYS = (
    "ref_traj_list", "reference_metrics", "simulation_metrics", "run_seeds"
)


def load_pickle(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def replace_waypoints(base, waypoint, base_path, waypoint_path):
    if base.get("schema_version") != "yaw_dual_input_v1":
        raise ValueError("Base dataset must use schema_version=yaw_dual_input_v1")
    if waypoint.get("schema_version") != "yaw_dual_input_v1":
        raise ValueError("Waypoint file must use schema_version=yaw_dual_input_v1")
    if waypoint.get("traj") != 4:
        raise ValueError("Replacement file must contain trajectory family 4")
    if base.get("dataset_profile") != waypoint.get("dataset_profile"):
        raise ValueError("Base and waypoint dataset profiles differ")

    labels = list(base.get("family_labels", []))
    indices = np.flatnonzero(np.asarray(labels, dtype=object) == "waypoint")
    if len(indices) != int(waypoint["n"]):
        raise ValueError(
            f"Base has {len(indices)} waypoint runs; replacement has {waypoint['n']}"
        )
    if len(indices) and not np.array_equal(
        indices, np.arange(indices[0], indices[0] + len(indices))
    ):
        raise ValueError("Waypoint runs must form one contiguous block")

    if not np.isclose(base["sim_dt"], waypoint["sim_dt"]):
        raise ValueError("Simulation time steps differ")
    if not np.array_equal(base["time"], waypoint["time"]):
        raise ValueError("Simulation time vectors differ")

    for key in ARRAY_KEYS:
        if key not in base or key not in waypoint:
            raise ValueError(f"Missing required array {key}")
        expected = (len(indices),) + tuple(base[key].shape[1:])
        if waypoint[key].shape != expected:
            raise ValueError(
                f"{key} replacement shape {waypoint[key].shape}, expected {expected}"
            )
        base[key][indices] = waypoint[key]

    for key in LIST_KEYS:
        if key not in waypoint:
            continue
        if key not in base or len(base[key]) != int(base["n"]):
            raise ValueError(f"Base dataset lacks complete {key}")
        if len(waypoint[key]) != len(indices):
            raise ValueError(f"Replacement {key} count is inconsistent")
        for source_index, target_index in enumerate(indices):
            base[key][int(target_index)] = waypoint[key][source_index]

    base["trajectory_profile_config"] = waypoint["trajectory_profile_config"]
    provenance = list(base.get("source_files", []))
    base["source_files_before_waypoint_replacement"] = provenance
    base["waypoint_replacement"] = {
        "base_dataset": str(Path(base_path).resolve()),
        "replacement_file": str(Path(waypoint_path).resolve()),
        "indices": indices.tolist(),
        "run_seeds": list(waypoint.get("run_seeds", [])),
        "profile_version": waypoint["trajectory_profile_config"].get("version"),
    }
    base["source_files"] = provenance + [str(Path(waypoint_path).resolve())]
    return base


def main():
    parser = argparse.ArgumentParser(
        description="Replace the waypoint block while preserving all other runs."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--waypoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = load_pickle(args.base)
    waypoint = load_pickle(args.waypoint)
    mixed = replace_waypoints(base, waypoint, args.base, args.waypoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(mixed, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {args.output.resolve()}")
    print(
        f"Replaced {len(mixed['waypoint_replacement']['indices'])} waypoint runs; "
        f"profile={mixed['waypoint_replacement']['profile_version']}"
    )


if __name__ == "__main__":
    main()
