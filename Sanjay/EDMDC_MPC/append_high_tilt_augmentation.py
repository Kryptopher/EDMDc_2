"""Append high-tilt excitation after a frozen base dataset without reindexing it."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


ARRAY_KEYS = ("t", "states", "U", "U_requested", "U_outer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--augmentation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.base.open("rb") as stream:
        base = pickle.load(stream)
    with args.augmentation.open("rb") as stream:
        augmentation = pickle.load(stream)
    for key in ARRAY_KEYS:
        if base[key].shape[1:] != augmentation[key].shape[1:]:
            raise ValueError(f"{key} per-run shape mismatch: {base[key].shape} vs {augmentation[key].shape}")
    if not np.isclose(base["sim_dt"], augmentation["sim_dt"]):
        raise ValueError("Simulation time steps differ")
    if base.get("schema_version") != "yaw_dual_input_v1" or augmentation.get("schema_version") != "yaw_dual_input_v1":
        raise ValueError("Both datasets must use yaw_dual_input_v1")
    base_count = int(base["n"])
    result = dict(base)
    for key in ARRAY_KEYS:
        result[key] = np.concatenate([base[key], augmentation[key]], axis=0)
    result["ref_traj_list"] = list(base["ref_traj_list"]) + list(augmentation["ref_traj_list"])
    result["family_labels"] = list(base.get("family_labels", ["unknown"] * base_count)) + list(augmentation["family_labels"])
    result["run_seeds"] = list(base.get("run_seeds", [])) + list(augmentation.get("run_seeds", []))
    result["simulation_metrics"] = list(base.get("simulation_metrics", [])) + list(augmentation.get("simulation_metrics", []))
    result["n"] = base_count + int(augmentation["n"])
    result["traj"] = "mixed_with_high_tilt_augmentation"
    result["dataset_profile"] = "acc_balanced_high_tilt_augmented_v1"
    result["source_files"] = [str(args.base.resolve()), str(args.augmentation.resolve())]
    result["augmentation"] = {
        "start_index": base_count,
        "count": int(augmentation["n"]),
        "validation_indices": [base_count + index for index in augmentation["augmentation_validation_local_indices"]],
        "test_indices": [base_count + index for index in augmentation["augmentation_test_local_indices"]],
        "profile_config": augmentation["trajectory_profile_config"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "output": str(args.output.resolve()), "total_runs": result["n"],
        "base_runs": base_count, **result["augmentation"],
        "base_indices_preserved": True,
        "alignment": "(x_k, u_k, x_{k+1})",
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
