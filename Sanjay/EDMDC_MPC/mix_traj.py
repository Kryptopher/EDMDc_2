import argparse
import pickle
import numpy as np
from pathlib import Path


def load_simulation_runs(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def infer_family_label(data, filename):
    traj = data.get("traj", None)
    if traj == 1:
        return "helix"
    elif traj == 2:
        return "fig8"
    elif traj == 3:
        return "lissajous"
    elif traj == 4:
        return "waypoint"
    elif traj == 5:
        return "hover_excitation"
    elif traj in ("prbs", "yaw_prbs"):
        return "yaw_prbs"
    elif isinstance(traj, str):
        return traj

    name = Path(filename).stem.lower()
    if "traj1" in name or "helix" in name:
        return "helix"
    if "traj2" in name or "fig8" in name:
        return "fig8"
    if "traj3" in name or "lissa" in name:
        return "lissajous"
    if "traj4" in name or "wayp" in name:
        return "waypoint"
    if "traj5" in name or "hover" in name:
        return "hover_excitation"
    if "prbs" in name:
        return "yaw_prbs"
    return "unknown"


def combine_run_files(file_list, output_file):
    datasets = [load_simulation_runs(f) for f in file_list]

    profiles = [d.get("dataset_profile", "custom") for d in datasets]
    if len(set(profiles)) != 1:
        raise ValueError(
            f"dataset_profile mismatch: {profiles}. Do not mix paper and "
            "ACC-balanced trajectory files."
        )
    profile_configs = [d.get("trajectory_profile_config") for d in datasets]
    if any(config != profile_configs[0] for config in profile_configs[1:]):
        raise ValueError(
            "trajectory_profile_config mismatch. All family files must come "
            "from the same generation configuration."
        )

    if not all(d.get("input_type") == "applied_wrench" for d in datasets):
        raise ValueError(
            "Every source file must log applied_wrench inputs. Regenerate legacy data "
            "with parallel_sim.py before mixing."
        )
    if not all("U_requested" in d for d in datasets):
        raise ValueError(
            "Every source file must retain U_requested actuator diagnostics. "
            "Regenerate legacy data with parallel_sim.py before mixing."
        )
    if not all("U_outer" in d for d in datasets):
        raise ValueError(
            "Every source file must retain U_outer desired-attitude commands. "
            "Regenerate the data with the dual-logging parallel_sim.py."
        )
    if not all(d.get("schema_version") == "yaw_dual_input_v1" for d in datasets):
        raise ValueError(
            "Every source file must use schema_version=yaw_dual_input_v1; "
            "do not mix files produced before and after the dual-logging change."
        )

    # Keep full yaw-aware logs:
    # states = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    # U      = realized [thrust, tau_roll, tau_pitch, tau_yaw]
    # U_requested is retained when each source file provides it.
    for f, d in zip(file_list, datasets):
        print(
            f"{Path(f).name}: states shape = {d['states'].shape}, "
            f"U shape = {d['U'].shape}, U_outer shape = {d['U_outer'].shape}"
        )

    # Compatibility checks against first file
    ref = datasets[0]
    for i, data in enumerate(datasets[1:], 1):
        if not np.isclose(ref["sim_dt"], data["sim_dt"]):
            raise ValueError(f"sim_dt mismatch at file {i}")
        if not np.array_equal(ref["time"], data["time"]):
            raise ValueError(f"time vector mismatch at file {i}")
        if ref["states"].shape[1:] != data["states"].shape[1:]:
            raise ValueError(f"states shape mismatch at file {i}")
        if ref["U"].shape[1:] != data["U"].shape[1:]:
            raise ValueError(f"U shape mismatch at file {i}")
        if ref["U_outer"].shape[1:] != data["U_outer"].shape[1:]:
            raise ValueError(f"U_outer shape mismatch at file {i}")
        if ref["t"].shape[1:] != data["t"].shape[1:]:
            raise ValueError(f"t shape mismatch at file {i}")

    t_combined = np.concatenate([d["t"] for d in datasets], axis=0)
    states_combined = np.concatenate([d["states"] for d in datasets], axis=0)
    U_combined = np.concatenate([d["U"] for d in datasets], axis=0)
    has_requested_inputs = True
    U_requested_combined = np.concatenate(
        [d["U_requested"] for d in datasets], axis=0
    )
    U_outer_combined = np.concatenate(
        [d["U_outer"] for d in datasets], axis=0
    )
    has_allocator_inputs = all("U_allocator" in d for d in datasets)
    U_allocator_combined = (
        np.concatenate([d["U_allocator"] for d in datasets], axis=0)
        if has_allocator_inputs else None
    )
    ref_combined = sum([list(d["ref_traj_list"]) for d in datasets], [])
    reference_metrics = sum(
        [list(d.get("reference_metrics", [])) for d in datasets], []
    )
    simulation_metrics = sum(
        [list(d.get("simulation_metrics", [])) for d in datasets], []
    )
    run_seeds = sum([list(d.get("run_seeds", [])) for d in datasets], [])

    family_labels = []
    for data, filename in zip(datasets, file_list):
        existing = list(data.get("family_labels", []))
        if existing:
            if len(existing) != int(data["n"]):
                raise ValueError(
                    f"family_labels length mismatch in {filename}: "
                    f"expected {data['n']}, got {len(existing)}"
                )
            family_labels.extend(existing)
        else:
            family_labels.extend(
                [infer_family_label(data, filename)] * int(data["n"])
            )

    combined_data = {
        "traj": "mixed",
        "n": sum(d["n"] for d in datasets),
        "sim_dt": ref["sim_dt"],
        "time": ref["time"],
        "t": t_combined,
        "states": states_combined,
        "U": U_combined,
        "ref_traj_list": ref_combined,
        "family_labels": family_labels,
        "source_files": [str(f) for f in file_list],
        "dataset_profile": ref.get("dataset_profile", "custom"),
        "trajectory_profile_config": ref.get("trajectory_profile_config"),
        "input_type": "applied_wrench",
        "input_labels": ["thrust", "tau_roll", "tau_pitch", "tau_yaw"],
        "outer_input_type": "desired_attitude",
        "outer_input_labels": ["thrust", "phi_des", "theta_des", "psi_des"],
        "schema_version": "yaw_dual_input_v1",
    }
    combined_data["U_requested"] = U_requested_combined
    combined_data["U_outer"] = U_outer_combined
    if has_allocator_inputs:
        combined_data["U_allocator"] = U_allocator_combined
    mismatch_identities = [d.get("hidden_plant_identity") for d in datasets]
    if all(identity is not None for identity in mismatch_identities):
        if any(identity != mismatch_identities[0] for identity in mismatch_identities[1:]):
            raise ValueError("hidden_plant_identity mismatch across family files")
        combined_data["hidden_plant_identity"] = mismatch_identities[0]
        combined_data["controller_parameter_source"] = "nominal"
    if len(reference_metrics) == combined_data["n"]:
        combined_data["reference_metrics"] = reference_metrics
    if len(simulation_metrics) == combined_data["n"]:
        combined_data["simulation_metrics"] = simulation_metrics
    if len(run_seeds) == combined_data["n"]:
        combined_data["run_seeds"] = run_seeds
    if any("trajectory_speed_scales" in data for data in datasets):
        speed_scales = sum([
            list(data["trajectory_speed_scales"])
            if "trajectory_speed_scales" in data
            else [float("nan")] * int(data["n"])
            for data in datasets
        ], [])
        if len(speed_scales) != combined_data["n"]:
            raise ValueError("trajectory_speed_scales length mismatch")
        combined_data["trajectory_speed_scales"] = speed_scales
    for metadata_key in (
        "quadratic_drag", "motor_lag_s", "identification_protocol_sha256"
    ):
        values = [data.get(metadata_key) for data in datasets]
        present = [value for value in values if value is not None]
        if present:
            if any(value != present[0] for value in present[1:]):
                raise ValueError(f"{metadata_key} mismatch across family files")
            combined_data[metadata_key] = present[0]

    with open(output_file, "wb") as f:
        pickle.dump(combined_data, f)

    print(f"Saved: {Path(output_file).resolve()}")
    print(f"Total runs:   {combined_data['n']}")
    print(f"t shape:      {combined_data['t'].shape}")
    print(f"states shape: {combined_data['states'].shape}")
    print(f"U shape:      {combined_data['U'].shape}")
    print(f"Requested U:  {'present' if has_requested_inputs else 'not available'}")
    print(f"Outer U:      {combined_data['U_outer'].shape}")
    print(f"Families:     {set(family_labels)}")
    print(f"Profile:      {combined_data['dataset_profile']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine yaw-aware trajectory families into one EDMDc dataset."
    )
    parser.add_argument(
        "--profile", choices=("paper", "acc_balanced", "compact"), default="paper",
        help=(
            "paper combines the old 300-run composition (default); "
            "acc_balanced combines the same counts from 60 s balanced runs."
        ),
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path(__file__).resolve().parent,
        help="Directory containing generated trajectory pickle files.",
    )
    parser.add_argument(
        "--runs-per-family", type=int, default=None,
        help="Compact profile only: count used to name each trajectory file (default: 50).",
    )
    parser.add_argument(
        "--prbs-runs", type=int, default=None,
        help="Compact profile only: bounded yaw-PRBS run count (default: 0).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output dataset path (default: runs_mixed_n<3*runs>.pkl in --input-dir).",
    )
    args = parser.parse_args()
    if args.profile in ("paper", "acc_balanced"):
        if args.runs_per_family is not None or args.prbs_runs is not None:
            parser.error(
                f"The {args.profile} profile has fixed 50/50/50/50/30/70 counts. "
                "Use --profile compact for a custom mix."
            )
        file_specs = [(1, 50), (2, 50), (3, 50), (4, 50), (5, 30)]
        prbs_runs = 70
    else:
        runs_per_family = 50 if args.runs_per_family is None else args.runs_per_family
        prbs_runs = 0 if args.prbs_runs is None else args.prbs_runs
        if runs_per_family < 1:
            parser.error("--runs-per-family must be at least 1")
        if prbs_runs < 0:
            parser.error("--prbs-runs cannot be negative")
        file_specs = [(traj, runs_per_family) for traj in (1, 2, 3)]

    input_dir = args.input_dir.resolve()
    output = args.output
    if output is None:
        output = input_dir / f"runs_mixed_n{sum(n for _, n in file_specs) + prbs_runs}.pkl"
    elif not output.is_absolute():
        output = Path.cwd() / output

    file_list = [
        input_dir / f"runs_traj{traj}_n{n}.pkl"
        for traj, n in file_specs
    ]
    if prbs_runs:
        file_list.append(input_dir / f"runs_prbs_n{prbs_runs}.pkl")
    combine_run_files(file_list=file_list, output_file=output)
