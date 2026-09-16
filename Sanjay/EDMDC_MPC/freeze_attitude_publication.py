"""Create a reproducibility manifest for the final attitude-command study."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SOURCE_FILES = (
    "Simulation.py", "quadcopter.py", "Closed_loop.py",
    "Cascaded_Controllers.py", "PID_Mixer.py", "EDMDc_training.py",
    "parallel_sim.py", "mix_traj.py", "generate_high_tilt_augmentation.py",
    "append_high_tilt_augmentation.py",
    "edmdc_mpc.py", "outer_command_mpc.py", "outer_ltv_mpc.py",
    "reactive_pid_tracking.py", "publication_scenarios.py",
    "aggressiveness_crossover.py", "interception_comparison_attitude.py",
    "audit_linear_tracking_guard.py", "plot_aggressive_tracking_comparison.py",
    "plot_confirmed_tracking_traces.py",
    "evaluate_edmdc_training.py", "finalize_attitude_results.py",
    "run_attitude_capacity_learning_study.py",
    "summarize_attitude_capacity_learning.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments):
    result = subprocess.run(
        ["git", "-c", f"safe.directory={REPO.as_posix()}", *arguments],
        cwd=REPO, check=True, text=True, capture_output=True,
    )
    return result.stdout.strip()


def checked(paths):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing files: {missing}")
    return {str(Path(path).resolve()): sha256(Path(path)) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "artifacts" / "acc_balanced_high_tilt_v1"
    parser.add_argument("--dataset", type=Path,
                        default=base / "runs_mixed_n360_high_tilt.pkl")
    parser.add_argument("--model", type=Path, default=base /
                        "attitude_capacity_learning_v1" / "models" /
                        "dictionary_physics42_f100.pkl")
    parser.add_argument("--tracking", type=Path,
                        default=base / "tracking_physics42_confirmation_fresh_n40")
    parser.add_argument("--capacity", type=Path,
                        default=base / "attitude_capacity_learning_v1")
    parser.add_argument("--nominal-interception", type=Path,
                        default=base / "interception_physics42_confirmation_n40")
    parser.add_argument("--output", type=Path,
                        default=base / "attitude_paper_final_v2" /
                        "publication_freeze_manifest.json")
    args = parser.parse_args()

    configurations = (
        Path(json.loads((args.tracking / "manifest.json").read_text())[key])
        for key in ("linear_config_path", "edmd_config_path", "reactive_config_path")
    )
    result_files = [
        args.tracking / "manifest.json", args.tracking / "episodes.csv",
        args.tracking / "summary.csv",
        args.tracking / "plots" / "tracking_comparison_summary.png",
        args.tracking / "plots" / "paired_position_rmse.png",
        args.tracking / "representative_rep8" / "waypoint_trajectory.png",
        args.capacity / "manifest.json", args.capacity / "heldout_summary.csv",
        args.capacity / "capacity_learning_curve.png",
        args.capacity / "heldout_evaluation" / "rolling_aggregate.csv",
        args.nominal_interception / "manifest.json",
        args.nominal_interception / "episodes.csv",
        args.nominal_interception / "plots_qualified" /
        "qualified_interception_traces_rep0.png",
        args.nominal_interception / "plots_qualified" /
        "qualified_interception_overall.png",
        base / "attitude_paper_final_v2" / "results_audit.json",
        ROOT / "paper_acc_revision" / "main_revised.tex",
        ROOT / "paper_acc_revision" / "PAPER_FIGURE_MANIFEST.md",
    ]
    source_paths = [ROOT / name for name in SOURCE_FILES]
    payload = {
        "freeze_id": "acc_attitude_command_publication_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_base_commit_at_freeze": git("rev-parse", "HEAD"),
        "source_base_commit_short_at_freeze": git("rev-parse", "--short", "HEAD"),
        "working_tree_clean": not bool(git("status", "--short")),
        "data": checked([args.dataset]),
        "accepted_model": checked([args.model]),
        "controller_configs": checked(list(configurations)),
        "source": checked(source_paths),
        "results_and_paper": checked(result_files),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output.resolve())
    if not payload["working_tree_clean"]:
        print("WARNING: hashes are frozen, but the Git working tree is not clean.")


if __name__ == "__main__":
    main()
