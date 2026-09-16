"""Run the feedforward-free PX4-like PID on held-out trajectories at 100 Hz."""

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from Closed_loop import ClosedLoopQuad
from Simulation import quad_sim
from edmdc_mpc import reference_yaw_arrays, wrap_angle_pi


DEFAULT_DATA = Path("artifacts/acc_balanced_waypoint_v2/runs_mixed_n300_waypoint_v2.pkl")
DEFAULT_INDICES = (39, 59, 129, 155, 210)


def metrics(states, reference, dt):
    position = np.asarray([point["pos"] for point in reference], dtype=float)
    velocity = np.asarray([point.get("vel", np.zeros(3)) for point in reference])
    yaw, yaw_rate = reference_yaw_arrays(reference, dt=dt)
    return {
        "position_component_rmse_m": float(np.sqrt(np.mean((states[:, :3] - position)**2))),
        "velocity_component_rmse_mps": float(np.sqrt(np.mean((states[:, 3:6] - velocity)**2))),
        "yaw_rmse_rad": float(np.sqrt(np.mean(wrap_angle_pi(states[:, 8] - yaw)**2))),
        "yaw_rate_rmse_radps": float(np.sqrt(np.mean((states[:, 11] - yaw_rate)**2))),
    }


def run_reactive_pid(reference, gain_multipliers=None, initial_yaw=None,
                     acceleration_limit_scale=1.0, sim=None):
    """Run one feedback-only episode with optional outer-loop gain scales."""
    sim = quad_sim() if sim is None else sim
    controller = sim.controller_PX4
    controller.use_trajectory_feedforward = False
    controller.acc_max_xy *= float(acceleration_limit_scale)
    controller.acc_max_z *= float(acceleration_limit_scale)
    gain_multipliers = {} if gain_multipliers is None else gain_multipliers
    controller.pos_p *= np.array([
        gain_multipliers.get("pos_xy", 1.0),
        gain_multipliers.get("pos_xy", 1.0),
        gain_multipliers.get("pos_z", 1.0),
    ])
    controller.vel_p *= np.array([
        gain_multipliers.get("vel_xy", 1.0),
        gain_multipliers.get("vel_xy", 1.0),
        gain_multipliers.get("vel_z", 1.0),
    ])
    controller.vel_i *= np.array([
        gain_multipliers.get("int_xy", 1.0),
        gain_multipliers.get("int_xy", 1.0),
        gain_multipliers.get("int_z", 1.0),
    ])
    closed_loop = ClosedLoopQuad(sim.quad, controller)
    initial_state = np.zeros(12)
    initial_state[8] = float(
        reference[0].get("yaw", 0.0) if initial_yaw is None else initial_yaw
    )
    time = np.arange(len(reference), dtype=float) * sim.dt
    _, states, _, applied, requested, outer = closed_loop.fct_simulate(
        time, sim.dt, reference, initial_state,
        return_requested=True, return_outer=True,
    )
    return sim, states, applied, requested, outer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--indices", default=",".join(map(str, DEFAULT_INDICES)))
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument(
        "--config", type=Path,
        help="Frozen validation-selected reactive PID configuration JSON.",
    )
    args = parser.parse_args()

    with args.data.open("rb") as stream:
        data = pickle.load(stream)
    dt = float(data["sim_dt"])
    if not np.isclose(dt, 0.01):
        raise ValueError(f"Reactive PID publication baseline requires dt=0.01 s, got {dt:g}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gain_multipliers = None
    if args.config is not None:
        config = json.loads(args.config.read_text())
        if config.get("controller") != "reactive_pid":
            raise ValueError("--config is not a reactive_pid configuration")
        if not np.isclose(float(config.get("controller_period_s", np.nan)), dt):
            raise ValueError("Reactive PID configuration rate does not match the dataset")
        gain_multipliers = config["gain_multipliers"]
    rows = []
    for index in [int(value) for value in args.indices.split(",") if value.strip()]:
        reference = data["ref_traj_list"][index]
        count = len(reference) if args.steps == 0 else min(args.steps, len(reference))
        reference = reference[:count]
        sim, states, applied, requested, outer = run_reactive_pid(
            reference, gain_multipliers
        )
        family = data["family_labels"][index]
        allocation_error = np.max(np.abs(applied - requested), axis=1)
        row = {
            "index": index,
            "family": family,
            "controller_period_s": dt,
            "controller_rate_hz": 1.0 / dt,
            "trajectory_feedforward": False,
            "gain_multipliers_json": json.dumps(
                {} if gain_multipliers is None else gain_multipliers,
                sort_keys=True,
            ),
            **metrics(states, reference, dt),
            "allocator_altered_steps": int(np.sum(allocation_error > 1e-8)),
            "allocator_max_abs_wrench_error": float(np.max(allocation_error)),
        }
        rows.append(row)
        np.savez_compressed(
            args.output_dir / f"run_{index}_{family}.npz",
            states=states, wrench=applied, requested_wrench=requested,
            outer_command=outer, controller_period_s=dt,
            controller_rate_hz=1.0 / dt, trajectory_feedforward=False,
        )
        print(row)

    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
