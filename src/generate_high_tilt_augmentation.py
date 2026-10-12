"""Generate balanced direct-attitude excitation for outer-command EDMDc.

The command interface is [thrust, phi_des, theta_des, psi_des].  Commands pass
through the existing PX4-like attitude/rate controller, shared motor allocator,
and nonlinear plant.  Logged transitions retain (x_k, u_k, x_{k+1}) alignment.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np

from Closed_loop import ClosedLoopQuad
from Simulation import quad_sim
from parallel_sim import INPUT_LABELS, OUTER_INPUT_LABELS, simulation_metrics


AMPLITUDE_BANDS_DEG = (7.5, 15.0, 22.5, 30.0)


class AttitudeReferenceAdapter:
    """Expose stored outer commands through the normal simulation boundary."""

    def __init__(self, controller):
        self.controller = controller

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def fct_step(self, state, reference, dt):
        command = np.array(reference["outer_command"], dtype=float)
        # Mild position/velocity containment prevents a 60 s direct-attitude
        # run from accumulating unrealistic translational drift.  The final
        # combined command (including this term) is what U_outer logs.
        position = np.asarray(state[:3], dtype=float)
        velocity = np.asarray(state[3:6], dtype=float)
        correction = -0.08 * position - 0.45 * velocity
        horizontal = np.linalg.norm(correction[:2])
        if horizontal > 1.5:
            correction[:2] *= 1.5 / horizontal
        correction[2] = float(np.clip(correction[2], -1.5, 1.5))
        yaw = float(state[8])
        c_yaw, s_yaw = np.cos(yaw), np.sin(yaw)
        command[1] += (correction[0] * s_yaw - correction[1] * c_yaw) / self.controller.quad.g
        command[2] += (correction[0] * c_yaw + correction[1] * s_yaw) / self.controller.quad.g
        angle_norm = np.hypot(command[1], command[2])
        combined_limit = np.deg2rad(32.0)
        if angle_norm > combined_limit:
            command[1:3] *= combined_limit / angle_norm
        command[0] += self.controller.quad.m * correction[2]
        return self.controller.fct_attitude_step(
            state, command, dt,
            yaw_rate_ref=reference["yaw_rate"],
        )

    def fct_reset(self):
        self.controller.fct_reset()


def smooth_window(time, ramp_seconds):
    duration = float(time[-1] + (time[1] - time[0]))
    up = np.clip(time / ramp_seconds, 0.0, 1.0)
    down = np.clip((duration - time) / ramp_seconds, 0.0, 1.0)
    value = np.minimum(up, down)
    return value * value * (3.0 - 2.0 * value)


def make_reference(time, seed, amplitude_deg, mass, gravity):
    """Create bounded circular/multisine attitude, yaw, and thrust excitation."""
    rng = np.random.default_rng(seed)
    time = np.asarray(time, dtype=float)
    envelope = smooth_window(time, ramp_seconds=3.0)
    frequency_hz = rng.uniform(0.12, 0.42)
    omega = 2.0 * np.pi * frequency_hz
    direction = rng.choice((-1.0, 1.0))
    phase = rng.uniform(-np.pi, np.pi)
    amplitude = np.deg2rad(amplitude_deg)

    # Circular excitation keeps total tilt near the selected band instead of
    # spending nearly all samples around zero as independent sinusoids would.
    phi_des = envelope * amplitude * np.cos(omega * time + phase)
    theta_des = envelope * amplitude * np.sin(direction * omega * time + phase)
    harmonic = 0.12 * amplitude * envelope
    phi_des += harmonic * np.sin(2.0 * omega * time + 0.7 * phase)
    theta_des += harmonic * np.cos(3.0 * omega * time - 0.4 * phase)
    # Cap the combined roll/pitch vector rather than each Euler component.
    # Component-wise clipping allowed simultaneous commands above 40 degrees.
    limit = np.deg2rad(30.0)
    angle_norm = np.hypot(phi_des, theta_des)
    scale = np.minimum(1.0, limit / np.maximum(angle_norm, 1e-12))
    phi_des *= scale
    theta_des *= scale

    yaw_rate_amplitude = rng.uniform(0.2, 0.8)
    yaw_frequency_hz = rng.uniform(0.04, 0.16)
    yaw_rate = envelope * yaw_rate_amplitude * np.sin(
        2.0 * np.pi * yaw_frequency_hz * time + rng.uniform(-np.pi, np.pi)
    )
    dt = float(time[1] - time[0])
    yaw = np.cumsum(yaw_rate) * dt

    hover_compensation = mass * gravity / np.maximum(
        0.55, np.cos(phi_des) * np.cos(theta_des)
    )
    thrust_scale = 1.0 + envelope * rng.uniform(0.04, 0.12) * np.sin(
        2.0 * np.pi * rng.uniform(0.08, 0.25) * time
        + rng.uniform(-np.pi, np.pi)
    )
    thrust = hover_compensation * thrust_scale

    return [
        {
            "pos": np.zeros(3), "vel": np.zeros(3), "acc": np.zeros(3),
            "yaw": float(yaw[k]), "yaw_rate": float(yaw_rate[k]),
            "outer_command": np.array(
                [thrust[k], phi_des[k], theta_des[k], yaw[k]], dtype=float
            ),
        }
        for k in range(len(time))
    ], {"frequency_hz": float(frequency_hz), "amplitude_deg": float(amplitude_deg)}


def run_one(task):
    run_index, duration, seed_start = task
    sim = quad_sim()
    sim.time = np.arange(0.0, duration, sim.dt)
    amplitude_deg = AMPLITUDE_BANDS_DEG[run_index % len(AMPLITUDE_BANDS_DEG)]
    seed = seed_start + run_index
    reference, design = make_reference(
        sim.time, seed, amplitude_deg, sim.quad.m, sim.quad.g
    )
    adapter = AttitudeReferenceAdapter(sim.controller_PX4)
    closed_loop = ClosedLoopQuad(sim.quad, adapter)
    initial_state = np.zeros(12)
    initial_state[8] = reference[0]["yaw"]
    t, states, _, applied, requested, outer = closed_loop.fct_simulate(
        sim.time, sim.dt, reference, initial_state,
        return_requested=True, return_outer=True,
    )
    # Tracking metrics are not meaningful for direct attitude excitation; the
    # standard diagnostic is retained only for numerical/allocation checks.
    diagnostics = simulation_metrics(states, applied, requested, reference)
    body_tilt = np.degrees(np.arccos(np.clip(
        np.cos(states[:, 6]) * np.cos(states[:, 7]), -1.0, 1.0
    )))
    command_tilt = np.degrees(np.arccos(np.clip(
        np.cos(outer[:, 1]) * np.cos(outer[:, 2]), -1.0, 1.0
    )))
    diagnostics.update({
        "body_tilt_deg_p50": float(np.percentile(body_tilt, 50)),
        "body_tilt_deg_p95": float(np.percentile(body_tilt, 95)),
        "body_tilt_deg_max": float(np.max(body_tilt)),
        "command_tilt_deg_p95": float(np.percentile(command_tilt, 95)),
        "body_rate_radps_p95": float(np.percentile(
            np.linalg.norm(states[:, 9:12], axis=1), 95
        )),
    })
    return t, states, applied, requested, outer, reference, diagnostics, design, seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=900000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 12 or args.runs % len(AMPLITUDE_BANDS_DEG):
        parser.error("--runs must be at least 12 and divisible by four")
    tasks = [(index, args.duration_seconds, args.seed_start) for index in range(args.runs)]
    with mp.Pool(min(args.workers, args.runs)) as pool:
        results = pool.map(run_one, tasks)
    t = np.stack([row[0] for row in results])
    states = np.stack([row[1] for row in results])
    applied = np.stack([row[2] for row in results])
    requested = np.stack([row[3] for row in results])
    outer = np.stack([row[4] for row in results])
    references = [row[5] for row in results]
    diagnostics = [row[6] for row in results]
    designs = [row[7] for row in results]
    seeds = [row[8] for row in results]
    # Last two complete four-band blocks are reserved for augmentation
    # validation and test.  These indices become global when appended.
    validation_local = list(range(args.runs - 8, args.runs - 4))
    test_local = list(range(args.runs - 4, args.runs))
    payload = {
        "traj": "high_tilt_attitude_excitation", "n": args.runs,
        "sim_dt": quad_sim.dt, "time": t[0], "t": t,
        "states": states, "U": applied, "U_requested": requested,
        "U_outer": outer, "ref_traj_list": references,
        "family_labels": ["high_tilt_attitude_excitation"] * args.runs,
        "run_seeds": seeds, "simulation_metrics": diagnostics,
        "excitation_designs": designs,
        "augmentation_validation_local_indices": validation_local,
        "augmentation_test_local_indices": test_local,
        "input_type": "applied_wrench", "input_labels": INPUT_LABELS,
        "outer_input_type": "desired_attitude",
        "outer_input_labels": OUTER_INPUT_LABELS,
        "dataset_profile": "high_tilt_attitude_augmentation_v1",
        "trajectory_profile_config": {
            "version": "high_tilt_attitude_augmentation_v1",
            "duration_seconds": args.duration_seconds,
            "amplitude_bands_deg": list(AMPLITUDE_BANDS_DEG),
            "frequency_hz": [0.12, 0.42], "yaw_rate_radps": [0.2, 0.8],
            "position_containment": {
                "position_gain": 0.08, "velocity_gain": 0.45,
                "acceleration_limit_mps2": 1.5,
                "combined_tilt_limit_deg": 32.0,
            },
        },
        "schema_version": "yaw_dual_input_v1",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "output": str(args.output.resolve()), "runs": args.runs,
        "validation_local_indices": validation_local, "test_local_indices": test_local,
        "body_tilt_p95_by_run_deg": [row["body_tilt_deg_p95"] for row in diagnostics],
        "allocator_altered_steps": int(sum(row["allocation_altered_steps"] for row in diagnostics)),
        "all_finite": bool(all(row["finite"] for row in diagnostics)),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
