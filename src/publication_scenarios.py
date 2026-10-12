"""Deterministic validation/test scenarios for the ACC publication campaign."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from Simulation import quad_sim


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "publication_protocol.json"
TRACKING_FAMILIES = {
    "helix": 1,
    "figure8": 2,
    "lissajous": 3,
    "waypoint": 4,
    "hover_excitation": 5,
}
INTERCEPTION_FAMILIES = ("straight", "accelerating", "helix", "weaving")


def load_protocol(path: Path = PROTOCOL_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    """Return a reproducible centered-random Latin-hypercube design."""
    if count < 1 or dimensions < 1:
        raise ValueError("count and dimensions must be positive")
    rng = np.random.default_rng(seed)
    result = np.empty((count, dimensions), dtype=float)
    for column in range(dimensions):
        strata = (np.arange(count, dtype=float) + rng.random(count)) / count
        result[:, column] = strata[rng.permutation(count)]
    return result


def _map(unit: float, bounds) -> float:
    lower, upper = map(float, bounds)
    return lower + float(unit) * (upper - lower)


def robustness_design(count: int, seed: int, protocol: dict | None = None) -> list[dict]:
    """Sample paired physical, sensing, delay, and reference perturbations."""
    protocol = load_protocol() if protocol is None else protocol
    limits = protocol["robustness"]
    # mass, Ixyz, drag, four motors, wind speed/direction, gust fraction/frequency/
    # phase, four noise magnitudes, two delays, speed, yaw, initial errors.
    design = latin_hypercube(count, 30, seed)
    rows = []
    for index, values in enumerate(design):
        wind_speed = _map(values[9], limits["wind_speed_mps"])
        wind_angle = 2.0 * np.pi * values[10]
        wind_z_fraction = 0.3 * (2.0 * values[11] - 1.0)
        horizontal = np.sqrt(max(0.0, 1.0 - wind_z_fraction**2))
        wind_direction = np.array([
            horizontal * np.cos(wind_angle),
            horizontal * np.sin(wind_angle),
            wind_z_fraction,
        ])
        wind = wind_speed * wind_direction
        gust_fraction = 0.15 + 0.35 * values[12]
        gust_direction = np.array([
            np.cos(wind_angle + np.pi / 2.0),
            np.sin(wind_angle + np.pi / 2.0),
            0.2 * (2.0 * values[13] - 1.0),
        ])
        gust_direction /= max(np.linalg.norm(gust_direction), 1e-12)
        measurement_delay = _map(
            values[20], limits["measurement_delay_seconds"]
        )
        command_delay = _map(values[21], limits["command_delay_seconds"])
        initial_position = 0.5 * (2.0 * values[24:27] - 1.0)
        initial_velocity = 0.2 * (2.0 * values[27:30] - 1.0)
        rows.append({
            "robustness_index": index,
            "robustness_seed": int(seed),
            "mass_scale": _map(values[0], limits["mass_scale"]),
            "inertia_scales": [
                _map(values[column], limits["inertia_scale"])
                for column in (1, 2, 3)
            ],
            "drag_scale": _map(values[4], limits["drag_scale"]),
            "motor_effectiveness": [
                _map(values[column], limits["motor_effectiveness"])
                for column in (5, 6, 7, 8)
            ],
            "wind_velocity_mps": wind.tolist(),
            "wind_gust_velocity_mps": (gust_fraction * wind_speed * gust_direction).tolist(),
            "wind_gust_frequency_radps": float(0.25 + 1.25 * values[14]),
            "wind_gust_phase_rad": float(2.0 * np.pi * values[15]),
            "position_noise_std_m": _map(
                values[16], limits["position_noise_std_m"]
            ),
            "velocity_noise_std_mps": _map(
                values[17], limits["velocity_noise_std_mps"]
            ),
            "attitude_noise_std_rad": _map(
                values[18], limits["attitude_noise_std_rad"]
            ),
            "body_rate_noise_std_radps": _map(
                values[19], limits["body_rate_noise_std_radps"]
            ),
            "measurement_delay_seconds": measurement_delay,
            "command_delay_seconds": command_delay,
            "measurement_delay_steps": int(round(measurement_delay / quad_sim.dt)),
            "command_delay_steps": int(round(command_delay / quad_sim.dt)),
            "speed_scale": float(0.75 + 0.60 * values[22]),
            "yaw_rate_scale": float(0.50 + 0.75 * values[23]),
            "initial_position_error_m": initial_position.tolist(),
            "initial_velocity_error_mps": initial_velocity.tolist(),
            "initial_yaw_error_rad": float(0.6 * (values[13] - 0.5)),
        })
    return rows


def sensor_noise_vector(scenario: dict | None) -> np.ndarray:
    if not scenario:
        return np.zeros(12, dtype=float)
    return np.array([
        *([float(scenario["position_noise_std_m"])] * 3),
        *([float(scenario["velocity_noise_std_mps"])] * 3),
        *([float(scenario["attitude_noise_std_rad"])] * 3),
        *([float(scenario["body_rate_noise_std_radps"])] * 3),
    ])


def configure_robust_plant(sim: quad_sim, scenario: dict | None) -> quad_sim:
    if not scenario:
        return sim
    import compare_mpc as cm

    return cm.perturb_plant(
        sim,
        scenario["mass_scale"],
        scenario["inertia_scales"],
        scenario["drag_scale"],
        scenario["motor_effectiveness"],
        wind_velocity=scenario["wind_velocity_mps"],
        wind_gust_velocity=scenario["wind_gust_velocity_mps"],
        wind_gust_frequency_radps=scenario["wind_gust_frequency_radps"],
        wind_gust_phase_rad=scenario["wind_gust_phase_rad"],
    )


def transform_reference(
    reference: list[dict], speed_scale: float = 1.0, yaw_rate_scale: float = 1.0
) -> list[dict]:
    """Scale path displacement/kinematics and integrate a bounded yaw rate."""
    if speed_scale <= 0.0 or yaw_rate_scale < 0.0:
        raise ValueError("speed scale must be positive and yaw scale nonnegative")
    output = []
    origin = np.asarray(reference[0]["pos"], dtype=float)
    yaw = float(reference[0].get("yaw", 0.0))
    for index, point in enumerate(reference):
        yaw_rate = float(np.clip(
            yaw_rate_scale * float(point.get("yaw_rate", 0.0)), -0.8, 0.8
        ))
        if index:
            yaw += yaw_rate * quad_sim.dt
        output.append({
            "pos": origin + speed_scale * (
                np.asarray(point["pos"], dtype=float) - origin
            ),
            "vel": speed_scale * np.asarray(point.get("vel", np.zeros(3)), dtype=float),
            "acc": speed_scale * np.asarray(point.get("acc", np.zeros(3)), dtype=float),
            "yaw": yaw,
            "yaw_rate": yaw_rate,
        })
    return output


def tracking_reference(
    family: str,
    run_index: int,
    duration_seconds: float = 60.0,
    speed_scale: float = 1.0,
    yaw_rate_scale: float = 1.0,
) -> list[dict]:
    """Generate an unseen deterministic ACC-balanced tracking reference."""
    if family not in TRACKING_FAMILIES:
        raise ValueError(f"Unknown tracking family: {family}")
    sim = quad_sim()
    sim.time = np.arange(0.0, duration_seconds, sim.dt)
    family_id = TRACKING_FAMILIES[family]
    rng = random.Random(1000 * family_id + int(run_index))
    reference = sim.fct_sample_trajectory(family_id, rng, profile="acc_balanced")
    origin = np.asarray(reference[0]["pos"], dtype=float).copy()
    for point in reference:
        point["pos"] = np.asarray(point["pos"], dtype=float) - origin
    return transform_reference(reference, speed_scale, yaw_rate_scale)


def initial_state_for_reference(reference: list[dict], scenario: dict | None) -> np.ndarray:
    state = np.zeros(12, dtype=float)
    state[8] = float(reference[0].get("yaw", 0.0))
    if scenario:
        state[:3] += np.asarray(scenario["initial_position_error_m"], dtype=float)
        state[3:6] += np.asarray(scenario["initial_velocity_error_mps"], dtype=float)
        state[8] += float(scenario["initial_yaw_error_rad"])
    return state
