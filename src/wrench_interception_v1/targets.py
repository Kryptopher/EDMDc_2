"""Frozen scripted target definitions copied without running the earlier campaign."""

import numpy as np

from dataclasses import dataclass

class Target:
    def position(self, t):
        raise NotImplementedError

    def velocity(self, t):
        raise NotImplementedError

    def acceleration(self, t):
        raise NotImplementedError

    def reference(self, t):
        position = np.asarray(self.position(t), dtype=float)
        velocity = np.asarray(self.velocity(t), dtype=float)
        acceleration = np.asarray(self.acceleration(t), dtype=float)
        horizontal_speed_sq = float(np.dot(velocity[:2], velocity[:2]))
        if horizontal_speed_sq > 1e-10:
            yaw = float(np.arctan2(velocity[1], velocity[0]))
            yaw_rate = float(
                (velocity[0] * acceleration[1] - velocity[1] * acceleration[0])
                / horizontal_speed_sq
            )
        else:
            yaw = 0.0
            yaw_rate = 0.0
        return {
            "pos": position,
            "vel": velocity,
            "acc": acceleration,
            "yaw": yaw,
            "yaw_rate": float(np.clip(yaw_rate, -0.8, 0.8)),
        }

@dataclass
class StraightTarget(Target):
    p0: np.ndarray
    velocity_vector: np.ndarray

    def position(self, t):
        return self.p0 + self.velocity_vector * t

    def velocity(self, t):
        return self.velocity_vector

    def acceleration(self, t):
        return np.zeros(3)

@dataclass
class AcceleratingTarget(Target):
    p0: np.ndarray
    v0: np.ndarray
    acceleration_vector: np.ndarray

    def position(self, t):
        return self.p0 + self.v0*t + 0.5*self.acceleration_vector*t*t

    def velocity(self, t):
        return self.v0 + self.acceleration_vector*t

    def acceleration(self, t):
        return self.acceleration_vector

@dataclass
class HelixTarget(Target):
    center: np.ndarray
    radius: float
    z0: float
    climb_rate: float
    omega: float
    phase: float

    def position(self, t):
        angle = self.phase + self.omega*t
        return np.array([
            self.center[0] + self.radius*np.cos(angle),
            self.center[1] + self.radius*np.sin(angle),
            self.z0 + self.climb_rate*t,
        ])

    def velocity(self, t):
        angle = self.phase + self.omega*t
        return np.array([
            -self.radius*self.omega*np.sin(angle),
            self.radius*self.omega*np.cos(angle),
            self.climb_rate,
        ])

    def acceleration(self, t):
        angle = self.phase + self.omega*t
        return np.array([
            -self.radius*self.omega**2*np.cos(angle),
            -self.radius*self.omega**2*np.sin(angle),
            0.0,
        ])

@dataclass
class WeavingTarget(Target):
    p0: np.ndarray
    vx: float
    amp_y: float
    omega_y: float
    amp_z: float
    omega_z: float
    phase_y: float
    phase_z: float

    def position(self, t):
        return np.array([
            self.p0[0] + self.vx*t,
            self.p0[1] + self.amp_y*np.sin(self.omega_y*t + self.phase_y),
            self.p0[2] + self.amp_z*np.sin(self.omega_z*t + self.phase_z),
        ])

    def velocity(self, t):
        return np.array([
            self.vx,
            self.amp_y*self.omega_y*np.cos(self.omega_y*t + self.phase_y),
            self.amp_z*self.omega_z*np.cos(self.omega_z*t + self.phase_z),
        ])

    def acceleration(self, t):
        return np.array([
            0.0,
            -self.amp_y*self.omega_y**2*np.sin(self.omega_y*t + self.phase_y),
            -self.amp_z*self.omega_z**2*np.sin(self.omega_z*t + self.phase_z),
        ])

def scenario_family(family, rng):
    if family == "straight":
        return StraightTarget(
            np.array([rng.uniform(9, 14), rng.uniform(-6, 6), rng.uniform(4, 8)]),
            np.array([rng.uniform(-1.3, -0.5), rng.uniform(-0.8, 0.8), rng.uniform(-0.1, 0.2)]),
        )
    if family == "accelerating":
        return AcceleratingTarget(
            np.array([rng.uniform(9, 13), rng.uniform(-4, 4), rng.uniform(4, 7)]),
            np.array([rng.uniform(-1.0, -0.4), rng.uniform(-0.4, 0.4), 0.0]),
            np.array([rng.uniform(-0.08, 0.03), rng.uniform(-0.12, 0.12), rng.uniform(-0.03, 0.03)]),
        )
    if family == "helix":
        radius = rng.uniform(3.5, 5.5)
        speed = rng.uniform(1.5, 2.8)
        return HelixTarget(
            np.array([rng.uniform(6.5, 9.0), rng.uniform(-1.5, 1.5)]),
            radius, rng.uniform(3.5, 6.0), rng.uniform(-0.05, 0.18),
            speed/radius, rng.uniform(-np.pi, np.pi),
        )
    if family == "weaving":
        return WeavingTarget(
            np.array([rng.uniform(9, 13), rng.uniform(-1, 1), rng.uniform(4, 7)]),
            rng.uniform(-1.1, -0.5), rng.uniform(2.0, 4.5),
            2*np.pi*rng.uniform(0.05, 0.11), rng.uniform(0.8, 2.2),
            2*np.pi*rng.uniform(0.06, 0.14), rng.uniform(-np.pi, np.pi),
            rng.uniform(-np.pi, np.pi),
        )
    raise ValueError(f"Unknown target family: {family}")
