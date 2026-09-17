"""Utilities that keep hidden plant parameters out of nominal controllers."""

import copy
import json
from pathlib import Path

import numpy as np

from Closed_loop import ClosedLoopQuad
from Simulation import quad_sim


DEFAULT_PROTOCOL = Path(__file__).with_name("model_mismatch_protocol.json")


def load_mismatch_protocol(path=DEFAULT_PROTOCOL):
    return json.loads(Path(path).read_text())


def plant_identity(plant_id, protocol=None):
    protocol = load_mismatch_protocol() if protocol is None else protocol
    try:
        return copy.deepcopy(protocol["plant_identities"][plant_id])
    except KeyError as exc:
        choices = ", ".join(protocol["plant_identities"])
        raise ValueError(f"Unknown plant identity {plant_id!r}; choose {choices}") from exc


def configure_hidden_plant(sim, identity):
    """Perturb only physical dynamics; controller/allocation stay nominal.

    ``quad_sim`` normally shares one quadcopter parameter object between the
    controller and plant.  That is convenient nominally but leaks true mass,
    drag and rotor effectiveness under mismatch.  The controller is therefore
    redirected to an immutable nominal copy before the physical object changes.
    """
    controller_quad = copy.deepcopy(sim.quad)
    sim.controller_PID.quad = controller_quad
    sim.controller_PX4.quad = controller_quad

    physical = sim.quad
    physical.m *= float(identity["mass_scale"])
    inertia_scale = np.asarray(identity["inertia_scales"], dtype=float)
    if inertia_scale.shape != (3,):
        raise ValueError("inertia_scales must contain three principal-axis scales")
    physical.I = np.diag(np.diag(physical.I) * inertia_scale)
    physical.k_drag_linear *= float(identity["linear_drag_scale"])
    physical.k_drag_angular *= float(identity["angular_drag_scale"])
    effectiveness = np.asarray(identity["motor_effectiveness"], dtype=float)
    if effectiveness.shape != (4,) or np.any(effectiveness <= 0.0):
        raise ValueError("motor_effectiveness must contain four positive values")
    physical.prop_efficiency *= effectiveness

    sim.sim_PID = ClosedLoopQuad(physical, sim.controller_PX4)
    sim.hidden_plant_identity = copy.deepcopy(identity)
    sim.controller_parameter_source = "nominal"
    return sim


def make_hidden_plant_sim(plant_id, protocol=None):
    protocol = load_mismatch_protocol() if protocol is None else protocol
    return configure_hidden_plant(quad_sim(), plant_identity(plant_id, protocol))


def parameter_snapshot(sim):
    controller = sim.controller_PX4.quad
    physical = sim.quad
    return {
        "controller": {
            "mass": float(controller.m),
            "inertia_diagonal": np.diag(controller.I).tolist(),
            "linear_drag": float(controller.k_drag_linear),
            "angular_drag": float(controller.k_drag_angular),
            "prop_efficiency": controller.prop_efficiency.tolist(),
        },
        "physical_plant": {
            "mass": float(physical.m),
            "inertia_diagonal": np.diag(physical.I).tolist(),
            "linear_drag": float(physical.k_drag_linear),
            "angular_drag": float(physical.k_drag_angular),
            "prop_efficiency": physical.prop_efficiency.tolist(),
            "quadratic_drag": float(physical.k_drag_quadratic),
            "motor_time_constant_s": float(physical.motor_time_constant_s),
        },
    }
