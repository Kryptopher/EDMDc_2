"""Matched direct-wrench MPCs with no inner PID and horizon-varying Jacobians.

The known-model LTV controller is an explicitly nominal-model benchmark. All
controllers use the same condensed QP, physical-unit objective, move blocks,
motor inequalities, reference preview and fallback. EDMD retains lifted state
recursion; it is not silently replaced by the analytical plant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import osqp
from scipy import sparse
from threadpoolctl import threadpool_limits

from edmdc_mpc import observables
from PID_Mixer import pid_mixer

_BLAS_LIMIT = threadpool_limits(limits=1, user_api="blas")


@dataclass
class WrenchConfig:
    dt: float = 0.01
    horizon_steps: int = 50
    block_steps: int = 5
    q: list = field(default_factory=lambda: [45, 45, 55, 8, 8, 10, 6, 6, 10, .25, .25, .4])
    r: list = field(default_factory=lambda: [.01, .4, .4, .2])
    rd: list = field(default_factory=lambda: [.02, .8, .8, .4])
    terminal_scale: float = 5.0
    trust: list = field(default_factory=lambda: [15, .7, .7, .25])
    max_iterations: int = 4000
    eps_abs: float = 1e-6
    eps_rel: float = 1e-6


class NominalWrenchDynamics:
    def __init__(self, quad, dt):
        if not np.allclose(quad.I, np.diag(np.diag(quad.I))):
            raise ValueError("This nominal benchmark requires diagonal inertia")
        self.mass, self.g, self.inertia = quad.m, quad.g, np.diag(quad.I).copy()
        self.drag, self.angular_drag, self.dt = quad.k_drag_linear, quad.k_drag_angular, dt

    def derivative(self, x, u):
        x, u = np.asarray(x), np.asarray(u)
        out = np.zeros(np.broadcast_shapes(x.shape, u.shape[:-1] + (12,)), dtype=np.result_type(x, u, float))
        phi, theta, psi = x[..., 6], x[..., 7], x[..., 8]
        sp, cp, st, ct, sy, cy = np.sin(phi), np.cos(phi), np.sin(theta), np.cos(theta), np.sin(psi), np.cos(psi)
        direction = np.stack([cy*st*cp + sy*sp, sy*st*cp - cy*sp, ct*cp], axis=-1)
        out[..., :3] = x[..., 3:6]
        out[..., 3:6] = (u[..., :1]*direction - self.drag*x[..., 3:6]) / self.mass
        out[..., 5] -= self.g
        p, q, r = x[..., 9], x[..., 10], x[..., 11]
        out[..., 6] = p + sp*(st/ct)*q + cp*(st/ct)*r
        out[..., 7] = cp*q - sp*r
        out[..., 8] = (sp*q + cp*r) / ct
        out[..., 9:12] = (u[..., 1:4] - self.angular_drag*x[..., 9:12] - np.cross(x[..., 9:12], self.inertia*x[..., 9:12])) / self.inertia
        return out

    def step(self, x, u):
        dt = self.dt
        a = self.derivative(x, u)
        b = self.derivative(x + .5*dt*a, u)
        c = self.derivative(x + .5*dt*b, u)
        d = self.derivative(x + dt*c, u)
        return x + (dt/6)*(a + 2*b + 2*c + d)

    def jacobians(self, states, inputs):
        """Batch complex-step derivatives of the actual discrete RK4 map."""
        states, inputs = np.atleast_2d(states), np.atleast_2d(inputs)
        count = len(states)
        x = np.broadcast_to(states[:, None, :], (count, 16, 12)).astype(complex).copy()
        u = np.broadcast_to(inputs[:, None, :], (count, 16, 4)).astype(complex).copy()
        epsilon = 1e-20
        x[:, np.arange(12), np.arange(12)] += 1j*epsilon
        u[:, 12 + np.arange(4), np.arange(4)] += 1j*epsilon
        derivative = np.imag(self.step(x, u)) / epsilon
        return derivative[:, :12].transpose(0, 2, 1), derivative[:, 12:].transpose(0, 2, 1)


def wrench_features(x, u):
    """Exact 10-feature physical wrench map, preserving complex perturbations."""
    x, u = np.asarray(x), np.asarray(u)
    phi, theta, psi = x[..., 6], x[..., 7], x[..., 8]
    sp, cp, st, ct, sy, cy = np.sin(phi), np.cos(phi), np.sin(theta), np.cos(theta), np.sin(psi), np.cos(psi)
    direction = np.stack([cy*st*cp + sy*sp, sy*st*cp - cy*sp, ct*cp], axis=-1)
    return np.concatenate([u, u[..., :1]*direction, u[..., 1:4]*x[..., 9:12]], axis=-1)


class LiftedWrenchDynamics:
    def __init__(self, model):
        if model.get("input_type") != "applied_wrench":
            raise ValueError("Direct controller rejects desired-attitude checkpoints")
        if model.get("input_lift_type") != "thrust_direction_rate_coupling":
            raise ValueError("Expected the physical 10-feature wrench map")
        self.A, self.B = np.asarray(model["A"]), np.asarray(model["B"])
        self.scaler = model["scaler"]
        self.mean, self.scale = self.scaler.mean_, self.scaler.scale_
        self.um, self.us = model["u_scaler"].mean_, model["u_scaler"].scale_
        self.indices = np.asarray(model["active_observable_indices"], dtype=int)
        assert len(self.um) == 10 and len(self.mean) == 12
        self.dt = float(model["dt"])

    def initialize(self, x):
        return observables((x - self.mean)/self.scale, self.scaler)[self.indices]

    def decode(self, z):
        return z[..., :12]*self.scale + self.mean

    def step(self, z, u):
        lifted = (wrench_features(self.decode(z), u) - self.um) / self.us
        return z @ self.A.T + lifted @ self.B.T

    def jacobians(self, zs, inputs):
        states, inputs = np.atleast_2d(self.decode(zs)), np.atleast_2d(inputs)
        count = len(states)
        x = np.broadcast_to(states[:, None, :], (count, 16, 12)).astype(complex).copy()
        u = np.broadcast_to(inputs[:, None, :], (count, 16, 4)).astype(complex).copy()
        epsilon = 1e-20
        x[:, np.arange(12), np.arange(12)] += 1j*epsilon
        u[:, 12 + np.arange(4), np.arange(4)] += 1j*epsilon
        jac = (np.imag(wrench_features(x, u))/epsilon).transpose(0, 2, 1) / self.us[None, :, None]
        F = np.broadcast_to(self.A, (count,) + self.A.shape).copy()
        F[:, :, :12] += np.einsum("ab,nbc->nac", self.B, jac[:, :, :12]) * self.scale[None, None, :]
        G = np.einsum("ab,nbc->nac", self.B, jac[:, :, 12:])
        return F, G


class WrenchMPC:
    def __init__(self, kind, sim, config=None, model=None, initial_yaw=0.0):
        if kind not in ("linear", "yaw_linear", "edmdc", "ltv"):
            raise ValueError(kind)
        self.kind, self.config = kind, config or WrenchConfig()
        c = self.config
        if c.horizon_steps % c.block_steps:
            raise ValueError("Horizon must contain complete uniform move blocks")
        self.N, self.Nc = c.horizon_steps, c.horizon_steps // c.block_steps
        self.nvar = 4*self.Nc
        self.uscale = np.array([10., 1., 1., .3])
        self.hover = np.array([sim.quad.m*sim.quad.g, 0., 0., 0.])
        self.reference_equilibrium = np.zeros(12)
        self.reference_equilibrium[8] = initial_yaw
        self.dynamics = LiftedWrenchDynamics(model) if kind == "edmdc" else NominalWrenchDynamics(sim.quad, c.dt)
        if not np.isclose(self.dynamics.dt, c.dt):
            raise ValueError("Controller/model dt mismatch")
        self.nz = len(self.dynamics.A) if kind == "edmdc" else 12
        if kind == "linear":
            F, G = self.dynamics.jacobians(self.reference_equilibrium, self.hover)
            self.fixed_F, self.fixed_G = F[0], G[0]
        self.previous_sequence = np.tile(self.hover, (self.N, 1))
        self.previous_applied = self.hover.copy()
        self.allocation = pid_mixer.fct_allocation_matrix(sim.quad.kT, sim.quad.kD, sim.quad.l)
        self.force_max = pid_mixer.fct_max_motor_forces(sim.quad.kT, sim.max_speed, sim.quad.prop_efficiency)
        self.umin = np.array([.05*self.hover[0], -1.1, -1.1, -.35])
        self.umax = np.array([sim.quad.m*(sim.quad.g + 10), 1.1, 1.1, .35])
        self.Q = np.tile(c.q, self.N).astype(float)
        self.Q[-12:] *= c.terminal_scale
        # Decision increments are block-constant; costs integrate across all
        # stage inputs, including the shorter first difference to last command.
        self.expansion = np.zeros((4*self.N, self.nvar))
        for i in range(self.N):
            j = i // c.block_steps
            self.expansion[4*i:4*i+4, 4*j:4*j+4] = np.diag(self.uscale)
        self.difference = self.expansion.copy()
        self.difference[4:] -= self.expansion[:-4]
        self.R = np.tile(c.r, self.N)
        self.Rd = np.tile(c.rd, self.N)
        self.input_P = self.expansion.T @ (self.R[:, None]*self.expansion) + self.difference.T @ (self.Rd[:, None]*self.difference)
        bounds_map = sparse.block_diag([np.diag(self.uscale)]*self.Nc, format="csc")
        motor_map = sparse.block_diag([self.allocation @ np.diag(self.uscale)]*self.Nc, format="csc")
        self.constraint = sparse.vstack([bounds_map, motor_map], format="csc")
        # Explicit full upper triangle keeps OSQP update sparsity unchanged.
        dense_pattern = np.triu(np.ones((self.nvar, self.nvar)))
        self.p_pattern = sparse.csc_matrix(dense_pattern)
        self.p_rows = self.p_pattern.indices
        self.p_cols = np.repeat(np.arange(self.nvar), np.diff(self.p_pattern.indptr))
        self.solver = None

    def _rollout(self, measured, nominal, reference=None):
        z = self.dynamics.initialize(measured) if self.kind == "edmdc" else measured.copy()
        zs = np.empty((self.N+1, self.nz))
        zs[0] = z
        if self.kind == "yaw_linear":
            if reference is None or np.asarray(reference).shape != (self.N, 12):
                raise ValueError(
                    "Yaw-scheduled hover-linear MPC requires an (N, 12) reference"
                )
            equilibria = np.zeros((self.N, 12))
            equilibria[:, 8] = np.asarray(reference)[:, 8]
            hover_inputs = np.broadcast_to(self.hover, (self.N, 4))
            scheduled_F, scheduled_G = self.dynamics.jacobians(
                equilibria, hover_inputs
            )
        for i in range(self.N):
            if self.kind == "linear":
                z = self.reference_equilibrium + self.fixed_F @ (z-self.reference_equilibrium) + self.fixed_G @ (nominal[i]-self.hover)
            elif self.kind == "yaw_linear":
                equilibrium = equilibria[i]
                z = (equilibrium + scheduled_F[i] @ (z-equilibrium)
                     + scheduled_G[i] @ (nominal[i]-self.hover))
            else:
                z = self.dynamics.step(z, nominal[i])
            zs[i+1] = z
        if self.kind == "linear":
            F = np.broadcast_to(self.fixed_F, (self.N, 12, 12))
            G = np.broadcast_to(self.fixed_G, (self.N, 12, 4))
        elif self.kind == "yaw_linear":
            F, G = scheduled_F, scheduled_G
        else:
            F, G = self.dynamics.jacobians(zs[:-1], nominal)
        predicted = self.dynamics.decode(zs[1:]) if self.kind == "edmdc" else zs[1:]
        sensitivity = np.zeros((self.nz, self.nvar))
        S = np.empty((12*self.N, self.nvar))
        for i in range(self.N):
            sensitivity = F[i] @ sensitivity
            j = i // self.config.block_steps
            sensitivity[:, 4*j:4*j+4] += G[i]*self.uscale
            physical = sensitivity[:12] * self.dynamics.scale[:, None] if self.kind == "edmdc" else sensitivity
            S[12*i:12*i+12] = physical
        return predicted, S

    def compute(self, measured, reference):
        started = time.perf_counter()
        c = self.config
        shifted = np.vstack([self.previous_sequence[1:], self.previous_sequence[-1:]])
        base = shifted[::c.block_steps].copy()
        nominal = np.repeat(base, c.block_steps, axis=0)
        status, iterations = "not_run", 0
        try:
            predicted, S = self._rollout(measured, nominal, reference=reference)
            residual = predicted - reference
            # A single local yaw branch avoids artificial 2*pi tracking errors.
            residual[:, 8] = np.arctan2(np.sin(residual[:, 8]), np.cos(residual[:, 8]))
            delta_prev = np.vstack([nominal[0]-self.previous_applied, np.diff(nominal, axis=0)]).reshape(-1)
            P = S.T @ (self.Q[:, None]*S) + self.input_P + 1e-9*np.eye(self.nvar)
            q = S.T @ (self.Q*residual.reshape(-1)) + self.expansion.T @ (self.R*(nominal-self.hover).reshape(-1)) + self.difference.T @ (self.Rd*delta_prev)
            low = np.maximum(self.umin-base, -np.asarray(c.trust)).reshape(-1)
            high = np.minimum(self.umax-base, np.asarray(c.trust)).reshape(-1)
            motor = base @ self.allocation.T
            low = np.r_[low, (1e-4-motor).reshape(-1)]
            high = np.r_[high, (self.force_max-1e-4-motor).reshape(-1)]
            upper = self.p_pattern.copy()
            upper.data = ((P+P.T)*.5)[self.p_rows, self.p_cols]
            if not np.isfinite(upper.data).all() or not np.isfinite(q).all():
                raise FloatingPointError("nonfinite_prediction")
            if self.solver is None:
                self.solver = osqp.OSQP()
                self.solver.setup(P=upper, q=q, A=self.constraint, l=low, u=high, verbose=False, polishing=False, warm_starting=True, max_iter=c.max_iterations, eps_abs=c.eps_abs, eps_rel=c.eps_rel)
            else:
                self.solver.update(Px=upper.data, q=q, l=low, u=high)
            self.solver.warm_start(x=np.zeros(self.nvar))
            solution = self.solver.solve()
            status, iterations = solution.info.status, int(solution.info.iter)
            accepted = status in ("solved", "solved inaccurate") and solution.x is not None
            if accepted:
                ax = self.constraint @ solution.x
                accepted = bool(max(np.max(low-ax), np.max(ax-high)) < 1e-4)
                if not accepted:
                    status = "rejected_constraint_residual"
            if accepted:
                optimum = nominal + (self.expansion @ solution.x).reshape(self.N, 4)
                self.previous_sequence = optimum
                command = optimum[0].copy()
            else:
                command = self.previous_applied.copy()
                self.previous_sequence = np.tile(command, (self.N, 1))
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
            status = f"prediction_error:{type(error).__name__}"
            accepted = False
            command = self.previous_applied.copy()
            self.previous_sequence = np.tile(command, (self.N, 1))
        return command, {"status": status, "accepted": accepted, "iterations": iterations, "elapsed_s": time.perf_counter()-started}


def plant_step(sim, state, command, dt):
    """Shared allocator then original nonlinear plant; no cascaded PID call."""
    omega, applied, forces, saturated = pid_mixer.fct_allocate_wrench(command, sim.quad.kT, sim.quad.kD, sim.quad.l, max_omega=sim.max_speed, prop_efficiency=sim.quad.prop_efficiency)
    derivative = lambda x: sim.quad.fct_dynamics(0., x, omega)
    a = derivative(state)
    b = derivative(state+.5*dt*a)
    c = derivative(state+.5*dt*b)
    d = derivative(state+dt*c)
    return state + dt*(a+2*b+2*c+d)/6, applied, forces, saturated


def physical_reference(reference, sim, dt):
    """Common preview, with exact Euler-rate -> body-rate conversion.

    No true future plant response or target state is read here. Acceleration
    and yaw belong to the prescribed reference available to every controller.
    """
    states = np.zeros((len(reference), 12))
    states[:, :3] = [point["pos"] for point in reference]
    states[:, 3:6] = [point["vel"] for point in reference]
    states[:, 8] = np.unwrap([point["yaw"] for point in reference])
    acc = np.asarray([point["acc"] for point in reference])
    force = sim.quad.m*(acc + [0, 0, sim.quad.g]) + sim.quad.k_drag_linear*states[:, 3:6]
    yaw = states[:, 8]
    magnitude = np.linalg.norm(force, axis=1)
    states[:, 6] = np.arcsin(np.clip((np.sin(yaw)*force[:, 0] - np.cos(yaw)*force[:, 1])/magnitude, -1, 1))
    states[:, 7] = np.arctan2(np.cos(yaw)*force[:, 0] + np.sin(yaw)*force[:, 1], force[:, 2])
    angles_dot = np.gradient(states[:, 6:9], dt, axis=0, edge_order=2)
    phi, theta = states[:, 6], states[:, 7]
    states[:, 9] = angles_dot[:, 0] - angles_dot[:, 2]*np.sin(theta)
    states[:, 10] = angles_dot[:, 1]*np.cos(phi) + angles_dot[:, 2]*np.sin(phi)*np.cos(theta)
    states[:, 11] = -angles_dot[:, 1]*np.sin(phi) + angles_dot[:, 2]*np.cos(phi)*np.cos(theta)
    return states
