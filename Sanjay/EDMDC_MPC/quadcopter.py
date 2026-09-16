# %% Import Libraries
import numpy as np

# %% Quadcopter Dynamics
class quadcopter:
    def __init__(self, m, g, l, I, kD, kT, k_drag_linear, k_drag_angular, prop_efficiency=None):
        self.m = m
        self.g = g
        self.l = l
        self.I = I
        self.kD = kD  # drag torque constant
        self.kT = kT  # thrust coefficient
        self.k_drag_linear = k_drag_linear
        self.k_drag_angular = k_drag_angular
        # Optional outer-loop-visible nonlinear/actuator dynamics. Defaults
        # preserve every legacy trajectory exactly.
        self.k_drag_quadratic = 0.0  # N / (m/s)^2
        self.motor_time_constant_s = 0.0
        self._motor_omega_state = None
        if prop_efficiency is None:
            self.prop_efficiency = np.ones(4)
        else:
            self.prop_efficiency = np.array(prop_efficiency, dtype=float)
        # Deterministic disturbance hooks used by the publication robustness
        # campaign.  All defaults are zero, so legacy simulations are exactly
        # unchanged.  Wind is represented as air velocity in world axes and
        # therefore enters through the existing linear relative-air drag.
        self.wind_velocity_world = np.zeros(3, dtype=float)
        self.wind_gust_velocity_world = np.zeros(3, dtype=float)
        self.wind_gust_frequency_radps = 0.0
        self.wind_gust_phase_rad = 0.0
        self.external_force_world = np.zeros(3, dtype=float)
        self.external_torque_body = np.zeros(3, dtype=float)
    
    def fct_R_matrix(self, phi, theta, psi):
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                       [np.sin(psi),  np.cos(psi), 0],
                       [0, 0, 1]])
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(phi), -np.sin(phi)],
                       [0, np.sin(phi),  np.cos(phi)]])
        return Rz @ Ry @ Rx

    def fct_W_matrix(self, phi, theta):
        return np.array([
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
        ])
    
    def fct_wind_force(self, t):
        # Constant bias + smooth gust
        F_bias = 0.8
        F_gust = 0.5 * np.sin(0.6*t) # smooth sinusoidal gust
        return np.array([F_bias + F_gust, 0.0, 0.0])

    def fct_wind_velocity(self, t):
        """Return the configured deterministic world-frame air velocity."""
        steady = np.asarray(self.wind_velocity_world, dtype=float)
        gust = np.asarray(self.wind_gust_velocity_world, dtype=float)
        phase = (
            float(self.wind_gust_frequency_radps) * float(t)
            + float(self.wind_gust_phase_rad)
        )
        return steady + gust * np.sin(phase)

    def fct_rotor_forces(self, omega):
        omega = np.array(omega, dtype=float)

        T = self.prop_efficiency * self.kT * omega**2
        tau = self.prop_efficiency * self.kD * omega**2

        return T, tau

    def fct_reset_motor_state(self):
        self._motor_omega_state = None

    def fct_apply_motor_dynamics(self, omega_command, dt):
        """Apply a discrete first-order rotor-speed response.

        The first command initializes the rotor state to avoid introducing an
        artificial takeoff spin-up transient into tracking experiments.
        """
        command = np.asarray(omega_command, dtype=float)
        tau = float(self.motor_time_constant_s)
        if tau <= 0.0:
            return command
        if self._motor_omega_state is None:
            self._motor_omega_state = command.copy()
        else:
            alpha = 1.0 - np.exp(-float(dt) / tau)
            self._motor_omega_state += alpha * (command - self._motor_omega_state)
        return self._motor_omega_state.copy()

    def fct_Rotor_torque(self, T, tau):
        # X-configuration
        arm = self.l / np.sqrt(2)
        u1 = np.sum(T)
        u2 = arm * (-T[0] - T[1] + T[2] + T[3]) # Roll
        u3 = arm * ( T[0] - T[1] - T[2] + T[3]) # Pitch
        u4 = tau[0] - tau[1] + tau[2] - tau[3]  # Yaw
        thrust = np.array([0, 0, u1])
        torque = np.array([u2, u3, u4])
        return thrust, torque

    def fct_dynamics(self, t, state, omega):
        omega = np.array(omega)
        T, tau = self.fct_rotor_forces(omega)
        thrust, torque = self.fct_Rotor_torque(T, tau)

        m, g, I = self.m, self.g, self.I
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state
        vel = np.array([vx, vy, vz])
        ang = np.array([phi, theta, psi])
        omega_body = np.array([p, q, r])

        R = self.fct_R_matrix(phi, theta, psi)
        gravity = np.array([0, 0, -g])
        relative_air_velocity = vel - self.fct_wind_velocity(t)
        speed_air = np.linalg.norm(relative_air_velocity)
        drag_world = (
            -self.k_drag_linear * relative_air_velocity
            - float(self.k_drag_quadratic) * speed_air * relative_air_velocity
        )
        external_force = np.asarray(self.external_force_world, dtype=float)
        acc = (1/m) * (R @ thrust + drag_world + external_force) + gravity

        damping = -self.k_drag_angular * omega_body
        external_torque = np.asarray(self.external_torque_body, dtype=float)
        omega_dot = np.linalg.inv(I) @ (
            torque + damping + external_torque
            - np.cross(omega_body, I @ omega_body)
        )
        euler_dot = self.fct_W_matrix(phi, theta) @ omega_body

        dstate = np.zeros(12)
        dstate[0:3] = vel
        dstate[3:6] = acc
        dstate[6:9] = euler_dot
        dstate[9:12] = omega_dot
        return dstate
