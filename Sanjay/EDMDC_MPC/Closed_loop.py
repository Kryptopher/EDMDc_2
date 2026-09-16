# %% Import libraries
import numpy as np
from scipy.integrate import solve_ivp

class ClosedLoopQuad:
    def __init__(self, quad, controller):
        self.quad = quad
        self.controller = controller

    def fct_simulate(self, time, dt, ref_traj, init_state,
                     return_requested=False, return_outer=False):
        state = np.array(init_state, dtype=float)
        states = np.zeros((len(time), len(state)))
        omegas = np.zeros((len(time), 4))
        control_inputs = np.zeros((len(time), 4))
        requested_inputs = np.zeros((len(time), 4))
        allocated_inputs = np.zeros((len(time), 4)
        )
        outer_inputs = np.zeros((len(time), 4))
        self.quad.fct_reset_motor_state()

        for i, t in enumerate(time):
            # Log the state before the command is applied so each training
            # tuple is aligned as (x_k, u_k, x_{k+1}).
            states[i] = state
            omega_cmd, u_allocated = self.controller.fct_step(state, ref_traj[i], dt)
            # The controller and physical plant may intentionally use different
            # parameter objects during model-mismatch studies.  Log the wrench
            # actually produced by the physical rotors, while retaining the
            # nominal allocator output as a separate diagnostic.
            omega_plant = self.quad.fct_apply_motor_dynamics(omega_cmd, dt)
            rotor_thrust, rotor_drag = self.quad.fct_rotor_forces(omega_plant)
            thrust, torque = self.quad.fct_Rotor_torque(
                rotor_thrust, rotor_drag
            )
            u_applied = np.r_[thrust[2], torque]
            control_inputs[i] = u_applied
            allocated_inputs[i] = u_allocated
            requested_inputs[i] = getattr(
                self.controller, "last_requested_wrench", u_applied
            )
            if not hasattr(self.controller, "last_outer_command"):
                raise RuntimeError(
                    "Controller must expose last_outer_command = "
                    "[thrust, phi_des, theta_des, psi_des]"
                )
            outer_inputs[i] = self.controller.last_outer_command

            def ode(t_local, s_local):
                return self.quad.fct_dynamics(t_local, s_local, omega_plant)

            sol = solve_ivp(ode, [t, t + dt], state, method="RK45")
            state = sol.y[:, -1]

            omegas[i] = omega_cmd
        self.controller.fct_reset()
        self.last_applied_inputs = control_inputs
        self.last_allocated_inputs = allocated_inputs
        self.last_requested_inputs = requested_inputs
        self.last_outer_inputs = outer_inputs
        if return_requested and return_outer:
            return (
                time, states, omegas, control_inputs,
                requested_inputs, outer_inputs,
            )
        if return_requested:
            return time, states, omegas, control_inputs, requested_inputs
        if return_outer:
            return time, states, omegas, control_inputs, outer_inputs
        return time, states, omegas, control_inputs
