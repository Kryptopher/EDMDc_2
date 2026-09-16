import os
import pickle
import numpy as np
import scipy.sparse as sp

from Helperfcts import helperfcts

try:
    from threadpoolctl import threadpool_limits
except ModuleNotFoundError:
    threadpool_limits = None

try:
    import osqp
except ModuleNotFoundError:
    osqp = None

# The MPC assembles many small dense matrices. Letting a multi-threaded BLAS
# fan out those operations produces latency spikes that are harmful at a fixed
# control rate. This limit can be overridden (or disabled with 0) per process.
_blas_thread_count = int(os.environ.get("EDMDC_BLAS_THREADS", "1"))
_blas_thread_limiter = (
    threadpool_limits(limits=_blas_thread_count, user_api="blas")
    if threadpool_limits is not None and _blas_thread_count > 0 else None
)

STATE_DIM = 12
STATE_LABELS = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r"]
REDUCED_10_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
RAW_INPUT_DIM = 4
RAW_INPUT_LABELS = ["thrust", "tau_roll", "tau_pitch", "tau_yaw"]
INPUT_LIFT_TYPE = "thrust_direction_rate_coupling"
LEGACY_INPUT_LIFT_TYPE = "thrust_direction"
OUTER_RAW_INPUT_LIFT_TYPE = "raw_outer_command"
OUTER_ATTITUDE_INPUT_LIFT_TYPE = "outer_attitude_error_thrust_vector"
INPUT_LIFT_LABELS = RAW_INPUT_LABELS + [
    "thrust_x", "thrust_y", "thrust_z",
    "tau_roll_p", "tau_pitch_q", "tau_yaw_r",
]
LEGACY_INPUT_LIFT_LABELS = RAW_INPUT_LABELS + ["thrust_x", "thrust_y", "thrust_z"]
OUTER_RAW_INPUT_LABELS = ["thrust", "phi_des", "theta_des", "psi_des"]
OUTER_ATTITUDE_INPUT_LIFT_LABELS = OUTER_RAW_INPUT_LABELS + [
    "desired_thrust_x", "desired_thrust_y", "desired_thrust_z",
    "sin_phi_des", "cos_phi_des",
    "sin_theta_des", "cos_theta_des",
    "sin_psi_des", "cos_psi_des",
    "phi_error", "theta_error", "psi_error",
]


def wrap_angle_pi(angle):
    """Wrap an angle or array of angles to [-pi, pi)."""
    return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def thrust_direction_from_state_phys(states_phys):
    """Return the world-frame thrust direction from physical 12-state samples."""
    states = np.asarray(states_phys, dtype=float)
    scalar = states.ndim == 1
    states_2d = np.atleast_2d(states)

    if states_2d.shape[1] < STATE_DIM:
        raise ValueError(
            f"Expected at least {STATE_DIM} state entries, got {states_2d.shape[1]}"
        )

    phi = states_2d[:, 6]
    theta = states_2d[:, 7]
    psi = states_2d[:, 8]

    s_phi, c_phi = np.sin(phi), np.cos(phi)
    s_theta, c_theta = np.sin(theta), np.cos(theta)
    s_psi, c_psi = np.sin(psi), np.cos(psi)

    dirs = np.column_stack([
        c_psi*s_theta*c_phi + s_psi*s_phi,
        s_psi*s_theta*c_phi - c_psi*s_phi,
        c_theta*c_phi,
    ])
    return dirs[0] if scalar else dirs


def lift_inputs_from_phys(states_phys, raw_inputs):
    """Map raw commands to the learned EDMDc input vector."""
    states = np.asarray(states_phys, dtype=float)
    raw = np.asarray(raw_inputs, dtype=float)
    scalar = states.ndim == 1 and raw.ndim == 1

    states_2d = np.atleast_2d(states)
    raw_2d = np.atleast_2d(raw)

    if raw_2d.shape[1] < RAW_INPUT_DIM:
        raise ValueError(
            f"Expected {RAW_INPUT_DIM} raw input channels, got {raw_2d.shape[1]}"
        )

    if states_2d.shape[0] == 1 and raw_2d.shape[0] > 1:
        states_2d = np.repeat(states_2d, raw_2d.shape[0], axis=0)
    elif raw_2d.shape[0] == 1 and states_2d.shape[0] > 1:
        raw_2d = np.repeat(raw_2d, states_2d.shape[0], axis=0)

    if states_2d.shape[0] != raw_2d.shape[0]:
        raise ValueError(
            f"State/input sample mismatch: {states_2d.shape[0]} vs {raw_2d.shape[0]}"
        )

    raw_4 = raw_2d[:, :RAW_INPUT_DIM]
    thrust = raw_4[:, :1]
    thrust_dir = thrust_direction_from_state_phys(states_2d)
    lifted_parts = [raw_4, thrust * thrust_dir]
    if raw_2d.shape[1] >= RAW_INPUT_DIM:
        rates = states_2d[:, 9:12]
        lifted_parts.append(raw_4[:, 1:4] * rates)
    lifted = np.hstack(lifted_parts)
    return lifted[0] if scalar else lifted


def outer_command_lift_from_phys(states_phys, raw_inputs, input_lift_type=None):
    """Lift [thrust, desired roll, pitch, yaw] using the current attitude.

    The nonlinear lift exposes the desired world thrust vector and the attitude
    errors that actually drive the inner attitude controller.  Raw mode remains
    available for a controlled ablation against the original affine interface.
    """
    lift_type = input_lift_type or OUTER_RAW_INPUT_LIFT_TYPE
    states = np.asarray(states_phys, dtype=float)
    raw = np.asarray(raw_inputs, dtype=float)
    scalar = states.ndim == 1 and raw.ndim == 1
    states_2d = np.atleast_2d(states)
    raw_2d = np.atleast_2d(raw)

    if states_2d.shape[1] < STATE_DIM:
        raise ValueError(
            f"Expected at least {STATE_DIM} state entries, got {states_2d.shape[1]}"
        )
    if raw_2d.shape[1] < RAW_INPUT_DIM:
        raise ValueError(
            f"Expected {RAW_INPUT_DIM} outer-command channels, got {raw_2d.shape[1]}"
        )
    if states_2d.shape[0] == 1 and raw_2d.shape[0] > 1:
        states_2d = np.repeat(states_2d, raw_2d.shape[0], axis=0)
    elif raw_2d.shape[0] == 1 and states_2d.shape[0] > 1:
        raw_2d = np.repeat(raw_2d, states_2d.shape[0], axis=0)
    if states_2d.shape[0] != raw_2d.shape[0]:
        raise ValueError(
            f"State/input sample mismatch: {states_2d.shape[0]} vs {raw_2d.shape[0]}"
        )

    raw_4 = raw_2d[:, :RAW_INPUT_DIM]
    if lift_type == OUTER_RAW_INPUT_LIFT_TYPE:
        return raw_4[0] if scalar else raw_4
    if lift_type != OUTER_ATTITUDE_INPUT_LIFT_TYPE:
        raise ValueError(f"Unsupported outer-command input lift {lift_type!r}")

    desired_attitude_state = np.zeros((raw_4.shape[0], STATE_DIM), dtype=float)
    desired_attitude_state[:, 6:9] = raw_4[:, 1:4]
    desired_thrust = raw_4[:, :1] * thrust_direction_from_state_phys(
        desired_attitude_state
    )
    desired_angles = raw_4[:, 1:4]
    angle_features = np.column_stack([
        np.sin(desired_angles[:, 0]), np.cos(desired_angles[:, 0]),
        np.sin(desired_angles[:, 1]), np.cos(desired_angles[:, 1]),
        np.sin(desired_angles[:, 2]), np.cos(desired_angles[:, 2]),
    ])
    attitude_error = desired_angles - states_2d[:, 6:9]
    attitude_error[:, 2] = wrap_angle_pi(attitude_error[:, 2])
    lifted = np.hstack([raw_4, desired_thrust, angle_features, attitude_error])
    return lifted[0] if scalar else lifted


def scaled_lifted_input_from_phys(
    state_phys, raw_input, u_scaler, input_lift_type=None
):
    if input_lift_type in (
        OUTER_RAW_INPUT_LIFT_TYPE, OUTER_ATTITUDE_INPUT_LIFT_TYPE
    ):
        lifted = outer_command_lift_from_phys(
            state_phys, raw_input, input_lift_type=input_lift_type
        )
    else:
        lifted = lift_inputs_from_phys(state_phys, raw_input)
    expected = int(getattr(u_scaler, "n_features_in_", np.asarray(lifted).shape[-1]))
    if np.asarray(lifted).shape[-1] < expected:
        raise ValueError(
            f"Input lift produced {np.asarray(lifted).shape[-1]} features; "
            f"model scaler expects {expected}"
        )
    lifted = np.asarray(lifted, dtype=float)[..., :expected]
    return u_scaler.transform(np.atleast_2d(lifted)).flatten()


def outer_command_lift_jacobians_scaled(
    state_phys, raw_input, state_scaler, u_scaler
):
    """Jacobians of the nonlinear outer lift in standardized coordinates.

    Returns derivatives with respect to standardized physical state and the
    first four standardized raw commands. This keeps MPC decision variables at
    [thrust, desired roll, desired pitch, desired yaw], never at 16 fictitious
    independently controllable lifted channels.
    """
    state = np.asarray(state_phys, dtype=float).reshape(-1)
    raw = np.asarray(raw_input, dtype=float).reshape(-1)[:RAW_INPUT_DIM]
    if state.size < STATE_DIM:
        raise ValueError(f"Expected {STATE_DIM} states, got {state.size}")
    if getattr(u_scaler, "n_features_in_", 0) != len(
        OUTER_ATTITUDE_INPUT_LIFT_LABELS
    ):
        raise ValueError("Outer attitude-error lift requires a 16-feature scaler")

    thrust, phi, theta, psi = raw
    s_phi, c_phi = np.sin(phi), np.cos(phi)
    s_theta, c_theta = np.sin(theta), np.cos(theta)
    s_psi, c_psi = np.sin(psi), np.cos(psi)
    direction = np.array([
        c_psi*s_theta*c_phi + s_psi*s_phi,
        s_psi*s_theta*c_phi - c_psi*s_phi,
        c_theta*c_phi,
    ])
    direction_derivatives = np.column_stack([
        np.array([
            -c_psi*s_theta*s_phi + s_psi*c_phi,
            -s_psi*s_theta*s_phi - c_psi*c_phi,
            -c_theta*s_phi,
        ]),
        np.array([
            c_psi*c_theta*c_phi,
            s_psi*c_theta*c_phi,
            -s_theta*c_phi,
        ]),
        np.array([
            -s_psi*s_theta*c_phi + c_psi*s_phi,
            c_psi*s_theta*c_phi + s_psi*s_phi,
            0.0,
        ]),
    ])

    n_lift = len(OUTER_ATTITUDE_INPUT_LIFT_LABELS)
    jac_u_phys = np.zeros((n_lift, RAW_INPUT_DIM), dtype=float)
    jac_x_phys = np.zeros((n_lift, STATE_DIM), dtype=float)
    jac_u_phys[:RAW_INPUT_DIM] = np.eye(RAW_INPUT_DIM)
    jac_u_phys[4:7, 0] = direction
    jac_u_phys[4:7, 1:4] = thrust * direction_derivatives
    jac_u_phys[7, 1] = c_phi
    jac_u_phys[8, 1] = -s_phi
    jac_u_phys[9, 2] = c_theta
    jac_u_phys[10, 2] = -s_theta
    jac_u_phys[11, 3] = c_psi
    jac_u_phys[12, 3] = -s_psi
    jac_u_phys[13:16, 1:4] = np.eye(3)
    jac_x_phys[13:16, 6:9] = -np.eye(3)

    lift_scale = np.asarray(u_scaler.scale_, dtype=float)
    raw_scale = np.asarray(u_scaler.scale_[:RAW_INPUT_DIM], dtype=float)
    state_scale = np.asarray(state_scaler.scale_[:STATE_DIM], dtype=float)
    jac_u_std = (jac_u_phys * raw_scale[None, :]) / lift_scale[:, None]
    jac_x_std = (jac_x_phys * state_scale[None, :]) / lift_scale[:, None]
    return jac_x_std, jac_u_std


# File I/O
def load_edmdc_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def load_simulation_runs(filename, input_source="applied_wrench"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    source = str(input_source).strip().lower()
    if source == "applied_wrench":
        key = "U"
    elif source == "outer_command":
        key = "U_outer"
    else:
        raise ValueError(
            "input_source must be applied_wrench or outer_command, "
            f"got {input_source!r}"
        )
    if key not in data:
        raise ValueError(
            f"Dataset lacks {key}; regenerate it with dual-input logging."
        )
    return data["t"], data["states"], data[key], data["ref_traj_list"]

# State lifting
# The current lifted model uses the full 12-state vector:
# [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
#
# Observable ordering must remain consistent with EDMDc_training.py:
#  - states
#  - sin/cos of roll, pitch, and yaw
#  - selected cross terms
#  - quadratic energy-like terms
#  - constant bias

def _scaler_state_dim(scaler):
    return int(getattr(scaler, "n_features_in_", len(getattr(scaler, "mean_", []))))


def _observables_10state(x_std, scaler):
    x = np.asarray(x_std).flatten()
    assert len(x) == 10, f"Expected 10-state vector, got {len(x)}"

    obs = list(x)  # 10 linear terms

    phi_rad   = x[6] * scaler.scale_[6] + scaler.mean_[6]
    theta_rad = x[7] * scaler.scale_[7] + scaler.mean_[7]

    obs.append(np.sin(phi_rad))
    obs.append(np.cos(phi_rad))
    obs.append(np.sin(theta_rad))
    obs.append(np.cos(theta_rad))

    obs.append(x[6] * x[8])   # phi * p
    obs.append(x[7] * x[9])   # theta * q
    obs.append(x[3] * x[6])   # vx * phi
    obs.append(x[4] * x[7])   # vy * theta

    obs.append(x[3]**2 + x[4]**2 + x[5]**2)   # v_sq
    obs.append(x[8]**2 + x[9]**2)             # omega_sq

    obs.append(x[3] * x[7])  # vx * theta
    obs.append(x[4] * x[6])  # vy * phi
    obs.append(x[5] * x[5])  # vz²
    obs.append(x[6] * x[6])  # phi²
    obs.append(x[7] * x[7])  # theta²
    obs.append(x[8] * x[9])  # p * q

    obs.append(1.0)

    return np.asarray(obs, dtype=float)


def _observables_12state(x_std, scaler):
    x = np.asarray(x_std).flatten()
    assert len(x) == STATE_DIM, f"Expected 12-state vector, got {len(x)}"

    obs = list(x)  # 12 linear terms

    phi_rad   = x[6] * scaler.scale_[6] + scaler.mean_[6]
    theta_rad = x[7] * scaler.scale_[7] + scaler.mean_[7]
    psi_rad   = x[8] * scaler.scale_[8] + scaler.mean_[8]

    s_phi, c_phi = np.sin(phi_rad), np.cos(phi_rad)
    s_theta, c_theta = np.sin(theta_rad), np.cos(theta_rad)
    s_psi, c_psi = np.sin(psi_rad), np.cos(psi_rad)

    obs.extend([
        s_phi, c_phi,
        s_theta, c_theta,
        s_psi, c_psi,
    ])

    obs.extend([
        x[6] * x[9],     # phi * p
        x[7] * x[10],    # theta * q
        x[8] * x[11],    # psi * r
        x[3] * x[6],     # vx * phi
        x[4] * x[7],     # vy * theta
        x[3] * x[8],     # vx * psi
        x[4] * x[8],     # vy * psi
        x[5] * x[7],     # vz * theta
    ])

    obs.extend([
        x[3]**2 + x[4]**2 + x[5]**2,       # v_sq
        x[9]**2 + x[10]**2 + x[11]**2,     # omega_sq
    ])

    obs.extend([
        x[5] * x[5],     # vz^2
        x[6] * x[6],     # phi^2
        x[7] * x[7],     # theta^2
        x[8] * x[8],     # psi^2
        x[9] * x[10],    # p * q
        x[10] * x[11],   # q * r
        x[9] * x[11],    # p * r
    ])

    obs.extend([
        x[0] * x[1],     # x * y
        x[0] * x[3],     # x * vx
        x[1] * x[4],     # y * vy
        x[0] * x[4],     # x * vy
        x[1] * x[3],     # y * vx
        x[3] * x[4],     # vx * vy
        x[0] * x[0],     # x^2
        x[1] * x[1],     # y^2
        x[3] * x[3],     # vx^2
        x[4] * x[4],     # vy^2
        x[0] * x[7],     # x * theta
        x[1] * x[6],     # y * phi
        x[3] * x[7],     # vx * theta
        x[4] * x[6],     # vy * phi
    ])

    vx, vy, vz = x[3], x[4], x[5]
    body_vx = c_theta*c_psi*vx + c_theta*s_psi*vy - s_theta*vz
    body_vy = (s_phi*s_theta*c_psi - c_phi*s_psi)*vx + \
              (s_phi*s_theta*s_psi + c_phi*c_psi)*vy + \
              s_phi*c_theta*vz
    body_vz = (c_phi*s_theta*c_psi + s_phi*s_psi)*vx + \
              (c_phi*s_theta*s_psi - s_phi*c_psi)*vy + \
              c_phi*c_theta*vz

    thrust_dir_x = c_psi*s_theta*c_phi + s_psi*s_phi
    thrust_dir_y = s_psi*s_theta*c_phi - c_psi*s_phi
    thrust_dir_z = c_theta*c_phi

    obs.extend([
        body_vx, body_vy, body_vz,
        thrust_dir_x, thrust_dir_y, thrust_dir_z,
    ])

    obs.append(1.0)

    return np.asarray(obs, dtype=float)


def observables(x_std, scaler):
    """
    Return the lifted observable vector for a standardized state.

    The observable definition must match the lifting used during training.
    A 10-state branch is kept only so old 10-state model files can still load.
    """
    expected_dim = _scaler_state_dim(scaler)
    if expected_dim == 10:
        return _observables_10state(x_std, scaler)
    if expected_dim == STATE_DIM:
        return _observables_12state(x_std, scaler)
    raise ValueError(f"Unsupported scaler state dimension: {expected_dim}")


def drop_to_10state(x12):
    """Convert a 12-state vector to the legacy reduced 10-state representation."""
    x = np.asarray(x12, dtype=float).flatten()
    if len(x) == 10:
        return x.copy()
    if len(x) < STATE_DIM:
        raise ValueError(f"Expected at least 12 entries, got {len(x)}")
    return x[REDUCED_10_INDICES].copy()


def drop_to_12state(x_state):
    """Convert a plant/logged state to the full EDMD 12-state representation."""
    x = np.asarray(x_state, dtype=float).flatten()
    if len(x) >= STATE_DIM:
        return x[:STATE_DIM].copy()
    if len(x) == 10:
        x12 = np.zeros(STATE_DIM)
        x12[0:6] = x[0:6]
        x12[6:8] = x[6:8]
        x12[9:11] = x[8:10]
        return x12
    raise ValueError(f"Expected 10-state or 12-state vector, got {len(x)}")


def lifted_state_from_x(x_state, scaler):
    """Map a physical state vector to the lifted observable space."""
    expected_dim = _scaler_state_dim(scaler)
    if expected_dim == 10:
        x_phys = drop_to_10state(x_state)
    elif expected_dim == STATE_DIM:
        x_phys = drop_to_12state(x_state)
    else:
        raise ValueError(f"Unsupported scaler state dimension: {expected_dim}")

    x_std = scaler.transform(x_phys.reshape(1, -1)).flatten()
    return observables(x_std, scaler)


# MPC solver
class EDMDcMPC_QP:
    """
    Quadratic-program MPC controller built on a lifted linear EDMDc model.

    The optimizer penalizes tracking error in the physical-state coordinates
    selected by Cz while optimizing control increments over the control horizon.
    """
    def __init__(self, A, B, Cz, N, NC, Q, R, Rd,
                 u_scaler, du_min, du_max, u_nominal_raw,
                 state_scaler=None, input_lift_type=None, raw_input_dim=None,
                 Q_terminal=None, allocation_matrix=None,
                 motor_force_min=None, motor_force_max=None):
        self.A  = np.asarray(A, dtype=float)
        self.B_model = np.asarray(B, dtype=float)
        self.Cz = np.asarray(Cz, dtype=float)
        self.N  = int(N)
        self.NC = int(NC)
        self.Q  = np.asarray(Q,  dtype=float)
        self.Q_terminal = (
            np.asarray(Q_terminal, dtype=float)
            if Q_terminal is not None else self.Q
        )
        self.R  = np.asarray(R,  dtype=float)
        self.Rd = np.asarray(Rd, dtype=float)

        self.u_scaler = u_scaler
        self.state_scaler = state_scaler
        self.input_lift_type = input_lift_type
        self.raw_input_dim = raw_input_dim

        self.du_min = np.asarray(du_min, dtype=float)
        self.du_max = np.asarray(du_max, dtype=float)

        self.nz   = self.A.shape[0]   # observable dimension
        self.model_nu = self.B_model.shape[1]
        nominal_input_size = np.asarray(u_nominal_raw, dtype=float).size
        self.uses_lifted_input = (
            self.input_lift_type in (INPUT_LIFT_TYPE, LEGACY_INPUT_LIFT_TYPE)
            or (self.model_nu in (len(INPUT_LIFT_LABELS), len(LEGACY_INPUT_LIFT_LABELS))
                and getattr(self.u_scaler, "n_features_in_", self.model_nu) == self.model_nu
                and (raw_input_dim == RAW_INPUT_DIM or nominal_input_size == RAW_INPUT_DIM))
        )
        if self.uses_lifted_input:
            if self.state_scaler is None:
                raise ValueError("state_scaler is required for thrust-direction input lifting")
            self.raw_input_dim = RAW_INPUT_DIM if self.raw_input_dim is None else int(self.raw_input_dim)
            self.nu = self.raw_input_dim
            self._lift_state_phys = np.zeros(STATE_DIM)
        else:
            self.nu = self.model_nu
        self.B = self.B_model[:, :self.nu] if self.uses_lifted_input else self.B_model

        self.nx   = self.Cz.shape[0]  # tracked physical-state dimension
        self.nvar = self.NC * self.nu
        self._p_rows, self._p_cols = np.triu_indices(self.nvar)
        self.A_pow = [np.eye(self.nz)]
        for _ in range(self.N):
            self.A_pow.append(self.A_pow[-1] @ self.A)

        if self.du_min.size != self.nu or self.du_max.size != self.nu:
            raise ValueError(
                f"du_min/du_max must have length {self.nu}, got "
                f"{self.du_min.size}/{self.du_max.size}"
            )

        self.allocation_matrix = None
        self.motor_force_min = None
        self.motor_force_max = None
        if allocation_matrix is not None or motor_force_max is not None:
            if allocation_matrix is None or motor_force_max is None:
                raise ValueError(
                    "allocation_matrix and motor_force_max must be supplied together"
                )
            if self.nu != RAW_INPUT_DIM:
                raise ValueError("Motor allocation constraints require four raw wrench inputs")
            allocation = np.asarray(allocation_matrix, dtype=float)
            force_max = np.asarray(motor_force_max, dtype=float).reshape(-1)
            force_min = (
                np.zeros_like(force_max) if motor_force_min is None
                else np.asarray(motor_force_min, dtype=float).reshape(-1)
            )
            if allocation.shape != (4, RAW_INPUT_DIM):
                raise ValueError("allocation_matrix must have shape (4, 4)")
            if force_min.shape != (4,) or force_max.shape != (4,):
                raise ValueError("Motor force bounds must each have length four")
            if np.any(force_min > force_max):
                raise ValueError("motor_force_min cannot exceed motor_force_max")
            self.allocation_matrix = allocation
            self.motor_force_min = force_min
            self.motor_force_max = force_max

        self._set_nominal_input(u_nominal_raw)
        self._du_prev = np.zeros(self.nvar)

        q_blocks = [sp.csc_matrix(self.Q) for _ in range(max(self.N - 1, 0))]
        q_blocks.append(sp.csc_matrix(self.Q_terminal))
        self.Qbar = sp.block_diag(q_blocks, format="csc").toarray()
        self.Rbar = sp.block_diag(
            [sp.csc_matrix(self.R) for _ in range(self.NC)], format="csc").toarray()
        self.D = self._build_difference_matrix()
        self.Rdbar = (
            sp.block_diag([sp.csc_matrix(self.Rd)
                           for _ in range(self.NC - 1)], format="csc").toarray()
            if self.NC > 1 else None
        )

        self._build_input_constraints()

        self.Sz, self.Su_model = self._build_prediction_matrices()
        Su_model_blocks = self.Su_model.reshape(
            self.N, self.nz, self.NC, self.model_nu
        )
        self.Su_model_phys = np.einsum(
            "ab,ibcd->iacd", self.Cz, Su_model_blocks
        ).reshape(self.N * self.nx, self.NC * self.model_nu)
        self._refresh_prediction_model()
        if osqp is None:
            raise ModuleNotFoundError(
                "osqp is required for EDMDcMPC_QP closed-loop optimization. "
                "Install osqp or use compare_three.py for rollout comparison."
            )
        self.prob = osqp.OSQP()
        self.prob.setup(P=self.P, q=np.zeros(self.nvar),
                        A=self.Aineq, l=self.l, u=self.u_bound,
                        warm_start=True, verbose=False, polish=False)

    def _set_nominal_input(self, u_nominal_raw):
        self.u_nom_raw = np.asarray(u_nominal_raw, dtype=float).flatten()
        if self.u_nom_raw.size < self.nu:
            raise ValueError(f"Expected at least {self.nu} nominal inputs, got {self.u_nom_raw.size}")
        self.u_nom_raw = self.u_nom_raw[:self.nu]

        if self.uses_lifted_input:
            self.u_nom_model_scaled = scaled_lifted_input_from_phys(
                self._lift_state_phys, self.u_nom_raw, self.u_scaler
            )
            self.u_nom_scaled = (
                (self.u_nom_raw - self.u_scaler.mean_[:self.nu])
                / self.u_scaler.scale_[:self.nu]
            )
        else:
            self.u_nom_scaled = self.u_scaler.transform(
                self.u_nom_raw.reshape(1, -1)).flatten()
            self.u_nom_model_scaled = self.u_nom_scaled

        if hasattr(self, "NC"):
            self.u_nom_horizon_scaled = np.tile(self.u_nom_scaled, self.NC)
            self.u_nom_model_horizon_scaled = np.tile(self.u_nom_model_scaled, self.NC)
        if hasattr(self, "Aineq"):
            self._update_input_constraint_bounds()

    def _build_input_constraints(self):
        """Build delta and per-motor feasibility constraints for every move."""
        self._delta_l = np.tile(self.du_min, self.NC)
        self._delta_u = np.tile(self.du_max, self.NC)
        identity = sp.eye(self.nvar, format="csc")

        if self.allocation_matrix is None:
            self.Aineq = identity
            self.l = self._delta_l.copy()
            self.u_bound = self._delta_u.copy()
            return

        # QP variables are standardized command deltas. Map those deltas to
        # physical motor thrust changes, while the bounds carry the current
        # feasible nominal wrench.
        raw_delta_scale = np.diag(self.u_scaler.scale_[:self.nu])
        motor_delta = self.allocation_matrix @ raw_delta_scale
        self._motor_delta_matrix = sp.kron(
            sp.eye(self.NC, format="csc"), sp.csc_matrix(motor_delta),
            format="csc",
        )
        self.Aineq = sp.vstack([identity, self._motor_delta_matrix], format="csc")
        self._update_input_constraint_bounds()

    def _update_input_constraint_bounds(self):
        if self.allocation_matrix is None:
            self.l = self._delta_l.copy()
            self.u_bound = self._delta_u.copy()
            return
        nominal_motor_forces = self.allocation_matrix @ self.u_nom_raw
        motor_l = np.tile(self.motor_force_min - nominal_motor_forces, self.NC)
        motor_u = np.tile(self.motor_force_max - nominal_motor_forces, self.NC)
        self.l = np.concatenate([self._delta_l, motor_l])
        self.u_bound = np.concatenate([self._delta_u, motor_u])

    def _input_lift_jacobian(self, state_phys=None):
        """
        Map raw standardized command deltas to lifted standardized input deltas.

        The learned model input starts with raw commands, then may include
        thrust-direction and torque-rate coupling channels. The QP still
        optimizes the 4 real commands, so lifted columns need local
        state-dependent sensitivities.
        """
        if not self.uses_lifted_input:
            return np.eye(self.model_nu)

        J = np.zeros((self.model_nu, self.nu))
        J[:self.nu, :self.nu] = np.eye(self.nu)

        state_phys = (
            self._lift_state_phys if state_phys is None
            else np.asarray(state_phys, dtype=float)
        )
        thrust_dir = thrust_direction_from_state_phys(state_phys)
        thrust_scale = self.u_scaler.scale_[0]
        for axis in range(3):
            lifted_idx = RAW_INPUT_DIM + axis
            if lifted_idx >= self.model_nu:
                continue
            J[lifted_idx, 0] = (
                thrust_scale * thrust_dir[axis] / self.u_scaler.scale_[lifted_idx]
            )
        rate_start = RAW_INPUT_DIM + 3
        rates = state_phys[9:12]
        for axis in range(3):
            lifted_idx = rate_start + axis
            raw_idx = 1 + axis
            if lifted_idx >= self.model_nu or raw_idx >= self.nu:
                continue
            J[lifted_idx, raw_idx] = (
                self.u_scaler.scale_[raw_idx] * rates[axis]
                / self.u_scaler.scale_[lifted_idx]
            )
        return J

    def _refresh_prediction_model(self):
        if self.uses_lifted_input:
            input_jacobian = self._input_lift_jacobian()
        else:
            input_jacobian = np.eye(self.model_nu)

        self.B = self.B_model @ input_jacobian
        input_jacobian_horizon = np.kron(np.eye(self.NC), input_jacobian)
        self.Su_phys = self.Su_model_phys @ input_jacobian_horizon
        self.P = self._as_osqp_upper(self._build_hessian())

    def _as_osqp_upper(self, matrix):
        """Store a full upper-triangular P pattern so OSQP can update it in place."""
        dense = np.asarray(matrix, dtype=float)
        values = dense[self._p_rows, self._p_cols]
        return sp.csc_matrix((values, (self._p_rows, self._p_cols)),
                             shape=(self.nvar, self.nvar))

    def _set_lift_state_from_z(self, z0):
        if not self.uses_lifted_input:
            return
        z0 = np.asarray(z0, dtype=float).flatten()
        x_std = z0[:STATE_DIM]
        self._lift_state_phys = self.state_scaler.inverse_transform(
            x_std.reshape(1, -1)
        ).flatten()
        self._set_nominal_input(self.u_nom_raw)

    def _build_prediction_matrices(self):
        Sz = np.zeros((self.N * self.nz, self.nz))
        for i in range(self.N):
            Sz[i*self.nz:(i+1)*self.nz, :] = self.A_pow[i+1]

        # The full lifted-input response is time invariant. Online updates only
        # change the local map from the real wrench to that lifted input.
        B_model_impulses = np.stack(
            [A_power @ self.B_model for A_power in self.A_pow[:-1]], axis=0
        )

        def build_toeplitz_response(impulses, input_dim):
            blocks = np.zeros((self.N, self.NC, self.nz, input_dim))
            for control_step in range(self.NC - 1):
                blocks[control_step:, control_step] = impulses[:self.N - control_step]

            # The final decision is held over the remainder of the prediction
            # horizon, so it accumulates every later impulse response.
            if self.NC <= self.N:
                cumulative = np.cumsum(impulses, axis=0)
                for prediction_step in range(self.NC - 1, self.N):
                    blocks[prediction_step, self.NC - 1] = cumulative[
                        prediction_step - self.NC + 1
                    ]
            return blocks.transpose(0, 2, 1, 3).reshape(
                self.N * self.nz, self.NC * input_dim
            )

        Su_model = build_toeplitz_response(B_model_impulses, self.model_nu)
        return Sz, Su_model

    def _build_difference_matrix(self):
        if self.NC <= 1:
            return None
        rows, cols, vals = [], [], []
        for k in range(self.NC - 1):
            for j in range(self.nu):
                r = k * self.nu + j
                rows.extend([r, r])
                cols.extend([k*self.nu+j, (k+1)*self.nu+j])
                vals.extend([-1.0, 1.0])
        return sp.coo_matrix(
            (vals, (rows, cols)),
            shape=((self.NC-1)*self.nu, self.NC*self.nu)).toarray()

    def _build_hessian(self):
        P = self.Su_phys.T @ self.Qbar @ self.Su_phys + self.Rbar
        if self.D is not None and self.Rdbar is not None:
            P = P + self.D.T @ self.Rdbar @ self.D
        return 0.5 * (P + P.T)

    def _build_q(self, z0, x_ref_std_horizon):
        # The learned EDMD model uses absolute standardized inputs. The QP
        # variables are deltas around the nominal input, so the nominal input
        # response must be part of the free trajectory.
        z_free = self.Sz @ z0 + self.Su_model @ self.u_nom_model_horizon_scaled
        x_free = (self.Cz @ z_free.reshape(self.N, self.nz).T).T.reshape(-1)
        x_ref = x_ref_std_horizon.reshape(-1)
        return np.asarray(
            self.Su_phys.T @ (self.Qbar @ (x_free - x_ref))
        ).reshape(-1)

    def compute(
        self, z0, x_ref_std_horizon, u_nominal_raw=None,
        u_nominal_raw_horizon=None,
    ):
        self._set_lift_state_from_z(z0)
        if u_nominal_raw_horizon is not None:
            nominal = np.asarray(u_nominal_raw_horizon, dtype=float)
            self._set_nominal_input(nominal[0] if nominal.ndim == 2 else nominal)
        elif u_nominal_raw is not None:
            self._set_nominal_input(u_nominal_raw)

        # The EDMDc model is re-linearized at the measured state every control
        # tick. Keeping that local input map fixed within one QP is more robust
        # than extrapolating the bilinear input lift far from logged data.
        if self.uses_lifted_input:
            self._refresh_prediction_model()
            self.prob.update(Px=self.P.data)

        q = self._build_q(z0, x_ref_std_horizon)
        self.prob.update(q=q, l=self.l, u=self.u_bound)
        self.prob.warm_start(x=self._du_prev)
        res = self.prob.solve()

        if res.info.status not in ("solved", "solved inaccurate"):
            print(f"Warning OSQP: {res.info.status}")
            du0 = self._du_prev[:self.nu]
        else:
            du_opt = np.asarray(res.x).reshape(-1)
            self._du_prev = du_opt.copy()
            du0 = du_opt[:self.nu]

        u0_scaled = self.u_nom_scaled + du0
        if self.uses_lifted_input:
            u0_raw = u0_scaled * self.u_scaler.scale_[:self.nu] + self.u_scaler.mean_[:self.nu]
        else:
            u0_raw = self.u_scaler.inverse_transform(
                u0_scaled.reshape(1, -1)).flatten()
        return u0_raw  # [thrust, tau_roll, tau_pitch, optional tau_yaw]


class OuterCommandEDMDcMPC_SQP:
    """Sequentially linearized MPC for an outer desired-attitude EDMDc model.

    The nonlinear 16-feature input lift is evaluated along a nominal predicted
    trajectory. Each SQP iteration includes both its state and command
    Jacobians, while the QP retains only four physical command decisions.
    """

    def __init__(
        self, A, B, Cz, N, NC, Q, R, Rd, state_scaler, u_scaler,
        u_min_raw, u_max_raw, du_max_raw, max_iterations=2,
        Q_terminal=None, trust_region_scale=1.0, move_block_steps=1,
        attitude_error_max_raw=None, first_move_rd_scale=1.0,
        command_slew_max_raw=None,
    ):
        if osqp is None:
            raise ModuleNotFoundError("osqp is required for outer-command MPC")
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.Cz = np.asarray(Cz, dtype=float)
        self.N = int(N)
        self.NC = int(NC)
        self.Q = np.asarray(Q, dtype=float)
        self.Q_terminal = (
            self.Q if Q_terminal is None else np.asarray(Q_terminal, dtype=float)
        )
        self.R = np.asarray(R, dtype=float)
        self.Rd = np.asarray(Rd, dtype=float)
        self.state_scaler = state_scaler
        self.u_scaler = u_scaler
        self.nz = self.A.shape[0]
        self.nx = self.Cz.shape[0]
        self.nu = RAW_INPUT_DIM
        self.nvar = self.NC * self.nu
        self.max_iterations = int(max_iterations)
        self.trust_region_scale = float(trust_region_scale)
        self.move_block_steps = int(move_block_steps)
        self.attitude_error_max_raw = (
            None if attitude_error_max_raw is None
            else np.broadcast_to(
                np.asarray(attitude_error_max_raw, dtype=float), (2,)
            ).copy()
        )
        self.first_move_rd_scale = float(first_move_rd_scale)
        if self.first_move_rd_scale < 0.0:
            raise ValueError("first_move_rd_scale must be nonnegative")
        if self.B.shape[1] != len(OUTER_ATTITUDE_INPUT_LIFT_LABELS):
            raise ValueError("Outer-command SQP requires the 16-feature input lift")
        if self.Cz.shape != (STATE_DIM, self.nz):
            raise ValueError("Cz must decode the 12 standardized physical states")
        if not 1 <= self.NC <= self.N or self.max_iterations < 1:
            raise ValueError("Require 1 <= NC <= N and at least one SQP iteration")
        if self.move_block_steps < 1:
            raise ValueError("move_block_steps must be positive")

        self.raw_mean = np.asarray(u_scaler.mean_[:self.nu], dtype=float)
        self.raw_scale = np.asarray(u_scaler.scale_[:self.nu], dtype=float)
        self.state_mean = np.asarray(
            state_scaler.mean_[:STATE_DIM], dtype=float
        )
        self.state_scale = np.asarray(
            state_scaler.scale_[:STATE_DIM], dtype=float
        )
        self.lift_mean = np.asarray(u_scaler.mean_, dtype=float)
        self.lift_scale = np.asarray(u_scaler.scale_, dtype=float)
        self.u_min_std = (
            np.asarray(u_min_raw, dtype=float) - self.raw_mean
        ) / self.raw_scale
        self.u_max_std = (
            np.asarray(u_max_raw, dtype=float) - self.raw_mean
        ) / self.raw_scale
        self.du_max_std = (
            np.asarray(du_max_raw, dtype=float) / self.raw_scale
        ) * self.trust_region_scale
        self.command_slew_max_std = (
            None if command_slew_max_raw is None else
            np.asarray(command_slew_max_raw, dtype=float) / self.raw_scale
        )
        if any(values.shape != (self.nu,) for values in (
            self.u_min_std, self.u_max_std, self.du_max_std
        )):
            raise ValueError("Outer-command bounds must contain four entries")

        q_blocks = [sp.csc_matrix(self.Q) for _ in range(max(self.N - 1, 0))]
        q_blocks.append(sp.csc_matrix(self.Q_terminal))
        self.Qbar = sp.block_diag(q_blocks, format="csc").toarray()
        self.Rbar = sp.block_diag(
            [sp.csc_matrix(self.R) for _ in range(self.NC)], format="csc"
        ).toarray()
        self.D = self._difference_matrix()
        self.Rdbar = (
            sp.block_diag(
                [sp.csc_matrix(self.Rd) for _ in range(self.NC - 1)],
                format="csc",
            ).toarray()
            if self.NC > 1 else None
        )
        self._p_rows, self._p_cols = np.triu_indices(self.nvar)
        initial_p = self._upper_matrix(np.eye(self.nvar))
        self._identity_constraints = sp.eye(self.nvar, format="csc")
        self.prob = osqp.OSQP()
        self.prob.setup(
            P=initial_p,
            q=np.zeros(self.nvar),
            A=self._identity_constraints,
            l=-np.ones(self.nvar),
            u=np.ones(self.nvar),
            warm_start=True,
            verbose=False,
            polish=False,
        )
        self.last_prediction = None
        self.last_commands_raw = None
        self.last_status = "not run"
        self.last_iterations = 0
        self.last_delta_norm = np.nan
        self.previous_command_raw = None
        # Optional online additive model-defect estimate. The default zero
        # preserves all previously frozen outer-command MPC behavior.
        self.additive_defect_z = np.zeros(self.nz, dtype=float)

    def _difference_matrix(self):
        if self.NC <= 1:
            return None
        D = np.zeros(((self.NC - 1) * self.nu, self.nvar), dtype=float)
        for k in range(self.NC - 1):
            row = slice(k * self.nu, (k + 1) * self.nu)
            D[row, k * self.nu:(k + 1) * self.nu] = -np.eye(self.nu)
            D[row, (k + 1) * self.nu:(k + 2) * self.nu] = np.eye(self.nu)
        return D

    def _upper_matrix(self, matrix):
        values = np.asarray(matrix, dtype=float)[self._p_rows, self._p_cols]
        return sp.csc_matrix(
            (values, (self._p_rows, self._p_cols)),
            shape=(self.nvar, self.nvar),
        )

    def _decode_state(self, z):
        x_std = self.Cz @ np.asarray(z, dtype=float)
        return x_std * self.state_scale + self.state_mean

    def _scaled_lift_fast(self, state, raw):
        lifted = outer_command_lift_from_phys(
            state, raw, input_lift_type=OUTER_ATTITUDE_INPUT_LIFT_TYPE
        )
        return (np.asarray(lifted, dtype=float) - self.lift_mean) / self.lift_scale

    def _raw_from_std(self, command_std):
        return np.asarray(command_std, dtype=float) * self.raw_scale + self.raw_mean

    def _std_from_raw(self, command_raw):
        return (np.asarray(command_raw, dtype=float) - self.raw_mean) / self.raw_scale

    def _prepare_reference_commands(self, commands_raw, state_phys):
        commands = np.asarray(commands_raw, dtype=float)
        if commands.ndim == 1:
            commands = np.repeat(commands.reshape(1, -1), self.NC, axis=0)
        if commands.shape[1] != self.nu:
            raise ValueError("Reference outer commands must have four columns")
        if len(commands) >= self.N:
            commands = commands[
                np.minimum(
                    np.arange(self.NC) * self.move_block_steps,
                    len(commands) - 1,
                )
            ]
        if len(commands) < self.NC:
            commands = np.vstack([
                commands,
                np.repeat(commands[-1:], self.NC - len(commands), axis=0),
            ])
        commands = commands[:self.NC].copy()
        commands[:, 3] = np.unwrap(
            np.concatenate([[state_phys[8]], commands[:, 3]])
        )[1:]
        return self._std_from_raw(commands)

    def _nominal_rollout_and_sensitivity(self, z0, command_std):
        z = np.asarray(z0, dtype=float).reshape(self.nz).copy()
        sensitivity = np.zeros((self.nz, self.nvar), dtype=float)
        predicted = np.zeros((self.N + 1, self.nz), dtype=float)
        output_sensitivity = np.zeros((self.N * self.nx, self.nvar), dtype=float)
        predicted[0] = z

        for k in range(self.N):
            move = min(k // self.move_block_steps, self.NC - 1)
            raw = self._raw_from_std(command_std[move])
            state = self._decode_state(z)
            lifted_scaled = self._scaled_lift_fast(state, raw)
            jac_x, jac_u = outer_command_lift_jacobians_scaled(
                state, raw, self.state_scaler, self.u_scaler
            )
            local_A = self.A + self.B @ jac_x @ self.Cz
            injection = np.zeros((self.B.shape[1], self.nvar), dtype=float)
            columns = slice(move * self.nu, (move + 1) * self.nu)
            injection[:, columns] = jac_u
            sensitivity = local_A @ sensitivity + self.B @ injection
            z = self.A @ z + self.B @ lifted_scaled + self.additive_defect_z
            predicted[k + 1] = z
            output_sensitivity[k*self.nx:(k+1)*self.nx] = self.Cz @ sensitivity
        return predicted, output_sensitivity

    def _nominal_rollout(self, z0, command_std):
        """Roll out the nonlinear lifted model without QP sensitivities."""
        z = np.asarray(z0, dtype=float).reshape(self.nz).copy()
        predicted = np.zeros((self.N + 1, self.nz), dtype=float)
        predicted[0] = z
        for k in range(self.N):
            move = min(k // self.move_block_steps, self.NC - 1)
            raw = self._raw_from_std(command_std[move])
            state = self._decode_state(z)
            lifted_scaled = self._scaled_lift_fast(state, raw)
            z = self.A @ z + self.B @ lifted_scaled + self.additive_defect_z
            predicted[k + 1] = z
        return predicted

    def compute(self, z0, x_ref_std_horizon, u_reference_raw_horizon,
                previous_command_raw=None):
        z0 = np.asarray(z0, dtype=float).reshape(self.nz)
        reference = np.asarray(x_ref_std_horizon, dtype=float)
        if reference.shape != (self.N, self.nx):
            raise ValueError(
                f"Reference state horizon must be {(self.N, self.nx)}, "
                f"got {reference.shape}"
            )
        state_phys = self._decode_state(z0)
        command_reference = self._prepare_reference_commands(
            u_reference_raw_horizon, state_phys
        )
        if previous_command_raw is None:
            previous_command_raw = self.previous_command_raw
        if previous_command_raw is None:
            previous_command_std = command_reference[0].copy()
        else:
            previous_command_raw = np.asarray(
                previous_command_raw, dtype=float
            ).reshape(self.nu).copy()
            # Desired yaw is periodic. Put the previous command on the same
            # branch as the current state before forming a command increment.
            previous_command_raw[3] = (
                state_phys[8]
                + wrap_angle_pi(previous_command_raw[3] - state_phys[8])
            )
            previous_command_std = self._std_from_raw(previous_command_raw)
        command_nominal = command_reference.copy()
        status = "not solved"
        delta_norm = np.inf

        for iteration in range(self.max_iterations):
            predicted, Su = self._nominal_rollout_and_sensitivity(
                z0, command_nominal
            )
            predicted_states = (self.Cz @ predicted[1:].T).T.reshape(-1)
            error = predicted_states - reference.reshape(-1)
            command_offset = (command_nominal - command_reference).reshape(-1)
            P = Su.T @ self.Qbar @ Su + self.Rbar
            q = Su.T @ (self.Qbar @ error) + self.Rbar @ command_offset
            if self.D is not None and self.Rdbar is not None:
                smooth = self.D @ command_nominal.reshape(-1)
                P += self.D.T @ self.Rdbar @ self.D
                q += self.D.T @ (self.Rdbar @ smooth)
            # Penalize the first optimized move against the command actually
            # applied at the preceding controller update. Without this term,
            # receding-horizon replanning can alternate its first action even
            # when every individual predicted sequence is internally smooth.
            if self.first_move_rd_scale > 0.0:
                first_error = command_nominal[0] - previous_command_std
                first_rd = self.first_move_rd_scale * self.Rd
                P[:self.nu, :self.nu] += first_rd
                q[:self.nu] += first_rd @ first_error
            P = 0.5 * (P + P.T) + 1e-9 * np.eye(self.nvar)

            lower_absolute = np.tile(self.u_min_std, self.NC) - command_nominal.reshape(-1)
            upper_absolute = np.tile(self.u_max_std, self.NC) - command_nominal.reshape(-1)
            trust = np.tile(self.du_max_std, self.NC)
            lower = np.maximum(lower_absolute, -trust)
            upper = np.minimum(upper_absolute, trust)
            if self.command_slew_max_std is not None:
                lower[:self.nu] = np.maximum(
                    lower[:self.nu],
                    previous_command_std - self.command_slew_max_std
                    - command_nominal[0],
                )
                upper[:self.nu] = np.minimum(
                    upper[:self.nu],
                    previous_command_std + self.command_slew_max_std
                    - command_nominal[0],
                )
            if self.attitude_error_max_raw is not None:
                # Keep desired roll/pitch within the input-lift domain observed
                # during identification.  Each control move is constrained
                # around the attitude of its nominal predicted state.
                for move in range(self.NC):
                    prediction_index = min(
                        move * self.move_block_steps + 1, self.N
                    )
                    predicted_state = self._decode_state(
                        predicted[prediction_index]
                    )
                    for local, raw_index in enumerate((1, 2)):
                        column = move * self.nu + raw_index
                        lower_raw = (
                            predicted_state[6 + local]
                            - self.attitude_error_max_raw[local]
                        )
                        upper_raw = (
                            predicted_state[6 + local]
                            + self.attitude_error_max_raw[local]
                        )
                        lower_std = (
                            lower_raw - self.raw_mean[raw_index]
                        ) / self.raw_scale[raw_index]
                        upper_std = (
                            upper_raw - self.raw_mean[raw_index]
                        ) / self.raw_scale[raw_index]
                        lower[column] = max(
                            lower[column],
                            lower_std - command_nominal[move, raw_index],
                        )
                        upper[column] = min(
                            upper[column],
                            upper_std - command_nominal[move, raw_index],
                        )
            if np.any(lower > upper + 1e-12):
                status = "infeasible command increment bounds"
                break
            P_upper = self._upper_matrix(P)
            self.prob.update(Px=P_upper.data, q=q, l=lower, u=upper)
            self.prob.warm_start(x=np.zeros(self.nvar))
            result = self.prob.solve()
            status = str(result.info.status)
            if status not in ("solved", "solved inaccurate"):
                break
            delta = np.asarray(result.x, dtype=float).reshape(self.NC, self.nu)
            command_nominal += delta
            delta_norm = float(np.max(np.abs(delta)))
            if delta_norm < 1e-4:
                iteration += 1
                break

        commands_raw = self._raw_from_std(command_nominal)
        # The final nonlinear rollout was previously recomputed only for this
        # diagnostic field. Preserve the final SQP linearization trajectory;
        # the applied command and optimization result are unchanged.
        self.last_prediction = predicted
        self.last_commands_raw = commands_raw
        self.last_status = status
        self.last_iterations = iteration + 1
        self.last_delta_norm = delta_norm
        self.previous_command_raw = commands_raw[0].copy()
        return commands_raw[0].copy()


# Reference processing
def reference_yaw_arrays(ref_traj, dt=None):
    """Return unwrapped yaw and yaw-rate references for a trajectory list."""
    yaw = np.unwrap([
        float(wp.get("yaw", 0.0)) for wp in ref_traj
    ])

    yaw_rate = np.zeros_like(yaw)
    explicit = [
        float(wp["yaw_rate"]) if "yaw_rate" in wp else np.nan
        for wp in ref_traj
    ]
    explicit = np.asarray(explicit, dtype=float)
    has_explicit = np.isfinite(explicit)
    if np.any(has_explicit):
        yaw_rate[has_explicit] = explicit[has_explicit]

    if np.any(~has_explicit) and dt is not None and len(yaw) > 1:
        computed = np.gradient(yaw, float(dt))
        yaw_rate[~has_explicit] = computed[~has_explicit]

    return yaw, yaw_rate


def extract_ref_xyz(ref_traj):
    return np.array([wp["pos"][:3] for wp in ref_traj], dtype=float)


def outer_command_reference_array(ref_traj, quad):
    """Build feedforward [thrust, roll_des, pitch_des, yaw_des] commands."""
    if not ref_traj:
        return np.empty((0, RAW_INPUT_DIM), dtype=float)
    yaw, _ = reference_yaw_arrays(ref_traj)
    commands = np.zeros((len(ref_traj), RAW_INPUT_DIM), dtype=float)
    for k, waypoint in enumerate(ref_traj):
        velocity = np.asarray(waypoint.get("vel", np.zeros(3)), dtype=float)
        acceleration = np.asarray(waypoint.get("acc", np.zeros(3)), dtype=float)
        force_world = (
            float(quad.m) * (
                acceleration + np.array([0.0, 0.0, float(quad.g)])
            )
            + float(quad.k_drag_linear) * velocity
        )
        rotation = helperfcts.fct_desired_rotation_from_force_and_yaw(
            force_world, yaw[k]
        )
        phi, theta, _ = helperfcts.fct_euler_from_R(rotation)
        commands[k] = [np.linalg.norm(force_world), phi, theta, yaw[k]]
    return commands


def hover_yaw_command_reference_array(ref_traj, hover_thrust_n):
    """Return information-minimal [hover thrust, 0, 0, yaw] commands."""
    if not ref_traj:
        return np.empty((0, RAW_INPUT_DIM), dtype=float)
    yaw, _ = reference_yaw_arrays(ref_traj)
    commands = np.zeros((len(ref_traj), RAW_INPUT_DIM), dtype=float)
    commands[:, 0] = float(hover_thrust_n)
    commands[:, 3] = yaw
    return commands


def reference_state_array(ref_traj, dt, quad=None):
    """Build a dynamically consistent 12-state reference trajectory.

    When a plant is supplied, desired roll and pitch follow from the force
    required by the reference acceleration. Body rates are then derived from
    that attitude history. Without a plant, the unavailable attitude channels
    remain zero for backward compatibility.
    """
    if dt is None or dt <= 0.0:
        raise ValueError("A positive reference sample time is required")
    if not ref_traj:
        return np.empty((0, STATE_DIM), dtype=float)

    X_ref = np.zeros((len(ref_traj), STATE_DIM), dtype=float)
    X_ref[:, 0:3] = np.asarray([wp["pos"][:3] for wp in ref_traj], dtype=float)
    X_ref[:, 3:6] = np.asarray([
        wp.get("vel", np.zeros(3))[:3] for wp in ref_traj
    ], dtype=float)
    yaw, yaw_rate = reference_yaw_arrays(ref_traj, dt=dt)
    X_ref[:, 8] = yaw
    X_ref[:, 11] = yaw_rate

    if quad is None:
        return X_ref

    for k, wp in enumerate(ref_traj):
        acc = np.asarray(wp.get("acc", np.zeros(3)), dtype=float)
        force_world = (
            float(quad.m) * (acc + np.array([0.0, 0.0, float(quad.g)]))
            + float(quad.k_drag_linear) * X_ref[k, 3:6]
        )
        R_des = helperfcts.fct_desired_rotation_from_force_and_yaw(
            force_world, yaw[k]
        )
        phi, theta, _ = helperfcts.fct_euler_from_R(R_des)
        X_ref[k, 6] = phi
        X_ref[k, 7] = theta

    angles = X_ref[:, 6:9].copy()
    angles[:, 2] = np.unwrap(angles[:, 2])
    if len(ref_traj) == 1:
        euler_dot = np.zeros_like(angles)
    else:
        edge_order = 2 if len(ref_traj) >= 3 else 1
        euler_dot = np.gradient(angles, dt, axis=0, edge_order=edge_order)
    for k, (phi, theta, _) in enumerate(angles):
        X_ref[k, 9:12] = np.linalg.solve(
            quad.fct_W_matrix(phi, theta), euler_dot[k]
        )
    # Retain explicitly capped yaw-rate commands instead of replacing them
    # with numerical differentiation noise.
    X_ref[:, 11] = yaw_rate
    return X_ref


def precompute_ref_std(ref_traj, scaler, n_states=None, dt=None, quad=None):
    """Build the standardized physical-state reference used by MPC."""
    T = len(ref_traj)
    expected_dim = _scaler_state_dim(scaler)
    if n_states is None or n_states != expected_dim:
        n_states = expected_dim

    if n_states >= STATE_DIM:
        X_ref = reference_state_array(ref_traj, dt=dt, quad=quad)
    else:
        X_ref = np.zeros((T, n_states))
        for k in range(T):
            X_ref[k, 0:3] = ref_traj[k]["pos"][:3]
            X_ref[k, 3:6] = ref_traj[k].get("vel", np.zeros(3))[:3]
    return scaler.transform(X_ref)


def build_ref_horizon(ref_std, k, N):
    T = ref_std.shape[0]
    h = np.zeros((N, ref_std.shape[1]))
    for i in range(N):
        h[i] = ref_std[min(k + i, T - 1)]
    return h

# Metrics
def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))
