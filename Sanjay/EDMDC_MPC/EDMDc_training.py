import os

import numpy as np
import pickle
from scipy.linalg import pinv
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt

from edmdc_mpc import (
    OUTER_ATTITUDE_INPUT_LIFT_LABELS,
    OUTER_ATTITUDE_INPUT_LIFT_TYPE,
    OUTER_RAW_INPUT_LABELS,
    OUTER_RAW_INPUT_LIFT_TYPE,
    outer_command_lift_from_phys,
)



# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
dt = float(os.environ.get("EDMDC_DT", "0.1"))
SHORT_HORIZON_SECONDS = float(os.environ.get("EDMDC_SHORT_HORIZON_SECONDS", "2.0"))
MPC_HORIZON = int(round(SHORT_HORIZON_SECONDS / dt))
ROLLING_WINDOW_STRIDE_SECONDS = 0.1
SWEEP_ROLLING_WINDOW_STRIDE_SECONDS = 1.0
STATE_DIM = 12
STATE_LABELS = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']
FULL_OBSERVABLE_LABELS = STATE_LABELS + [
    'sin_phi','cos_phi','sin_theta','cos_theta','sin_psi','cos_psi',
    'phi*p','theta*q','psi*r','vx*phi','vy*theta','vx*psi','vy*psi','vz*theta',
    'v_sq','omega_sq','vz^2','phi^2','theta^2','psi^2','p*q','q*r','p*r',
    'x*y','x*vx','y*vy','x*vy','y*vx','vx*vy',
    'x^2','y^2','vx^2','vy^2','x*theta','y*phi','vx*theta','vy*phi',
    'body_vx','body_vy','body_vz','thrust_dir_x','thrust_dir_y','thrust_dir_z',
    'bias',
]
OBSERVABLE_SET = os.environ.get("EDMDC_OBSERVABLE_SET", "full56").strip().lower()
OBSERVABLE_SETS = {
    "state13": list(range(12)) + [55],
    "trig19": list(range(18)) + [55],
    "physics42": list(range(35)) + list(range(49, 56)),
    "full56": list(range(56)),
}
if OBSERVABLE_SET not in OBSERVABLE_SETS:
    raise ValueError(
        "EDMDC_OBSERVABLE_SET must be one of " + ", ".join(OBSERVABLE_SETS)
    )
OBSERVABLE_INDICES = OBSERVABLE_SETS[OBSERVABLE_SET]
OBSERVABLE_LABELS = [FULL_OBSERVABLE_LABELS[i] for i in OBSERVABLE_INDICES]
TRAIN_FRACTION = float(os.environ.get("EDMDC_TRAIN_FRACTION", "1.0"))
if not 0.0 < TRAIN_FRACTION <= 1.0:
    raise ValueError("EDMDC_TRAIN_FRACTION must be in (0, 1]")
RAW_INPUT_DIM = 4
INPUT_SOURCE = os.environ.get("EDMDC_INPUT_SOURCE", "applied_wrench").strip().lower()
if INPUT_SOURCE == "applied_wrench":
    DATA_INPUT_KEY = "U"
    RAW_INPUT_LABELS = ["thrust", "tau_roll", "tau_pitch", "tau_yaw"]
    INPUT_LIFT_TYPE = "thrust_direction_rate_coupling"
    INPUT_LIFT_LABELS = RAW_INPUT_LABELS + [
        "thrust_x", "thrust_y", "thrust_z",
        "tau_roll_p", "tau_pitch_q", "tau_yaw_r",
    ]
    MODEL_INPUT_TYPE = "applied_wrench"
    MODEL_U_TYPE = "wrench"
elif INPUT_SOURCE == "outer_command":
    DATA_INPUT_KEY = "U_outer"
    RAW_INPUT_LABELS = list(OUTER_RAW_INPUT_LABELS)
    requested_outer_lift = os.environ.get(
        "EDMDC_OUTER_INPUT_LIFT", OUTER_RAW_INPUT_LIFT_TYPE
    ).strip().lower()
    outer_lift_aliases = {
        "raw": OUTER_RAW_INPUT_LIFT_TYPE,
        OUTER_RAW_INPUT_LIFT_TYPE: OUTER_RAW_INPUT_LIFT_TYPE,
        "attitude_error": OUTER_ATTITUDE_INPUT_LIFT_TYPE,
        "nonlinear": OUTER_ATTITUDE_INPUT_LIFT_TYPE,
        OUTER_ATTITUDE_INPUT_LIFT_TYPE: OUTER_ATTITUDE_INPUT_LIFT_TYPE,
    }
    if requested_outer_lift not in outer_lift_aliases:
        raise ValueError(
            "EDMDC_OUTER_INPUT_LIFT must be raw, attitude_error, or "
            f"{OUTER_ATTITUDE_INPUT_LIFT_TYPE}; got {requested_outer_lift!r}"
        )
    INPUT_LIFT_TYPE = outer_lift_aliases[requested_outer_lift]
    INPUT_LIFT_LABELS = (
        list(OUTER_RAW_INPUT_LABELS)
        if INPUT_LIFT_TYPE == OUTER_RAW_INPUT_LIFT_TYPE
        else list(OUTER_ATTITUDE_INPUT_LIFT_LABELS)
    )
    MODEL_INPUT_TYPE = "desired_attitude"
    MODEL_U_TYPE = "outer_command"
else:
    raise ValueError(
        "EDMDC_INPUT_SOURCE must be applied_wrench or outer_command, "
        f"got {INPUT_SOURCE!r}"
    )
DATA_FILE = Path(os.environ.get("EDMDC_DATA_FILE", SCRIPT_DIR / "runs_mixed_n300.pkl"))
MODEL_FILE = Path(os.environ.get("EDMDC_MODEL_FILE", SCRIPT_DIR / "edmdc_model_yaw_wrench.pkl"))
PLOT_DIR = Path(os.environ.get(
    "EDMDC_PLOT_DIR", MODEL_FILE.parent / f"{MODEL_FILE.stem}_plots"
))


def save_training_figure(fig, stem):
    """Save a training diagnostic in both review and publication formats."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(PLOT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

TARGET_FAMILIES = (
    "helix", "fig8", "lissajous", "waypoint", "hover_excitation", "yaw_prbs",
)

def indices_from_env(variable, default):
    """Read a comma-separated deterministic split from the environment."""
    raw = os.environ.get(variable)
    if raw is None:
        return list(default)
    try:
        return [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(f"{variable} must be a comma-separated list of integers") from exc


# Each trajectory family has a separate validation and test run. Lambda selection
# uses validation only; the test split is reserved for final reported diagnostics.
validation_indices = indices_from_env(
    "EDMDC_VALIDATION_INDICES", [38, 58, 128, 154, 209]
)
# These preserve the five held-out runs used by the original paper.  The new
# validation split is adjacent but disjoint, so regularization is not selected
# on the reported test cases.  All yaw-PRBS runs remain available for training.
test_indices = indices_from_env(
    "EDMDC_TEST_INDICES", [39, 59, 129, 155, 210]
)

# The screenshots inspect the first 2 seconds, so give that transient more
# influence in the least-squares fit.
EARLY_TRANSIENT_SECONDS = 2.0
EARLY_TRANSIENT_WEIGHT = 8.0
EARLY_TRANSIENT_STEPS = int(round(EARLY_TRANSIENT_SECONDS / dt))

# Optional operating-envelope balancing for augmentation studies.  The legacy
# behavior remains unchanged unless EDMDC_TILT_WEIGHTING=balanced is set.
TILT_WEIGHTING = os.environ.get("EDMDC_TILT_WEIGHTING", "none").strip().lower()
TILT_WEIGHT_BINS_DEG = np.asarray([
    float(value) for value in os.environ.get(
        "EDMDC_TILT_WEIGHT_BINS_DEG", "0,5,10,15,20,25,30,45,90"
    ).split(",") if value.strip()
])
TILT_WEIGHT_CAP = float(os.environ.get("EDMDC_TILT_WEIGHT_CAP", "12.0"))
if TILT_WEIGHTING not in ("none", "balanced"):
    raise ValueError("EDMDC_TILT_WEIGHTING must be none or balanced")
if len(TILT_WEIGHT_BINS_DEG) < 2 or np.any(np.diff(TILT_WEIGHT_BINS_DEG) <= 0):
    raise ValueError("EDMDC_TILT_WEIGHT_BINS_DEG must be strictly increasing")
if TILT_WEIGHT_CAP < 1.0:
    raise ValueError("EDMDC_TILT_WEIGHT_CAP must be at least 1")

ENFORCE_KINEMATIC_ROWS = True

# Tikhonov regularization candidates
LAMBDA_CANDIDATES = [
    0.0, 1e-3, 1e-2, 5e-2, 1e-1,
    0.2, 0.3, 0.5, 0.75, 1.0,
    1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
    7.5, 10.0, 20.0, 30.0, 50.0, 100.0,
]

# Kept as diagnostic metadata. Model selection uses the validation rolling
# prediction metrics below; plotting diagnostics must not influence it.
FIRST_ROLLOUT_SCORE_WEIGHT = 0.0
WORST_ROLLOUT_SCORE_WEIGHT = 0.0
PLOT_STATE_SCORE_WEIGHTS = np.array([
    1.5, 1.5, 0.3,   # x, y, z
    1.2, 1.2, 0.3,   # vx, vy, vz
    0.2, 0.2, 0.7,   # phi, theta, psi
    0.2, 0.2, 0.7,   # p, q, r
], dtype=float)

# These are suggested "move forward" limits for the short-horizon training check.
# Adjust them if your project has stricter or looser tracking requirements.
SHORT_HORIZON_LIMITS = {
    "rolling_pos": 0.20,
    "rolling_vel": 0.25,
    "y": 0.20,
    "vy": 0.20,
    "yaw": 0.25,
    "r": 0.25,
}

# Position alone is not a suitable selection objective once yaw is part of the
# learned dynamics and controller. Normalize selected validation errors by the
# corresponding acceptance limits to create an explicit, dimensionless score.
# The held-out test split remains untouched until final evaluation.
VALIDATION_SCORE_LIMITS = {
    "rolling_pos": SHORT_HORIZON_LIMITS["rolling_pos"],
    "rolling_vel": SHORT_HORIZON_LIMITS["rolling_vel"],
    "yaw": SHORT_HORIZON_LIMITS["yaw"],
    "r": SHORT_HORIZON_LIMITS["r"],
}


# Load simulation data
def load_simulation_runs(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    if data.get("input_type") != "applied_wrench":
        raise ValueError(
            "EDMDc training requires applied_wrench data. Regenerate the simulation "
            "data with parallel_sim.py and mix_traj.py."
        )
    if "U_requested" not in data:
        raise ValueError(
            "Dataset lacks U_requested diagnostics; regenerate it with parallel_sim.py."
        )
    if DATA_INPUT_KEY not in data:
        raise ValueError(
            f"Dataset lacks {DATA_INPUT_KEY}; regenerate it with the dual-logging "
            "parallel_sim.py before training this input source."
        )
    if INPUT_SOURCE == "outer_command":
        if data.get("schema_version") != "yaw_dual_input_v1":
            raise ValueError(
                "Outer-command training requires schema_version=yaw_dual_input_v1."
            )
        if data.get("outer_input_type") != "desired_attitude":
            raise ValueError("U_outer must declare outer_input_type=desired_attitude")
    return (
        data["t"], data["states"], data[DATA_INPUT_KEY],
        data["ref_traj_list"],
    )


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
    """
    Construct the configured EDMDc input vector.

    Applied-wrench models add thrust direction and torque/rate products. The
    outer-command model uses the selected raw or attitude-error lift.
    """
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
    if INPUT_SOURCE == "outer_command":
        return outer_command_lift_from_phys(
            states if scalar else states_2d,
            raw if scalar else raw_2d,
            input_lift_type=INPUT_LIFT_TYPE,
        )

    thrust = raw_4[:, :1]
    thrust_dir = thrust_direction_from_state_phys(states_2d)
    rates = states_2d[:, 9:12]
    torque_rate = raw_4[:, 1:4] * rates
    lifted = np.hstack([raw_4, thrust * thrust_dir, torque_rate])
    return lifted[0] if scalar else lifted


def scaled_lifted_input_from_phys(state_phys, raw_input, u_scaler):
    lifted = lift_inputs_from_phys(state_phys, raw_input)
    return u_scaler.transform(np.atleast_2d(lifted)).flatten()

t_all, states_all, U_all, ref_traj_list = load_simulation_runs(DATA_FILE)

if U_all.shape[2] < RAW_INPUT_DIM:
    raise ValueError(
        f"Expected logged inputs to include yaw torque ({RAW_INPUT_DIM} channels), "
        f"but got {U_all.shape[2]}."
    )

n_runs   = t_all.shape[0]
# Every non-validation/test run, including optional PRBS excitation data, is
# available for identification. Held-out trajectory runs remain untouched.
target_indices = list(range(n_runs))
validation_indices = [i for i in validation_indices if i in target_indices]
test_indices = [i for i in test_indices if i in target_indices]
if not validation_indices:
    raise ValueError("No validation indices are available; set EDMDC_VALIDATION_INDICES.")
if not test_indices:
    raise ValueError("No test indices are available; set EDMDC_TEST_INDICES.")
if set(validation_indices) & set(test_indices):
    raise ValueError("Validation and test indices must be disjoint.")
train_indices = [
    i for i in target_indices
    if i not in validation_indices and i not in test_indices
]
if not train_indices:
    raise ValueError("No training indices remain after validation/test splitting.")
all_train_indices = list(train_indices)
if TRAIN_FRACTION < 1.0:
    # Deterministically spread the retained runs across the complete ordered
    # training list; validation and test indices remain unchanged.
    keep = max(1, int(round(TRAIN_FRACTION * len(all_train_indices))))
    positions = np.linspace(0, len(all_train_indices) - 1, keep, dtype=int)
    train_indices = [all_train_indices[i] for i in np.unique(positions)]

print("Target families:", ", ".join(TARGET_FAMILIES))
print("Validation indices:", validation_indices)
print("Held-out test indices:", test_indices)
print(f"Loaded file: {DATA_FILE.name}")
print(f"Training input source: {INPUT_SOURCE} ({DATA_INPUT_KEY})")
print("Training raw input labels:", RAW_INPUT_LABELS)
print("Total runs:", n_runs)
print("Target runs:", len(target_indices))
print("Training runs:", len(train_indices))
print("Available training runs:", len(all_train_indices))
print("Training fraction:", TRAIN_FRACTION)
print("Observable set:", OBSERVABLE_SET, f"({len(OBSERVABLE_INDICES)} terms)")
print("t shape:", t_all.shape)
print("states shape (raw 12):", states_all.shape)
print("U shape (raw):", U_all.shape)
print("ref count:", len(ref_traj_list))

# Downsample to the training time step
sim_dt = t_all[0, 1] - t_all[0, 0]
ratio = dt / sim_dt
step = int(round(ratio))

if not np.isclose(ratio, step, rtol=1e-6, atol=1e-8):
    raise ValueError(
        f"EDMD dt={dt} must be an integer multiple of simulation dt={sim_dt}"
    )

print(f"\nDownsampling: sim_dt={sim_dt}, edmd_dt={dt}, step={step}")

idx = np.arange(0, t_all.shape[1], step)
U_raw = U_all

t_all = t_all[:, idx]
states_all = states_all[:, idx, :]
ref_traj_list = [ref_traj[::step] for ref_traj in ref_traj_list]

# The state transition from t[k] to t[k+1] spans several 0.01 s simulator
# control updates. Use the interval-average command as the EDMD input for that
# 0.1 s transition instead of just the first 0.01 s command sample.
U_interval = np.zeros((U_raw.shape[0], len(idx), U_raw.shape[2]))
for k in range(len(idx) - 1):
    U_interval[:, k, :] = np.mean(U_raw[:, idx[k]:idx[k + 1], :], axis=1)
U_interval[:, -1, :] = U_interval[:, -2, :]
U_all = U_interval

if states_all.shape[2] != STATE_DIM:
    raise ValueError(
        f"Expected simulation states with {STATE_DIM} entries "
        f"[x,y,z,vx,vy,vz,phi,theta,psi,p,q,r], got {states_all.shape[2]}"
    )

if U_all.shape[2] != RAW_INPUT_DIM:
    raise ValueError(
        f"Expected {RAW_INPUT_DIM} inputs {RAW_INPUT_LABELS}, got {U_all.shape[2]}"
    )

print(f"Downsampled shape: states={states_all.shape}, U={U_all.shape}")

# Build training snapshots
Xc_list, Xn_list, U_list, W_list = [], [], [], []

for run in train_indices:
    states_run = states_all[run]
    U_run = U_all[run]

    if states_run.shape[0] < 2:
        continue

    n_transitions = states_run.shape[0] - 1
    sample_weights = np.ones(n_transitions, dtype=float)
    early_steps = min(EARLY_TRANSIENT_STEPS, n_transitions)
    sample_weights[:early_steps] = EARLY_TRANSIENT_WEIGHT

    Xc_list.append(states_run[:-1, :])
    Xn_list.append(states_run[1:, :])
    U_list.append(U_run[:-1, :].T)
    W_list.append(sample_weights)

Xc = np.vstack(Xc_list).T          # (12, K)
Xn = np.vstack(Xn_list).T          # (12, K)
U_train = np.hstack(U_list)        # (4, K)
sample_weights = np.concatenate(W_list)

tilt_weight_diagnostics = {"mode": TILT_WEIGHTING}
if TILT_WEIGHTING == "balanced":
    # Exact body-z tilt from Euler roll/pitch.  Weight populated bins toward
    # equal total influence, while capping rare-bin leverage for robustness.
    tilt_deg = np.degrees(np.arccos(np.clip(
        np.cos(Xc[6, :]) * np.cos(Xc[7, :]), -1.0, 1.0
    )))
    bin_index = np.clip(
        np.digitize(tilt_deg, TILT_WEIGHT_BINS_DEG) - 1,
        0, len(TILT_WEIGHT_BINS_DEG) - 2,
    )
    counts = np.bincount(bin_index, minlength=len(TILT_WEIGHT_BINS_DEG) - 1)
    populated = counts > 0
    target_count = float(np.mean(counts[populated]))
    multipliers = np.ones_like(counts, dtype=float)
    multipliers[populated] = np.minimum(
        TILT_WEIGHT_CAP, target_count / counts[populated]
    )
    transition_weights = multipliers[bin_index]
    transition_weights /= np.mean(transition_weights)
    sample_weights *= transition_weights
    tilt_weight_diagnostics = {
        "mode": TILT_WEIGHTING,
        "bins_deg": TILT_WEIGHT_BINS_DEG.tolist(),
        "counts": counts.tolist(),
        "multipliers_before_normalization": multipliers.tolist(),
        "cap": TILT_WEIGHT_CAP,
        "transition_weight_mean": float(np.mean(transition_weights)),
        "transition_weight_min": float(np.min(transition_weights)),
        "transition_weight_max": float(np.max(transition_weights)),
    }

print("\n========== SNAPSHOT DEBUG ==========")
print("Xc shape:", Xc.shape)
print("Xn shape:", Xn.shape)
print("U_train shape:", U_train.shape)
print("Sample weights shape:", sample_weights.shape)
print("Early transient weight:", EARLY_TRANSIENT_WEIGHT)
print("Tilt weighting:", tilt_weight_diagnostics)
print("Number of transitions per run:", states_all.shape[1] - 1)
print("Expected total transitions:", len(train_indices) * (states_all.shape[1] - 1))
print("====================================")

# Scale lifted model inputs
X_all_for_u_scaler = states_all[train_indices].reshape(-1, states_all.shape[2])
U_all_raw_flat = U_all[train_indices].reshape(-1, U_all.shape[2])
U_all_lifted_flat = lift_inputs_from_phys(X_all_for_u_scaler, U_all_raw_flat)
U_train_lifted = lift_inputs_from_phys(Xc.T, U_train.T)

u_scaler = StandardScaler()
u_scaler.fit(U_all_lifted_flat)
U_norm = u_scaler.transform(U_train_lifted).T

print("\n========== INPUT SCALER DEBUG ==========")
print("Raw input labels:", RAW_INPUT_LABELS)
print("Lifted input labels:", INPUT_LIFT_LABELS)
print("Input lift type:", INPUT_LIFT_TYPE)
print("Lifted input shape:", U_train_lifted.shape)
print("Input scaler mean:", u_scaler.mean_)
print("Input scaler scale:", u_scaler.scale_)
print("Scaled U_train mean (approx):", np.mean(U_norm, axis=1))
print("Scaled U_train std  (approx):", np.std(U_norm, axis=1))
print("========================================")

# Scale states
X_all_flat = states_all[train_indices].reshape(-1, states_all.shape[2])
scaler = StandardScaler()
scaler.fit(X_all_flat)
Xc_s = scaler.transform(Xc.T).T
Xn_s = scaler.transform(Xn.T).T

print("\n========== STATE SCALER DEBUG ==========")
print("State scaler mean:", scaler.mean_)
print("State scaler scale:", scaler.scale_)
print("Scaled Xc mean (approx):", np.mean(Xc_s, axis=1))
print("Scaled Xc std  (approx):", np.std(Xc_s, axis=1))
print("Scaled Xn mean (approx):", np.mean(Xn_s, axis=1))
print("Scaled Xn std  (approx):", np.std(Xn_s, axis=1))
print("========================================")

# Legacy 10-state lifting kept only for reference.
# The active training lift is the 12-state observables() below.
#
# [ 0- 9] 10 linear states
# [10-13] sin(phi), cos(phi), sin(theta), cos(theta)
# [14-17] phi*p, theta*q, vx*phi, vy*theta
# [18-19] v_sq, omega_sq
# [20-25] vx*theta, vy*phi, vz², phi², theta², p*q
# [26]    bias

def observables_legacy_10state(x, scaler):
    """
    Return the lifted observable vector for a standardized 10-state input.
    """
    x = np.asarray(x).flatten()
    assert len(x) == 10, f"Expected 10-state vector, got {len(x)}"

    obs = list(x)  # 10 linear terms

    # ----- Trig terms (unscale to radians first) -----
    phi_rad   = x[6] * scaler.scale_[6] + scaler.mean_[6]
    theta_rad = x[7] * scaler.scale_[7] + scaler.mean_[7]

    s_phi   = np.sin(phi_rad)
    c_phi   = np.cos(phi_rad)
    s_theta = np.sin(theta_rad)
    c_theta = np.cos(theta_rad)

    obs += [s_phi, c_phi, s_theta, c_theta]

    # ----- Cross terms (angle × rate, velocity × angle) -----
    obs.append(x[6] * x[8])   # phi * p
    obs.append(x[7] * x[9])   # theta * q
    obs.append(x[3] * x[6])   # vx * phi
    obs.append(x[4] * x[7])   # vy * theta

    # ----- Energy-like terms -----
    v_sq = x[3]**2 + x[4]**2 + x[5]**2
    omega_sq = x[8]**2 + x[9]**2
    obs.append(v_sq)
    obs.append(omega_sq)

    # ----- Targeted quadratic (velocity-angle, angle², gyroscopic) -----
    obs.append(x[3] * x[7])   # vx * theta
    obs.append(x[4] * x[6])   # vy * phi
    obs.append(x[5] * x[5])   # vz²
    obs.append(x[6] * x[6])   # phi²
    obs.append(x[7] * x[7])   # theta²
    obs.append(x[8] * x[9])   # p * q

    # ----- Bias -----
    obs.append(1.0)

    return np.array(obs, dtype=float)


def observables_full(x, scaler):
    """
    Return the lifted observable vector for a standardized 12-state input.
    Must match edmdc_mpc.py exactly.
    """
    x = np.asarray(x).flatten()
    assert len(x) == STATE_DIM, f"Expected 12-state vector, got {len(x)}"

    obs = list(x)  # 12 linear terms

    phi_rad = x[6] * scaler.scale_[6] + scaler.mean_[6]
    theta_rad = x[7] * scaler.scale_[7] + scaler.mean_[7]
    psi_rad = x[8] * scaler.scale_[8] + scaler.mean_[8]

    s_phi, c_phi = np.sin(phi_rad), np.cos(phi_rad)
    s_theta, c_theta = np.sin(theta_rad), np.cos(theta_rad)
    s_psi, c_psi = np.sin(psi_rad), np.cos(psi_rad)

    obs += [
        s_phi, c_phi,
        s_theta, c_theta,
        s_psi, c_psi,
    ]

    obs += [
        x[6] * x[9],     # phi * p
        x[7] * x[10],    # theta * q
        x[8] * x[11],    # psi * r
        x[3] * x[6],     # vx * phi
        x[4] * x[7],     # vy * theta
        x[3] * x[8],     # vx * psi
        x[4] * x[8],     # vy * psi
        x[5] * x[7],     # vz * theta
    ]

    obs += [
        x[3]**2 + x[4]**2 + x[5]**2,
        x[9]**2 + x[10]**2 + x[11]**2,
    ]

    obs += [
        x[5] * x[5],     # vz^2
        x[6] * x[6],     # phi^2
        x[7] * x[7],     # theta^2
        x[8] * x[8],     # psi^2
        x[9] * x[10],    # p * q
        x[10] * x[11],   # q * r
        x[9] * x[11],    # p * r
    ]

    # Lateral trajectory-shape terms for x/y/vx/vy. These help the 100 Hz
    # model capture figure-8 and lissajous curvature without changing inputs.
    obs += [
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
    ]

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

    obs += [
        body_vx, body_vy, body_vz,
        thrust_dir_x, thrust_dir_y, thrust_dir_z,
    ]

    obs.append(1.0)

    return np.array(obs, dtype=float)


def observables(x, scaler):
    """Return the configured capacity-ablation subset of the full lift."""
    return observables_full(x, scaler)[OBSERVABLE_INDICES]


# Test observable dimension
n_obs_test = len(observables(np.zeros(STATE_DIM), scaler))
print(f"\nObservable dimension: {n_obs_test}")

# Lifted snapshot matrices
Psi = np.column_stack([observables(Xc_s[:, k], scaler) for k in range(Xc_s.shape[1])])
Phi = np.column_stack([observables(Xn_s[:, k], scaler) for k in range(Xn_s.shape[1])])

print("\n========== LIFTING DEBUG ==========")
print("Psi shape:", Psi.shape)
print("Phi shape:", Phi.shape)

Omega = np.vstack([Psi, U_norm])
print("Omega shape:", Omega.shape)

try:
    svals = np.linalg.svd(Omega, compute_uv=False)
    print("Omega singular values (first 10):", svals[:10])
    print("Omega singular values (last 10):", svals[-10:])
    if svals[-1] == 0:
        print("Omega condition number: inf (smallest singular value is zero)")
    else:
        print("Omega condition number:", svals[0] / svals[-1])
except Exception as e:
    print("SVD failed:", e)

print("===================================")

# ============================================================
# REGULARIZATION SWEEP
# ============================================================
def enforce_kinematic_rows(A, B, scaler, dt):
    """Enforce exact standardized integrator rows for positions and angles."""
    A = A.copy()
    B = B.copy()
    bias_idx = A.shape[1] - 1

    A[bias_idx, :] = 0.0
    B[bias_idx, :] = 0.0
    A[bias_idx, bias_idx] = 1.0

    if not ENFORCE_KINEMATIC_ROWS:
        return A, B

    for state_idx, rate_idx in [(0, 3), (1, 4), (2, 5),
                                (6, 9), (7, 10), (8, 11)]:
        A[state_idx, :] = 0.0
        B[state_idx, :] = 0.0
        A[state_idx, state_idx] = 1.0
        A[state_idx, rate_idx] = dt * scaler.scale_[rate_idx] / scaler.scale_[state_idx]
        A[state_idx, bias_idx] = dt * scaler.mean_[rate_idx] / scaler.scale_[state_idx]

    return A, B


def train_edmdc(Psi, Phi, U_norm, n_obs, lam, sample_weights=None):
    """Fit EDMDc via weighted Tikhonov-regularized least squares."""
    Omega = np.vstack([Psi, U_norm])
    Phi_fit = Phi

    if sample_weights is not None:
        sqrt_w = np.sqrt(np.asarray(sample_weights, dtype=float)).reshape(1, -1)
        Omega = Omega * sqrt_w
        Phi_fit = Phi * sqrt_w

    G = Omega @ Omega.T
    Y = Phi_fit @ Omega.T
    if lam != 0:
        G = G + lam * np.eye(G.shape[0])
    AB = Y @ pinv(G)

    A = AB[:, :n_obs]
    B = AB[:, n_obs:]

    return enforce_kinematic_rows(A, B, scaler, dt)


def rolling_horizon_rmse(states, inputs, A, B, scaler, u_scaler,
                         horizon, observables_fn, stride=1):
    n_total = states.shape[0]
    n_windows = n_total - horizon
    if n_windows <= 0:
        raise ValueError("Trajectory is shorter than the evaluation horizon.")
    stride = max(1, int(stride))

    pos_rmse_list = []
    vel_rmse_list = []
    full_rmse_list = []
    per_state_rmse_list = []

    start_indices = list(range(0, n_windows, stride))
    if start_indices[-1] != n_windows - 1:
        start_indices.append(n_windows - 1)

    for start in start_indices:
        states_seg = states[start:start + horizon + 1]
        inputs_seg = inputs[start:start + horizon]

        psi_pred = np.zeros((A.shape[0], horizon + 1))
        psi_pred[:, 0] = observables_fn(
            scaler.transform(states_seg[0].reshape(1, -1)).flatten(),
            scaler
        )

        for k in range(1, horizon + 1):
            x_prev_phys = scaler.inverse_transform(
                psi_pred[:STATE_DIM, k - 1].reshape(1, -1)
            ).flatten()
            u_k_s = scaled_lifted_input_from_phys(
                x_prev_phys, inputs_seg[k - 1], u_scaler
            )
            psi_pred[:, k] = A @ psi_pred[:, k - 1] + B @ u_k_s

        x_pred = scaler.inverse_transform(psi_pred[:STATE_DIM, :].T)

        err = states_seg - x_pred
        pos_err = err[:, 0:3]
        vel_err = err[:, 3:6]

        pos_rmse_list.append(np.sqrt(np.mean(pos_err**2)))
        vel_rmse_list.append(np.sqrt(np.mean(vel_err**2)))
        full_rmse_list.append(np.sqrt(np.mean(err**2)))
        per_state_rmse_list.append(np.sqrt(np.mean(err**2, axis=0)))

    return (
        float(np.mean(pos_rmse_list)),
        float(np.mean(vel_rmse_list)),
        float(np.mean(full_rmse_list)),
        np.mean(np.asarray(per_state_rmse_list), axis=0),
    )


def single_rollout_metrics(states, inputs, A, B, scaler, u_scaler,
                           horizon, observables_fn):
    """Evaluate the same initial free rollout shown in the diagnostic plots."""
    M = min(horizon + 1, states.shape[0])
    states_short = states[:M]
    inputs_short = inputs[:M]

    psi_pred = np.zeros((A.shape[0], M))
    psi_pred[:, 0] = observables_fn(
        scaler.transform(states_short[0, :].reshape(1, -1)).flatten(),
        scaler
    )

    for k in range(1, M):
        x_prev_phys = scaler.inverse_transform(
            psi_pred[:STATE_DIM, k - 1].reshape(1, -1)
        ).flatten()
        u_k_s = scaled_lifted_input_from_phys(
            x_prev_phys, inputs_short[k - 1, :], u_scaler
        )
        psi_pred[:, k] = A @ psi_pred[:, k - 1] + B @ u_k_s

    x_pred = scaler.inverse_transform(psi_pred[:STATE_DIM, :].T).T
    err = states_short.T - x_pred
    rmse_each = np.sqrt(np.mean(err**2, axis=1))
    pos_rmse = np.sqrt(np.mean(err[0:3, :]**2))
    vel_rmse = np.sqrt(np.mean(err[3:6, :]**2))
    weighted_state_rmse = np.sqrt(
        np.mean(PLOT_STATE_SCORE_WEIGHTS * rmse_each**2)
    )
    return (
        float(pos_rmse),
        float(vel_rmse),
        float(np.sqrt(np.mean(err**2))),
        rmse_each,
        float(weighted_state_rmse),
    )


family_names = {
    38: "helix-validation",
    39: "helix",
    58: "figure-8-validation",
    59: "figure-8",
    128: "lissajous-validation",
    129: "lissajous",
    154: "waypoint-validation",
    155: "waypoint",
    209: "hover-excitation-validation",
    210: "hover-excitation",
}

n_obs = Psi.shape[0]
h = MPC_HORIZON
rolling_stride = max(1, int(round(ROLLING_WINDOW_STRIDE_SECONDS / dt)))
sweep_rolling_stride = max(1, int(round(SWEEP_ROLLING_WINDOW_STRIDE_SECONDS / dt)))

print(f"\n{'='*60}")
print(f"REGULARIZATION SWEEP ({len(LAMBDA_CANDIDATES)} candidates)")
print(f"Selecting lambda with rolling {h}-step RMSE on validation trajectories")
print(
    f"Sweep rolling-window stride: {sweep_rolling_stride} steps "
    f"({sweep_rolling_stride * dt:.2f} s)"
)
print(f"{'='*60}")

best_lam = 0
best_score = float("inf")
best_avg_roll_pos = float("inf")
best_avg_roll_vel = float("inf")
best_avg_yaw = float("inf")
best_avg_r = float("inf")
best_avg_plot_score = float("inf")
sweep_rows = []

for lam in LAMBDA_CANDIDATES:
    A_try, B_try = train_edmdc(
        Psi, Phi, U_norm, n_obs, lam,
        sample_weights=sample_weights,
    )

    per_traj = {}
    total_score = 0.0
    total_roll_pos = 0.0
    total_roll_vel = 0.0
    total_yaw = 0.0
    total_r = 0.0
    total_plot_score = 0.0
    for tidx in validation_indices:
        name = family_names.get(tidx, str(tidx))
        try:
            pos_r, vel_r, _, per_state = rolling_horizon_rmse(
                states_all[tidx], U_all[tidx],
                A_try, B_try, scaler, u_scaler, h,
                observables, stride=sweep_rolling_stride
            )
            _, _, _, _, plot_score = single_rollout_metrics(
                states_all[tidx], U_all[tidx],
                A_try, B_try, scaler, u_scaler, h,
                observables
            )
            yaw_r = per_state[8]
            r_r = per_state[11]
            per_traj[name] = (pos_r, vel_r, yaw_r, r_r)
            total_roll_pos += pos_r
            total_roll_vel += vel_r
            total_yaw += yaw_r
            total_r += r_r
            total_score += (
                pos_r / VALIDATION_SCORE_LIMITS["rolling_pos"]
                + vel_r / VALIDATION_SCORE_LIMITS["rolling_vel"]
                + yaw_r / VALIDATION_SCORE_LIMITS["yaw"]
                + r_r / VALIDATION_SCORE_LIMITS["r"]
            ) / 4.0
            total_plot_score += plot_score
        except Exception:
            per_traj[name] = (float("inf"),) * 4
            total_score = float("inf")
            total_roll_pos = float("inf")
            total_roll_vel = float("inf")
            total_yaw = float("inf")
            total_r = float("inf")
            total_plot_score = float("inf")

    avg_roll_pos = total_roll_pos / len(validation_indices)
    avg_roll_vel = total_roll_vel / len(validation_indices)
    avg_yaw = total_yaw / len(validation_indices)
    avg_r = total_r / len(validation_indices)
    avg_plot_score = total_plot_score / len(validation_indices)
    worst_roll_pos = max(metrics[0] for metrics in per_traj.values())
    worst_roll_vel = max(metrics[1] for metrics in per_traj.values())
    score = total_score / len(validation_indices)
    detail = "  ".join(
        f"{name}:pos={metrics[0]:.4f},vel={metrics[1]:.4f},"
        f"yaw={metrics[2]:.4f},r={metrics[3]:.4f}"
        for name, metrics in per_traj.items()
    )
    print(
        f"  lam={lam:.0e}  score={score:.4f}  "
        f"roll_pos={avg_roll_pos:.4f}  roll_vel={avg_roll_vel:.4f}  "
        f"yaw={avg_yaw:.4f}  r={avg_r:.4f}  "
        f"worst_pos={worst_roll_pos:.4f}  worst_vel={worst_roll_vel:.4f}  "
        f"plot_score={avg_plot_score:.4f}  "
        f"{detail}"
    )
    sweep_rows.append({
        "lambda": lam,
        "score": score,
        "rolling_pos": avg_roll_pos,
        "rolling_vel": avg_roll_vel,
        "yaw": avg_yaw,
        "r": avg_r,
        "worst_roll_pos": worst_roll_pos,
        "worst_roll_vel": worst_roll_vel,
        "plot_score": avg_plot_score,
        "per_trajectory": {
            name: {
                "rolling_pos": metrics[0],
                "rolling_vel": metrics[1],
                "yaw": metrics[2],
                "r": metrics[3],
            }
            for name, metrics in per_traj.items()
        },
    })

    if score < best_score:
        best_score = score
        best_avg_roll_pos = avg_roll_pos
        best_avg_roll_vel = avg_roll_vel
        best_avg_yaw = avg_yaw
        best_avg_r = avg_r
        best_avg_plot_score = avg_plot_score
        best_lam = lam

print(
    f"\nBest lambda: {best_lam:.0e} "
    f"(score={best_score:.4f}, rolling pos={best_avg_roll_pos:.4f}, "
    f"rolling vel={best_avg_roll_vel:.4f}, "
    f"yaw={best_avg_yaw:.4f}, r={best_avg_r:.4f}, "
    f"plot score={best_avg_plot_score:.4f})"
)

finite_sweep_rows = [
    row for row in sweep_rows
    if np.isfinite(row["score"])
]
if finite_sweep_rows:
    lam_values = np.array([row["lambda"] for row in finite_sweep_rows], dtype=float)
    score_values = np.array([row["score"] for row in finite_sweep_rows], dtype=float)
    rolling_values = np.array([row["rolling_pos"] for row in finite_sweep_rows], dtype=float)
    velocity_values = np.array([row["rolling_vel"] for row in finite_sweep_rows], dtype=float)
    plot_values = np.array([row["plot_score"] for row in finite_sweep_rows], dtype=float)

    fig_sweep, ax_sweep = plt.subplots(figsize=(9, 5))
    ax_sweep.plot(lam_values, score_values, marker="o", linewidth=2.0, label="selection score")
    ax_sweep.plot(lam_values, rolling_values, marker="s", linewidth=1.6, label="rolling position RMSE")
    ax_sweep.plot(lam_values, velocity_values, marker="d", linewidth=1.6, label="rolling velocity RMSE")
    ax_sweep.plot(lam_values, plot_values, marker="^", linewidth=1.6, label="first-rollout plot score")
    ax_sweep.axvline(best_lam, color="black", linestyle="--", linewidth=1.2, label=f"chosen lambda={best_lam:.0e}")
    ax_sweep.set_xscale("symlog", linthresh=1e-3)
    ax_sweep.set_xlabel("lambda")
    ax_sweep.set_ylabel("RMSE / score")
    ax_sweep.set_title("EDMDc Lambda Selection")
    ax_sweep.grid(True, which="both", alpha=0.3)
    ax_sweep.legend()
    fig_sweep.tight_layout()
    save_training_figure(fig_sweep, "regularization_sweep")

# ============================================================
# FINAL MODEL WITH BEST LAMBDA
# ============================================================
A, B = train_edmdc(
    Psi, Phi, U_norm, n_obs, best_lam,
    sample_weights=sample_weights,
)

rho = np.max(np.abs(np.linalg.eigvals(A)))

print(f"\n========== FINAL MODEL (lambda={best_lam:.0e}) ==========")
print("A shape:", A.shape)
print("B shape:", B.shape)

eigvals = np.linalg.eigvals(A)
abs_eigs = np.sort(np.abs(eigvals))

print("Max abs eigenvalue of A:", np.max(np.abs(eigvals)))
print("Top 10 abs eigenvalues:", abs_eigs[-10:])
print("Any NaN in A?", np.isnan(A).any())
print("Any Inf in A?", np.isinf(A).any())
print("Any NaN in B?", np.isnan(B).any())
print("Any Inf in B?", np.isinf(B).any())
print("=================================")

print("\n========== B ROW NORMS ==========")
labels_10 = STATE_LABELS
for i, lbl in enumerate(labels_10):
    print(f"  {lbl:>6s}: {np.linalg.norm(B[i,:]):.6f}")
print("  --- lifted rows ---")
lifted_labels = ['sin_phi','cos_phi','sin_theta','cos_theta',
                 'phi*p','theta*q','vx*phi','vy*theta',
                 'v_sq','omega_sq',
                 'vx*theta','vy*phi','vz²','phi²','theta²','p*q',
                 'bias']
lifted_labels = ['sin_phi','cos_phi','sin_theta','cos_theta','sin_psi','cos_psi',
                 'phi*p','theta*q','psi*r','vx*phi','vy*theta','vx*psi','vy*psi','vz*theta',
                 'v_sq','omega_sq',
                 'vz^2','phi^2','theta^2','psi^2','p*q','q*r','p*r',
                 'x*y','x*vx','y*vy','x*vy','y*vx','vx*vy',
                 'x^2','y^2','vx^2','vy^2','x*theta','y*phi','vx*theta','vy*phi',
                 'body_vx','body_vy','body_vz','thrust_dir_x','thrust_dir_y','thrust_dir_z',
                 'bias']
lifted_labels = OBSERVABLE_LABELS[STATE_DIM:]
for i, lbl in enumerate(lifted_labels):
    print(f"  {lbl:>12s}: {np.linalg.norm(B[STATE_DIM+i,:]):.6f}")
print("==================================")


# ============================================================
# Held-out short-horizon evaluation
# ============================================================
labels = STATE_LABELS
units  = ['m','m','m','m/s','m/s','m/s','rad','rad','rad','rad/s','rad/s','rad/s']

print("\n========== SHORT-HORIZON EVALUATION ==========")
print(f"Evaluation horizon: {h} steps ({h * dt:.2f} s)")
print(f"Rolling-window stride: {rolling_stride} steps ({rolling_stride * dt:.2f} s)")
print("==============================================")

summary_rows = []

for test_idx in test_indices:
    t_test = t_all[test_idx]
    states_test = states_all[test_idx].copy()
    U_test = U_all[test_idx]
    ref_test = ref_traj_list[test_idx]

    name = family_names.get(test_idx, f"idx {test_idx}")

    M = min(h + 1, states_test.shape[0])
    t_short = t_test[:M]
    states_short = states_test[:M]
    U_short = U_test[:M]

    Psi_pred = np.zeros((n_obs, M))
    Psi_pred[:, 0] = observables(
        scaler.transform(states_short[0, :].reshape(1, -1)).flatten(),
        scaler
    )

    for k in range(1, M):
        x_prev_phys = scaler.inverse_transform(
            Psi_pred[:STATE_DIM, k - 1].reshape(1, -1)
        ).flatten()
        u_k_s = scaled_lifted_input_from_phys(
            x_prev_phys, U_short[k - 1, :], u_scaler
        )
        Psi_pred[:, k] = A @ Psi_pred[:, k - 1] + B @ u_k_s

    x_pred = scaler.inverse_transform(Psi_pred[:STATE_DIM, :].T).T
    err = states_short.T - x_pred

    rmse_each = np.sqrt(np.mean(err**2, axis=1))
    rmse_total = np.sqrt(np.mean(err**2))

    X_test_s = scaler.transform(states_short)
    one_step_pred = np.zeros_like(states_short.T)
    one_step_pred[:, 0] = states_short[0]

    for k in range(states_short.shape[0] - 1):
        psi_k = observables(X_test_s[k], scaler)
        u_k_s = scaled_lifted_input_from_phys(
            states_short[k], U_short[k], u_scaler
        )
        psi_next = A @ psi_k + B @ u_k_s
        x_next_pred = scaler.inverse_transform(
            psi_next[:STATE_DIM].reshape(1, -1)
        ).flatten()
        one_step_pred[:, k + 1] = x_next_pred

    err_one = states_short.T - one_step_pred
    rmse_one = np.sqrt(np.mean(err_one**2))
    pos_rmse_roll, vel_rmse_roll, full_rmse_roll, per_state_rmse_roll = rolling_horizon_rmse(
        states_test, U_test, A, B, scaler, u_scaler, h,
        observables, stride=rolling_stride
    )

    print(f"\n--- {name} (idx={test_idx}) ---")
    print(f"One-step total RMSE:              {rmse_one:.4f}")
    print(f"Single {h}-step rollout RMSE:       {rmse_total:.4f}")
    print(f"Rolling {h}-step position RMSE:     {pos_rmse_roll:.4f}")
    print(f"Rolling {h}-step velocity RMSE:     {vel_rmse_roll:.4f}")
    print(f"Rolling {h}-step full-state RMSE:   {full_rmse_roll:.4f}")
    print(
        f"Rolling weak states: y={per_state_rmse_roll[1]:.4f}m  "
        f"vy={per_state_rmse_roll[4]:.4f}m/s  "
        f"yaw={per_state_rmse_roll[8]:.4f}rad  "
        f"r={per_state_rmse_roll[11]:.4f}rad/s"
    )
    print("Per-state short-horizon RMSE (single rollout from initial state):")
    for lbl, val in zip(labels, rmse_each):
        print(f"  {lbl}: {val:.4f}")

    summary_rows.append((
        name, test_idx, rmse_one, rmse_total,
        pos_rmse_roll, vel_rmse_roll, full_rmse_roll,
        per_state_rmse_roll, rmse_each,
    ))

    # 3D short-horizon trajectory plot
    x_sim = states_short[:, 0]
    y_sim = states_short[:, 1]
    z_sim = states_short[:, 2]

    x_edmd = x_pred[0, :]
    y_edmd = x_pred[1, :]
    z_edmd = x_pred[2, :]

    fig_trajectory = plt.figure(figsize=(7, 5))
    ax = fig_trajectory.add_subplot(111, projection="3d")
    ax.plot(x_sim, y_sim, z_sim, linewidth=2, label="True")
    ax.plot(x_edmd, y_edmd, z_edmd, '--', linewidth=2, label="EDMDc")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(f"{name}: short-horizon rollout ({h} steps)")
    ax.legend()
    ax.grid(True)
    plot_stem = (
        f"run_{test_idx:03d}_{name.lower().replace(' ', '_')}"
    ).replace("-", "_")
    fig_trajectory.tight_layout()
    save_training_figure(fig_trajectory, f"{plot_stem}_trajectory")

    # Per-state time-series plots
    n_states = len(labels)
    n_cols = 4
    n_rows = int(np.ceil(n_states / n_cols))
    fig_states, axs = plt.subplots(n_rows, n_cols, figsize=(20, 7))
    axs = np.asarray(axs).reshape(n_rows, n_cols)
    for i in range(n_states):
        row, col = divmod(i, n_cols)
        ax = axs[row, col]
        ax.plot(t_short, states_short[:, i], label='True')
        ax.plot(t_short, x_pred[i], '--', label='EDMDc')
        ax.set_title(f"{labels[i]} (RMSE {rmse_each[i]:.3f})")
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f"{labels[i]} [{units[i]}]")
        ax.grid(True)
        if i == 0:
            ax.legend()
    for j in range(n_states, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axs[row, col].axis("off")
    fig_states.suptitle(
        f"{name}: short-horizon state prediction", fontsize=14, y=0.98
    )
    fig_states.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    save_training_figure(fig_states, f"{plot_stem}_all_states")

print("\n========== SHORT-HORIZON SUMMARY ==========")
for name, idx, rmse_one, rmse_roll, pos_rmse_roll, vel_rmse_roll, full_rmse_roll, per_state_rmse_roll, rmse_each in summary_rows:
    print(
        f"{name:<18s} idx={idx:<4d} "
        f"one-step={rmse_one:.4f}  "
        f"single-rollout={rmse_roll:.4f}  "
        f"rolling-pos={pos_rmse_roll:.4f}  "
        f"rolling-vel={vel_rmse_roll:.4f}  "
        f"rolling-full={full_rmse_roll:.4f}"
    )
print("===========================================")

print("\n========== SINGLE-ROLLOUT PLOT CHECK ==========")
print("These are the subplot RMSE values from the first 2-second free rollout.")
for name, idx, _, _, _, _, _, _, rmse_each in summary_rows:
    print(
        f"{name:<18s} "
        f"x={rmse_each[0]:.4f}  y={rmse_each[1]:.4f}  "
        f"vx={rmse_each[3]:.4f}  vy={rmse_each[4]:.4f}  "
        f"psi={rmse_each[8]:.4f}  r={rmse_each[11]:.4f}"
    )
print("==============================================")

print("\n========== SHORT-HORIZON GATE ==========")
print(
    "Limits: "
    f"pos<{SHORT_HORIZON_LIMITS['rolling_pos']:.2f}m, "
    f"velocity<{SHORT_HORIZON_LIMITS['rolling_vel']:.2f}m/s, "
    f"y<{SHORT_HORIZON_LIMITS['y']:.2f}m, "
    f"vy<{SHORT_HORIZON_LIMITS['vy']:.2f}m/s, "
    f"yaw<{SHORT_HORIZON_LIMITS['yaw']:.2f}rad, "
    f"r<{SHORT_HORIZON_LIMITS['r']:.2f}rad/s"
)
gate_all_pass = True
for name, idx, _, _, pos_rmse_roll, vel_rmse_roll, _, per_state_rmse_roll, _ in summary_rows:
    checks = {
        "rolling_pos": pos_rmse_roll,
        "rolling_vel": vel_rmse_roll,
        "y": per_state_rmse_roll[1],
        "vy": per_state_rmse_roll[4],
        "yaw": per_state_rmse_roll[8],
        "r": per_state_rmse_roll[11],
    }
    passing = all(value <= SHORT_HORIZON_LIMITS[key] for key, value in checks.items())
    gate_all_pass = gate_all_pass and passing
    status = "PASS" if passing else "REVIEW"
    print(
        f"{status:>6s}  {name:<18s} "
        f"pos={checks['rolling_pos']:.4f}  "
        f"vel={checks['rolling_vel']:.4f}  "
        f"y={checks['y']:.4f}  "
        f"vy={checks['vy']:.4f}  "
        f"yaw={checks['yaw']:.4f}  "
        f"r={checks['r']:.4f}"
    )
print("Decision:", "GOOD ENOUGH TO MOVE FORWARD" if gate_all_pass else "DO NOT TUNE YET")
print("========================================")

# Save model
model_data = {
    "A": A,
    "B": B,
    "scaler": scaler,
    "u_scaler": u_scaler,
    "dt": dt,
    "n_obs": n_obs,
    "lambda": best_lam,
    "lambda_selection_score": best_score,
    "lambda_selection_rolling_pos": best_avg_roll_pos,
    "lambda_selection_rolling_vel": best_avg_roll_vel,
    "lambda_selection_yaw": best_avg_yaw,
    "lambda_selection_r": best_avg_r,
    "lambda_selection_plot_score": best_avg_plot_score,
    "regularization_sweep": sweep_rows,
    "validation_score_limits": VALIDATION_SCORE_LIMITS,
    "first_rollout_score_weight": FIRST_ROLLOUT_SCORE_WEIGHT,
    "worst_rollout_score_weight": WORST_ROLLOUT_SCORE_WEIGHT,
    "plot_state_score_weights": PLOT_STATE_SCORE_WEIGHTS,
    "short_horizon_seconds": SHORT_HORIZON_SECONDS,
    "short_horizon_steps": h,
    "rolling_window_stride_seconds": rolling_stride * dt,
    "sweep_rolling_window_stride_seconds": sweep_rolling_stride * dt,
    "target_families": list(TARGET_FAMILIES),
    "target_indices": target_indices,
    "validation_indices": validation_indices,
    "train_indices": train_indices,
    "available_train_indices": all_train_indices,
    "train_fraction": TRAIN_FRACTION,
    "early_transient_seconds": EARLY_TRANSIENT_SECONDS,
    "early_transient_weight": EARLY_TRANSIENT_WEIGHT,
    "tilt_weighting": tilt_weight_diagnostics,
    "enforce_kinematic_rows": ENFORCE_KINEMATIC_ROWS,
    "state_labels": labels,
    "raw_input_dim": RAW_INPUT_DIM,
    "raw_u_labels": RAW_INPUT_LABELS,
    "u_labels": INPUT_LIFT_LABELS,
    "input_lift_type": INPUT_LIFT_TYPE,
    "input_lift_labels": INPUT_LIFT_LABELS,
    "observable_set": OBSERVABLE_SET,
    "observable_labels": OBSERVABLE_LABELS,
    "active_observable_indices": OBSERVABLE_INDICES,
    "source_file": DATA_FILE.name,
    "test_indices": test_indices,
    "input_source": INPUT_SOURCE,
    "data_input_key": DATA_INPUT_KEY,
    "u_type": MODEL_U_TYPE,
    "input_type": MODEL_INPUT_TYPE,
    "downsampling": {
        "source_dt_seconds": float(sim_dt),
        "state_method": (
            f"take every {step}th state" if step > 1 else "native"
        ),
        "input_method": "interval_mean" if step > 1 else "native",
        "input_description": (
            f"mean input over each {step}-sample transition"
            if step > 1 else "native aligned input"
        ),
        "transition_alignment": (
            f"(x[{step}k], mean(u[{step}k:{step}(k+1)]), x[{step}(k+1)])"
            if step > 1 else "(x[k], u[k], x[k+1])"
        ),
    },
}

with open(MODEL_FILE, "wb") as f:
    pickle.dump(model_data, f)

print(f"\nSaved model to {MODEL_FILE.name}")
print(f"A: {A.shape}, B: {B.shape}, n_obs: {n_obs}, lambda: {best_lam:.0e}")
print(f"Saved open-loop rollout plots to {PLOT_DIR.resolve()}")

if "agg" in plt.get_backend().lower():
    # Headless paper runs save the numeric model and console log.  Calling
    # show() on Agg only emits a warning, which PowerShell can misreport as a
    # failed training process even though the model was written successfully.
    plt.close("all")
else:
    plt.show()
