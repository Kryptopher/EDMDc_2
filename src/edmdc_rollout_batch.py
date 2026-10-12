"""Batched equivalent of the training script's mean-window rollout metric."""
import numpy as np


def rolling_horizon_rmse_batch(states, inputs, A, B, scaler, u_scaler,
                                horizon, observables_fn, lift_fn, stride=1):
    windows = len(states)-horizon
    if windows <= 0:
        raise ValueError("Trajectory is shorter than the evaluation horizon.")
    starts = list(range(0, windows, max(1, int(stride))))
    if starts[-1] != windows-1:
        starts.append(windows-1)
    starts = np.asarray(starts)
    standardized = (states[starts]-scaler.mean_) / scaler.scale_
    z = np.stack([observables_fn(x, scaler) for x in standardized])
    squared = (states[starts]-(z[:, :12]*scaler.scale_+scaler.mean_))**2
    for k in range(horizon):
        physical = z[:, :12]*scaler.scale_+scaler.mean_
        lifted = (lift_fn(physical, inputs[starts+k])-u_scaler.mean_) / u_scaler.scale_
        z = z @ A.T + lifted @ B.T
        error = states[starts+k+1]-(z[:, :12]*scaler.scale_+scaler.mean_)
        squared += error**2
    mse = squared/(horizon+1)
    return (float(np.mean(np.sqrt(np.mean(mse[:, :3], axis=1)))),
            float(np.mean(np.sqrt(np.mean(mse[:, 3:6], axis=1)))),
            float(np.mean(np.sqrt(np.mean(mse, axis=1)))),
            np.mean(np.sqrt(mse), axis=0))
