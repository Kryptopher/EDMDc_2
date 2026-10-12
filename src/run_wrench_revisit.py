"""Auditable, matched direct-wrench linear/EDMDc/LTV tracking experiments."""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import platform
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np

from Simulation import quad_sim
from publication_scenarios import tracking_reference
from wrench_revisit_mpc import WrenchConfig, WrenchMPC, physical_reference, plant_step

ROOT = Path(__file__).resolve().parent
LABELS = {"linear": "Fixed hover-linear MPC", "yaw_linear": "Yaw-scheduled hover-linear MPC", "edmdc": "Wrench EDMDc-MPC", "ltv": "Nominal-model LTV MPC"}
COLORS = {"linear": "#0072B2", "yaw_linear": "#56B4E9", "edmdc": "#D55E00", "ltv": "#009E73"}
STATES = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_case(task):
    started = time.time()
    c = WrenchConfig(**task["config"])
    sim = quad_sim()
    duration = task["duration"]
    count = int(round(duration/c.dt))
    ref_duration = max(30.0, duration + c.horizon_steps*c.dt + c.dt)
    reference = tracking_reference(task["family"], task["seed"], ref_duration, speed_scale=task["scale"], yaw_rate_scale=1.0)
    if task["yaw_mode"] == "fixed":
        heading = float(reference[0]["yaw"])
        for point in reference:
            point["yaw"], point["yaw_rate"] = heading, 0.0
    desired = physical_reference(reference, sim, c.dt)
    model = None
    if task["controller"] == "edmdc":
        with Path(task["model"]).open("rb") as stream:
            model = pickle.load(stream)
    ctrl = WrenchMPC(task["controller"], sim, c, model=model, initial_yaw=desired[0, 8])
    x = np.zeros(12)
    x[:6], x[8] = desired[0, :6], desired[0, 8]
    states, requested, applied, forces, solve, statuses, accepted, iterations = [x.copy()], [], [], [], [], [], [], []
    stop_reason = "completed"
    for k in range(count):
        command, info = ctrl.compute(x, desired[k+1:k+c.horizon_steps+1])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            next_x, actual, motor_forces, clipped = plant_step(sim, x, command, c.dt)
        requested.append(command)
        applied.append(actual)
        forces.append(motor_forces)
        solve.append(info["elapsed_s"])
        statuses.append(info["status"])
        accepted.append(info["accepted"])
        iterations.append(info["iterations"])
        states.append(next_x.copy())
        ctrl.previous_applied = actual.copy()
        x = next_x
        if not np.isfinite(x).all():
            stop_reason = "nonfinite_state"
            break
        if np.max(np.abs(x[6:8])) > np.deg2rad(80):
            stop_reason = "euler_attitude_envelope_80deg"
            break
        if np.linalg.norm(x[:3]-desired[k+1, :3]) > 50:
            stop_reason = "position_error_over_50m"
            break
    states, requested, applied = np.asarray(states), np.asarray(requested), np.asarray(applied)
    solve = np.asarray(solve)
    reference_used = desired[:len(states)]
    error = states-reference_used
    error[:, 8] = np.arctan2(np.sin(error[:, 8]), np.cos(error[:, 8]))
    finite = np.isfinite(error).all(axis=1)
    rms = np.sqrt(np.mean(error[finite]**2, axis=0))
    complete = stop_reason == "completed"
    row = {key: task[key] for key in ("case_id", "family", "seed", "scale", "yaw_mode", "controller")}
    alteration = np.max(np.abs(requested-applied), axis=1)
    row.update(completed=complete, stop_reason=stop_reason, simulated_seconds=(len(states)-1)*c.dt,
               position_rmse_m=float(np.sqrt(np.mean(error[:, :3]**2))) if complete else None,
               prefix_position_rmse_m=float(np.sqrt(np.mean(error[finite, :3]**2))),
               velocity_rmse_mps=float(np.sqrt(np.mean(error[:, 3:6]**2))) if complete else None,
               yaw_rmse_rad=float(rms[8]) if complete else None,
               solver_failures=int(np.count_nonzero(~np.asarray(accepted))),
               allocator_altered_steps=int(np.count_nonzero(alteration > 1e-7)),
               allocator_max_abs_change=float(np.max(alteration)),
               mean_solve_ms=float(1000*np.mean(solve)), p95_solve_ms=float(1000*np.percentile(solve, 95)),
               max_solve_ms=float(1000*np.max(solve)), deadline_misses=int(np.count_nonzero(solve > c.dt)),
               actual_max_abs_tilt_deg=float(np.degrees(np.max(np.abs(states[finite, 6:8])))),
               reference_max_abs_tilt_deg=float(np.degrees(np.max(np.abs(reference_used[:, 6:8])))),
               reference_max_speed_mps=float(np.max(np.linalg.norm(reference_used[:, 3:6], axis=1))),
               reference_sha256=hashlib.sha256(desired.tobytes()).hexdigest(),
               wall_seconds=time.time()-started)
    case_dir = Path(task["output"]) / "traces" / task["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / (task["controller"] + ".npz")
    np.savez_compressed(path, time=np.arange(len(states))*c.dt, states=states, reference=reference_used,
                        U_requested=requested, U=applied, motor_forces=np.asarray(forces), solve_seconds=solve,
                        solver_status=np.asarray(statuses), solver_accepted=np.asarray(accepted), iterations=np.asarray(iterations))
    row["trace"] = str(path.resolve())
    path.with_suffix(".json").write_text(json.dumps(row, indent=2))
    return row


def plot_results(output, rows):
    folder = output / "plots"
    folder.mkdir(exist_ok=True)
    for case_id in dict.fromkeys(row["case_id"] for row in rows):
        group = [row for row in rows if row["case_id"] == case_id]
        loaded = {row["controller"]: np.load(row["trace"]) for row in group}
        longest = max(loaded.values(), key=lambda item: len(item["time"]))
        fig, axes = plt.subplots(4, 3, figsize=(14, 11), sharex=True)
        for j, ax in enumerate(axes.flat):
            ax.plot(longest["time"], longest["reference"][:, j], "k--", linewidth=1.1, label="Reference")
            for kind, trace in loaded.items():
                ax.plot(trace["time"], trace["states"][:, j], color=COLORS[kind], linewidth=.9, label=LABELS[kind])
            ax.set_title(STATES[j]); ax.grid(alpha=.25)
            ax.set_ylabel("m" if j<3 else "m/s" if j<6 else "rad" if j<9 else "rad/s")
        axes[0, 0].legend(fontsize=7)
        for ax in axes[-1]: ax.set_xlabel("Time [s]")
        fig.suptitle(case_id + " — direct wrench; all states")
        fig.tight_layout(); fig.savefig(folder / f"{case_id}_states.png", dpi=160); plt.close(fig)
        fig = plt.figure(figsize=(12, 5))
        a = fig.add_subplot(121, projection="3d")
        a.plot(*longest["reference"][:, :3].T, "k--", label="Reference")
        b = fig.add_subplot(122)
        for kind, trace in loaded.items():
            a.plot(*trace["states"][:, :3].T, color=COLORS[kind], label=LABELS[kind])
            b.plot(trace["time"], np.linalg.norm(trace["states"][:, :3]-trace["reference"][:, :3], axis=1), color=COLORS[kind], label=LABELS[kind])
        a.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        a.legend(fontsize=7); b.set(xlabel="Time [s]", ylabel="Euclidean position error [m]"); b.grid(alpha=.25)
        fig.suptitle(case_id); fig.tight_layout(); fig.savefig(folder / f"{case_id}_trajectory.png", dpi=160); plt.close(fig)
        fig, axes = plt.subplots(5, 1, figsize=(11, 11), sharex=True)
        for kind, trace in loaded.items():
            for j in range(4):
                axes[j].plot(trace["time"][:-1], trace["U"][:, j], color=COLORS[kind], label=LABELS[kind])
                axes[j].plot(trace["time"][:-1], trace["U_requested"][:, j], color=COLORS[kind], linestyle="--", alpha=.4)
            axes[4].plot(trace["time"][:-1], 1000*trace["solve_seconds"], color=COLORS[kind])
        for ax, label in zip(axes, ["Thrust [N]", "Roll torque [Nm]", "Pitch torque [Nm]", "Yaw torque [Nm]", "Optimization [ms]"]):
            ax.set_ylabel(label); ax.grid(alpha=.25)
        axes[0].legend(fontsize=8); axes[4].axhline(10, color="black", linestyle="--"); axes[4].set_xlabel("Time [s]")
        fig.suptitle(case_id + " — solid: applied, dashed: requested")
        fig.tight_layout(); fig.savefig(folder / f"{case_id}_inputs_timing.png", dpi=160); plt.close(fig)
        for trace in loaded.values(): trace.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--controllers", default="linear,edmdc,ltv")
    parser.add_argument("--families", default="helix,figure8,lissajous,waypoint")
    parser.add_argument("--scales", default="1,1.75")
    parser.add_argument("--yaw-modes", default="fixed,path")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=940000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--role", choices=("smoke", "validation", "confirmation"), default="validation")
    args = parser.parse_args()
    controllers = args.controllers.split(",")
    assert set(controllers) <= set(LABELS)
    assert set(args.yaw_modes.split(",")) <= {"fixed", "path"}
    if "edmdc" in controllers and args.model is None:
        parser.error("--model is required for EDMDc")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Choose a new output folder; existing results are never overwritten")
    config = WrenchConfig(**json.loads(args.config.read_text())) if args.config else WrenchConfig()
    sources = ["run_wrench_revisit.py", "wrench_revisit_mpc.py", "EDMDc_training.py", "edmdc_mpc.py", "Simulation.py", "quadcopter.py", "PID_Mixer.py", "publication_scenarios.py"]
    manifest = {
        "role": args.role, "architecture": "direct_applied_wrench_no_cascade", "status": "running",
        "created_unix": time.time(), "config": asdict(config), "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model": str(args.model.resolve()) if args.model else None,
        "model_sha256": sha256(args.model) if args.model else None,
        "sources": {name: sha256(ROOT/name) for name in sources},
        "controllers": LABELS, "reference_contract": "Identical prescribed position/velocity/acceleration/heading preview; attitude/body rates derived identically; hover input reference; no true future plant state; no PID nominal.",
        "linear_contract": "Constant discrete nominal hover Jacobians at initial reference yaw, retaining equilibrium offset.",
        "yaw_linear_contract": "Nominal hover Jacobians scheduled at every horizon sample using only the prescribed reference yaw; zero-velocity hover equilibrium and affine equilibrium offsets retained.",
        "ltv_contract": "Known nominal rigid-body model, relinearized along shifted prior-command nonlinear prediction, one QP/update, full affine defect retained.",
        "edmdc_contract": "Recursive learned lifted state; full state and input derivatives of wrench features along nominal prediction; one QP/update.",
        "state_feedback": "exact, noiseless", "plant": "original quadcopter derivative with shared motor allocation, RK4 at 0.01 s", "mismatch": False,
        "rmse": "sqrt(mean(position component errors squared)); incomplete runs have null headline RMSE and separately labeled prefix RMSE",
        "deadline_semantics": "Desktop optimization times logged; overruns do not delay simulated updates.",
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "osqp")}},
    }
    tasks = []
    for family_i, family in enumerate(args.families.split(",")):
        for rep in range(args.replicates):
            seed = args.base_seed + 1000*family_i + rep
            for scale in map(float, args.scales.split(",")):
                for yaw_mode in args.yaw_modes.split(","):
                    case_id = f"{family}_{seed}_scale{scale:g}_{yaw_mode}"
                    for controller in controllers:
                        tasks.append(dict(case_id=case_id, family=family, seed=seed, scale=scale, yaw_mode=yaw_mode, controller=controller, duration=args.duration, config=asdict(config), model=str(args.model.resolve()) if args.model else None, output=str(args.output.resolve())))
    manifest["episode_count"] = len(tasks)
    path = args.output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    rows = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(run_case, task): task for task in tasks}
            for future in as_completed(pending):
                row = future.result()
                rows.append(row)
                write_csv(args.output / "episodes.csv", rows)
                print(json.dumps(row), flush=True)
                (args.output / "progress.json").write_text(json.dumps({"completed": len(rows), "total": len(tasks), "last": row}, indent=2))
        for case in {r["case_id"] for r in rows}:
            assert len({r["reference_sha256"] for r in rows if r["case_id"] == case}) == 1
        summary = []
        for yaw_mode in args.yaw_modes.split(","):
            for controller in controllers:
                group = [r for r in rows if r["controller"] == controller and r["yaw_mode"] == yaw_mode]
                complete = [r for r in group if r["completed"]]
                summary.append(dict(controller=controller, yaw_mode=yaw_mode, episodes=len(group), completed=len(complete),
                    mean_position_rmse_completed_m=float(np.mean([r["position_rmse_m"] for r in complete])) if complete else None,
                    median_position_rmse_completed_m=float(np.median([r["position_rmse_m"] for r in complete])) if complete else None,
                    solver_failures=sum(r["solver_failures"] for r in group), allocator_altered_steps=sum(r["allocator_altered_steps"] for r in group),
                    mean_episode_solve_ms=float(np.mean([r["mean_solve_ms"] for r in group])), max_solve_ms=max(r["max_solve_ms"] for r in group), deadline_misses=sum(r["deadline_misses"] for r in group)))
        write_csv(args.output / "summary.csv", summary)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        plot_results(args.output, rows)
        manifest.update(status="complete", completed_unix=time.time())
        print(json.dumps(summary, indent=2), flush=True)
    except Exception:
        manifest.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
