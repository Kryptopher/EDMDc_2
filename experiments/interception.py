"""Interception under oracle, Current/CV and delayed/CV target information (Tables IV and V).

Runs any subset of the four predictor/configuration cells (Linear-A, EDMDc-B, EDMDc-A,
Linear-B) on the 40 frozen scenarios. The episode logic is a path-independent port of
provenance/original/artifacts/target_information_ablation_20261002/scripts/runtime.py.

Examples:
    python experiments/interception.py --cells Linear-A EDMDc-B --modes cv --output results_local/interception
    python experiments/interception.py --seeds 3103001001 --cells EDMDc-B --modes cv --save-traces --output out
"""
import argparse
import datetime
import pickle
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _common import CONFIGS, MODEL_PATH, array_sha256, check_model, read_json, write_json
from Simulation import quad_sim
from wrench_revisit_mpc import WrenchConfig, WrenchMPC, physical_reference, plant_step
from targets import scenario_family

MODES = {"oracle": 0, "cv": 0, "delayed_cv_100ms": 10, "delayed_cv_250ms": 25}
DT = 0.01
STEPS = 2000  # 20 s


@dataclass(frozen=True)
class Measurement:
    sample: int
    position: tuple
    velocity: tuple


class CausalSensor:
    """Simulator-owned buffer; the policy only receives one immutable measurement."""

    def __init__(self, history, delay):
        self.__history = history
        self.delay = delay
        self.reads = []

    def receive(self, k):
        j = k - self.delay
        assert j >= -25
        pv = self.__history[j + 25]
        self.reads.append((k, j))
        return Measurement(j, tuple(pv[:3]), tuple(pv[3:6]))


def causal_reference(packet, k, N, delay, previous_yaw, sim):
    """Constant-velocity horizon reference built only from one measurement packet."""
    assert packet.sample == k - delay
    p = np.asarray(packet.position, dtype=float)
    v = np.asarray(packet.velocity, dtype=float)
    yaw = previous_yaw if np.linalg.norm(v[:2]) <= 1e-8 else float(np.arctan2(v[1], v[0]))
    yaw = previous_yaw + float(np.arctan2(np.sin(yaw - previous_yaw), np.cos(yaw - previous_yaw)))
    forecasts = [dict(pos=p + (delay + i) * DT * v, vel=v, acc=np.zeros(3), yaw=yaw) for i in range(1, N + 1)]
    return physical_reference(forecasts, sim, DT), yaw


def build_scenario(case, sim):
    """Regenerate the true target reference and the preloaded measurement history."""
    target = scenario_family(case["family"], np.random.default_rng(case["seed"]))
    truth = physical_reference([target.reference(t) for t in np.arange(2202) * DT], sim, DT)
    history = np.asarray([np.r_[target.position(j * DT), target.velocity(j * DT)] for j in range(-25, STEPS)])
    if array_sha256(truth) != case["reference_sha256"] or array_sha256(history) != case["measurement_history_sha256"]:
        raise RuntimeError(f"Regenerated scenario {case['case_id']} does not match the frozen hash")
    return truth, history


def metrics(states, truth):
    finite = np.all(np.isfinite(states), axis=1)
    valid = len(states) if finite.all() else int(np.flatnonzero(~finite)[0])
    x, r = states[:valid], truth[:valid]
    sep = np.linalg.norm(x[:, :3] - r[:, :3], axis=1)
    speed = np.linalg.norm(x[:, 3:6] - r[:, 3:6], axis=1)
    entry = np.flatnonzero(sep <= .75)
    joint = (sep <= .75) & (speed <= 3)
    hits = np.flatnonzero(np.convolve(joint.astype(int), np.ones(21, dtype=int), "valid") == 21) if valid >= 21 else []
    end = int(hits[0] + 20) if len(hits) else None
    first = int(entry[0]) if len(entry) else None
    return dict(captured=end is not None, qualified_time_s=None if end is None else end * DT,
                scored_time_s=20. if end is None else end * DT,
                first_entry_s=None if first is None else first * DT,
                relative_speed_at_first_entry_mps=None if first is None else float(speed[first]),
                minimum_separation_m=float(np.min(sep)),
                maximum_abs_roll_deg=float(np.degrees(np.max(np.abs(x[:, 6])))),
                maximum_abs_pitch_deg=float(np.degrees(np.max(np.abs(x[:, 7])))),
                qualifying_sample_end=end)


def episode(task):
    nominal, plant = quad_sim(), quad_sim()
    case = task["case"]
    truth, history = build_scenario(case, nominal)
    x = np.asarray(case["initial_state"], dtype=float)
    config = task["config"]
    N = config["horizon_steps"]
    model = None
    if task["predictor"] == "edmdc":
        with Path(task["model"]).open("rb") as f:
            model = pickle.load(f)
    controller = WrenchMPC(task["predictor"], nominal, WrenchConfig(**config), model, initial_yaw=x[8])
    sensor = None if task["mode"] == "oracle" else CausalSensor(history, task["delay_samples"])
    states, requested, applied, accepted = [x.copy()], [], [], []
    saturated, yaw, stop = [], x[8], "completed"
    for k in range(STEPS):
        if sensor is None:
            future = truth[k + 1:k + N + 1]
        else:
            future, yaw = causal_reference(sensor.receive(k), k, N, task["delay_samples"], yaw, nominal)
        u, info = controller.compute(x, future)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            xn, actual, _, sat = plant_step(plant, x, u, DT)
        states.append(xn.copy()); requested.append(u.copy()); applied.append(actual.copy())
        accepted.append(info["accepted"]); saturated.append(bool(np.any(sat)))
        controller.previous_applied = actual.copy()
        x = xn
        if not np.isfinite(x).all():
            stop = "nonfinite_state"; break
        if np.max(np.abs(x[6:8])) > np.deg2rad(80):
            stop = "euler_attitude_envelope_80deg"; break
        # Aborts are judged against the true target, never the forecast.
        if np.linalg.norm(x[:3] - truth[k + 1, :3]) > 50:
            stop = "position_error_over_50m"; break
    states, req, ap = np.asarray(states), np.asarray(requested), np.asarray(applied)
    row = {k: task[k] for k in ("task_id", "cell", "mode", "delay_samples")}
    row.update(case_id=case["case_id"], family=case["family"], seed=case["seed"])
    row.update(metrics(states, truth), completed=stop == "completed", stop_reason=stop,
               simulated_seconds=(len(states) - 1) * DT,
               solver_failures=int(np.count_nonzero(~np.asarray(accepted))),
               allocator_events=int(np.count_nonzero(np.max(np.abs(req - ap), axis=1) > 1e-7)),
               rotor_saturated_steps=int(np.count_nonzero(saturated)),
               reference_sha256=array_sha256(truth))
    if task.get("trace_dir"):
        folder = Path(task["trace_dir"]) / task["task_id"]
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "trace.npz", states=states, reference=truth[:len(states)],
                            time=np.arange(len(states)) * DT, U_requested=req, U=ap)
        row["trace"] = (folder / "trace.npz").as_posix()
    return row


def safe_episode(task):
    try:
        return episode(task)
    except Exception:
        row = {k: task[k] for k in ("task_id", "cell", "mode", "delay_samples")}
        row.update(case_id=task["case"]["case_id"], family=task["case"]["family"], seed=task["case"]["seed"],
                   completed=False, captured=False, qualified_time_s=None, scored_time_s=20.,
                   stop_reason="execution_exception", exception=traceback.format_exc())
        return row


def main():
    controllers = read_json(CONFIGS / "interception_controllers.json")
    scenarios = read_json(CONFIGS / "interception_scenarios.json")["scenarios"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", nargs="+", default=list(controllers["cells"]), choices=list(controllers["cells"]))
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="subset of scenario seeds (default: all 40)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_model()
    chosen = [c for c in scenarios if args.seeds is None or c["seed"] in args.seeds]
    if args.seeds and len(chosen) != len(set(args.seeds)):
        parser.error("unknown seed requested")
    tasks = []
    for cell in args.cells:
        spec = controllers["cells"][cell]
        config = controllers["configurations"][spec["configuration"]]["config"]
        for mode in args.modes:
            for case in chosen:
                tasks.append(dict(task_id=f"{cell}_{mode}_{case['seed']}", cell=cell, mode=mode,
                                  delay_samples=MODES[mode], predictor=spec["predictor"], config=config,
                                  case=case, model=str(MODEL_PATH),
                                  trace_dir=str(args.output / "traces") if args.save_traces else None))
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(safe_episode, t) for t in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            print(f"[{i}/{len(tasks)}] {row['task_id']}: captured={row['captured']} "
                  f"t_q={row['qualified_time_s']} stop={row['stop_reason']}", flush=True)
    rows.sort(key=lambda r: r["task_id"])
    write_json(args.output / "episodes.json", rows)
    write_json(args.output / "run_info.json", {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                               "cells": args.cells, "modes": args.modes, "episodes": len(rows)})
    print(f"Wrote {len(rows)} episodes to {args.output / 'episodes.json'}")


if __name__ == "__main__":
    main()
