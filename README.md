# Drone Interception via EDMDc-Based Model Predictive Control

Code, frozen configurations, and result records for the paper

> S. Maharjan, D. Johnson, A. Bangal, Y. Atteia, T. Hlayel, and A. Stein,
> "Drone Interception via EDMDc-Based Model Predictive Control," submitted to the American Control Conference (ACC) 2027.

## Check the paper's numbers (under a minute)

After installing the requirements (below), run from the repository root:

```bash
python experiments/analyze_tracking.py       # Table III, Sec. V-A
python experiments/analyze_interception.py   # Tables IV and V, Secs. V-C to V-E
python experiments/analyze_selection.py      # Table II, Sec. IV-C
python experiments/analyze_timing.py         # Sec. V-B, Fig. 3
python experiments/analyze_model.py          # Sec. IV-B
```

These scripts read only the frozen records in `results/` and the model in `data/models/`; none of them
needs the training dataset. The table below is generated from their output by
`python experiments/make_claim_table.py --readme`.

<!-- CLAIM-TABLE:BEGIN -->
| Paper | Value | Check | Result file |
|---|---|---|---|
| §IV-B | λ = 0 from a 22-value grid; 2,051,658 training transitions (342 runs); K 42×42, B_u 42×10 | `python experiments/analyze_model.py` | `data/models/wrench_physics42_dt001.pkl`, `results/training/training_physics42.log` |
| §IV-B | Regression data matrix full rank (σ_min = 0.0941); condition number 1.37e+05 | `python experiments/analyze_model.py` | `data/models/wrench_physics42_dt001.pkl`, `results/training/training_physics42.log` |
| §IV-B | Model file SHA-256 `090b8dd3106dcf3646dcf991fe23c030d3fe27435443a5f8fae19c1c377b2db4`; content digest (numerical fields) `f4e736d380d64679b991c4ae7b0d9a379d43fe8502c9a8b8c051dc1e90351cf4` | `python experiments/analyze_model.py` | `data/models/wrench_physics42_dt001.pkl`, `results/training/training_physics42.log` |
| Table II, §IV-C | Configuration A (hover-linear): c04, Np/Nc/b = 50/5/10; 5/25 candidates completed all screening episodes | `python experiments/analyze_selection.py` | `results/selection/*.json` |
| Table II, §IV-C | Configuration B (EDMDc): c22, Np/Nc/b = 100/20/5; 12/25 candidates completed all screening episodes | `python experiments/analyze_selection.py` | `results/selection/*.json` |
| Table III | Hover-linear MPC: position 6.958 mm, attitude 2.158 mrad, yaw 2.560 mrad (n = 200) | `python experiments/analyze_tracking.py` | `results/tracking/blind_test_episodes.json` |
| Table III | EDMDc-MPC: position 1.576 mm, attitude 4.462 mrad, yaw 7.142 mrad (n = 200) | `python experiments/analyze_tracking.py` | `results/tracking/blind_test_episodes.json` |
| Table III | Trajectory-linear MPC: position 0.086 mm, attitude 0.182 mrad, yaw 0.057 mrad (n = 200) | `python experiments/analyze_tracking.py` | `results/tracking/blind_test_episodes.json` |
| §V-A | EDMDc lower position RMSE in 191/200; paired difference -5.382 mm, 95% CI [-5.989, -4.772] mm | `python experiments/analyze_tracking.py` | `results/tracking/blind_test_episodes.json` |
| §V-B | Matched-configuration mean update time: Hover-linear MPC 2.599 ms, EDMDc-MPC 2.910 ms, Trajectory-linear MPC 11.640 ms | `python experiments/analyze_timing.py` | `results/timing/*` |
| §V-B, Fig. 3 | Four bundles, mean (P95, max): Linear-A 1.559 (1.987, 5.167) ms; EDMDc-A 1.634 (2.089, 6.853) ms; Linear-B 3.563 (3.954, 9.080) ms; EDMDc-B 4.190 (5.209, 9.862) ms; updates over 10 ms: 0 | `python experiments/analyze_timing.py` | `results/timing/*` |
| Table IV | Linear-A, Oracle: 40/40 captures, 0 attitude aborts, success mean 5.8097 s, penalized mean 5.8097 s | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| Table IV | Linear-A, Current/CV: 33/40 captures, 0 attitude aborts, success mean 6.6655 s, penalized mean 8.9990 s | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| Table IV | Linear-A, 100-ms delay/CV: 32/40 captures, 0 attitude aborts, success mean 6.7888 s, penalized mean 9.4310 s | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| Table IV | Linear-A, 250-ms delay/CV: 31/40 captures, 0 attitude aborts, success mean 7.1416 s, penalized mean 10.0347 s | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| Table IV | EDMDc-B, Oracle: 40/40 captures, 0 attitude aborts, success mean 2.2335 s, penalized mean 2.2335 s | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| Table IV | EDMDc-B, Current/CV: 40/40 captures, 0 attitude aborts, success mean 2.2910 s, penalized mean 2.2910 s | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| Table IV | EDMDc-B, 100-ms delay/CV: 40/40 captures, 0 attitude aborts, success mean 2.3413 s, penalized mean 2.3413 s | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| Table IV | EDMDc-B, 250-ms delay/CV: 37/40 captures, 0 attitude aborts, success mean 2.5335 s, penalized mean 3.8435 s | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| §V-C | Current/CV joint captures n = 33: EDMDc-B − Linear-A -4.4491 s, 95% CI [-4.9612, -3.9836]; EDMDc-B earlier in 33/33 | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| §V-C | Current/CV failure-penalized difference -6.7080 s | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| §V-C | Helix captures: Linear-A 3/10, EDMDc-B 10/10 (other families: straight 10/10 vs 10/10, accelerating 10/10 vs 10/10, weaving 10/10 vs 10/10) | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| §V-D | 100-ms delay/CV joint captures n = 32: -4.5509 s, 95% CI [-5.2056, -3.9838]; EDMDc-B earlier in 32/32 | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| §V-D | 250-ms delay/CV joint captures n = 31: -4.7255 s, 95% CI [-5.7165, -3.9687]; EDMDc-B earlier in 31/31 | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| §V-E | Oracle: EDMDc-B earlier in 40/40 | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| §V-E | EDMDc-B Current/CV − oracle 0.0575 s, 95% CI [-0.0115, 0.1350] | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| Table V | Hover-linear, config A: 40/40 captures, 0 attitude aborts, mean score 5.8097 s | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| Table V | EDMDc, config A: 40/40 captures, 0 attitude aborts, mean score 5.9505 s | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| Table V | Hover-linear, config B: 2/40 captures, 38 attitude aborts, mean score 19.1015 s | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| Table V | EDMDc, config B: 40/40 captures, 0 attitude aborts, mean score 2.2335 s | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| §V-E | Config A, EDMDc − hover-linear 0.14075 s, 95% CI [0.09525, 0.18425] (paper: 0.1408, [0.0953, 0.1843]); EDMDc later in 34/40 | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| §V-E | Causal crossed captures (CV/100 ms/250 ms): EDMDc-A 33/32/31 vs Linear-A 33/32/31; Linear-B 2/2/1 vs EDMDc-B 40/40/37 | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| §II-A (context) | EDMDc-B Current/CV peak tilt: mean 58.75°, max 70.73° (abort limit 80°) | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| §IV-E | Initial separation 6.4–16.7 m (mean 12.5 m) | `python experiments/analyze_interception.py` | `results/interception/episodes.json` |
| §IV-B | Retraining reproduces the content digest above (all numerical fields bitwise identical); model file SHA-256 is `090b8dd3…` when trained from the `.pkl` and `78822050…` from the `.npz`, because the stored input file name differs (see `DATA.md`) | `python experiments/train_edmdc.py --data <dataset> --output <dir>` (needs the training dataset, see `DATA.md`) | dataset, see `DATA.md` |
| (supplementary) | Held-out open-loop prediction RMSE | `python experiments/evaluate_prediction.py --data <dataset>` (needs the training dataset, see `DATA.md`) | `results/prediction/heldout_rolling_aggregate.csv` |
<!-- CLAIM-TABLE:END -->

Only the last two rows need the training dataset (see `DATA.md`).

## About

This repository contains only what is needed to reproduce the paper's results: a 42-observable,
wrench-input EDMDc predictor of the full 12-state quadrotor, embedded in a constrained MPC and
compared with a yaw-scheduled hover-linear MPC baseline on the same nonlinear plant (RK4, 100 Hz).

## Installation

Python 3.10 was used for all results (3.10.11, Windows 11).

```bash
conda create -n edmdc python=3.10
conda activate edmdc
pip install -r requirements.txt
```

The pinned versions (`numpy 1.26.4`, `scipy 1.14.1`, `osqp 1.0.5`, …) are the ones used in the paper.
Other versions may change results in the last digits.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Simulator, plant, allocation, EDMDc identification, and MPC code. These files are **byte-identical** to the versions used for the paper (SHA-256 in `PROVENANCE.md`); do not edit them if you want to reproduce exact numbers. |
| `experiments/` | Command-line entry points for every experiment and analysis (below). |
| `configs/` | Frozen controller configurations, the 25 search candidates, scenario seed lists, and training settings. |
| `data/models/` | The EDMDc model used in the paper (`wrench_physics42_dt001.pkl`, 41 KB). |
| `results/` | Frozen per-episode result records behind every table, plus the small inputs the figures need. |
| `figures/` | Figure scripts (Figs. 2–4) and their outputs; `figures/paper/` holds the EPS files of Figs. 1–4 as submitted (Fig. 1 is a drawing with no generating script). |
| `provenance/` | Unmodified original scripts and protocol files, for audit only. They contain the authors' original absolute paths and are not meant to be run from this repository. |
| `DATA.md` | Training dataset (not in git; published on Zenodo as a 540 MB `.npz`), its structure, and how it was generated. |

## Reproducing the paper

Every command below runs from the repository root. Analysis scripts read the frozen records in
`results/` by default; pass the `--episodes`/`--results` output of a rerun to analyze your own runs.

| Paper item | Re-run | Analyze |
|---|---|---|
| EDMDc model (Sec. IV-A/B: λ = 0, 2,051,658 transitions) | `python experiments/train_edmdc.py --data <dataset.npz> --output results_local/training` | prints the model SHA-256 and compares it with the paper model |
| Held-out prediction RMSE (optional, not tabulated in the paper) | `python experiments/evaluate_prediction.py --data <dataset.npz>` | printed |
| Table III, Fig. 2 (matched tracking, 200 scenarios) | `python experiments/tracking.py --output results_local/tracking` | `python experiments/analyze_tracking.py [--episodes results_local/tracking/episodes.json]` |
| Controller selection (Sec. IV-C, Table II) | `python experiments/selection.py --output results_local/selection` | `python experiments/analyze_selection.py [--results results_local/selection]` |
| Tables IV and V, delay study, crossed study, Fig. 4 | `python experiments/interception.py --output results_local/interception` | `python experiments/analyze_interception.py [--episodes results_local/interception/episodes.json]` |
| Fig. 3 (controller-update timing) | `python experiments/timing.py --output results_local/timing` | printed / `timing_summary.json` |
| Figures 2, 3, 4 | — | `python figures/fig2_tracking.py`, `python figures/fig3_timing.py`, `python figures/fig4_interception.py` |

Useful options: `--workers N` (parallel episodes), `--seeds …`/`--cells …`/`--modes …` (interception
subsets), `--families …`/`--replicates …` (tracking subsets), `--candidates …` (selection subsets),
`--save-traces` (interception state traces).

Quick check (about a minute): reproduce two Current/CV episodes shown in Fig. 4.

```bash
python experiments/interception.py --cells Linear-A EDMDc-B --modes cv --seeds 3103001001 --output results_local/quick
# expected: Linear-A qualifies at 5.79 s, EDMDc-B at 2.13 s
```

## What was verified for this release

Run from this repository with the pinned environment:

- **Analyses** reproduce every number in Tables II–V and the Section V text from the frozen records,
  including the bootstrap intervals (e.g. Current/CV joint-capture difference −4.4491 s, 95% CI [−4.9612, −3.9836] s).
- **Interception**: re-simulated episodes are bit-for-bit identical to the paper's saved traces
  (Current/CV Linear-A and EDMDc-B, seed 3103001001; oracle crossed EDMDc-A and Linear-B, seed 3103002000).
- **Tracking**: the hover-linear and EDMDc re-runs are bit-identical. The trajectory-linear comparator
  differs at the 1e-13 relative level due to floating-point non-determinism, which was already recorded
  for the original study.
- **Selection**: all 48 screening episodes of candidates c04 and c22 reproduce exactly.
- **Training**: retraining reproduces every numerical field of the paper model bitwise (content
  digest `f4e736d380d64679b991c4ae7b0d9a379d43fe8502c9a8b8c051dc1e90351cf4`), from both the original
  `.pkl` and the published `.npz`. The model file hash additionally depends on the stored input file
  name; see `DATA.md`.
- **Prediction**: held-out RMSE matches the frozen evaluation exactly.
- **Figures**: Figs. 2–4 regenerated from `results/` are pixel-identical to the paper's renderings.

## Notes and limitations

- **Interception scenarios** are regenerated from their seeds; every reference trajectory is checked
  against the frozen hash in `configs/interception_scenarios.json`.
- **Timing** numbers depend on hardware. The paper's numbers were measured on an AMD Ryzen 9 9950X
  with one worker on an otherwise idle machine; run `timing.py` the same way.
- **Selection** episodes use the known-future (scripted) target reference of the original selection
  campaign; the evaluation in Tables IV–V uses the causal Current/CV and delayed references, plus the
  oracle reference for comparison.
- **Held-out runs** (indices 39, 59, 129, 155, 210, 356–359) had been inspected in earlier research
  before the final model was trained; they are not a never-seen test set.
- **Training data** generation scripts are included in `src/`, but three of them changed after the dataset
  was generated, so regenerating the dataset is not guaranteed to be byte-identical. The published dataset
  (see `DATA.md`) is the authoritative input.

## License

MIT — see `LICENSE`. Citation metadata: `CITATION.cff`.
