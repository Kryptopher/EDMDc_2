# Current EDMD Workflow

This folder uses Darren's current PX4-like simulation/controller stack.

## ACC revision: frozen architecture

The ACC revision uses the 100 Hz **outer-command** EDMDc model, with physical
input `[thrust, phi_des, theta_des, psi_des]`, followed by the shared cascaded
attitude/rate controller and motor allocator. Wrench-model studies remain
diagnostic ablations and are not the paper's controller architecture.

The current fair tracking confirmation is
`artifacts/acc_balanced_high_tilt_v1/tracking_physics42_confirmation_fresh_n40`.
On 40 unseen 1.75x trajectories, position RMSE is 0.097 m for EDMDc-MPC,
0.774 m for reactive PID, and 1.140 m for the unguarded yaw-scheduled
hover-linear MPC. The linear controller has the best median (0.027 m) but an
11.637 m worst-case tail; EDMDc has 0.310 m P90 and 0.507 m worst-case error.
There are zero solver failures and zero allocator-altered commands. These
results supersede the historical diagnostic comparisons later in this file.

Simulation state:

```text
[x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
```

Logged control input:

```text
[thrust, tau_roll, tau_pitch, tau_yaw]
```

`U` is the wrench actually applied to the plant after motor allocation and
per-motor thrust limits. `U_requested` is saved alongside it for actuator
diagnostics. Applied-wrench EDMDc remains the default, so every transition in
that model uses the input the plant actually received. Set
`EDMDC_INPUT_SOURCE=outer_command` to train the matched outer-command model.

The simulator also records the original paper's outer-loop interface, extended
with yaw:

```text
U_outer = [thrust, phi_des, theta_des, psi_des]
```

All three arrays are aligned with the state log as `(x_k, u_k, x_{k+1})`:

- `U`: applied motor-feasible wrench (the current EDMDc training input)
- `U_requested`: wrench before motor allocation, for actuator diagnostics
- `U_outer`: desired-attitude command given to the inner attitude/rate loop

This dual-input schema permits a controlled input/state ablation between the
current wrench model, a 10-state/3-command reduction, and a true
12-state/4-command yaw extension on the same yaw-aware flights. A strict
reproduction of the paper's fixed-yaw assumption still requires a separately
generated fixed-yaw dataset.
New files declare `schema_version="yaw_dual_input_v1"`; `mix_traj.py` rejects
mixed old/new family files.

The yaw channel is applied yaw torque, not a desired yaw angle. Reference yaw
and yaw rate remain in each trajectory dictionary. Path simulations start at
their reference heading and cap the heading rate at 0.8 rad/s, avoiding an
artificial, infeasible yaw step at startup.

## Regenerate EDMD Data

```bash
python parallel_sim.py --workers 4
python mix_traj.py
python EDMDc_training.py
python compare_three.py
python compare_mpc.py
python Intercept_comparison.py
```

The default `paper` profile deliberately matches the old paper's data regime:
100-second runs at 100 Hz, with 50 helix, 50 figure-eight, 50 Lissajous,
50 waypoint, 30 hover-excitation, and 70 PRBS runs (300 total). It preserves
the original deterministic seed convention and the original five held-out test
runs. EDMDc then uses the original 0.1 s identification sample time and 2 s
MPC horizon by default, while retaining the yaw-aware state and applied-wrench
data contract.

### ACC-balanced 60-second dataset

The optional `acc_balanced` profile is the revised identification experiment;
the unchanged `paper` profile remains the exact baseline. It keeps the same
50/50/50/50/30/70 family counts, seed convention, run ordering, and default
held-out indices, but uses 60-second runs (1.8 million native 100 Hz samples).
Parametric paths use a fixed 5-second ramp instead of spending 30% of the run
accelerating and 30% decelerating. Helix cruise speeds are sampled from
1.0--2.5 m/s. The faster parametric families are uniformly path-scaled only
when needed to enforce 5 m/s and 6 m/s^2 reference limits. In profile version
`acc_balanced_waypoint_v2`, waypoint paths use a natural cubic spline evaluated
analytically in physical time instead of a short sample-based Gaussian filter.
They are limited to 3.5 m/s and 4.0 m/s^2, leaving feedback margin below the
cascade's 4 m/s and 5 m/s^2 XY limits. Waypoint and hover references also
receive a derivative-consistent 5-second startup ramp.

Every generated family file records `dataset_profile`, the complete
`trajectory_profile_config`, deterministic `run_seeds`, and per-run reference
coverage metrics. It also records per-run closed-loop tracking errors,
finite-state checks, attitude extrema, and requested-versus-applied allocation
diagnostics in `simulation_metrics`. The mixer rejects files from different
profiles or profile configurations. The reference-only audit writes both a
per-run CSV and a JSON summary:

```powershell
conda activate drone
python audit_trajectory_profile.py --profile acc_balanced `
  --output-dir artifacts\acc_balanced_reference_audit
```

Generate, mix, and train the full revised dataset from this folder with:

```powershell
conda activate drone
python parallel_sim.py --profile acc_balanced --workers 4 `
  --output-dir artifacts\acc_balanced_data
python mix_traj.py --profile acc_balanced `
  --input-dir artifacts\acc_balanced_data

$env:EDMDC_DATA_FILE = "artifacts/acc_balanced_data/runs_mixed_n300.pkl"
$env:EDMDC_MODEL_FILE = "artifacts/acc_balanced_data/edmdc_model_yaw_wrench_dt001.pkl"
$env:EDMDC_DT = "0.01"
$env:MPLBACKEND = "Agg"
python EDMDc_training.py
```

Use the existing default validation indices `[38, 58, 128, 154, 209]` and
held-out test indices `[39, 59, 129, 155, 210]`; their family mapping is
unchanged. Do not combine this dataset with the 100-second paper files.

To replace only the waypoint family in an existing ACC-balanced v1 dataset
while leaving the other 250 runs and all indices unchanged:

```powershell
python -c 'from parallel_sim import save_runs; save_runs(4, 50, "artifacts/acc_balanced_waypoint_v2/runs_traj4_n50.pkl", duration=60, workers=4, profile="acc_balanced")'
python replace_waypoint_family.py `
  --base artifacts\acc_balanced_data\runs_mixed_n300.pkl `
  --waypoint artifacts\acc_balanced_waypoint_v2\runs_traj4_n50.pkl `
  --output artifacts\acc_balanced_waypoint_v2\runs_mixed_n300_waypoint_v2.pkl
```

The replacement dataset records the original source list, replacement file,
replaced indices, seeds, and profile version in `waypoint_replacement`.

### ACC-balanced observable/family campaign

`acc_edmdc_overnight.py` runs a resumable identification-only campaign at
`dt=0.01 s`. It searches all 512 observable-group dictionaries across 34
named family-weight patterns, 24 regularization values, four early-transient
weights, and constrained/unconstrained kinematic rows. The original validation
and test indices remain present; indices 268 and 269 add a supplementary,
disjoint yaw-PRBS validation/test pair. Test runs are locked until the
validation ranking and selected model are frozen.

The campaign keeps fixed training-split scalers across family patterns so the
family experiment isolates regression weighting rather than normalization.
Stage 2 performs true rolling 1/2/5-second validation rollouts for the best
candidate from every observable dictionary and every family pattern. It then
exports the selected compact model and the matched best 56-observable model,
runs frozen six-family and legacy-five held-out audits, and writes plots,
hashes, per-family tables, and `summary.md`.

The active 2026-08-19 campaign is in:

```text
artifacts/acc_edmdc_observable_family_2026-08-19
```

Monitor it without interrupting the run:

```powershell
Get-Content artifacts\acc_edmdc_observable_family_2026-08-19\status.json
Get-Content artifacts\acc_edmdc_observable_family_2026-08-19\progress.json
```

The orchestrator and Stage-2 rollout files are resumable. Re-running the same
command after an interruption continues from completed artifacts rather than
discarding the campaign.

### Selected-observable 0.1-second model

`train_selected_observables_dt010.py` freezes the selected 35-observable
dictionary from the 0.01-second campaign and takes every tenth state. Its
publication default identifies
`(x[10k], mean(u[10k:10(k+1)]), x[10(k+1)])`, because the 100 Hz cascaded
controller changes the applied wrench inside each 0.1-second transition. The
old instantaneous-input construction remains available with
`--input-downsampling decimate` as an explicit ablation. The script searches
family weights, transient weights,
regularization, and kinematic-row enforcement using validation runs before it
exports validation-tuned and exactly hyperparameter-matched models:

```powershell
python train_selected_observables_dt010.py `
  --data artifacts\acc_balanced_data\runs_mixed_n300.pkl `
  --source-model artifacts\acc_edmdc_observable_family_2026-08-19\models\selected_observable_family.pkl `
  --output-dir artifacts\acc_edmdc_selected35_dt010_interval_mean `
  --input-downsampling interval_mean
```

The frozen held-out 0.01/0.1-second rollout comparison and its PNG/PDF plots
are in `artifacts/acc_edmdc_selected35_dt010/heldout_dt_comparison`.

The corresponding full-episode controller comparison is in
`artifacts/acc_edmdc_dt_tracking`. It uses a common 2-second prediction
horizon, 0.2-second control horizon, and 0.1-second offboard period for the
0.01- and 0.1-second EDMDc/linear MPC pairs. PID and NMPC are common baselines
and are run once per split. The controller configuration was frozen from the
six validation runs before evaluating the six test runs. Rebuild the combined
tables, plots, hashes, and summary with:

```powershell
python analyze_dt_tracking_results.py `
  --data artifacts\acc_balanced_data\runs_mixed_n300.pkl `
  --input-dir artifacts\acc_edmdc_dt_tracking `
  --output-dir artifacts\acc_edmdc_dt_tracking\analysis
```

The matched sample-time interception comparison is in
`artifacts/acc_edmdc_dt_interception`. Validation and test each contain three
deterministic straight, accelerating, helical, and weaving targets. The test
split was evaluated only after the 2-second horizon, 1.5-second target lead,
0.1-second offboard period, and controller weights were frozen. Rebuild its
combined tables and figures with:

```powershell
python analyze_dt_interception_results.py `
  --data artifacts\acc_balanced_data\runs_mixed_n300.pkl `
  --input-dir artifacts\acc_edmdc_dt_interception `
  --output-dir artifacts\acc_edmdc_dt_interception\analysis
```

Regenerate all data and models after changes to the allocator or simulator.
`mix_traj.py` and `EDMDc_training.py` intentionally reject legacy datasets
that do not declare `input_type="applied_wrench"`. The mixer additionally
requires `U_requested`, `U_outer`, and the dual-input schema version.

The PRBS count and seed range are kept from the old code, but its implementation
is intentionally yaw-aware: it excites a bounded yaw-rate reference through the
same controller, so `U` remains a physical applied wrench rather than a log of
desired roll/pitch angles.

For a first reproducible validation set with disjoint train/validation/test
trajectory runs, use the simulator's intended 45-second duration:

```bash
mkdir -p artifacts/yaw_validation
python parallel_sim.py --profile compact --runs-per-family 4 --prbs-runs 4 --duration 45 --output-dir artifacts/yaw_validation
python mix_traj.py --profile compact --runs-per-family 4 --prbs-runs 4 --input-dir artifacts/yaw_validation
EDMDC_DATA_FILE=artifacts/yaw_validation/runs_mixed_n16.pkl \\
EDMDC_MODEL_FILE=artifacts/yaw_validation/edmdc_model_yaw_wrench.pkl \\
EDMDC_VALIDATION_INDICES=2,6,10 EDMDC_TEST_INDICES=3,7,11 \\
MPLBACKEND=Agg python EDMDc_training.py
```

For a quick deterministic smoke test that does not depend on Git LFS assets:

```bash
mkdir -p artifacts/smoke
python parallel_sim.py --profile compact --runs-per-family 3 --prbs-runs 3 --duration 8 --output-dir artifacts/smoke
python mix_traj.py --profile compact --runs-per-family 3 --prbs-runs 3 --input-dir artifacts/smoke
EDMDC_DATA_FILE=artifacts/smoke/runs_mixed_n12.pkl \\
EDMDC_MODEL_FILE=artifacts/smoke/edmdc_model_yaw_wrench.pkl \\
EDMDC_VALIDATION_INDICES=1,4,7 EDMDC_TEST_INDICES=2,5,8 \\
MPLBACKEND=Agg python EDMDc_training.py
EDMDC_DATA_FILE=artifacts/smoke/runs_mixed_n12.pkl \\
EDMDC_MODEL_FILE=artifacts/smoke/edmdc_model_yaw_wrench.pkl \\
EDMDC_TEST_INDICES=2,5,8 MPLBACKEND=Agg python compare_three.py
EDMDC_DATA_FILE=artifacts/smoke/runs_mixed_n12.pkl \\
EDMDC_MODEL_FILE=artifacts/smoke/edmdc_model_yaw_wrench.pkl \\
MPLBACKEND=Agg python compare_mpc.py --test-indices 2,5,8 --steps 100
EDMDC_DATA_FILE=artifacts/smoke/runs_mixed_n12.pkl \\
EDMDC_MODEL_FILE=artifacts/smoke/edmdc_model_yaw_wrench.pkl \\
MPLBACKEND=Agg python Intercept_comparison.py --cases straight --tmax 1
```

The shortened smoke trajectory is a software integration check only: reducing
the duration while retaining the full-scale path makes that reference much more
aggressive than the intended experiment. Do not use its model metrics in the
paper or to tune the controller.

Install dependencies with `python -m pip install -r requirements.txt` from the
repository root. The training script uses separate deterministic validation and
test runs: lambda selection uses validation only, balances rolling position,
rolling velocity, yaw, and yaw-rate error, and the final diagnostics use the
untouched test split.

`compare_mpc.py` applies the full optimized correction for both EDMDc-MPC and
linear MPC. This keeps the comparison attributable to their prediction models,
rather than controller-specific post-processing.

The simulator and inner controller run at 0.01 s. Identified-model time steps
are read from each model artifact, and prediction/controller comparisons honor
the artifact's declared input-downsampling method. In particular, the
publication 0.1-second model and its matched linear baseline both use the
interval-mean applied wrench; direct every-tenth decimation is a separately
labeled ablation. MPC horizon lengths are computed from the requested duration, so a
2-second horizon is 200 model steps at 0.01 s and 20 model steps at 0.1 s.
The PX4-like nominal controller always continues updating at 0.01 s.

Optional `--edmd-correction-blend` and `--linear-correction-blend` arguments
support validation-only hybrid-controller studies. Their paper-comparison
defaults remain 1.0 so both models apply the full optimized correction.

The MPC also uses the same wrench-to-motor allocation matrix as the plant. Its
per-motor force inequalities prevent the optimizer from requesting a wrench
that would be altered by motor clipping.

The shared MPC module limits BLAS to one thread by default (`EDMDC_BLAS_THREADS=1`)
because its small, repeatedly assembled matrices otherwise create timing jitter.
Set the variable to `0` to leave the process's BLAS configuration unchanged.

Outputs:

```text
runs_traj1_n50.pkl ... runs_traj4_n50.pkl
runs_traj5_n30.pkl
runs_prbs_n70.pkl
runs_mixed_n300.pkl
edmdc_model_yaw_wrench.pkl
```

`edmdc_mpc.py` contains the shared state/input lifting utilities used by the
trained model.

`compare_three.py` visualizes the current PID/PX4 simulation trace against EDMDc
and a linear least-squares baseline using the same logged wrench inputs. The
model traces use 1-second reset windows so the plot shows prediction quality
over the full trajectory without one long open-loop drift hiding the path.
`final_comparison.py` calls the same comparison entrypoint.

`compare_mpc.py` runs closed-loop PID/PX4, EDMD-MPC, and linear-MPC through the
current yaw-wrench plant. By default it evaluates the full trajectory; use a
command like `python compare_mpc.py --steps 1500` for a shorter development run.
The restored MPC entrypoints `final_comparison.py`, `tunerfull.py`,
`Intercept_comparison.py`, `intercept_comparison_w_pred.py`, and
`Real data edmdc mpc.py` now call this same current-simulator MPC comparison.

`Intercept_comparison.py` runs and logs moving-target interception benchmarks
with the current yaw-wrench simulator. `intercept_comparison_w_pred.py` calls
the same updated interception workflow.

## Four-controller ACC benchmark

`controller_benchmark.py` is the definitive full-episode trajectory comparison
for the revised controller stack. It compares the reactive cascaded PID with
matched linear MPC, compact EDMDc-MPC, and genuine nonlinear direct-shooting
MPC. The three predictive controllers command the complete motor-feasible
wrench directly around the fixed hover wrench; they do not use a cascaded PID
nominal. Only the PID baseline uses the cascade. This keeps the controller
architecture consistent with the applied-wrench identification contract.

Desired-attitude models are a separate architecture: their command is
`[thrust, phi_des, theta_des, psi_des]` and must pass through the cascaded
attitude/rate controller. They must not be evaluated with the direct-wrench
runner, which rejects checkpoints not declaring `input_type="applied_wrench"`.

The validation-only 0.01-second direct-wrench tuning campaign is stored in
`artifacts/direct_wrench_dt001_tuning_v2` (global search) and
`artifacts/direct_wrench_dt001_tuning_v3_local` (long-episode local search).
The frozen validation winner is `direct_wrench_dt001_tuned_v2_config.json`.
It uses a 0.25-second prediction/control horizon and 0.01-second updates. The
held-out test split was intentionally not evaluated because 30-second
validation still shows lateral drift on figure-eight, Lissajous, and waypoint
tracking; tuning weights alone is therefore not sufficient for a paper claim.

The MPC weights are specified in physical units and transformed internally to
the standardized coordinates used by the learned and linear models. Reference
roll/pitch and body rates are derived from desired acceleration and yaw instead
of being incorrectly set to zero on curved paths. Compact EDMD models are
solved in their declared active-observable space.

The earlier development comparison used a 2 s prediction horizon, 0.2 s
control horizon, a 10 Hz offboard update, and a shared input-cost multiplier
of 0.2. Publication gains are now selected by the validation-only campaign
described below:

```bash
python controller_benchmark.py --split validation --steps 0 \
  --horizon-seconds 2 --control-horizon-seconds 0.2 \
  --offboard-period 0.1 --r-scale 0.2 \
  --nmpc-dt 0.1 --nmpc-control-steps 3 --nmpc-max-iterations 2
```

The current applied-wrench model remains the default. To train the matched
12-state outer-command model from the same mixed dataset, set:

```bash
EDMDC_INPUT_SOURCE=outer_command \
EDMDC_DATA_FILE=artifacts/acc_paper_data_v3/runs_mixed_n300.pkl \
EDMDC_MODEL_FILE=artifacts/acc_paper_data_v3/edmdc_model_yaw_outer.pkl \
EDMDC_DT=0.01 MPLBACKEND=Agg python EDMDc_training.py
```

That command retains the four raw outer commands as the controlled baseline.
To test whether exposing the inner cascade's actual nonlinear drivers improves
velocity prediction, train a separate checkpoint with:

```bash
EDMDC_INPUT_SOURCE=outer_command \
EDMDC_OUTER_INPUT_LIFT=attitude_error \
EDMDC_DATA_FILE=artifacts/acc_paper_data_v3/runs_mixed_n300.pkl \
EDMDC_MODEL_FILE=artifacts/acc_paper_data_v3/edmdc_model_yaw_outer_attitude_error.pkl \
EDMDC_DT=0.01 MPLBACKEND=Agg python EDMDc_training.py
```

This 16-column lift contains the four raw commands, desired world-frame thrust
vector, sine/cosine of desired roll/pitch/yaw, and wrapped attitude errors.
The checkpoint records the exact lift type and labels. Keep both checkpoints:
the raw model is the ablation baseline and the attitude-error model is accepted
only if it improves the frozen validation velocity metrics without degrading
the untouched held-out position/yaw rollouts.

### Outer-command EDMDc-MPC

`outer_command_mpc.py` is the matched controller for the nonlinear
attitude-error checkpoint. It uses sequential QP linearization along the
predicted attitude trajectory. The learned model has 16 lifted input features,
but the optimizer always has exactly four physical decisions:
`[thrust, phi_des, theta_des, psi_des]`. Those commands pass through
`QuadPX4LikeController.fct_attitude_step`, retaining the simulated Pixhawk-like
attitude/rate loop, motor allocation, and plant boundary.

At the default 0.01 s model time step and 0.1 s offboard period, each optimized
move is held for ten model steps. The default 2 s prediction horizon, 0.5 s
control horizon, and two SQP iterations can be smoke-tested with:

```powershell
python outer_command_mpc.py `
  --data artifacts\acc_balanced_waypoint_v2\runs_mixed_n300_waypoint_v2.pkl `
  --model artifacts\acc_balanced_waypoint_v2\edmdc_outer_attitude_error_dt001.pkl `
  --output-dir artifacts\acc_balanced_waypoint_v2\outer_mpc_smoke `
  --indices 39,155 --steps 300
```

Yaw-rate reference is known trajectory feedforward to the retained inner loop;
it is not an independently optimized EDMDc input. Reports therefore continue
to declare four optimized outer commands and should separately disclose this
reference feedforward convention.

`compare_three.py` reads the model metadata and automatically selects `U` or
`U_outer`, so it cannot accidentally evaluate an outer-command model using
wrench data.

`Intercept_comparison.py` now provides analytic target acceleration and yaw
rate for straight, accelerating, helical, and weaving targets. Predictive
controllers receive a frozen 1.5 s target forecast; reactive PID sees only the
current target. Validation and test scenarios use disjoint deterministic seed
ranges. The final held-out command is:

```bash
python Intercept_comparison.py --split test --runs-per-family 3 --tmax 10 \
  --horizon-seconds 2 --offboard-period 0.1 --intercept-lead 1.5 \
  --r-scale 0.2 --nmpc-dt 0.1 --nmpc-control-steps 3 \
  --nmpc-max-iterations 2
```

The nonlinear MPC uses bounded rotor forces and accepts a candidate plan only
when it improves on the nominal cascade and stays within predicted attitude and
rate envelopes. Rejected plans fall back to the reactive cascade and are logged
as solver fallbacks. It is an accuracy/safety benchmark, not a real-time claim.

Final machine-readable tables, plots, hashes, and interpretation are in:

```text
artifacts/controller_revision/controller_revision_summary.md
artifacts/controller_revision/tracking_validation_final.csv
artifacts/controller_revision/tracking_test_final.csv
artifacts/controller_revision/interception_validation_final.csv
artifacts/controller_revision/interception_test_final.csv
artifacts/controller_revision/controller_revision_manifest.json
```

`interception_test_frozen.csv` is retained as the untouched raw held-out output;
`interception_test_final.csv` adds the frozen controller/model configuration to
each row and is the definitive analysis file.

## Frozen ACC publication campaign

`PUBLICATION_PROTOCOL.md` and `publication_protocol.json` define the current
confirmatory protocol. The primary 0.1 s model uses states at `10k` and the
mean motor-feasible applied wrench over the corresponding ten 0.01 s plant
intervals. Direct input decimation is an explicitly named ablation.

The first validation-only search is:

```powershell
& 'C:\Users\sanja\miniconda3\envs\drone\python.exe' publication_tuning.py `
  --model-file artifacts\acc_publication_v1\models_dt010_interval_mean\models\edmdc_selected35_dt010_tuned.pkl `
  --linear-model-file artifacts\acc_publication_v1\matched_linear.pkl `
  --output-dir artifacts\acc_publication_v1\tuning `
  --pid-candidates 32 --mpc-candidates 48 `
  --tracking-runs-per-family 3 --interception-runs-per-family 5 --workers 4
```

After that file creates `tuning/selected_validation_config.json`, the complete
restartable campaign is:

```powershell
& 'C:\Users\sanja\miniconda3\envs\drone\python.exe' run_publication_campaign.py `
  --output-dir artifacts\acc_publication_v1 --workers 4
```

The runner logs each subprocess under `artifacts/acc_publication_v1/logs`,
writes restart state to `campaign_status.json`, completes the preview and NMPC
validation searches, trains all observable/input/family/no-yaw ablations,
runs nominal and 500-scenario robustness tests, executes the 0.01-versus-0.1 s
controller comparison, and generates statistics plus PNG/PDF paper figures.
It refuses to open confirmatory controller tests until
`frozen_configuration.json` exists and passes source/model/protocol hash checks.

Long stages are restart-safe at their natural evidence boundary. NMPC tuning
and controller benchmarks atomically save every completed episode; observable
ablations save every completed model; unseen prediction generation saves every
trajectory family; and prediction evaluation saves every completed model.
Each partial artifact has a contract containing task/data/model hashes. An
identical command resumes missing work, while a changed duration, model,
scenario grid, or evaluation horizon is rejected instead of silently mixing
experiments.

For an unattended Windows run that is independent of the current terminal,
start the master hidden and redirect its console output:

```powershell
$project = (Get-Location).Path
$stdout = Join-Path $project 'artifacts\acc_publication_v1\logs\campaign_master.log'
$stderr = Join-Path $project 'artifacts\acc_publication_v1\logs\campaign_master.err.log'
$master = Start-Process `
  -FilePath 'C:\Users\sanja\miniconda3\envs\drone\python.exe' `
  -ArgumentList @('run_publication_campaign.py', '--output-dir',
    'artifacts\acc_publication_v1', '--workers', '4', '--nmpc-candidates', '18') `
  -WorkingDirectory $project -WindowStyle Hidden `
  -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
```

`finalize_publication_campaign.ps1` can watch that PID, restart the master from
its atomic checkpoints up to five times, and then run
`generate_acc_publication_report.py` plus
`audit_acc_publication_campaign.py`. The definitive completion files are:

```text
artifacts/acc_publication_v1/campaign_complete.json
artifacts/acc_publication_v1/ACC_PUBLICATION_EVIDENCE_REPORT.md
artifacts/acc_publication_v1/completion_audit.json
artifacts/acc_publication_v1/completion_audit.md
artifacts/acc_publication_v1/logs/finalize_campaign.log
```

NMPC timing is part of the evidence rather than hidden: any nonzero deadline
miss rate or 95th-percentile solve time above the 0.1 s offboard period labels
it an offline nonlinear oracle, not a real-time controller claim.

The new publication artifacts supersede the older
`artifacts/controller_revision` files for ACC claims; those older files remain
useful development evidence only.

## Yaw-aware 100 Hz outer-controller audit

The latest architecture-matched held-out comparison keeps the purely reactive
PID as a separately tuned baseline and compares it with the original
trajectory-feedforward PID, EDMDc-MPC, exact-model trajectory LTV-MPC, and
exact-model NMPC. All five use the same 100 Hz outer-command boundary and the
same inner attitude/rate cascade and allocator. See:

```text
artifacts/acc_balanced_waypoint_v2/controller_comparison_100hz_final/
```

The directory contains the full CSV, linear/log PNG and PDF plots, solve-time
summary, and `CONTROLLER_COMPARISON_100HZ.md`. The reactive PID gains were
selected using only validation indices and frozen in:

```text
artifacts/acc_balanced_waypoint_v2/reactive_pid_tuning_100hz/selected_config.json
```

`outer_ltv_mpc.py` is the yaw-aware linear baseline. Its main
`trajectory_ltv` mode linearizes the exact architecture along the future
reference; `hover_lti` is retained only as a fixed-linear-model ablation.
`outer_nmpc.py` uses the exact nonlinear architecture, but current timing
classifies it as an offline oracle. Install its optional dependency with
`python -m pip install -r requirements_nmpc.txt`.

The working fixed-linear baseline is implemented by
`train_yaw_aligned_linear_model.py` and `yaw_aligned_linear_mpc.py`. It uses a
single fixed error-state A matrix in reference-yaw-aligned coordinates and a
single causal hover B matrix; it does not relinearize online. Its frozen
validation-selected configuration and held-out comparison are in:

```text
artifacts/acc_balanced_waypoint_v2/yaw_aligned_linear_causal_frozen_config.json
artifacts/acc_balanced_waypoint_v2/controller_comparison_100hz_with_fixed_lti/
```

Do not use the earlier global inertial-coordinate LTI artifact as the main
linear baseline. Do not describe the working model as fully data-identified:
its A matrix is learned, while its B matrix is a one-time causal hover
finite-difference Jacobian.

## Locked nominal confirmation (post-freeze)

The primary nominal evidence now uses 25 fresh, post-freeze 60-second runs
(five unseen seeds for each trajectory family) at a 100 Hz controller rate.
The controller weights and trust regions were selected only on the preserved
validation indices before this dataset was generated. Dataset and frozen
configuration hashes are recorded in the manifest and generated report:

```text
artifacts/acc_balanced_waypoint_v2/locked_nominal_confirmation_n25.manifest.json
artifacts/acc_balanced_waypoint_v2/locked_nominal_analysis/LOCKED_NOMINAL_REPORT.md
artifacts/acc_balanced_waypoint_v2/locked_nominal_analysis/per_run_metrics.csv
artifacts/acc_balanced_waypoint_v2/locked_nominal_analysis/paired_differences.csv
artifacts/acc_balanced_waypoint_v2/locked_nominal_analysis/locked_nominal_rmse.pdf
artifacts/acc_balanced_waypoint_v2/locked_nominal_analysis/edmd_reproduction_check.json
```

This confirmation supersedes all previously viewed five-index comparisons for
nominal paper claims. Across matched runs, EDMDc-MPC has a statistically
supported velocity improvement over PID + feedforward, but its position
advantage is not statistically resolved and its yaw error is worse. Both the
fixed yaw-aligned LTI-MPC and exact trajectory LTV-MPC outperform EDMDc-MPC in
the deterministic matched-model simulation. Do not claim overall nominal
EDMDc superiority from these data.

All tested MPC controllers meet the 10 ms deadline with zero failed solves and
zero allocator-altered commands in this locked set. The very small linear-MPC
errors are ideal-simulation results, not expected flight-test accuracy. NMPC is
kept outside the primary comparison because the available converged setup uses
a shorter horizon and is not real-time at 100 Hz. Model-mismatch experiments
must use a new locked scenario manifest and must not overwrite these nominal
artifacts.

The EDMD portion was independently rerun after adding complete controller
configuration fields to every CSV row and NPZ artifact. The reproduced state,
wrench, and outer-command traces are bit-for-bit identical to the original 25
runs. `verify_locked_nominal_reproduction.py` performs the trace comparison and
checks every logged setting against the frozen validation-selected config.

## Hidden-plant model-mismatch study

`model_mismatch_protocol.json` defines three fixed parameter-only hidden plant
identities. The simulated cascaded controller, feedforward calculation, and
motor allocator retain nominal parameters; only the physical plant receives
the altered mass, principal inertias, linear/angular drag, and individual motor
effectiveness. This prevents true-parameter leakage through the shared plant
object used by older robustness code.

Generate a complete response dataset and identify EDMDc from one hidden plant:

```powershell
python generate_mismatched_identification_data.py `
  --plant-id heavy_asymmetric `
  --output-dir artifacts\model_mismatch_v1\heavy_asymmetric\identification `
  --workers 4

python train_mismatched_edmd.py `
  --data artifacts\model_mismatch_v1\heavy_asymmetric\identification\runs_mixed_n300.pkl `
  --output artifacts\model_mismatch_v1\heavy_asymmetric\edmdc_identified_dt001.pkl
```

Run the paired held-out comparison. The nominal and plant-identified EDMDc
models receive exactly the same hidden plant and references. Fixed LTI and LTV
retain nominal physics. All MPC weights, horizons, trust regions, and PID gains
remain frozen from the nominal validation stage:

```powershell
python evaluate_hidden_plant_mismatch.py `
  --plant-id heavy_asymmetric `
  --nominal-edmd-model artifacts\acc_balanced_waypoint_v2\edmdc_outer_attitude_error_dt001.pkl `
  --identified-edmd-model artifacts\model_mismatch_v1\heavy_asymmetric\edmdc_identified_dt001.pkl `
  --fixed-model artifacts\acc_balanced_waypoint_v2\yaw_aligned_linear_causal_dt001.pkl `
  --matched-config artifacts\acc_balanced_waypoint_v2\matched_mpc_tuning_v1\selected_config.json `
  --reactive-config artifacts\acc_balanced_waypoint_v2\reactive_pid_tuning_100hz\selected_config.json `
  --output artifacts\model_mismatch_v1\heavy_asymmetric\paired_tracking.csv `
  --workers 4

python analyze_hidden_plant_mismatch.py `
  artifacts\model_mismatch_v1\heavy_asymmetric\paired_tracking.csv `
  --output-dir artifacts\model_mismatch_v1\heavy_asymmetric\analysis
```

Repeat without changing the protocol for `light_low_drag` and
`yaw_inertia_motor`. Evaluation CSVs are restartable and protected by hashes of
the protocol, task contract, models, and frozen controller configurations.
Sensor noise, wind, and delays are excluded here and must be introduced only
in a subsequent, separately labeled experiment.

The first `heavy_asymmetric` run revealed that strict zero-shot MPC is not
trim-feasible: the hidden plant requires 39.026 N at hover versus 32.748 N
nominal, outside the frozen +/-3 N correction region. That result is retained
as `paired_tracking.csv`, but it is a trim-feasibility stress test rather than
a fair dynamic-model comparison.

The primary calibrated comparison is `paired_tracking_calibrated.csv`. Hover
thrust is estimated as the median of 65,838 low-motion samples from the
identification response—no true plant parameter is read. Plant-identified
EDMDc uses the full response dataset; the fixed LTI and LTV baselines receive
only this same scalar hover calibration while retaining nominal dynamics. Its
analysis is under `analysis_calibrated/`.

### Linear-MPC tuning and yaw-coordinate ablation

The validation-only fixed-zero-yaw search is stored in
`simple_hover_lti_tuning/`. Its frozen winner reduces held-out position RMSE
from 68.07 m to 21.30 m, but still diverges on four trajectory families. It has
zero failed solves and zero allocator clipping. This is retained as a useful
ablation, not the recommended linear baseline: a zero-yaw input matrix maps
roll and pitch into the wrong inertial axes as path yaw changes, and cost
weights cannot correct that structural coordinate error.

The final fair physics-only baseline is a level-hover translational model whose
roll/pitch input map is rotated using known reference yaw. Yaw itself is pure
feedforward: the MPC optimizes only thrust, desired roll, and desired pitch,
while desired yaw passes unchanged to the inner attitude loop. It uses no
trajectory-identification data, nonlinear prediction, velocity/acceleration
operating-point linearization, yaw-rate linearization, or online state-dependent
relinearization. The validation search and frozen configuration are in
`hover_linear_yaw_feedforward_tuning/`; held-out results and analysis are in
`analysis_primary_hover_linear_yaw_feedforward/`.

Across 50 untouched heavy-asymmetric test trajectories, mean position RMSE is
0.001289 m for hover-linear MPC with yaw feedforward, 0.043676 m for
plant-identified EDMDc-MPC, and 0.384218 m for tuned reactive PID. Corresponding
velocity RMSEs are 0.001557, 0.023945, and 0.147818 m/s; yaw RMSEs are 0.011143,
0.013112, and 0.069421 rad. Linear MPC has statistically supported position and
velocity advantages in the paired analysis. Its QP averages 0.693 ms (0.829 ms
episode-average p95), with zero failed solves and zero allocator-altered
commands.

These ideal deterministic results do not support a claim that EDMDc globally
outperforms a properly yaw-aware physics linear MPC. Noise, delay, wind, stronger
unmodeled effects, interception, and flight data must be evaluated separately;
the result must not be weakened by presenting the coordinate-mismatched
fixed-zero-yaw controller as the only linear baseline.

### Frozen-controller yaw/inertia/motor mismatch

The second locked plant, `yaw_inertia_motor`, was evaluated without retuning
the controller settings selected above. A separate 300-run response dataset
was generated and used to identify EDMDc; all physics-controller dynamics and
PID gains remained frozen. The physical plant has altered principal inertias,
30% greater angular damping, 5% greater mass, and asymmetric motor
effectiveness. The shared hover trim (36.134 N) was estimated only from the
new response data.

Across 50 held-out trajectories, hover-linear MPC with yaw feedforward obtains
0.001093 m position, 0.001559 m/s velocity, and 0.018635 rad yaw RMSE.
Plant-identified EDMDc-MPC obtains 0.027144 m, 0.015355 m/s, and 0.020123 rad;
reactive PID obtains 0.375641 m, 0.143253 m/s, and 0.073809 rad. EDMDc improves
strongly over PID, but the frozen linear MPC retains statistically supported
position and velocity advantages. The EDMDc/linear yaw difference is not
statistically resolved. Results are under
`model_mismatch_v1/yaw_inertia_motor/analysis_primary_hover_linear_yaw_feedforward/`.

This mismatch is primarily inside the attitude/motor subsystem and is strongly
rejected by the shared inner cascaded PID. A useful next mismatch phase must
therefore perturb outer-loop-visible behavior--for example quadratic drag,
wind, delay, state-estimation noise, or an actuator time constant--rather than
only increasing parameter errors already hidden by the inner loop.

### Information-matched MPC correction

The millimetre-level feedforward linear-MPC result above is retained as a
physics-informed ablation, not the primary simple baseline. That controller
received acceleration-derived inverse-dynamics thrust/attitude, dynamically
consistent desired attitude/body-rate states, and an affine reference defect.
The information-matched comparison removes direct acceleration-derived
thrust/attitude commands and dynamically consistent attitude/body-rate target
states from both MPCs. Each receives position/velocity preview and the nominal
command `[response_trim_thrust, 0, 0, yaw_reference]`; yaw is feedforward-only
and attitude/body-rate costs are zero. A later causality audit found that the
reported hover-linear error model still includes its affine reference defect,
computed using the next position/velocity reference. Thus it implicitly
encodes reference acceleration even though acceleration is not passed as a
controller input. Do not describe this result as having no affine defect.

Both MPCs were tuned on fresh validation trajectories over identical candidate
domains, then frozen and evaluated on 50 new paired confirmation trajectories.
Mean position RMSE is 0.028744 m for hover-linear MPC, 0.052472 m for EDMDc-MPC,
and 0.389843 m for reactive PID. The paired EDMDc-minus-linear position
difference is +0.023727 m with bootstrap interval [-0.015190, +0.044850], so it
is not statistically resolved. Linear MPC has a resolved velocity advantage;
EDMDc retains resolved improvements over reactive PID in all three metrics.

Linear MPC wins 49/50 position pairs but has one 1.379 m aggressive
hover-excitation failure; EDMDc reduces that episode to 0.502 m. The current
EDMDc identification data contain 99.9th-percentile roll/pitch attitude errors
of only 0.113/0.078 rad, while no-feedforward EDMDc-MPC requests up to
0.554/0.514 rad. This behavior-policy/control-policy distribution shift is the
next modeling issue to fix with bounded roll/pitch/thrust excitation data. See
`analysis_information_matched/INFORMATION_MATCHED_REPORT.md`.

### Linear-MPC causality and reference-information audit

The 50 confirmation trajectories were rerun with the frozen hover-linear MPC
under three reference contracts. Reported preview plus affine defect gives
0.005326 m position component RMSE. Keeping preview but zeroing the defect gives
0.012869 m; additionally replacing the horizon by the current reference gives
0.012861 m. All variants have zero failed solves and zero allocator clipping.
The mean one-step residual of the hover-linear model against the hidden plant
is 1.55e-6 m in position and 3.10e-4 m/s in velocity for the reported mode.

Therefore the affine defect is responsible for much of the exceptionally low
reported position error, while ordinary future preview contributes almost
nothing once that defect is removed. There is no future plant-state leakage:
the optimizer uses the measured current state, precomputed reference samples,
and applies its command before advancing the plant. The inner yaw loop does
receive the known reference yaw rate directly. The current reporting metric is
component RMSE over time and xyz; the corresponding mean 3-D vector RMSE is
0.009225 m for the reported linear controller.

Run 56003 remains an isolated hover-excitation transient: its reported position
component RMSE is 0.2309 m. Excluding it descriptively, the reported linear mean
is 0.000722 m; without the defect it is 0.006829 m. The outlier is retained in
all inferential results. Most importantly, the strict current-only/no-defect
linear result of 0.012861 m still beats the augmented EDMDc result of 0.037843
m on the same cases. The linear advantage is therefore not solely an artifact
of affine reference forcing. See `linear_causality_audit/`.

### Outer-command excitation augmentation

The command-policy distribution shift above was tested directly on the
`heavy_asymmetric` hidden plant. `generate_outer_excitation_data.py` adds 100
deterministic 60 s hover-contained identification runs. Independent bounded
multisines perturb thrust, desired roll, and desired pitch at the offboard
attitude-command boundary; the existing attitude/rate controller, motor
allocator, and physical plant remain in the loop. The original 300 runs remain
first in the combined file, so the locked validation and test indices keep
their original meaning.

Across 600,000 added 100 Hz samples, allocator clipping is zero. Absolute
roll/pitch command-to-state errors have 99th percentiles of 0.465/0.412 rad,
99.9th percentiles of 0.604/0.546 rad, and maxima of 0.771/0.661 rad. This
covers the command regime that was absent from the original behavior data.
Generate and train the augmented model with:

```powershell
python generate_outer_excitation_data.py `
  --plant-id heavy_asymmetric `
  --base-data artifacts\model_mismatch_v1\heavy_asymmetric\identification\runs_mixed_n300.pkl `
  --output-dir artifacts\model_mismatch_v1\heavy_asymmetric\identification_augmented `
  --runs 100 --workers 4

python train_mismatched_edmd.py `
  --data artifacts\model_mismatch_v1\heavy_asymmetric\identification_augmented\runs_mixed_augmented_n400.pkl `
  --output artifacts\model_mismatch_v1\heavy_asymmetric\edmdc_identified_augmented_dt001.pkl `
  --allow-augmented-data
```

The untouched five-run open-loop test has mean two-second rolling position
RMSE 0.0154 m and velocity RMSE 0.0220 m/s; all state gates pass. A separate
fresh validation search selected a 0.2 s prediction horizon and 0.05 s control
horizon. On 50 new paired 60 s confirmation trajectories, the augmented EDMDc
obtains 0.03784 m position, 0.07322 m/s velocity, and 0.01244 rad yaw RMSE.
The information-matched hover-linear MPC obtains 0.00533 m, 0.01101 m/s, and
0.01099 rad; reactive PID obtains 0.37878 m, 0.13855 m/s, and 0.06174 rad.
There are zero failed solves and zero allocator-altered commands. Mean solve
times are 1.47 ms for EDMDc and 0.884 ms for linear MPC.

On those exact same 50 trajectories, the original EDMDc obtains 0.04258 m,
0.08731 m/s, and 0.01381 rad. Augmentation therefore changes position by
-11.1% (paired bootstrap interval crosses zero), velocity by -16.1% (resolved),
and yaw by -9.9% (resolved). It improves the learned controller, but does not
reverse the baseline ordering: linear MPC wins 49/50 position pairs, and the
EDMDc-minus-linear position gap is +0.03252 m with 95% bootstrap interval
[+0.02398, +0.03989]. This result must be reported honestly; broader training
coverage alone is not evidence that EDMDc should beat a well-matched hover
linear model in these modest-attitude, deterministic trajectories.

Artifacts are under `identification_augmented/`,
`edmdc_identified_augmented_dt001_plots/`,
`information_matched_tuning_augmented/`,
`analysis_information_matched_augmented/`, and
`analysis_excitation_ablation/`.

### Exploratory nonlinear stress screen

Zero-default physical hooks add quadratic relative-air drag
`-k_q |v_air| v_air` and a first-order rotor-speed response. A regression run
verified that zero values reproduce a previously saved 60 s hidden-plant run
bit-for-bit for time, all states, applied/requested wrench, and outer commands.
The screening protocol is non-confirmatory and keeps all three controller
configurations frozen.

The valid 60 s screen contains 330 episodes: 11 conditions, five families, two
paired runs, and three controllers. A discarded 30 s pilot is retained under
`nonlinear_stress_screen_30s_confounded/`; shortening generation compressed the
trajectories and therefore changed their dynamics, so it is excluded from all
selection and claims.

The first identification target is `qdrag_0p08_fast`: 1.5x trajectory
amplitude/kinematics and quadratic-drag coefficient 0.08 N/(m/s)^2. Mean
position component RMSE is 0.0454 m for frozen hover-linear MPC, 0.989 m for
frozen EDMDc-MPC, and 0.607 m for reactive PID. All episodes are finite, with
zero failed solves and zero allocator clipping. Frozen EDMDc has not been
identified on this new plant and its result is not a fair final model
comparison; it is a coverage diagnostic.

This condition degrades the linear baseline materially, remains Markov in the
existing 12 physical states, and does not rely on numerical failure or
infeasible actuation. Motor lag is retained as a separate future ablation:
because rotor speed is an unobserved dynamic state, the current 12-state/current
input EDMDc formulation would require motor-state or input-delay augmentation
for a fair identification claim. The next phase is quadratic-drag response-data
generation spanning speed scales 1.0--1.5, EDMDc retraining, validation-only
controller selection, and untouched 1.5x confirmation seeds. See
`nonlinear_stress_screen/` and `nonlinear_stress_protocol.json`.

### Quadratic-drag identification and held-out control result

The predeclared `qdrag_0p08_fast` follow-up is complete. The identification
set contains the original 300-family composition at 60 s and speed scales
1.0--1.5, plus 100 independent outer-command excitation runs, all on the
`heavy_asymmetric` plant with quadratic drag 0.08 N/(m/s)^2 and zero motor
lag. The mixed 400-run dataset has SHA-256
`636197241483f80d5b20974d3136f99afbac68ab8635cce8b3ea9684fc501227`.
It is finite and has zero allocator clipping. The command-excitation runs
reach 99.9th-percentile absolute roll/pitch errors of 0.602/0.545 rad.

The selected 56-observable model uses lambda 0.3 and has SHA-256
`9853c5631490cab3eb3970790e30ad78a4dba0df8635c5d41a2b51451fe52824`.
On the five locked two-second rollout tests, family position RMSE ranges from
0.0145 to 0.0491 m and mean position RMSE is 0.0333 m; all rollout gates pass.
This verifies useful prediction but is not evidence of closed-loop control
quality.

EDMD-only controller tuning used fresh run starts 68000 and 69000 at 1.5x
speed. Its best candidate still had 2.832 m mean validation position RMSE and
8.332 m worst-case RMSE. With that configuration frozen, 50 new paired 60 s
cases starting at 70000 give 0.1062 m position RMSE for the previously frozen
information-matched hover-linear MPC, 0.5729 m for reactive PID, and 2.1799 m
for the condition-identified EDMDc-MPC. EDMDc minus PID is +1.6070 m with 95%
bootstrap interval [+0.9161, +2.3911]. All controllers have zero failed solves
and zero allocator-altered commands. The negative result is therefore not a
solver or clipping artifact.

A same-case 2x2 diagnostic separates model and tuning effects. Old model/old
configuration gives 0.9536 m; new model/old configuration 1.1411 m; old
model/new configuration 2.0515 m; and new model/new configuration 2.1799 m.
The validation-selected configuration accounts for most of the degradation,
while the new identification also degrades control under fixed weights. This
demonstrates that short open-loop rollout RMSE alone is an inadequate
control-oriented model-selection objective. Do not use this EDMD controller
as the ACC headline result. The next EDMD step should diagnose lifted-model
local input response and multi-step Jacobians, then select models on
validation closed-loop cost while always including the incumbent controller
configuration.

Artifacts are under `heavy_asymmetric_qdrag008/identification_augmented/`,
`heavy_asymmetric_qdrag008/edmdc_qdrag008_augmented_dt001_plots/`,
`heavy_asymmetric_qdrag008/information_matched_tuning_speed1p5/`, and
`heavy_asymmetric_qdrag008/heldout_speed1p5_analysis/`.

### Corrected simple hover-linear MPC baseline

The fixed zero-yaw hover LTI model is not a valid baseline for yawing paths
when its roll/pitch input matrix is applied directly in world coordinates. A
25-case structural audit makes the failure mechanism explicit. With path yaw,
the fixed input map gives 181.4 m mean position RMSE; locking yaw to zero gives
0.0115 m. Applying only the known yaw-coordinate transformation gives 0.0132 m
without an affine reference defect. Adding the defect reduces this to 0.0030 m
but implicitly provides reference-acceleration information and is excluded
from the simple baseline.

The simple baseline is therefore a fixed level-hover translational model with
the roll/pitch input map rotated by known reference yaw. It receives
position/velocity preview and yaw feedforward, but no acceleration
feedforward, affine reference defect, trajectory linearization, or learned
dynamics. Commands still pass through the same cascaded attitude/rate PID and
shared motor allocator.

Two apparent instability cliffs were traced to artificial optimization trust
bounds rather than the plant or QP. At 1.5x speed with quadratic drag 0.08, a
0.35 rad tilt trust bound was active for 50.6% of one aggressive run and gave
1.493 m RMSE; 0.45 rad reduced that same case to 0.0565 m. Separately, the
8 N thrust trust bound caused alternating saturation during vertical hover
excitation and gave 0.553 m; 12 N reduced the same case to 0.0287 m. Physical
limits are retained independently, and neither case had allocator clipping.

Fresh one-factor validation selected 0.65 rad tilt trust from
0.35--0.65 rad, then 12 N thrust trust from 8--20 N. The 12 N value is the
smallest point on the validation performance plateau. All validation
trajectories were generated at their full 60 s duration; the earlier practice
of generating 12 s or 45 s references is avoided because it changes trajectory
dynamics.

After freezing the result, 50 untouched 60 s confirmation trajectories at
100 Hz (run start 99000) give 0.02359 m position, 0.01373 m/s velocity, and
0.01137 rad yaw RMSE for simple hover-linear MPC. Reactive PID gives 0.63142 m,
0.23765 m/s, and 0.07088 rad. The paired linear-minus-PID position difference
is -0.60783 m with 95% bootstrap interval [-0.67941, -0.53664]. The worst
linear position RMSE is 0.04645 m; there are zero failed solves and zero
allocator-altered commands. These are the corrected simple-linear baseline
results. The earlier 79000 confirmation block is diagnostic because its
outlier was used to identify the thrust-bound problem and must not be used as
final confirmation.

Artifacts are under `heavy_asymmetric/simple_linear_structure_audit/`,
`heavy_asymmetric_qdrag008/simple_linear_structure_audit_speed1p5/`,
`heavy_asymmetric_qdrag008/simple_linear_trust_validation/`,
`heavy_asymmetric_qdrag008/simple_linear_thrust_validation/`, and
`heavy_asymmetric_qdrag008/simple_linear_vs_pid_final_v2_analysis/`.

### Final paired three-controller comparison

The frozen condition-trained EDMDc-MPC was subsequently run on the exact same
50 final confirmation references above; neither linear MPC nor EDMDc-MPC was
retuned. Mean position RMSE is 0.02359 m for simple yaw-compensated
hover-linear MPC, 0.63142 m for reactive PID, and 2.47580 m for EDMDc-MPC.
Velocity RMSE is 0.01373, 0.23765, and 0.65717 m/s, respectively. EDMDc minus
linear position is +2.45221 m with 95% bootstrap interval
[+1.71472, +3.26423]; EDMDc minus PID is +1.84438 m with interval
[+1.13169, +2.63958]. Linear MPC beats PID in all 50 position pairs and beats
EDMDc in 49/50. EDMDc beats PID in only 20/50 position pairs.

That condition-trained EDMDc comparison is an ablation, not the default EDMDc
result. Replacing it with the normal augmented EDMDc model and its original
frozen controller on the same quadratic-drag/1.5x cases gives 1.13009 m
position, 0.45764 m/s velocity, and 0.01253 rad yaw RMSE. This is better than
the condition-trained EDMDc result but remains out-of-distribution relative to
the normal model's training regime. On this stress condition, normal EDMDc is
worse than linear MPC by 1.10650 m in paired position RMSE and its difference
from PID is +0.49867 m with a bootstrap interval that crosses zero. The normal
model's previously reported 0.03784 m position result applies to the original
1.0x, zero-quadratic-drag condition and must not be conflated with this stress
test. See `heavy_asymmetric_qdrag008/final_three_controller_normal_edmd_analysis/`.

### Fresh normal-condition comparison after baseline correction

The normal augmented EDMDc was also restored to its original zero-quadratic-
drag, 1.0x condition and compared on 50 new paired 60 s references starting at
106000. The corrected simple linear baseline was independently selected on
full-duration validation runs, uses a fixed hover model with yaw-coordinate
input rotation, and has no affine reference defect.

Mean position RMSE is 0.01125 m for corrected simple linear MPC, 0.12562 m for
normal augmented EDMDc-MPC, and 0.38550 m for reactive PID. Corresponding
velocity RMSE is 0.00727, 0.11322, and 0.15135 m/s. EDMD improves position over
PID by 0.25988 m with a paired 95% bootstrap interval [0.12708, 0.35320] m, but
linear MPC has a resolved 0.11437 m advantage over EDMD.

The higher EDMD mean relative to the earlier 0.03784 m block is caused by two
new tail failures, not a different EDMD model: helix 106009 gives 2.61889 m and
hover-excitation 110008 gives 1.07382 m. EDMD median RMSE is 0.03638 m; after
descriptively excluding those two runs its mean is 0.05393 m, consistent with
the earlier normal-condition tables. Both failures remain in every reported
inferential result. There are zero QP failures and zero allocator-altered
commands. See `heavy_asymmetric/final_normal_three_controller_analysis/`.

Mean solve time is 1.64 ms for linear MPC and 2.80 ms for EDMDc-MPC. All three
controllers have zero failed solves and zero allocator-altered commands. The
EDMDc position/velocity failure therefore cannot be attributed to the QP or
actuator clipping. Its yaw RMSE (0.01146 rad) remains close to linear MPC
(0.01137 rad) because yaw is feedforward-only in both MPC formulations.

The final tables, bootstrap comparisons, and plot are under
`heavy_asymmetric_qdrag008/final_three_controller_analysis/`.
