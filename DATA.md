# Data

## Included in this repository

| File | Size | SHA-256 | Purpose |
|---|---|---|---|
| `data/models/wrench_physics42_dt001.pkl` | 41 KB | `090b8dd3106dcf3646dcf991fe23c030d3fe27435443a5f8fae19c1c377b2db4` | EDMDc model used in every paper result |

The interception scenarios are not stored; `experiments/interception.py` regenerates each target
trajectory from its seed and checks it against `configs/interception_scenarios.json`.

## Training dataset (not in git)

The training dataset is published separately on Zenodo as a compressed NumPy archive:

**Zenodo DOI: _to be added after upload_**

Download it to `data/runs_mixed_n360_high_tilt.npz` (ignored by git). **The `.npz` archive is the file to
verify against.**

| File | Format | Size | SHA-256 |
|---|---|---|---|
| `runs_mixed_n360_high_tilt.npz` (published) | NumPy `.npz`, compressed, no pickled objects | 540,170,594 B | `0ee76c173f0dfba149fdd0ff3f4cfcd3d1b7ab785164edad822ddac9da64c109` |
| `runs_mixed_n360_high_tilt.pkl` (original, not published) | Python pickle | 893,563,750 B | `17f9045739ec05f500004d28bb242ea63f2fe0f53ca5cf1058c747531c505089` |

The archive was produced from the original pickle by `tools/pkl_to_npz.py` and verified bitwise: every
array (all `float64`, no downcasting), dtype, key, list and scalar of the pickle is reproduced exactly by
`experiments/dataset_io.load_dataset()`. The conversion is deterministic: re-running `tools/pkl_to_npz.py`
on the original pickle reproduces the archive byte-for-byte (same SHA-256). `train_edmdc.py` and
`evaluate_prediction.py` also accept the original `.pkl` if you have it.

### Retraining check: content digest, not file hash

The training script stores the name of its input file inside the model (`source_file`). A model
trained from the `.npz` therefore records `runs_mixed_n360_high_tilt.npz`, while the paper model
records `runs_mixed_n360_high_tilt.pkl`. That one string makes the two model files differ, even though
every number in them is identical, so the file hash is not a usable reproduction check:

| Model | Trained from | File SHA-256 | Content digest (numerical fields) |
|---|---|---|---|
| Paper model (`data/models/wrench_physics42_dt001.pkl`) | `.pkl` | `090b8dd3106dcf3646dcf991fe23c030d3fe27435443a5f8fae19c1c377b2db4` | `f4e736d380d64679b991c4ae7b0d9a379d43fe8502c9a8b8c051dc1e90351cf4` |
| Retrained from the published archive | `.npz` | `788220505e8c1ebedb533e4c12abb4bb4d1ea127d251b07ffcfaf05dcf0a00b6` | `f4e736d380d64679b991c4ae7b0d9a379d43fe8502c9a8b8c051dc1e90351cf4` |

The acceptance criterion for retraining is that **all numerical fields are bitwise identical**, checked
by the content digest (`experiments/model_digest.py`): SHA-256 over every array (K, B_u, …), numeric
scalar (λ, dt, selection scores, …), scaler statistic, the λ sweep and the run indices, ignoring text
fields. `train_edmdc.py` prints this digest, compares it with the paper model, and also reports the
file hash expected for the input format used. `analyze_model.py` reports both values for the
included model.

Archive contents (axis 0 is always the run index, so run boundaries are preserved):

| Key | Shape | dtype | Content |
|---|---|---|---|
| `states` | 360 × 6000 × 12 | float64 | position, velocity, roll/pitch/yaw, body rates |
| `U` | 360 × 6000 × 4 | float64 | applied wrench (training input) |
| `U_requested` | 360 × 6000 × 4 | float64 | requested wrench before allocation |
| `U_outer` | 360 × 6000 × 4 | float64 | outer-loop desired-attitude command |
| `t`, `time` | 360 × 6000, 6000 | float64 | time stamps (0.01 s) |
| `ref_pos`, `ref_vel`, `ref_acc` | 360 × 6000 × 3 | float64 | reference trajectory |
| `ref_yaw`, `ref_yaw_rate` | 360 × 6000 | float64 | reference heading |
| `ref_outer_command`, `has_outer_command` | 360 × 6000 × 4, 360 | float64, bool | reference attitude command of the 60 high-tilt runs |
| `family_labels`, `run_seeds` | 360 | str, int64 | trajectory family and seed of each run |
| `metadata_json` | scalar | str | remaining metadata (generation settings, per-run metrics), type-preserving JSON |

Split by complete run: validation runs 38, 58, 128, 154, 209, 352–355; held-out test runs 39, 59, 129,
155, 210, 356–359; the remaining 342 runs give 2,051,658 training transitions. Original metadata:
`provenance/original/artifacts/acc_balanced_high_tilt_v1/runs_mixed_n360_high_tilt.json`.

## How the dataset was generated

The dataset was built in four steps with the scripts in `src/`. Intermediate files (SHA-256):

1. `parallel_sim.py --profile acc_balanced` → trajectory-family files (`runs_traj1..5_n*.pkl`,
   `runs_prbs_n70.pkl`): 50 helix, 50 figure-eight, 50 Lissajous, 50 random-waypoint, 30 hover,
   70 PRBS runs, 60 s each, PID-driven wrench commands.
2. `mix_traj.py --profile acc_balanced` → `runs_mixed_n300.pkl`
   (`da73c30152edf2694f07d3b921d39d7f7b5809993a7cef506cd2058f45225561`).
3. `replace_waypoint_family.py --base runs_mixed_n300.pkl --waypoint runs_traj4_n50.pkl --output runs_mixed_n300_waypoint_v2.pkl`
   replaces the random-waypoint block (runs 150–199) with the speed-limited waypoint profile
   `acc_balanced_waypoint_v2` (waypoint file `f4e362e85e1065c2310eefd31c3b288fc95c8c58606a0a877670da81e1e68ca8`;
   output `5e10c52897777b9c62f72b4191dc7cdddad5e2b96042a9d33cd28f712c1f6fb6`).
   The first 300 runs of the final dataset are identical to this file.
4. `generate_high_tilt_augmentation.py --runs 60 --output runs_high_tilt_attitude_n60.pkl` (tilt bands
   7.5°, 15°, 22.5°, 30°; `26f5f099925261a10d78078eb273688f658cae0072f4e07ee50b7c724921a5ce`), then
   `append_high_tilt_augmentation.py --base runs_mixed_n300_waypoint_v2.pkl --augmentation runs_high_tilt_attitude_n60.pkl --output runs_mixed_n360_high_tilt.pkl`.

Caveat: `parallel_sim.py`, `mix_traj.py`, and `Closed_loop.py` were modified after the dataset was
generated and no snapshot of the exact earlier versions exists, so steps 1–2 are documented but not
guaranteed to regenerate byte-identical files. The published archive (hash above) is the authoritative
input.
