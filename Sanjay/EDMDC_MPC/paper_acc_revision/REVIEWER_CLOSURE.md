# Reviewer-comment closure status

## Addressed in `main_revised.tex`

- Yaw and yaw rate are included in the 12-state model.
- The four raw outer commands and 16-dimensional input lift are distinguished.
- Snapshot alignment is stated as `(x_k, u_k, x_{k+1})`.
- Training-run counts, split indices, transition count, sampling rate, ridge
  penalty, model dimensions, prediction horizon, command bounds, and selected
  EDMDc weights are reported.
- The PID information disadvantage is disclosed: PID is reactive, whereas
  both MPCs receive preview.
- The linear baseline is described accurately as yaw-scheduled hover-linear,
  not as a matched nonlinear model.
- Claims of identical costs, universal EDMDc superiority, sub-millisecond
  solution time, real-flight validation, and universal robustness are removed.
- Interception requires proximity, relative speed, and dwell; dwell-end times
  are reported.
- Fresh paired sample counts, bootstrap intervals, solver failures, allocator
  alterations, solve-time statistics, and parameter-mismatch scope are stated.
- Negative sharp/evasive stress evidence is disclosed as a limitation.
- The target CPU, Python/OSQP versions and settings, and sample-wise maximum
  tracking solve time are recorded.
- The hover-linear attitude-error guard was isolated on an independent
  validation split and removed. A fresh confirmation retains a sparse
  aggressive failure tail, so the paper no longer attributes the old tail
  entirely to linearization.
- A model-matched 13/19/42/56-term observable-capacity ablation is complete.
  The 42-term physics dictionary was selected by validation error and
  parsimony, then confirmed on fresh closed-loop tests.
- A fixed-split learning curve at 10/25/50/100 percent of the training runs is
  complete and reported without claiming a monotonic test-set improvement.
- Publication-critical source, the selected model, controller configs,
  lightweight results/figures, paper, and data/model hashes are frozen in the
  scoped publication commit. The 893 MB dataset is identified by SHA-256
  rather than stored directly in Git.

## Still required before final submission

1. Add quantitative flight tracking results when the in-progress hardware
   implementation is ready. Until those data exist, retain the platform only
   as future work and make no experimental-performance claim.
The draft does not fabricate results for unfinished reviewer requests.
