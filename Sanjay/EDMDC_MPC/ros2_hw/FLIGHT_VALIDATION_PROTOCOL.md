# ACC flight-validation protocol

This protocol is intentionally staged. Do not begin interception or evasive
tests until the preceding tracking stage passes. A pilot with an immediate
mode-switch/kill capability must supervise every armed test.

## Frozen controller contract

- Control rate: 100 Hz.
- Command: collective thrust plus desired roll, pitch, and yaw.
- Pixhawk retains the tuned cascaded attitude/rate loops and motor mixing.
- Log state estimates, references, requested outer commands, applied/estimated
  actuator outputs, controller status, solve time, battery voltage, flight
  mode, failsafe state, and timestamps from a common clock.
- Never identify or evaluate on the same flight. Freeze identification,
  validation, and test flight lists before computing final metrics.

## Stage 0: bench and prop-off

Verify command signs, ENU/NED conversions, yaw wrapping, thrust scaling,
timestamp alignment, 100 Hz delivery, stale-command failsafe, mode takeover,
and logging. Exercise the full command range with propellers removed. No
autonomous arming is permitted.

## Stage 1: low-risk tracking

Run hover, vertical steps, slow straight segments, and low-speed circles with
the tuned Pixhawk/PID path first. Require no estimator reset, failsafe, timing
overrun, command-limit violation, or unexpected mode change. Then run linear
MPC and EDMDc-MPC on the same reference files in separate flights.

## Stage 2: paper tracking set

Use at least five repetitions per controller and trajectory family. Randomize
controller order across batteries. Use identical geometric references and
report position/velocity/yaw RMSE, median and P90 error, maximum tilt,
allocator saturation, missed deadlines, mean/P95/P99/max solve time, and
battery voltage. Increase reference aggressiveness only after reviewing the
previous tier. The final held-out tier must not be used for tuning.

## Stage 3: interception

Begin with a virtual target and geofenced open space. Apply the paper's
qualified-capture definition: separation at most 0.75 m and relative speed at
most 3 m/s for 0.2 s. Report both first sphere entry and dwell completion.
Physical-target interception is a later safety review and is not required to
claim quantitative hardware trajectory tracking.

## Minimum paper evidence

For an experimental claim, include the vehicle mass/inertia estimate,
autopilot and companion-computer versions, estimator source, update rates,
controller configuration hashes, number of successful/attempted flights,
tracking metrics with dispersion, sampled worst-case solve time, saturation
counts, and representative synchronized plots. Until these data exist, the
paper remains explicitly simulation-only.
