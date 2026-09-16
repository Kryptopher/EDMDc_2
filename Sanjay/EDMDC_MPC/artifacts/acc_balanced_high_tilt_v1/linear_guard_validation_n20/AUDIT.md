# Hover-linear attitude-guard audit

The 0.12 rad state-to-command attitude-error guard used in the earlier frozen
comparison was audited as a single changed factor on 20 fresh 1.75x validation
trajectories (run indices 760000--763004). Model, weights, horizon, plant,
preview, and references were held fixed.

| Guard | Mean RMSE [m] | Median [m] | P90 [m] | Worst [m] | Runs >1 m |
|---|---:|---:|---:|---:|---:|
| 0.12 rad | 7.654 | 5.038 | 18.565 | 22.974 | 10/20 |
| 0.25 rad | 3.691 | 0.024 | 12.376 | 17.912 | 6/20 |
| 0.40 rad | 1.864 | 0.024 | 10.036 | 11.106 | 5/20 |
| Unguarded | **1.533** | **0.024** | **9.170** | **10.733** | **3/20** |

The guard caused most of the waypoint tail: family-mean waypoint error fell
from 14.024 m at 0.12 rad to 0.038 m without the guard. It also amplified the
Lissajous tail. It did not explain every failure: the unguarded controller
still produced 4.153 m figure-eight and 1.933 m Lissajous family-mean errors.
Thus the earlier tail cannot be attributed solely to local linearization, but
the 0.12 rad optimization guard was an important confound and was removed.

The unguarded setting was selected using this validation split only. The final
40-case confirmation uses new 770000-series seeds. It retains a low median
(0.026 m) but a 16.739 m worst case, showing a genuine sparse aggressive tail
after the guard artifact is removed. All guard-audit and confirmation runs had
zero QP failures and zero allocator-altered steps.
