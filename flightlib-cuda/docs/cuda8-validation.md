# CUDA-8 API, package and systems validation

Base checkpoint: 54d6021. Version 0.1.0 and frozen v1 tensor contracts remain
unchanged. [Core API](core-api.md) contains the public Quadrotor/QuadrotorDynamics
method checklist, data shapes, helper semantics, ownership rules and explicit
scene/sensor boundary. Core and Environment share the same equations.

## Behavior and installation

Initial full verify report:
`.local-ci/20261117T033627.944024Z-verify/report.json` (passed).
Final verification after helper specialization:
`.local-ci/20261117T035556.258500Z-verify/report.json` (passed, all 24 steps).

- Frozen CPU suite: 36 tests, integrity manifest unchanged.
- Native contracts/runtime/lifecycle/dynamics/environment gates pass. Lifecycle
  includes all nine optional operations against actual CPU methods for 33 mixed
  model/active lanes, including rejected-input preservation. Dynamics compares
  separate acceptance/step, combined run and graph replay against CPU trajectories.
- Python: 15 passed, one multi-GPU skip. The new Core tests replay frozen CPU
  fall, rate-reversal and 250-step YAML hover traces through native and graph
  submissions, comparing all state/motor/time fields at every interval.
  Variable-substep-capacity motor trajectories remain covered by native tests;
  Core's frozen schedule capacity is explicit, not silently resized.
- Partial state/reset/command operations, invalid-command no-advance semantics,
  independent model IDs, inactive lanes, rejected/live setup, stale graphs,
  metadata persistence, copied host model configuration, stable typed buffers,
  alias rejection, nondefault streams and exported owner lifetime are covered.
- `verify` installs Python and C++ artifacts to a fresh report-local prefix,
  executes the non-RL example using that module, then configures/builds/runs an
  independent `find_package` C++ consumer. Legacy Environment and binding tests
  continue to pass. Profiler analysis suite: 11 tests passed.

No tolerance in an existing fixture was loosened. The new long YAML-hover replay
uses CUDA-7's documented 3e-4 normalized budget for state/acceleration and 1e-4
for torque/bias; the shorter original native tests keep their prior thresholds.
Optional inverse-motor and motor-speed comparisons allow .002rad/s to cover the
qualified CPU SIMD square-root approximation, and other helper values use 1e-5
relative-to-max(1,abs(reference)). Invalid/inactive decisions and clocks are exact.

## Core stepping and residency

Hardware: RTX 5070 Ti, Ryzen 7 9800X3D, Docker/WSL, CUDA 13.3, PyTorch 2.9.1+cu130.
All cases use ten warmups and five 100-step samples, dt=.01, initial z=5/time=2,
constant rotor force=3 and active lanes throughout. Core has no task/auto-reset.
Median completed wall time in microseconds per batch control interval:

| Lanes | Accepted step | Accepted graph | Accept + step | Accept + step graph |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 9.39 | 8.86 | 16.87 | 10.12 |
| 33 | 9.30 | 8.99 | 16.78 | 10.66 |
| 16,384 | 9.63 | 10.32 | 17.76 | 12.55 |
| 65,536 | 28.31 | 25.91 | 30.84 | 32.66 |

Graph replay helps the small command-plus-step workload; it is not universally
faster at large batches or for an already single-kernel step. Choose based on
measured workload. These are amortized timings with completion at sample ends,
not per-step CPU round-trip latency guarantees.

All 16 cases pass independent Nsight trace gates: one dynamics node for accepted
commands, two nodes for command-plus-step; one graph API call per replay. No
hot-loop memcpy/memset, allocation/free or internal host synchronization is
recorded. Python allocator deltas are zero; storage pointers remain stable.
Completion waits occur at the declared measurement boundaries. Shared dynamics
functions still compile with 254 registers, zero spill stores/loads and zero
stack bytes. Optional helper kernels do not run during these workloads.

Arena usage remains 8.617MiB at 16,384 lanes and 34.460MiB at 65,536, including
the existing unused task reservations. Core staging/external metadata is a
separate 140 bytes/lane. Optional helper inputs/outputs are caller-owned and
allocated only when explicitly requested. This is not a measurement of total
driver/context VRAM or hardware bandwidth.

Report: `.local-ci/profile-20261117T034138.602049Z/report.json` (passed, 16 cases,
all traced). Source fingerprint:
`3d7888027124b0d9523433098fffa47816f54f255aa973233806ac0d0bba5610`.
Python extension SHA256:
`3609c620256bbd6b7356a499f38ea33a3f50e17c0174a6683268232ec68689ea`.

### Optional helper kernel qualification

Final resource inspection found a 200-byte per-thread stack in the original
generic helper dispatch. Before committing, helper launches were specialized by
operation. All nine compiled helper kernels now have zero stack/local bytes and
20–72 registers per thread; the dynamics kernel is unchanged. The independent
helper probe records exactly nine launches per iteration and no hot-loop memory
transfers, allocation/free or internal host waits at 1 and 16,384 lanes.
Report: `.local-ci/profile-20261117T035317.377732Z/report.json` (passed).

A four-trial alternating comparison used the prior installed generic module and
the specialized module with identical 16,384-lane inputs, ten warmups and five
100-iteration samples. Median of trial medians for all nine calls: 82.20us
generic versus 73.01us specialized (0.888x). Raw evidence:
`.local-ci/cuda8-helper-pair.json`; local reproducer `.local-ci/cuda8-helper-pair.py`.
The specialized probe module SHA256 is
`5468829b556ebbb319eaddd8d0d0c29002d97f525320b2b065ca42f3f44f61ef`.
Core stepping measurements above predate this optional-helper-only change;
their kernels do not execute the helper path. Full differential verification
and installed-package checks passed again after specialization.

## Matched CPU behavior and retained Environment cost

The production Python Core was also compared against the unchanged optimized
sequential Flightlib implementation, with identical model, initial state,
commands, interval counts and schedule. All 25 final state fields, motor speeds,
motor forces and float clock match the documented gate. Each GPU case also
checks equal complete telemetry across active lanes. This complements the
per-interval differential tests; it does not replace them.

At 1,024 lanes, accepted Core stepping measured 10.59us versus 609.20us sequential
CPU; command-plus-step graph measured 11.34us versus 602.75us. CPU was faster at
one lane. This is **not** a new best-parallel-CPU speedup claim; the earlier
[thread/process comparison](cpu-parallel-performance.md) remains separate.
Report `.local-ci/profile-20261117T034352.067779Z/report.json`: all six cases and
CPU telemetry checks passed.

The retained native Environment and graph paths were compared with the passing
CUDA-6 baseline `profile-20261117T013741.581504Z`, using matched 50% initial reset
density, dimensions, intervals and sample counts. The first run
`profile-20261117T034522.906630Z` failed: one-lane non-graph timing was 1.234x
baseline, while the other seven rows were within 1.032x. It remains recorded as
a failed run. One complete repeat, with no code or threshold change, passed all
eight rows at .912–1.028x baseline:
`.local-ci/profile-20261117T034545.362284Z/report.json`.
The wall-time budget stayed 1.15x. Clocks and WSL scheduling remain uncontrolled;
this repeat does not erase the initial outlier or establish a latency guarantee.

## Qualification boundary

The method checklist covers the scoped headless state/model/control/motor core,
including separately callable mathematical helpers and external scene metadata.
It does not claim arbitrary-NaN equivalence, bitwise cross-platform trajectories,
rendering/collider equivalence, new physical models or sim-to-real certification.
Existing device random-reset support remains in Environment; Core reset takes
explicit state just as CPU Quadrotor does. Rendering/sensors and RL wrappers stay
in their designated extensions. Hardware counters, WDDM sanitizer and multi-GPU
coverage gaps remain explicit. CUDA-9 is integrated release qualification.

Reproduction commands are in [the profiler README](../../flightmare-profile/README.md)
and [core API documentation](core-api.md). Raw reports retain their complete
source identities; optional helper specialization is identified separately above.
