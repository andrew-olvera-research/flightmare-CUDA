# Accelerated Flightlib scope

This milestone reproduces the scoped headless Flightlib core in a scalable,
device-resident package. It is a GPU acceleration extension, not a new scene
simulator or a promise to replace every legacy class. Version remains 0.1.0.

## What defines parity

The reference is Flightlib's state/control/model behavior: preparation and
updates, command acceptance, both command modes, resets versus state setters,
Flightmare equations and operation ordering, all 25 state fields, motors, clocks,
world-box constraints and diagnostics. Preserve legacy quirks. Keep existing
task/observation/reward/reset coverage and Python Environment integration.
CUDA-9 now extends the final scope to headless usage and RL wrappers, resident
rollout collection and a train/evaluate example; see the
[CUDA-9 usage plan](history/cuda9-usage-plan.md). This supersedes the earlier RL deferral.

Behavioral fidelity to this reference is separate from adding physical fidelity.
No new aerodynamics, collision response or replacement physics engine is implied.

Scene collision detection is external in the original implementation:
[UnityBridge](../../flightlib/src/bridges/unity_bridge.cpp) sends poses and stores
the collision boolean returned by Unity. [Quadrotor](../../flightlib/src/objects/quadrotor.cpp)
has a collision getter/setter, not a mesh collision solver. The existing
[QuadrotorEnv terminal test](../../flightlib/src/envs/quadrotor_env/quadrotor_env.cpp)
checks altitude independently. Preserve/document that external result boundary;
do not synthesize collision results or confuse world-box behavior with scene
collision. Exact Unity collider fidelity requires its scene/prefab implementation.

## Remaining implementation layers

| Issue | Deliverable | Gate |
| --- | --- | --- |
| [CUDA-7](https://github.com/andrew-olvera-research/flightmare-CUDA/issues/7) | State fidelity audit implemented | [Support matrix and numerical contract](flightlib-support.md), Eigen allocation fix, extended differential gates; [validation evidence](cuda7-validation.md) |
| [CUDA-8](https://github.com/andrew-olvera-research/flightmare-CUDA/issues/8) | Resident core API and package implemented | [Core API/public-method checklist](core-api.md): lifecycle, stepping, optional CPU helpers, model/motor/state views, explicit streams/snapshots; installed Python/C++ consumers |
| [CUDA-9](https://github.com/andrew-olvera-research/flightmare-CUDA/issues/9) | Final headless usage and RL package, then qualification (planned) | Resident tensor wrapper, explicit host adapter, configuration/method audit, train-save-load-evaluate example, wrapper parity/residency, clean install and matched performance |

Order: completed CUDA-6 -> CUDA-7 -> CUDA-8 -> CUDA-9. CUDA-8 implements only
actual supported gaps identified by CUDA-7, rather than speculating about a new
API hierarchy. Accessors may be borrowed views or explicit setup metadata instead
of CPU-style per-drone objects. Any external collision flag passthrough must be
distinguished from computed collision detection and retain contract discipline.

The parallel benchmark exposed a numerical branch discrepancy for symmetric
commands. CUDA-7 reproduced it against both frozen and optimized CPU, fixed the
allocation evaluation ordering and added a reward/observation regression. The
support matrix records the remaining CPU SIMD approximation boundary and the
separate long-rollout drift budget. Angle equivalence does not excuse reward
differences; the original oracle and its tolerances remain unchanged.

CUDA-9 keeps performance evidence honest: matched work/hardware/configuration,
best tested OpenMP/multiprocessing configuration, per-step completion versus
amortized throughput, and separate host/device traffic and allocator counts.
Hardware-counter, sanitizer and multi-GPU gaps remain explicit. Relevant hot-path
changes require performance checks alongside parity; documentation changes do not
require rebuilding the simulator.

## Deferred work, preserved for later

- Scene assets and instances, body proxies, overlap/ray queries, broad/narrow
  phase, containment/touching/overflow rules and backend selection. The earlier
  [collision plan](history/cuda7-collision-plan.md) is retained as a historical design
  resource, not a current implementation instruction or Unity-equivalence claim.
- Rendering overhaul, scene fidelity and sensors, including timing/state exchange.
- Shared swarm rewards/termination, distributed learning, a general RL algorithm
  library and automatic legacy TensorFlow checkpoint conversion. Headless
  wrappers and the bounded training/evaluation example are now in CUDA-9.
- ROS/PX4/SITL, new platforms and additional physical fidelity.

Frozen geometry declarations remain reserved; their presence does not require
implementing geometry to finish this package. This rescope changes no tensor
definitions, allocations, kernels or equations. Future compatibility changes
still need review. No PhysX dependency is introduced.

Parent scope is [CUDA-1](https://github.com/andrew-olvera-research/flightmare-CUDA/issues/1).
Local implementation checkpoint: ef242b2; parallel baseline/design checkpoint:
25207b6. Issue titles/bodies #1/#7/#8/#9 were respecified with this boundary.
Implementation provenance and closure evidence remain separate from planning;
the rescope does not close completed-layer issues or claim unimplemented APIs.
