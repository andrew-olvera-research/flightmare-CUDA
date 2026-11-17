# Accelerated Flightlib scope

This milestone reproduces the scoped headless Flightlib core in a scalable,
device-resident package. It is a GPU acceleration extension, not a new scene
simulator or a promise to replace every legacy class. Version remains 0.1.0.

## What defines parity

The reference is Flightlib's state/control/model behavior: preparation and
updates, command acceptance, both command modes, resets versus state setters,
Flightmare equations and operation ordering, all 25 state fields, motors, clocks,
world-box constraints and diagnostics. Preserve legacy quirks. Keep existing
task/observation/reward/reset coverage and Python Environment integration as
compatibility consumers; do not expand the RL wrapper or training API now.

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
| [CUDA-7](https://github.com/andrew-olvera-research/flightmare-CUDA/issues/7) | State fidelity and numerical parity completion | Source-anchored support matrix, demonstrated gap fixes and disposition of optimized-CPU Euler-branch/reward discrepancy |
| [CUDA-8](https://github.com/andrew-olvera-research/flightmare-CUDA/issues/8) | Resident core API and standalone package | Supported reset/setState/setCommand/step and state/model/motor access usable without task Environment; explicit lifetime/stream/snapshot rules and non-RL example |
| [CUDA-9](https://github.com/andrew-olvera-research/flightmare-CUDA/issues/9) | Integrated qualification and release readiness | Differential verification, clean build/install, matched parallel CPU/GPU performance and documented limitations |

Order: completed CUDA-6 -> CUDA-7 -> CUDA-8 -> CUDA-9. CUDA-8 implements only
actual supported gaps identified by CUDA-7, rather than speculating about a new
API hierarchy. Accessors may be borrowed views or explicit setup metadata instead
of CPU-style per-drone objects. Any external collision flag passthrough must be
distinguished from computed collision detection and retain contract discipline.

The parallel benchmark exposed a numerical branch discrepancy for symmetric
commands under an optimized CPU build. CUDA-7 must reproduce and investigate it
against frozen CPU, optimized CPU and CUDA; reward differences cannot be accepted
merely because Euler angles describe an equivalent orientation. Do not silently
relax tolerances or rewrite the oracle to produce a passing result.

CUDA-9 keeps performance evidence honest: matched work/hardware/configuration,
best tested OpenMP/multiprocessing configuration, per-step completion versus
amortized throughput, and separate host/device traffic and allocator counts.
Hardware-counter, sanitizer and multi-GPU gaps remain explicit. Relevant hot-path
changes require performance checks alongside parity; documentation changes do not
require rebuilding the simulator.

## Deferred work, preserved for later

- Scene assets and instances, body proxies, overlap/ray queries, broad/narrow
  phase, containment/touching/overflow rules and backend selection. The earlier
  [collision plan](cuda7-collision-plan.md) is retained as a historical design
  resource, not a current implementation instruction or Unity-equivalence claim.
- Rendering overhaul, scene fidelity and sensors, including timing/state exchange.
- RL wrapper, vector-environment adapters, rollout/training tooling, shared swarm
  rewards/termination and expanded policy integration.
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
