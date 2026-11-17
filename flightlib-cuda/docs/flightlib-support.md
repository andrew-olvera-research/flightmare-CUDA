# Flightlib core support and numerical contract

CUDA-7 audit, v0.1.0. This is the scoped headless acceleration extension described
in [flightlib-scope.md](flightlib-scope.md). It preserves the Flightmare equations;
it does not certify sim-to-real transfer, new physics, or every legacy class.
The frozen Flightlib source and oracle hash manifest remain unchanged.

CUDA-8 completion: the API gaps recorded in this checkpoint are now exposed by
the [resident Core API and complete public-method checklist](core-api.md),
including optional low-level helpers and external size/collision metadata.
CUDA-9 now plans the final headless usage/RL layer in
[cuda9-usage-plan.md](cuda9-usage-plan.md), superseding the wrapper deferral in
this historical CUDA-7 matrix. The numerical and randomization boundaries below remain authoritative. The
table's CUDA-8 references describe the original audit handoff, not current gaps.

## Operation matrix

| CPU source / operation | CUDA support | Evidence / remaining boundary |
| --- | --- | --- |
| [QuadState fields and validity](../../flightlib/include/flightlib/common/quad_state.hpp) | Native `state.x[25,N]`, time[N], motors[4,N]; xyz, wxyz, world velocity, body rates, acceleration, torque, angular/acceleration biases | `lifecycle_test`, `dynamics_test`: all fields, times, motors; bias fields persist without introducing noise forces. Native borrowed views exist; standalone Python core access is CUDA-8. |
| [Quadrotor reset / setState](../../flightlib/src/objects/quadrotor.cpp) | Selected-lane `resetState` clears motors; `setState` preserves them; both retain accepted commands and task counters | Invalid inputs leave lane state intact. `lifecycle_test` checks mixed active/selected/rejected lanes; `randomizedModels` continues trajectories after both operations. Task reset has its own documented command/episode semantics. |
| [Command validity](../../flightlib/src/common/command.cpp), [setCommand](../../flightlib/src/objects/quadrotor.cpp) | Rotor thrust and collective/body-rate modes, finite timestamp, XOR legacy adapter, model clamps, persistent last accepted command | `lifecycle_test`: ambiguity, NaNs, rejection and saturation. `dynamics_test`: mode changes and rate reversals. Standalone Python command API remains CUDA-8. |
| [QuadrotorDynamics preparation / updateParams](../../flightlib/src/dynamics/quadrotor_dynamics.cpp) | YAML or `ModelConfig`; P resident models with per-lane IDs; mass, arm, motor range/time constant, polynomial, kappa and rate limits | `lifecycle_test` compares derived tables to CPU; `randomizedModels` varies all inputs and updates live models. Preserve the legacy cached thrust cap, including after YAML changes. |
| [updateDynamics](../../flightlib/src/objects/quadrotor.cpp) | `configure` replaces model/topology/schedule tables at an explicit synchronous setup boundary; preserves state, motors, commands and counters | Live trajectory update tested. Fixed capacities, complete validation before upload, stale graph rejection. Sampling parameters is caller-owned; no per-step host randomization or device parameter sampler is promised. |
| [RK4](../../flightlib/src/common/integrator_rk4.cpp), [run / motors / flight controller](../../flightlib/src/objects/quadrotor.cpp), [derivative](../../flightlib/src/dynamics/quadrotor_dynamics.cpp) | One fused D kernel; local motor/control/RK stages; exact float remaining-time schedule; default maximum substep 2.5ms | `dynamics_test`: dt=0, .0025, .006125, .01 plus randomized .006125, .02, .05. `configure` can select another maximum step, but the legacy fidelity gate uses 2.5ms. No new integrator or dynamics model. |
| [World constraints](../../flightlib/src/objects/quadrotor.cpp) | Per-lane box[6,N], applied after the control interval | Original trajectory suite covers all axes. Preserve old-position x/y clamp and z-to-lower-bound behavior, including retained vertical velocity. This is not contact response. |
| [getState / getMotorOmega / getMotorThrusts / getDynamics / pose / mass](../../flightlib/include/flightlib/objects/quadrotor.hpp) | C++ views contain physical state, motor and derived model data; `ModelConfig` contains setup metadata | No per-drone CPU object is required. CUDA-8 must expose standalone core Python views/accessors and explicit snapshots. Raw source parameters are host configuration, not reconstructible in full from every derived table. |
| [setSize / getSize / collision flag / cameras](../../flightlib/include/flightlib/objects/quadrotor.hpp), [UnityBridge collision exchange](../../flightlib/src/bridges/unity_bridge.cpp) | No scene geometry computation or camera backend | Size is scene metadata, not an inertia input. Reserved collision tensors do not implement Unity flag passthrough. If exposed later, externally supplied flags must remain distinct from computed geometry and task termination. Rendering/sensors are deferred. |
| [QuadrotorEnv observations / rewards / termination](../../flightlib/src/envs/quadrotor_env/quadrotor_env.cpp) | Existing task consumer: Eigen ZYX convention, legacy action scaling/reward, altitude terminal test, explicit/random reset | `environment_test` compares actual CPU outputs/rewards, including saturated symmetry and Euler boundaries. `max_t` does not create a new timeout. Existing Python Environment is retained; no RL wrapper expansion. |
| [Random reset transform](../../flightlib/src/envs/quadrotor_env/quadrotor_env.cpp), [CPU RNG and setSeed](../../flightlib/include/flightlib/envs/env_base.hpp) | Device lane-key/counter RNG, ten uniform samples; same position/velocity construction and normalized four-component quaternion sampling | `rngAndBranches` checks seed, grouping, selection partition, normalization/ranges and reset counters. CPU uses random-device-seeded mt19937; its `setSeed` calls `srand` and does not reseed that engine. CPU/GPU sample-sequence identity is explicitly unsupported. |
| Failures / topology / graph execution | Per-lane status; failed task output invalidated; no silent numerical failure; fixed E*A lane identities | Runtime, lifecycle, dynamics and environment tests check rejection/isolation, mixed active lanes, graph invalidation and exact graph/native replay. Diagnostic failure handling is an explicit CUDA contract, not a claim that the CPU handles overflow transactionally. |

Model domains are those validated by `prepareModels`; this is a stricter finite,
positive, nonsingular setup boundary than some individual legacy setters.
Arbitrary thrust polynomials can still yield invalid inverse roots during a step;
those lanes report `NumericalFailure`. The audit does not promise every tuple
accepted by a permissive CPU setter will produce usable dynamics.

Explicit state input supports caller-sampled positions, velocities, quaternions
and bias values without adding normalization. Keep the legacy run quirk: CPU
normalizes the old quaternion then overwrites it with RK4 output. The CUDA path
does not introduce a normalization or reinterpret bias fields as sensor noise.
Model updates do not automatically recompute task action scaling; configure task
parameters explicitly when that behavior is intended.

## Numerical discrepancy and disposition

The parallel benchmark's apparent optimized-CPU issue also reproduces in the
frozen O0 CPU build. With the repository YAML model, identity state, 20ms interval
and saturated equal rotor commands, CPU allocation produces tiny nonzero torque
(approximately -7.97e-8, 3.95e-8, -1.73e-9 Nm). Its tiny negative yaw selects
Eigen's alternate ZYX Euler branch near (pi, pi, pi). The old CUDA allocation
cancelled to zero, producing (0,0,0) and a different orientation reward.

Eigen's fixed-size column-major packet product uses a rounded first multiply
followed by explicit `pmadd` intrinsics (`Core/ProductEvaluators.h`). On this
native x86/FMA reference build those are fused even with `-ffp-contract=off`.
CUDA globally retains `--fmad=false`; explicit `fmaf` now matches that sequence
in forward/inverse allocation and the motor polynomial. Inertia preparation
also retains Flightlib's double scalar intermediate before Eigen float storage.
No equation, tensor, kernel boundary or global fast-math switch changes.

`scalarEnv(true)` runs 120 saturated symmetric YAML steps and checks both returned
and transition observations, reward, done, validity and episode/clock behavior
against the actual CPU environment. It runs against frozen and optimized CPU.
A negative-control build with the old allocation accumulation failed at step 0,
observation field 3: GPU 0 versus CPU 3.141593. Angle equivalence is not an
acceptance rule; the existing observation/reward thresholds remain in force.

This is a qualified arithmetic profile, not bitwise parity with arbitrary CPU
architectures or Eigen builds. In particular, Eigen's default `EIGEN_FAST_MATH`
packet square root uses an x86 reciprocal-square-root estimate and one Newton
iteration (`Core/arch/SSE/MathFunctions.h`), even in these strict compiler builds.
CUDA's square root need not yield the same last bits. The randomized test reports
the CPU packet-versus-scalar inverse-motor difference separately and bounds it
at .002 rad/s. Transcendental arithmetic and accumulated FP32 drift remain bounded
by the following gates; we do not emulate the x86 estimate or alter the oracle.

## Differential coverage and tolerances

All relative budgets below use `max(1,abs(CPU reference))`. These are regression
budgets for the specified trajectories, not universal error bounds.

| Fixture | Coverage / gate |
| --- | --- |
| Original dynamics suite | 33 lanes, two models, 300 intervals per dt through .01s, nonunit quaternion, both commands, world limits, failures. Integrated fields 3e-4; acceleration/torque/bias 1e-4; motors .02 rad/s and 1e-4 N. Unchanged. |
| New randomized trajectories | 17 lanes, seven models (default, actual YAML, five deterministic sampled models), 160 intervals at .006125/.02/.05s. Saturated startup, mode changes, live mass/arm/tau/kappa update at 60, setState at 100, reset at 120. All 25 fields and motors compared every interval. Position/quaternion/velocity/rates **and acceleration** 3e-4; torque/bias 1e-4; same motor thresholds. |
| Task compatibility | Existing observation gate 5e-4, scalar reward 2e-4, direct Euler fixtures 2e-6. New saturated fixture uses these unchanged gates. Discrete outputs, time and counters exact. |
| Random/graph execution | GPU reproducibility across lane grouping, selection partition and native/graph paths exact; CPU random sequence excluded. |

The new 50ms randomized case reaches five seconds before its explicit state
replacement. Its acceleration drift exceeds the original short-step suite's
1e-4 gate (observed about 1.46e-4), hence the explicitly separate 3e-4 budget,
matching its integrated-state budget. Existing tests are not weakened. Numerical
rounding can still cross discontinuous Euler or terminal boundaries in untested
trajectories; passing these fixtures is not proof of universal reward equivalence.

The sampled parameter ranges are mass .5–1.8kg, arm .15–.35m, motor tau .01–.08s,
motor minimum 100–200rad/s, maximum 1800–3000rad/s, kappa .01–.025, polynomial
coefficients scaled .85–1.15, and rate limits 2–8rad/s. YAML supplies the additional
fast-motor case. Larger domains and longer open-loop trajectories require their
own measured qualification before claiming equivalent training behavior.

## Remaining layers

CUDA-8 now exposes native lifecycle/dynamics operations as a standalone resident
core package, with model/motor/state access, optional helpers and explicit
stream/ownership rules; the separate Core interface does not depend on the task
API. CUDA-9 adds headless wrappers, resident RL usage and a training/evaluation
example, then qualifies clean installation and the integrated checkpoint,
including matched best-tested parallel CPU performance. Geometry, collision
solvers, rendering, sensors and additional physical models remain deferred.
See [CUDA-7 validation evidence](cuda7-validation.md).
