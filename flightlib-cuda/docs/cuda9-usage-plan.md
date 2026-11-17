# CUDA-9: headless usage and RL completion

Status: usage/RL implementation and benchmark milestone implemented, version 0.1.0.
Baseline: CUDA-8 commit e01d077; planning checkpoint 23082b9.
See [implemented contracts/method mapping](cuda9-usage-contract.md) and
[validation/performance evidence](cuda9-validation.md). Tensor/host wrappers,
scalar and lane-zero execution, configuration/seed lifecycle, resident collection,
GAE/PPO, save/load/evaluation and installed-package checks are implemented.
The public README/migration presentation and clean release qualification remain
the next turn's operations. This milestone does not close or publish CUDA-9.
This is the final implementation layer for this public rewrite. It extends the
previous core-only scope to the headless user and RL interfaces. Rendering,
scene collision engines, cameras and all sensors remain later-project work.

## What completion and release mean

Completion means users can install the packages, configure the original
quadrotor task, run single/batched simulation, collect resident rollouts, train,
save, reload and evaluate a policy, with documented behavioral and performance
evidence. Core equations and the original oracle remain unchanged.

Release qualification means a reproducible build/install and validated committed
checkpoint. The local `release` profile runs `verify` and requires a clean tree
and an unused version tag. It does not publish, upload wheels, push or create a
tag. Keep 0.1.0; tagging/publication is a separate deliberate action, not another
implementation milestone. No cloud CI or new service is required.

"Full parity" here means the supported headless simulation/task/usage surface,
with every legacy entry classified. It does not mean universal bitwise floating
point equivalence, identical CPU/GPU random sequences, identical learning curves,
or compatibility with every old TensorFlow checkpoint. Those distinctions must
appear in the public documentation, not be hidden behind a parity claim.

## Findings from the checked-in usage layer

| Source | Existing behavior / gap to address |
| --- | --- |
| `flightlib/src/wrapper/pybind_wrapper.cpp` | `flightgym.QuadrotorEnv_v1` exposes configuration constructors, reset/step/testStep, seed, close, dimensions, extra-info names and curriculum hooks. Audit every binding, including stubs. |
| `flightlib/src/envs/vec_env.cpp` | Per-lane terminal reward is added and reset observation returned in the same step. `testStep` advances lane zero only. `isTerminalState` is an empty function, not a working query. |
| `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp` | 12 observations, four normalized-action inputs, physical thrust scaling, altitude termination. `max_t` is metadata, not an enforced timeout. Single-environment stepping does not auto-reset. |
| `flightrl/rpg_baselines/envs/vec_env_wrapper.py` | NumPy copies every step; per-lane reward lists and info dictionaries on CPU. Episode return/length and reset bookkeeping belong in the new wrapper. Async/reflection methods are currently unimplemented. |
| `flightrl/rpg_baselines/envs/env_wrapper.py` | Single-env usage exists, but gate/state methods reference APIs not provided by the checked-in `flightgym` binding. Classify these honestly rather than invent a gate simulator. |
| `flightrl/rpg_baselines/ppo/ppo2.py` and `common/distributions.py` | TensorFlow/NumPy rollout and learner; policy applies tanh before environment submission. Moving physics alone cannot make this pipeline resident. Preserve the action/log-probability convention in a qualified tensor example. |
| `flightrl/examples/run_drone_control.py` | Training, save/load and evaluation user journey; configurations depend on `FLIGHTMARE_PATH`. Installed packages need packaged defaults and explicit paths instead. |
| `flightlib/src/envs/test_env.cpp` | `TestEnv_v0` is a fixed-array binding demonstration, not another task requiring a physics implementation. |

## Package and execution boundaries

Use a separate `flightrl-cuda` distribution (`flightrl_cuda` import) depending on
`flightlib_cuda`. It owns adapters, configuration, episode accounting and rollout
utilities. It does not duplicate equations, rewards, reset sampling or simulation
storage. Keep legacy `flightgym` and `rpg_baselines` names intact; GPU selection
must be explicit and must never silently fall back to CPU.

```
user configuration / train / evaluate
                  |
       flightrl_cuda tensor vector API
                  |
       flightlib_cuda.Environment ---- Core (independent non-RL use)
                  |
       persistent arena + existing CUDA task/dynamics/reset kernels

legacy-shaped NumPy adapter -> same tensor API -> same simulation
```

The NumPy adapter is an explicit transfer boundary, with separate measurements.
An old CPU-array learner cannot become transfer-free through a wrapper alone.
The accelerated path uses the existing PyTorch stack and no TensorFlow runtime.
Legacy learner/checkpoint migration is documented separately: retained upstream
code remains available, but automatic TF checkpoint conversion and matching all
PPO2 internals are not completion claims for this simulator extension.

## Implementation order within the one ticket

### 1. Freeze usage semantics and configuration

- Publish a source-anchored public-method matrix: implemented equivalent,
  deliberate compatibility no-op, original unimplemented method, or deferred
  rendering/sensor method. Include reset/step/testStep, seed, close, dimensions,
  spaces, extra info, episode info, curriculum and single-env behavior.
- Parse both legacy vector and task/model YAML files (and explicit YAML text),
  plus explicit overrides with documented precedence: packaged defaults,
  supplied configuration, then overrides. Preserve legacy model caches, action
  scaling and reward coefficients. Do not silently assume Core's default model
  or .01s step when the legacy task YAML specifies .02s and another model.
- Package defaults using resource lookup, without source-tree cwd, sys.path
  edits or mandatory `FLIGHTMARE_PATH`. Accept legacy path configuration only as
  an explicit compatibility path. Separate CPU-only `num_threads`, unused scene
  identifiers and rendering fields from meaningful device/task configuration.
  Classify legacy inert camera keys; reject requests for actual rendering rather
  than returning fabricated observations. Report unsupported active settings.
- Freeze setup lifecycle: device selection, fixed E/A/P/schedule capacities,
  model/task updates, deterministic CUDA seed/reseed rules, partial reset and
  graph invalidation/recapture. Expose any missing Environment lifecycle bindings
  using existing native primitives; add only genuinely missing operations.

### 2. Resident tensor vector wrapper

- Primary learner shapes: action `[N,4]`, observation `[N,12]`, reward/done/valid
  `[N]`, N=E*A. Expose zero-copy grouped `[E,A,...]` views. Slots are independent
  agents, not a shared reward/termination team task. Keep dtype and lane order
  explicit; boolean presentation of byte masks must not hide per-step casts.
- Bind persistent action storage once. Support policy writes directly into it;
  an ergonomic `step(actions)` may do one explicit device copy into that buffer.
  Reject wrong-device/layout/type inputs instead of silently staging on CPU.
- Preserve original action-to-thrust behavior; wrapper does not add clipping,
  observation normalization, reward scaling or a time limit. Policy tanh is a
  learner decision matching the example, not a simulator clamp.
- Same-step autoreset returns reset observation plus terminal reward/done;
  separately retain the pre-reset transition observation and terminal mask.
  Default truncation is false; rollout horizon T is not episode termination.
  Numerical failures remain explicit `valid/status`, never fake normal endings.
- Provide single-env/no-autoreset mode and lane-zero `testStep` compatibility
  through narrowly scoped task execution controls if necessary. Do not step all
  lanes then restore them, or alter existing default task behavior.
- Accumulate episode return/length and emit completed episode records on device,
  including the terminal step and reward exactly once. Define selected/manual
  reset accounting and don't inherit stale Python reward lists. Optional legacy
  bookkeeping quirks are classified explicitly in the compatibility matrix.
- Define reset/step, split submission/collection, observation access, metadata,
  seed/reseed and idempotent close. `step_wait` in the tensor API orders consumer
  access on device; host synchronization belongs to the host adapter. Reject
  overlapping submissions. Retained exports must keep storage alive after close.
- Avoid fresh output dictionaries, lane loops, `.item()`, `.cpu()` and `.numpy()`
  in the resident loop. Reuse output/metadata structures and persistent events.

### 3. Host compatibility and usable RL example

- Provide explicitly selected single/vector NumPy adapters with familiar
  spaces, reset/step returns, extra-info/episode dictionaries, seed and close.
  No implicit global replacement of `flightgym`. Match supported legacy methods
  and document original stubs; a useful new terminal query must be distinguished
  from the old no-op. Unsupported Unity/sensor calls fail clearly.
- Isolate optional Gym/legacy learner dependencies from the tensor package.
  Pin and validate any claimed framework adapter in its own environment; naming
  something VecEnv is not proof that a particular framework accepts it. A modern
  five-return API, if provided, must map terminated/truncated explicitly.
- Supply a compact PyTorch PPO train/evaluate entry point as a usage example,
  not a new general-purpose RL framework. Include resident policy inference,
  fixed-capacity rollout history, GAE/returns, PPO updates, periodic metrics,
  save/load and deterministic evaluation. Explicitly document supported policy
  shape and algorithm settings; do not claim TF optimizer/checkpoint identity.
- Freeze history shapes `[T,N,12]` observations, `[T,N,4]` actions and `[T,N]`
  rewards/dones/valid/values/log-probabilities; optional terminal observation
  history only when needed. Borrowed simulator outputs must be copied into the
  correct resident history slot before the next step overwrites them.
- Test GAE/bootstrap masks independently: terminal, horizon boundary and invalid
  sample handling. Store the policy sample needed for the chosen tanh/log-prob
  convention. Include optimizer/RNG/config metadata for training resume, and
  distinguish a learner resume with reset environments from exact mid-rollout
  simulator restoration; the latter is not promised without full RNG/state data.

### 4. Systems and behavioral qualification

- CPU differential traces cover wrapper-visible observations, scaled actions,
  rewards, terminal reward, auto-reset routing, terminal observations and episode
  accounting. Use matched explicit resets/random inputs where needed; CPU seed
  sequence identity is not supported. Existing oracle/tolerances stay unchanged.
- Focus regression cases on partial resets, invalid/inactive lanes, non-warp
  tails, multi-slot flattening, config overrides, reseeding, lifetime/streams,
  stale graphs, no-autoreset and lane-zero execution. No repetitive getter tests.
- Extend the Python systems harness with wrapper-only, policy+wrapper, resident
  rollout collection, learner-update and host-adapter workloads. Measure startup,
  capture and checkpoint/logging separately from hot-loop execution. Include
  launch counts, bytes/copies, scratch and rollout VRAM, allocator behavior,
  completed latency and amortized throughput. Attribute wrapper overhead against
  the exact same direct Environment workload.
- Retain existing three-node fused task path (or one graph submission with three
  nodes). Account separately for action/history copies and optional statistics
  kernels; fuse bookkeeping only if measurements justify it. No per-lane launches
  or host decisions. Graph capture of the entire learner is not a prerequisite;
  use supported environment graphs and explicit streams first.
- Hard gate: no host/device traffic or host synchronization inside the resident
  collection loop, and no simulator/wrapper/rollout allocation after setup.
  Policy/optimizer temporary allocations must be measured with bounded steady
  memory; cached scratch is not evidence of zero allocations. Logging/checkpoint
  exports are explicit boundaries. Device history traffic is necessary work,
  accounted separately from avoidable simulator intermediates.
- Use paired repeats and existing 1.15x matched wall-time regression gates for
  unchanged paths; establish a separately reviewed wrapper/rollout baseline.
  Benchmark GAE/learner launch overhead too, not only physics. Preserve best-tested
  OpenMP/multiprocess CPU comparisons for equivalent environment work. Compare
  training throughput only with identical learner/workload; no sequential-CPU or
  different-framework ratio may be presented as a training speedup.

### 5. Finish the public package

- Fresh build/install/import of core and RL packages from outside the checkout;
  run installed native/core, tensor rollout and train-save-load-evaluate examples.
  Record supported software/hardware versions and optional adapter dependencies.
- Make root README a usable entry point: quick start, CPU-to-GPU migration,
  configuration reference, shapes/lifetimes, errors, performance reproduction,
  capabilities and explicit exclusions. Preserve upstream attribution/license.
- Run local verify plus applicable profiling/install gates, update the support
  matrix/changelog, commit and qualify the clean checkpoint. Record unresolved
  sanitizer/hardware-counter/multi-GPU coverage; unavailable tools are not passes.
- Close CUDA-9 only with evidence for all implemented items. No rendering or
  sensor project is required, and no mandatory CUDA-10 follows this ticket.

## Final boundary

In scope: existing quadrotor dynamics/model/state core; existing task semantics;
usable single/vector headless APIs; resident RL integration; explicit host
compatibility; training/evaluation example; reproducible packaging and validation.

Deferred: Unity fidelity, visual rendering, all sensors, scene collision queries,
ROS/PX4/SITL, new physical models/platforms, team tasks, distributed training,
automatic TensorFlow checkpoint conversion and a general RL algorithm library.
Legacy already-unimplemented APIs are labeled, not silently advertised as parity.

Dependencies: [core API](core-api.md), [numerical contract](flightlib-support.md),
[scope](flightlib-scope.md), [local validation](../../docs/local-development.md).
