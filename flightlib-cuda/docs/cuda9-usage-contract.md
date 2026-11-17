# CUDA-9 usage contracts

Implementation contract for `flightrl-cuda` 0.1.0. This package composes the
existing Environment; it does not own a second dynamics implementation. The
frozen v1 simulation arena, state layout, model preparation and equations remain
unchanged. Packaging/public README and release qualification are separate final
operations. See [validation](cuda9-validation.md) for the measured envelope.

## Storage and ordering

N = E * A, lane = environment * A + slot. Slots are independent tasks. All
learner tensors are dense float32 unless listed otherwise. Construction fixes
capacity, device and the submission Torch stream. Nondefault construction streams
are supported; submission from another stream is rejected. Core's more general
stream API remains available separately.

| Owner | Tensor | Shape / type | Lifetime |
| --- | --- | --- | --- |
| Existing arena | Physical state, models, clocks, task outputs | Frozen v1 definitions | Simulation ownership |
| Environment input binding | Actions | `[E,A,4]`, flat borrowed `[N,4]` | Persistent; mutate values only |
| TensorEnv result | Observation, transition observation | `[N,12]` | Borrowed arena views, overwritten by subsequent work |
| TensorEnv result | Reward; done and valid | `[N]` float32; `[N]` uint8 | Borrowed arena views |
| TensorEnv | Running/completed episode return | `[N]` float32 | Persistent wrapper-owned storage |
| TensorEnv | Running/completed episode length | `[N]` int64 | Persistent wrapper-owned storage |
| TensorEnv grouped result | Each result field | `[E,A,...]` | Zero-copy views of the flat result |
| Rollout | Observation | `[T,N,12]` | Owned history, copied before the next step |
| Rollout | Action, latent Gaussian sample, fixed/resampled noise | `[T,N,4]` each | Owned history |
| Rollout | Reward, value, log probability, advantage, return | `[T,N]` each | Owned history |
| Rollout | Done, valid | `[T,N]` uint8 | Owned history |
| Rollout | Bootstrap value | `[N]` | Owned history |
| Policy | Inference workspace | N by fixed network widths | Persistent, policy graph bindings |

`step()` uses persistent actions. `step(actions)` validates device, float32,
shape, contiguity and absence of autograd, then makes at most one explicit D2D
copy. `step_async` submits; `step_wait` orders statistics on the construction
stream and returns the same Transition object. It does not wait on the CPU.
Overlapping submissions are errors. Retained output tensors keep the underlying
storage alive after idempotent `close`; close is an explicit synchronization.

Reset respects `engine.selected`, explicit reset state/time and validity status.
Only successful selected resets clear wrapper statistics. Seed/reseed restarts
the existing per-lane CUDA RNG keys/counters and invalidates graph use, without
changing physical state. CUDA reproducibility is supported, CPU RNG sequence
identity is not. Accepted Config/model metadata is copied to prevent accidental
mutation from misrepresenting device configuration. `configure` validates setup,
preserves physical state and requires recapture; capacity/device changes require
a new wrapper. Reset mode/autoreset changes also require recapture.

## Work decomposition

For each resident collection step:

1. Copy current borrowed observation to its history row; replay policy inference.
2. Save latent sample/action/value/log probability; copy action to the fixed input.
3. Replay the existing three-node environment graph: action preparation,
   fused dynamics, and fused observation/reward/terminal/reset.
4. Launch one episode-statistics kernel, one thread per lane. No lane decisions
   leave the device. Copy reward/done/valid to history.

One final policy inference provides the horizon bootstrap. GAE is one kernel,
one thread per lane looping backward over T; no scratch history is required.
Terminal and invalid transitions block bootstrap/recurrence. The independent CPU
implementation is a reverse Torch loop. All stored history is owned, not an alias
of the simulation's reusable output buffer.

Policy inference is captured separately from the environment graph; nested
framework/native graph capture is not supported. PPO updates use ordinary Torch
autograd, gradient clipping and Adam. Their temporary allocations and memory peak
are distinct from the allocation-free collection contract. Save/log/evaluation
exports are explicit host boundaries.

The example network is a float32 12-128-128 shared ReLU MLP with four actor
outputs, one value output and learned diagonal Gaussian log standard deviation.
Actions are tanh of Gaussian latent samples. PPO ratios use the stored latent
sample; its fixed tanh Jacobian cancels between old/new log probabilities. The
entropy term is latent Gaussian entropy. Full-batch clipped PPO is an example,
not an assertion of TensorFlow PPO2 optimizer/checkpoint equivalence. Checkpoints
contain policy, optimizer, configuration, update count and RNG metadata. Loading
returns that metadata; it does not restore a simulator mid-episode or silently
restore global RNG state.

## Task and host API mapping

Source anchors: [native binding](../../flightlib/src/wrapper/pybind_wrapper.cpp),
[vector environment](../../flightlib/src/envs/vec_env.cpp),
[task](../../flightlib/src/envs/quadrotor_env/quadrotor_env.cpp),
[legacy vector adapter](../../flightrl/rpg_baselines/envs/vec_env_wrapper.py),
[legacy single adapter](../../flightrl/rpg_baselines/envs/env_wrapper.py).

| Legacy behavior / method | New interface and disposition |
| --- | --- |
| Configuration constructors | `Config.from_yaml` accepts vector/task paths or text, then explicit overrides. Defaults are package resources. No mandatory source cwd or FLIGHTMARE_PATH. |
| Vector reset / step | `TensorEnv` and explicit `NumpyEnv`. Same-step autoreset, terminal reward and reset observation; separate pre-reset observation retained. No added clipping, normalization or timeout. |
| Scalar QuadrotorEnv step | `TensorEnv(auto_reset=False)`, `SingleNumpyEnv` for N=1. No vector terminal bonus and no automatic reset. |
| `testStep` | `TensorEnv.test_step`, `NumpyEnv.testStep`. Only lane zero advances using existing kernels with one thread. Other rows are deliberately stale; never use as a batched rollout step. |
| `setSeed` / wrapper seed | `seed`, plus host alias `setSeed`. Reproducible CUDA streams, not CPU random-sequence identity. |
| `close` | Explicit synchronized, idempotent lifetime boundary. Retained tensor exports stay alive. |
| Dimensions / extra-info names | Host getters; 12 observations, four actions, N lanes, empty task extra-info names as upstream. Tensor result shapes are the primary resident interface. |
| Spaces / sample actions | Host properties use optional pinned Gymnasium Box spaces. Action bounds [-1,1] describe the policy convention and do not clamp the simulator. |
| Episode return/length | Fused device accounting, copied to host info only in NumpyEnv. Manual successful reset clears selected accumulators; no stale Python reward lists. |
| `reset_and_update_info` | Explicit host snapshot of running episode statistics, followed by reset. |
| `curriculumUpdate` / callback | Deliberate no-op: the source QuadrotorEnv has no curriculum update. |
| Vector `isTerminalState` | Source implementation is empty. Not advertised as a working compatibility query; actual stepped terminal/valid flags are exposed. |
| Wrapper `step_async` / `step_wait` | Original wrapper raises unimplemented errors. The tensor API supplies real ordered submission/collection. |
| `get_attr`, `set_attr`, `env_method` | Original stubs; not new arbitrary reflection promises. |
| `connectUnity`, `disconnectUnity`, rendering/video/sensors | Deferred; host Unity entry points explicitly raise. No fake observations or successful connections. |
| `stepUnity` | Legacy wrapper references an absent native binding; not a working headless capability to reproduce. |
| Single-wrapper `init`, gate/state bridge calls | Not supplied by the checked-in flightgym binding. Physical state is available through Environment/Core; no gate simulator is invented. |
| `TestEnv_v0`, repr | Binding demo/debug representation, not additional physics/task functionality. Legacy module remains intact. |

NumpyEnv is an explicit four-return host adapter with copied outputs; it is not a
Stable Baselines VecEnv subclass or Gymnasium five-return Env. Optional
`flightrl-cuda[spaces]` installs Gymnasium 1.2.2 for Box properties only. No direct
third-party trainer compatibility or automatic replacement of flightgym is
claimed. `max_t` remains original metadata, not truncation; rollout horizon is
not termination. Invalid numerical samples carry valid/status, never fake normal
terminal events. The example CLI rejects invalid or nonfinite training batches.

Rendering/sensors, scene collision geometry, ROS/PX4, new physical models,
distributed RL and automatic TensorFlow checkpoint conversion remain outside
this public rewrite. Public migration/quick-start presentation and clean release
qualification remain the next operations; no tag or publication is performed.
