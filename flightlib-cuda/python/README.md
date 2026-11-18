# Resident Python environment

For physical stepping without task/reward/reset behavior, use the standalone
[`Core` API](../docs/core-api.md). It includes both command modes, selected
state/reset operations, optional derivative/controller/motor helpers, borrowed
model/state/motor views, and [a non-RL example](examples/core.py). `Environment`
below remains a compatibility consumer with unchanged task semantics.

CUDA-6 exposes the existing Flightlib CUDA environment through PyTorch tensors.
Physics, task semantics and the v1 tensor contract are unchanged. The extension
uses pybind11 and DLPack without linking libtorch. Version: 0.1.0.

## Build locally

From the repository root, with the dev container running:

```sh
docker compose exec -T dev bash scripts/setup-cuda-python.sh
docker compose exec -T dev python3 scripts/local_ci.py verify
```

The setup script installs pinned PyTorch 2.9.1/cu130 into the persistent
`/build/flightlib-cuda-python` virtual environment. It uses the container's pytest
and pybind11 development package. Verify builds the extension in
`/build/flightlib-cuda-python-native/python`. To run the complete example:

```sh
docker compose exec -T -e PYTHONPATH=/build/flightlib-cuda-python-native/python dev /build/flightlib-cuda-python/bin/python flightlib-cuda/python/examples/rollout.py
```

For a standalone build, configure `flightlib-cuda` with
`FLIGHTLIB_CUDA_BUILD_PYTHON=ON`, `FLIGHTLIB_CUDA_BUILD_RUNTIME=ON`,
`Python_EXECUTABLE=/build/flightlib-cuda-python/bin/python` and
`CMAKE_CUDA_ARCHITECTURES=120-real` (select your GPU architecture), then build.
`cmake --install <build-dir> --prefix <destination>` installs the importable
`flightlib_cuda` directory; add that destination to PYTHONPATH. This is a local
CMake installation, not a published wheel.

## Use

```python
import torch
from flightlib_cuda import Environment

env = Environment(1024, slots=4, seed=7)
env.reset(random=True)
env.capture(random=True)                 # setup only; no simulation step
for _ in range(100):
    env.actions.fill_(0)                 # replace with GPU policy, in place
    outputs = env.step(graph=True, random=True)
env.synchronize()                       # explicit completion boundary
```

All returned tensors stay on the selected CUDA device. `step()` returns the same
cached dictionary and storage each time. Graph mode replays preparation, dynamics
and fused task/reset as three GPU kernel nodes under one host graph launch.
Native mode uses the identical operations. `capture()` drains registered work
and instantiates the graph; first replay/upload is a separate warmup cost.

| Tensor | Shape | Type |
| --- | --- | --- |
| actions | [E,A,4] | float32 |
| selected reset mask | [N] | uint8 |
| reset_state / state | [25,N] | float32 |
| reset_time / time | [N] | float32 |
| observation / transition_observation | [E,A,12] | float32 |
| reward | [E,A] | float32 |
| done / valid | [E,A] | uint8 |
| status | [N] | uint32 |

N=E*A. `legacy=True` requires A=1, accepts actions[E,4] and exposes
observations[E,12], reward/done/valid[E] as views. Explicit reset uses the
selected mask, reset_state and reset_time; reset observations are not valid
training transitions. Graph reset mode must match the mode selected at capture.
`transition_observation` retains the pre-reset transition; `observation` returns
the reset observation for terminal lanes. Check `valid` for failed lanes.

## Ownership and streams

Inputs must be dense, correctly typed CUDA tensors on the simulation device,
without requires_grad. Binding rejects aliases with the arena or other inputs,
wrong shapes/devices and unverifiable allocation extents. No implicit conversion
or copy occurs. Default PyTorch CUDA allocation is tested; custom/expandable
memory pools whose ranges cannot be verified by cuMemGetAddressRange are rejected.

`bind(actions, selected, reset_state, reset_time)` retains detached storage
owners. Mutate values in place. Do not resize or replace the storage of a bound
tensor; detected pointer changes fail before launch. Replacing a Python tensor
variable does not change the registration: bind explicitly. Rebinding is a setup
operation, invalidates the Python graph, and retains old registered allocations
until the simulation arena is released. Never rebind per step.

Borrowed output tensors keep the arena alive after the Environment is deleted,
but later steps overwrite their values. Clone/copy explicitly for trajectory
history. Exported state is writable; direct edits must obey physical invariants
and ordering, or use reset inputs for validated updates.

Step/reset bridge the current PyTorch stream to the private simulation stream
and back with persistent events. Same-stream policy writes and output reads need
no CPU synchronization. For another stream, use:

```python
consumer = torch.cuda.Stream()
with env.access(consumer):
    saved.copy_(env.outputs['observation'])  # saved allocated beforehand
```

The context waits for prior simulation work and joins external use back to the
owner before future steps or destruction. Low-level equivalents are `wait(stream)`
before access and `record_usage(stream)` after queued work. An explicit
`step(stream=...)` names the producer/consumer stream; it does not move arbitrary
PyTorch operations onto that stream. When final external reads follow the last
step, record their usage before releasing the owner. Concurrent host submission
to one environment is unsupported. External PyTorch graph capture is rejected;
use the explicit environment capture API. This simulator is not differentiable.

`configure_task(...)` and `configure(control_dt)` are synchronous setup boundaries
that invalidate captured graphs. The latter uploads the stored host models,
model_ids, active mask and bounds; changing allocated capacities requires a new
Environment. Rejected configuration leaves the prior graph valid. Input value
changes and resets do not require recapture. `snapshot()` is the explicit CPU
export of task outputs and performs transfers; it is outside resident stepping.

See [systems evidence](../docs/cuda6-performance.md) and the
[profiling harness](../../flightmare-profile/README.md) for measured scope/limits.

### Collective thrust and body rates (CTBR)

Use the same resident four-channel action tensor for rotor forces or CTBR.
Select CTBR at setup, before graph capture.

```python
from flightlib_cuda import Environment, ModelConfig, ctbr_task_config

plant = ModelConfig()
plant.rate_limit = [6., 6., 4.]       # body x/y/z, rad/s
plant.rate_gain = [16.6, 16.6, 5.]   # proportional gains, 1/s
env = Environment(128, models=[plant])
env.configure_task(ctbr_task_config(plant))
```

For normalized actions, zero requests hover acceleration and zero angular
rates. Collective -1 and +1 request 0 and 19.62 m/s². Each signed rate input
is scaled by its axis limit. For physical inputs in m/s² and rad/s, set
the task's action_mean to four zeros and action_scale to four ones before
calling configure_task. Within each physics substep the controller measures
body rate, computes torque using plant inertia and gyroscopic compensation,
and mixes total force and torque through the plant's cached inverse allocation
matrix. Each rotor saturates independently. No extra per-step launch is needed.

For the RL wrapper set quadrotor_env.action_mode to ctbr in the task YAML.
Tune quadrotor_dynamics.rate_gain per plant and omega_max for command limits.
Start from hover, command one axis at a time, and inspect the measured rate
(state rows 10–12), settling time and rotor saturation. Reduce the axis gain
for overshoot or oscillation; increase it for sluggish unsaturated response.
Use the same gains and action scaling when evaluating a trained policy.
