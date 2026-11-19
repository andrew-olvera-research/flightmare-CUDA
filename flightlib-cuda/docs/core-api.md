# Resident Flightlib core API (CUDA-8)

`flightlib_cuda.Core` exposes the headless physical core independently of the
existing task `Environment`. The C++ entry points are in `lifecycle.hpp`,
`dynamics.hpp` and `operations.hpp`. Both use the same persistent arena, model
tables, kernels and frozen v1 tensor semantics.

The [non-RL Python example](../python/examples/core.py) runs multiple environments
and drones, commands/steps on device, edits selected state and explicitly exports
a CPU snapshot. `verify` installs to a fresh directory, executes that example
using the installed module, and builds/runs an independent C++ consumer through
`find_package(flightlib_cuda 0.2 CONFIG REQUIRED)` and `flightlib_cuda::runtime`.

## Resident interface

```python
from flightlib_cuda import Core, CommandMode

sim = Core(environments=1024, slots=4, control_dt=.01)
sim.state_input[2].fill_(5)
sim.reset()
sim.command_mode.fill_(CommandMode.ROTOR_THRUST)
sim.command_payload.fill_(3)  # N per rotor; [4,N]
sim.capture(commands=True)
for _ in range(100):
    sim.step(commands=True, graph=True)
with sim.access():
    device_position = sim.position  # borrowed [3,N], not history
snapshot = sim.snapshot()           # deliberate synchronized host copies
```

N=E*A; lane=e*A+slot. Core outputs retain component-major storage rather than
materializing per-drone objects. `state[25,N]`, `time[N]`, `motor_omega[4,N]`,
`motor_thrust[4,N]` and `status[N]` are exported once with retained DLPack owners.
`position`, `quaternion` (wxyz), `velocity`, `body_rates`, `acceleration`, `torque`,
`angular_bias` and `acceleration_bias` are zero-copy slices. Model parameter views
are indexed by P, with `topology['model_id'][N]` selecting each lane's model.
All floating tensors are float32 even if PyTorch's default dtype differs.

`state_selected[N]` uint8, `state_input[25,N]` and `time_input[N]` feed `reset`
and `set_state`. `command_selected[N]` uint8, `command_mode[N]` uint8,
`command_time[N]` and `command_payload[4,N]` feed commands. Bind alternatives once
using `bind_state` / `bind_command`; no implicit dtype conversion, contiguous
copy, host staging or tensor resize occurs on submission. Masks must be 0 or 1.

| Operation | Semantics |
| --- | --- |
| `reset()` | Assign selected state/time and zero their motors. Accepted commands and task counters remain intact. Inputs initially contain the legacy zero state (identity quaternion), so resetting a fresh Core gives that state. For later zero resets, refill staging buffers explicitly. |
| `set_state()` | Assign selected state/time, preserving running motors and accepted commands. No quaternion normalization is added. |
| `set_command()` | Accept and clamp selected inputs. Rejected commands leave accepted commands intact. Does not step physics. |
| `step()` | Advance every active lane using its last accepted command and configured schedule: one fused dynamics kernel. |
| `step(commands=True)` | Accept selected commands and advance in one host submission: command kernel then fused dynamics. A rejected selected command does **not** advance that lane. Unselected active lanes use prior accepted commands. |
| `capture(commands=...)` | Capture either repeated workload at setup without executing it. Replay must use the same `commands` setting. One graph API launch; one or two kernel nodes. |
| `configure(...)` | Explicit synchronous model/topology/world-box/schedule setup. Optional `models`, `model_ids`, `active`, `bounds`, `control_dt`. Rejected validation preserves live configuration and graph; successful setup preserves state/motors/commands but invalidates graphs. |
| `snapshot()` | Explicit host copies of physical state, motors, status, accepted commands and external scene metadata. Never called by step. |

Typed modes are `CommandMode.ROTOR_THRUST` (four rotor forces in N) and
`COLLECTIVE_BODY_RATES` (collective acceleration, body rates in rad/s). Preserve
the CPU's numeric collective clamp using its cached thrust limits, despite the
units difference. `accepted_command` exposes read-only-by-contract `mode`,
`time`, `payload`; it is separate from proposed inputs. A later plain `step`
can still run the old command after a rejected command operation.

Statuses are `OK`, `INACTIVE`, `INVALID_COMMAND`, `INVALID_STATE`, `INVALID_RESET`,
`NUMERICAL_FAILURE`. Inspect them on device. Each operation overwrites status;
it is not a sticky error history. Unselected active lifecycle lanes report OK
without changing data. Rejected lane updates retain prior data; failures do not
invalidate neighboring lanes. Exceptions indicate host schema/setup/ownership
errors. CUDA launch failures poison the runtime as before.

`models` returns copies of accepted host `ModelConfig` objects. Modify a copy and
pass it to `configure(models=...)`; editing it alone cannot silently alter device
parameters. `parameters` exposes mass, diagonal inertia/inverse, allocation and
inverse, thrust polynomial, motor time constant inverse, speed/thrust bounds,
body-rate limits, rate gains and gravity. `topology` exposes IDs, active mask and
world bounds; `substep_dt` exposes the exact float schedule (None for zero dt).
Parameter/topology/accepted-command tensors are read-only by contract. Physical
views may be edited under `access`, but direct writes bypass lifecycle validation;
prefer the validated state/command inputs for routine changes.

Fixed E/A/P/substep capacities are intentional. A timestep with a different
number of substeps requires a new Core; it is not hidden reallocation in `step`.
`prepare_schedule(dt)` reports the legacy sequence. All underlying model values
are configurable at setup; sampling/randomization policy stays caller-owned.
Core does not introduce automatic task reset, rewards, sensors or noise dynamics.

## Optional low-level operations

Public CPU helpers are available without forcing users to duplicate equations.
Allocate input/output tensors once and bind an `Operation`:

```python
import torch
from flightlib_cuda import CoreOperation
samples = sim.state.clone()
derivative = torch.empty_like(samples)
op = sim.bind_operation(CoreOperation.DERIVATIVE, samples, derivative)
op.run()  # fills derivative on device; no allocation or host readback
```

| CoreOperation | Input | Output | CPU counterpart / side effects |
| --- | --- | --- | --- |
| `DERIVATIVE` | [25,N] state | [25,N] derivative | `QuadrotorDynamics::dState`; does not change physical state. A bound operation is the callable replacement for `getDynamicsFunction`. |
| `CONTROLLER` | [3,N] measured body rates | [4,N] desired rotor force | `runFlightCtl`; uses accepted collective/rate command and current state rates for the legacy cross term. Does not change motors. CPU ignores its dt argument. |
| `MOTORS` | [4,N] desired rotor force | [8,N] omega then thrust | `runMotors`; updates persistent motor state only, with `run(dt=...)`. Does not advance time or pose. |
| `OMEGA_TO_THRUST` | [4,N] angular speed | [4,N] force | `motorOmegaToThrust`, raw polynomial without force clamp. |
| `THRUST_TO_OMEGA` | [4,N] force | [4,N] angular speed | `motorThrustToOmega`, raw inverse without speed clamp. |
| `CLAMP_THRUST` | [4,N] force | [4,N] clamped force | Scalar/vector `clampThrust`; same cached legacy bounds. |
| `CLAMP_MOTOR_OMEGA` | [4,N] angular speed | [4,N] clamped speed | `clampMotorOmega`. |
| `CLAMP_BODY_RATES` | [3,N] body rates | [3,N] clamped rates | `clampBodyrates`. |
| `CONSTRAIN_WORLD` | [25,N] previous state | [25,N] constrained current state | `constrainInWorldBox`; also writes current physical state, retaining time/motors. Previous-state time is not an input because constraints use previous xyz only. |

All helpers use current per-lane models and topology and update shared status.
Input/output must be disjoint dense float32 CUDA allocations, including no alias
with the arena. Inactive/invalid lanes retain previous output; check status before
using it. Nonfinite inputs are rejected, and nonfinite derived results report
numerical failure without partial lane writes. This is the explicit CUDA finite
input/failure contract, not an imitation of every legacy NaN edge case.
Helper handles retain Core and tensor owners; storage must remain fixed. These
are optional single launches, not extra stages inside normal stepping. Shared
device functions keep their formulas identical to the fused dynamics path.
Standalone helper graph capture is not exposed; fused `step` is the efficient
repeated simulation path. No helper scratch buffers are allocated by Core.

## Public CPU surface checklist

Source anchors: [Quadrotor](../../flightlib/include/flightlib/objects/quadrotor.hpp),
[QuadrotorDynamics](../../flightlib/include/flightlib/dynamics/quadrotor_dynamics.hpp),
[QuadState](../../flightlib/include/flightlib/common/quad_state.hpp),
[Command](../../flightlib/include/flightlib/common/command.hpp).

| CPU surface | Accelerated equivalent / disposition |
| --- | --- |
| constructors, `init`, reset overloads, state setter | Core/configure and explicit lifecycle inputs; identity/zero reset data can be reused. Core owns one batched arena, not one CPU object per lane. |
| both `run` overloads, `setCommand` | `step`, `step(commands=True)`, `set_command`; invalid-command distinction tested. |
| state, position, quaternion, mass, motors, dynamics getters | Resident views, `parameters`, copied host `models`, explicit snapshot. `getJ` / inverse are diagonal [3,P]; reconstruct a full matrix only if the caller needs it. Allocation is [16,P], row*4+column. |
| `valid`, `updateParams`, `setMass`, `setArmLength`, `setMotortauInv`, `updateDynamics`, `setWorldBox` | Validated `ModelConfig`, `load_model`, `configure`; motor_tau is seconds (inverse of setter argument). Strict positive/finite setup domain remains documented in CUDA-7. |
| derivative/controller/motor/conversion/clamp functions | Bound operations listed above; `getAllocationMatrix` is the parameter view. Collective minimum/maximum are four times the stored rotor thrust bounds. |
| `getSize` / `setSize`, `getCollision` / `setCollision` | `size[3,N]` float32 and `collision[N]` bool, separate caller-writable scene metadata. Defaults ones/false; reset, step and configure leave them unchanged. They do not affect inertia, constraints or termination. |
| cameras, Unity/ROS exchange, IMU/sensor outputs, rendering | Use Environment and its [sensor, renderer and ROS adapters](../../README.md#choose-an-interface) for these operations. |
| stream insertion / CPU object copies / STL callbacks | Python repr/object identity and C++ implementation details are not serialized ABI promises. Numerical/controller behavior is provided through the batched API above. |

`collision` is external metadata, not the reserved geometry `collisions.hit/valid`
query result. No engine updates it here. C++ consumers likewise own external
scene metadata separately from the frozen simulation/query views.

## Ownership, streams and validation

Input owners are retained for simulation lifetime (including replaced bindings).
Rebinding belongs to setup, not a per-step loop. Exported views retain the arena
after the Core wrapper is destroyed. Changing input values is supported;
resizing storage is not. A successful command rebind invalidates the wrapper's
captured graph; state-input rebind does not, because a step graph does not read it.

Each submission orders the selected producer stream into the private simulation
stream and back with persistent events, without host synchronization. Use
`wait` before another consumer stream reads and `record_usage` afterward, or
`with sim.access(stream): ...`. This also covers external writes and helper
buffers. Arbitrary enclosing framework graph capture is rejected. Snapshot and
setup are explicit completion boundaries. No automatic autograd is supported.

The core shares the existing arena allocation plan, including unused task
reservations; it executes no task kernels. Separate staging plus size/collision
metadata uses 140 bytes/lane, all allocated at setup. Caller-bound helper buffers
are additional, explicitly owned allocations. No frozen buffer changes occur.

Validation includes complete physical-state/motor/time replay against frozen CPU
traces, all nine helpers against actual CPU methods, mixed model/active/selected
lanes, live setup, invalid commands, native versus graph submission, nondefault
streams, alias/dtype/layout rejection and installed Python/C++ consumers. Existing
Environment tests remain in the gate. CUDA-7 numerical budgets and cross-platform
limitations still apply; long YAML-hover trace uses the documented 3e-4 state
and acceleration budget, 1e-4 torque/bias budget. CUDA-9 remains integrated release
qualification; whole Flightmare rendering/RL parity is outside this core package.
