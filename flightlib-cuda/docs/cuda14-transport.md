# CUDA-14 external consumer transport

Use a subscription when a renderer or host reader needs to retain a capture across simulation steps. Use the ordinary borrowed Environment tensors for a policy or estimator that finishes reading them on the submission stream. Flightlib publishes poses and delivered IMU or odometry measurements. Adapters own scene assets, camera calibration, rendered images and protocol serialization.

## Installed API

Pass a tuple of `flightlib_cuda.Subscription` objects to `Environment` or `flightrl_cuda.TensorEnv` at construction. The matching handles are in `env.subscriptions`, in declaration order. Each handle has an independent bounded queue, destination, and pressure policy. This example subscribes one world with 1,000 actors and uses a second CUDA stream for the viewer.

```python
import torch
from flightlib_cuda import Environment, ImuConfig, Subscription

env = Environment(1, 1000, imu=ImuConfig(rate_hz=200), subscriptions=(
    Subscription("pose", period_ns=30_000_000, depth=3),
    Subscription("imu", actors=((0, 7), (0, 12)), destination="host", depth=4),
))
viewer, telemetry = env.subscriptions
render_stream = torch.cuda.Stream()
env.reset_state[2] = 5
env.reset()
env.capture(random=False)

for _ in range(30):
    env.step(graph=True, random=False)
    frame = viewer.acquire(render_stream)
    if frame is not None:
        with torch.cuda.stream(render_stream), frame:
            positions = frame.values[:, :3].clone()  # replace with renderer submission
    packet = telemetry.acquire()                     # nonblocking host poll
    if packet is not None:
        with packet:
            timestamps = packet.packet_acquired_ns.clone()
env.close_subscriptions()
```

A worker may call `acquire(wait=True)` to wait for the oldest queued host capture. It returns `None` immediately when the queue is empty or closed. A host subscriber with pending sensor headers must poll or wait to enqueue its payload transfer. Give each channel one reader and each environment one step/reset/configuration owner. The reader can run on a different host thread. [Installed TensorEnv example](../../flightrl-cuda/flightrl_cuda/transport_example.py)

## Capture semantics

- **Selection and cadence.** Specify unique `(environment, slot)` pairs, or omit `actors` for all lanes. The exported lane ID is `environment * slots_per_environment + slot`. A pose subscription shares one snapshot among all cameras consuming that world at that capture time. A positive integer `period_ns` must span at least one control tick. The first request is one period after startup. Capture occurs on the first completed tick at or after each absolute deadline. Sensor subscriptions publish a batch every control tick containing all newly delivered records for their selection, including drops and invalid measurements with a sequence number. Empty batches are possible.
- **Time and identity.** Frame `sequence` starts at one per subscription and includes skipped captures. `requested_ns` is the absolute request time. `acquired_ns` is the actual capture tick. Both use simulation nanoseconds from startup, independent of sensors and of per-actor episode resets. A pose frame includes per-actor episode, validity, world position XYZ in metres and a unit body-to-world WXYZ quaternion. Rows preserve the actor selection order. Inactive or invalid poses, including zero-norm attitudes, are zero-filled and masked. On a terminal tick the sampled pose and episode belong to the ending transition. The same step still returns the ordinary autoreset observation to its caller.
- **Sensor batches.** `frame.values` follows the six- or ten-channel sensor schema. `packet_sequence`, `episode`, `scheduled_ns`, `packet_acquired_ns`, `available_ns` and `delivered_ns` come directly from the sensor engine. Per-lane rows preserve acquisition order. Relative ordering between lanes is unspecified. A valid lease can outlive a reset. A consumer should match generation, lane and time before applying it. [Sensor semantics](cuda13-sensors.md)
- **Length and ownership.** Host tensors expose exactly `frame.count` rows, including zero. Device sensor tensors expose capacity and a one-element CUDA `frame.count` tensor. Only the first count rows are populated. A GPU consumer can mask by device count without reading it on the CPU. Device pose tensors always have exactly one row per selected actor. All frame fields are borrowed until `release()` or context exit. Enqueue all GPU reads on the stream supplied to `acquire`. Release records a completion event on that stream. Copy on that stream to retain data independently.
- **Pressure and lifetime.** Default viewers skip the new capture when all slots are busy. With `required=True`, a full channel raises `Backpressure` before simulation time advances or any channel publishes. Drain or release the outstanding lease, allow its completion event to finish, then retry the same step. Inspect `channel.stats` for published, acquired, dropped, backpressure, high-water and host byte counts. A channel can disconnect with `channel.close()` while the environment continues. Queued captures are discarded and staging work is drained. Existing leases remain usable until released. `env.close_subscriptions()` disconnects all channels. `TensorEnv.close()` does this automatically.

## Execution and storage

Construction fixes the actor selection, graph variants, staging capacity and pinned buffers. Graph capture is explicit. On a due pose tick with a free slot, the runtime runs through the terminal decision, packs selected poses, and finishes the reset on the owning stream. A tick with no accepted pose capture can use the original fused task/reset graph. A slow GPU reader holds the snapshot allocation and its own completion event. Only the gather has to precede reuse of the original simulation state.

For host poses the count is known at setup. For host measurements a gather compacts due records on device, then a dedicated transfer stream copies a four-byte count. After that event completes, the reader submits exactly the populated prefix into a pinned host slot. The device staging allocation can be recycled at transfer completion even while the consumer retains that host frame. Set `staging_depth` below `depth` to trade device memory for less tolerance of transfer delays. Depth and staging depth are per subscription. GPU leases are same-process and same-device. Process transport and renderer-specific pose or image uploads belong to the adapters.

The packed row widths are 48 bytes for a pose, 80 for IMU, and 96 for odometry. Python exposes typed views without copying individual fields. The native layout and entry points are in the [transport header](../include/flightlib_cuda/transport.hpp).

## Qualification on RTX 5070 Ti

One world with 1,000 agents, 10 ms control ticks, captured graphs, no policy or renderer. CUDA-13's existing completed-step benchmark measured 17.87 µs with sensors disabled. Rerunning the same benchmark on CUDA-14 gave **17.82 µs** and the same 558,056-byte native arena.

The separate transport benchmark measures completed wall time, including Python consumption, over nine batches of 20 steps. The device viewer acquires and releases four poses each tick. For the host cases the benchmark deliberately waits on the submitting thread for both the count and the payload. These measurements include that blocking reader and a 200 Hz ideal IMU running on all 1,000 actors.

| Subscription | Median µs/step | Actual device-to-host bytes/step | GPU kernels/step |
| --- | ---: | ---: | ---: |
| Disabled | 18.42 | 0 | 3 |
| 4 device poses | 34.32 | 0 | 5 |
| 1 host IMU | 156.12 | 164 | 5 |
| 4 host IMUs | 151.18 | 644 | 5 |

Nsight Systems recorded 20 graph launches in 20 steps for every case. The host payload doubled each actor's 80-byte record because two IMU packets were due per tick. Each batch added one four-byte header. The device cases recorded zero memory copies. None of the four traces contained a device-wide synchronization, allocation, or event creation during the measured window. The benchmark synchronizes its producer stream once at the end of each timing batch.

The regression suite exercises a 1,000-actor held GPU frame alongside host IMU, release before consumer completion, queue wrap, required-step retry, selective reset, terminal versus returned poses, a threaded host reader, disconnect and in-flight shutdown. Run the usual `scripts/local_ci.py verify` in the dev container for these tests and the installed-package smoke test. To reproduce the performance measurements in that container:

```sh
export PYTHONPATH=/build/flightlib-cuda-python-native/python
PY=/build/flightlib-cuda-python/bin/python
$PY tools/sensors/bench_sensors.py
$PY tools/transport/bench_transport.py disabled
$PY tools/transport/bench_transport.py device --selected 4
$PY tools/transport/bench_transport.py host --selected 1
$PY tools/transport/bench_transport.py host --selected 4
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop -o /tmp/transport-host \
  $PY tools/transport/bench_transport.py host --selected 4 --trace
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_api_sum /tmp/transport-host.nsys-rep
```
