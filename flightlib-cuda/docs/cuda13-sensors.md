# CUDA-13 sensor API

Use the IMU for angular rate and specific force. Use local odometry for a position, attitude and velocity measurement with a known mount. Both run on the environment's CUDA stream. Each option allocates its queues once at construction. Leaving both disabled selects the original three-kernel headless step and allocates zero sensor-buffer bytes.

```python
from flightlib_cuda import Environment, ImuConfig, OdometryConfig

env = Environment(
    1, 1000, control_dt=0.01,
    imu=ImuConfig(rate_hz=200, latency_s=0.003, seed=71,
                  gyro_noise_density=0.0002, accel_noise_density=0.002),
    odometry=OdometryConfig(rate_hz=50, latency_s=0.010, position_std=0.01),
)
env.reset_state[2] = 5
env.reset()
env.capture(random=False)
env.step(graph=True, random=False)
with env.access():
    imu = env.imu
    # Example for a collector that retains history across steps.
    angular_rate = imu.values[..., :3].clone()
    acquisition_time = imu.acquired_ns.clone()
    valid = imu.valid.bool()
```

Pass the same keyword arguments to `flightrl_cuda.TensorEnv` and read `env.imu` and `env.odometry` after stepping. The stock policy observation retains its existing shape. To build a custom batched sensor or estimator, see the virtual range-to-beacon measurement and planar yaw integrator. Both use only packets delivered to the consumer. [Runnable Python example](../../flightrl-cuda/flightrl_cuda/sensor_example.py)

## Measurements

| Stream | Channels in order | Mount and error model |
| --- | --- | --- |
| IMU | Angular rate XYZ (rad/s), specific force XYZ (m/s²), both in sensor axes | Body-origin IMU with a configurable sensor-to-body WXYZ unit quaternion. The accelerometer projects the integrator's world acceleration minus model gravity into the sensor frame. Level hover gives +g on sensor Z with an identity mount. Ideal free fall gives zero. |
| Local odometry | World position XYZ (m), sensor-to-world quaternion WXYZ, world velocity XYZ (m/s) | Body-frame offset in metres and sensor-to-body rotation. Velocity includes the rotated angular-velocity cross offset term. Independent Gaussian position and velocity errors use world axes. Attitude errors use a sensor-frame rotation vector in radians. |

The defaults are ideal measurements. Each stream has its own seed, rate, phase, fixed latency and independent per-acquisition dropout probability. IMU error parameters take either one scalar or a sensor-axis XYZ triple. Set fixed bias and saturation limits in output units. A zero limit disables clipping on that axis.

For nominal period T in seconds, the per-sample white noise standard deviation is N/√T. The bias starts at the configured offset and accumulates independent Gaussian increments with standard deviation K√T on valid physics samples, including acquisitions subsequently dropped in transport. N is specified in output units per √Hz and K in output units per second per √Hz. The optional increment readout term differences consecutive independent uniform cumulative errors and divides by T. Its peak-to-peak width is in radians or m/s. It models short-timescale correlations in floating-point measurements. [Kalibr noise convention](https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model)

## Timing and ownership

- Each stream exposes borrowed CUDA tensors. Values are float32 with shape `[packet slot, lane, channel]`. The lane is `environment * slots_per_environment + agent`. Validity is uint8 and each metadata field is int64, both `[packet slot, lane]`. Use `sequence == 0` for padding and `valid != 0` for usable measurements. Drops and invalid acquisitions preserve their nonzero sequence and timestamps, with zero-filled values.
- The sensor clock starts at zero and advances in integer nanoseconds per environment step, independently of the floating-point dynamics reset time. The rounded period sets absolute deadlines within the episode. A positive phase sets the first deadline relative to episode start. Zero phase starts after one full period.
- `scheduled_ns` is the deadline. `acquired_ns` is the first integration boundary at or after that deadline. `available_ns` adds the configured latency. `delivered_ns` is the first completed control-step boundary at or after availability. The stream can deliver multiple samples per lane per step in acquisition order. For 200 Hz sensing with a 10 ms step and 3 ms latency, the sample acquired at 5 ms arrives at 10 ms and the one acquired at 10 ms arrives at 20 ms.
- With the default 2.5 ms maximum physics substep, a 10 ms control period supports up to 400 Hz. Shorter control periods can support higher rates. Setup rejects sensor periods shorter than the longest actual substep. Rebuild the environment to change its sensor schedule or integer control period.
- Sensor-enabled dynamics record pose, velocity, angular rate and applied acceleration at every substep in shared device storage. Each enabled sensor adds one graph node to construct, queue and deliver packets. Failed dynamics invalidate acquisitions for that step. A discontinuous world-bounds correction invalidates an acquisition at the corrected endpoint. Check task validity separately when retaining training transitions.
- The clock continues across resets. Sequence starts at one for each new lane episode. A terminal step exposes due packets from the ending episode. Subsequent steps discard pending packets from that episode. Explicit partial reset clears delivered packets for selected lanes. An inactive lane publishes no packets. Reactivation within the same episode preserves the sequence gap.
- Read tensors on the submitting stream. For a second stream, wrap external work in `with env.access(other_stream):` (or `env.engine.access(other_stream)` with TensorEnv). Copy the values **and** the identity, timing and validity fields your collector needs before the next step or reset overwrites the borrowed storage.

## Stationary IMU qualification

We fit each axis on the first 1,552,625 samples of EPFL's 250 Hz NavChip recording. We reserved the following 1,552,625 samples for comparison. The statistic is Allan deviation, which measures how much adjacent time-averaged readings differ. A constant offset or gravity component cancels. At each of six averaging times from 4 ms to 4.096 s, we compared the held-out data with actual CUDA output from eight independent lanes, each generating 16,384 samples. The table summarizes the 36 axis/time combinations. [Dataset description](https://smac-group.github.io/imudata/) · [Pinned recording](https://github.com/SMAC-Group/imudata/blob/eec6e5b627a31a4a1c286e0d1475c6556705b2a6/data/navchip.rda) · [Coefficients, checksums and per-axis results](cuda13-navchip.json)

| Fitted IMU model | Median absolute relative deviation error | Worst error |
| --- | ---: | ---: |
| White noise + bias random walk | 8.03% | 61.90% |
| Same model with increment readout error | 6.55% | 19.73% |

At 4 ms the basic model produced 38–42% of the observed gyro deviation. Adding the readout term brought all three axes to 90–120%. This comparison qualifies stationary variability over the measured timescales on one sensor. Dynamic response, mounting vibration, temperature dependence, long-term drift and odometry hardware accuracy require independent measurements.

To tune a device, supply a uniformly sampled stationary CSV with headers `gx,gy,gz,ax,ay,az` (rad/s, m/s²), or a numeric NumPy array with six columns in that order. The fitter estimates nonnegative coefficients against relative Allan variance on the training half. Use the first three and last three fitted components as the gyro and accelerometer triples in `ImuConfig`. Fixed biases and saturation limits require separate measurements. To reproduce the published comparison, download the pinned RDA and convert it once in an analysis environment with NumPy and `rdata` installed.

```python
import numpy as np
import rdata

np.save("navchip.npy", np.asarray(rdata.read_rda("navchip.rda")["navchip"]))
```

Run from the repository root inside the prepared dev container, with the recording accessible at the path supplied.

```sh
export PYTHONPATH=/build/flightlib-cuda-python-native/python
PY=/build/flightlib-cuda-python/bin/python
$PY tools/sensors/qualify_imu.py navchip.npy --rate 250
$PY tools/sensors/qualify_imu.py navchip.npy --rate 250 --increment-readout
$PY tools/sensors/bench_sensors.py
```

## Headless performance

RTX 5070 Ti, one world, 10 ms per step, graph replay, explicit reset, no rendering, collision queries or policy. Measurements are medians over nine batches of 50 completed steps. The IMU uses 200 Hz, white noise, bias random walk and 3 ms latency. Odometry uses 50 Hz, position noise and 10 ms latency. Native arena sizes include the existing simulation and exclude Torch's allocator.

| Agents | Disabled | IMU | IMU and odometry | Native arena bytes, disabled / IMU / both |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 15.86 µs | 30.23 µs | 34.68 µs | 63,588 / 149,344 / 200,544 |
| 1,000 | 17.87 µs | 32.33 µs | 38.62 µs | 558,056 / 1,409,984 / 1,913,536 |

Nsight Systems recorded 20 graph launches for 20 steps in each 1,000-agent case traced. With both sensors, each graph ran five kernels. With sensors disabled, it ran three. Neither capture recorded a memory copy or device allocation. To reproduce, run the command below once per mode and use `nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_api_sum` on each report.

```sh
for mode in disabled both; do
  nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop -o /tmp/sensors-$mode \
    $PY tools/sensors/bench_sensors.py --trace $mode
done
```
