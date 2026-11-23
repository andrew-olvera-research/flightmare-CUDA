# Flightmare CUDA

A GPU quadrotor simulator for flying many aircraft in one world or running thousands of independent racing environments. It preserves Flightmare's motor model, body-rate controller and rigid-body RK4 integration. State, swept collision queries, packet-level communications, sampled sensors and optional FPV cameras stay on device and can share a CUDA graph with your policy.

## Flight dynamics and control

Each aircraft has a motor model, body-rate controller and rigid-body RK4 integrator. The CPU/CUDA trajectory tests compare all 25 state components and motor outputs across seven aircraft models and control periods of 6.125–50 ms. Benchmarks below use an RTX 5070 Ti and Ryzen 7 9800X3D. One transition advances one aircraft by one control interval. [Trajectory tests](flightlib-cuda/tests/dynamics_test.cpp) · [CPU comparison and chart data](tools/rl_comparison/README.md)

| Workload | CUDA transitions/s | CPU transitions/s (16 threads) |
| --- | ---: | ---: |
| Split-S collection with state observations and course collisions, 131,072 aircraft | 53.33 million | 4.31 million |
| Same workload including PPO updates | 8.07 million | 2.75 million |

With peer interactions disabled, collection for 1,000 aircraft reached 5.90 million transitions/s in one world and 5.89 million/s in 1,000 independent worlds.

![Measured CUDA and CPU scaling and the effect of environment grouping](docs/figures/headless-scaling.png)

## Course and aircraft collisions

Swept queries detect an aircraft crossing a course obstacle or another aircraft between control steps. On a 32,768-transition Split-S replay, CPU and CUDA agreed on all 231 contacts and every terminal flag. The simulator reports contact and can terminate the episode. [Geometry and contact semantics](docs/simulation.md) · [Collision replay](tools/rl_comparison/README.md)

| Formation workload | Completed simulation speed |
| --- | ---: |
| 1,000 aircraft in one world, 10 ms control period, course and peer checks, 250 candidate pairs | 53× real time for the whole formation |

The number of nearby pairs affects this timing. [Formation benchmark](https://github.com/andrew-olvera-research/flightmare-CUDA/blob/v0.2.0/flightlib-cuda/docs/cuda12-swarm-collision.md#qualification)

## Packet-level mesh networking

Enable `NetworkConfig` to step communications alongside physics in the captured CUDA graph. Senders share airtime. Concurrent offers can wait, collide or expire. A broadcast uses one transmission, and each subscribed receiver gets its own delivery outcome. The inbox exposes packet identity, payload handle and arrival time so applications can implement forwarding, duplicate detection and freshness policies. The loss profile can use distance, persistent outages or recent packet outcomes on each directed link. [Network API and example](tools/transport/resident_channel.md)

### GPU step cost

Each column advances 1,000 aircraft by 10 ms of simulated time. The world layout determines which senders compete for airtime. Entries are **median / p95 GPU time in µs** for the complete graph step, measured with CUDA events and summarized across three warmed rounds. Sender counts are per world. Each active sender offers 1,000-byte broadcasts and each receiver subscribes to all active sources. [Benchmark code](tools/transport/bench_resident.py)

| Traffic per step | 1 world × 1,000 aircraft | 100 worlds × 10 aircraft |
| --- | ---: | ---: |
| Network disabled | 16.6 / 40.9 | 16.6 / 38.5 |
| 1 sender, 1 offer, zero sampled loss | 36.9 / 38.9 | 24.6 / 25.2 |
| 8 senders, 1 offer each, zero sampled loss | 62.9 / 69.6 | 30.7 / 31.7 |
| 8 senders, 3 offers each, ordinary 10% loss sampler | 148.5 / 161.8 | 51.2 / 51.4 |
| Same traffic and loss probability, five-bit receiver history enabled | 149.5 / 163.8 | 51.2 / 53.2 |

In the single-world, three-offer comparison, enabling receiver history raised the median by 1.0 µs and total native arena storage from 2.48 to 8.53 MB. Calibrating one link from eight pilot updates took 0.59 ms median on the CPU, outside the GPU step and excluding upload. [Calibration method and cost](tools/transport/history_transfer.md)

### Network fidelity

| Validation reference | What the comparison established |
| --- | --- |
| Wi-SUN receiver logs, three copies per update | Using eight local pilot updates per block of 32, the GPU met every limit across three seeds in 8 of 21 windows. The other 13 lacked enough events for at least one metric. Every supported check passed. In the final 55e4 window, 178 of 12,334 updates lost all three copies, versus 201–213 simulated. Repeat loss was 10.8% measured and 18.3–19.5% simulated. [Cross-session results](tools/transport/history_transfer.md) |
| Bristol roadside broadcasts to two moving vehicles | The distribution of which vehicles received each broadcast passed on a held-out same-channel drive. A separate vehicle-to-vehicle comparison exceeded tolerance in 4 of 12 directional distance bands. One-way deadline qualification still needs a measured clock-error bound. [Bristol data and limits](tools/transport/resident_channel.md#evidence) |
| 802.11g hardware throughput, 1–8 contenders | Simulated aggregate throughput was within 7.3% of the digitized hardware mean at every tested load for saturated unicast at 54 Mbps. [Experiment and per-load results](tools/transport/shared_channel.md) |
| ns-3 acknowledged unicast, 1, 4 and 8 senders | All three loads met the 20% p95 wait and 10% throughput tolerances. At eight senders the CUDA model gave 3.49 ms p95 wait and 21.78 Mbps, versus 3.38 ms and 22.51 Mbps in ns-3. [Matched workload and trials](tools/transport/session_service.md) |

These results qualify the listed packet workloads. Deployment-specific deadlines, interference conditions and application behavior need target-network measurements.

## Sampled sensors

IMU, odometry and barometer packets carry acquisition times and support scheduled delivery. On held-out stationary NavChip data, the fitted IMU had 6.55% median absolute relative error in Allan deviation, a measure of noise at different averaging times. Cube Orange pressure replay had 24.73 Pa held-out RMSE using GNSS altitude as reference. [Sensor interfaces and validation](docs/simulation.md#geometry-and-measurements)

| Sensors in one world | 100 aircraft, µs/step | 1,000 aircraft, µs/step |
| --- | ---: | ---: |
| Disabled | 16.18 | 17.87 |
| 200 Hz IMU | 30.31 | 32.48 |
| 200 Hz IMU and 50 Hz odometry | 34.54 | 39.08 |

Times are the median of nine warmed, 50-step batches. [Sensor benchmark](tools/sensors/bench_sensors.py)

## FPV cameras and authored scenes

Newton renders RGB and optional axial depth into device memory. Tiled rendering shares scene geometry across cameras and supports packaged courses and imported Unity or OpenUSD scenes. A 256×192 packed framebuffer uses 192 KiB per camera. [Camera setup, supported assets and benchmarks](tools/newton/README.md)

| Split-S, 256×192 RGB | Tiled image-conditioned collection | Including PPO updates |
| --- | ---: | ---: |
| 32 cameras | 49,978 transitions/s | 21,066 transitions/s |
| 256 cameras | 83,037 transitions/s | 64,631 transitions/s |

## ROS 2 and PX4 lockstep

The single-aircraft ROS 2 bridge publishes stamped sensor data and simulation time. The PX4 adapter advances the aircraft by 4 ms per actuator exchange. In four aided takeoff runs, the onboard estimator had 0.0090–0.0094 m position RMS error during flight. [ROS 2 topics and timing](tools/ros2/README.md) · [PX4 fixture and results](tools/px4/README.md#acceptance-and-results)

| 32 simulated seconds with PX4, ROS publication and logging | Real-time factor | p95 wall time per tick |
| --- | ---: | ---: |
| Camera off | 3.73–3.76× | 1.21–1.22 ms |
| 256×192 RGB at 50 Hz | 2.79–2.81× | 1.62–1.66 ms |

## Install

From this checkout, start Docker Compose on a machine with an NVIDIA GPU visible to Linux containers.

```sh
docker compose up -d --build dev
docker compose exec -T dev bash scripts/install-headless.sh --with-rl
docker compose exec -T dev /build/flightlib-cuda-python/bin/python -m flightrl_cuda.custom_collector
```

Omit `--with-rl` for the standalone simulator. [Core Python example](flightlib-cuda/python/examples/core.py) · [GPU targets and API](docs/simulation.md) · [Newton camera](tools/newton/README.md) · [ROS 2](tools/ros2/README.md) · [PX4 SITL](tools/px4/README.md)

Contributions are welcome on bounded neighbor observations, measured packet-network profiles, replayable peer aircraft and sensor models with held-out recordings. Include a reproducible fixture and the quantity it checks. [Build and test guide](docs/validation.md)

Derived from **Flightmare: A Flexible Quadrotor Simulator** (Song et al., CoRL 2020). The original CPU simulator remains the numerical reference. [Upstream Flightmare](https://github.com/uzh-rpg/flightmare) · [MIT license](LICENSE)
