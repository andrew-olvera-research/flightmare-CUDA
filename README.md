# Flightmare CUDA

A GPU quadrotor simulator for flying many aircraft in one world or running thousands of independent racing environments. It preserves Flightmare's motor model, body-rate controller and rigid-body RK4 integration. State, swept collision queries, sampled sensors and optional FPV cameras stay on device and can share a CUDA graph with your policy.

## Simulation fidelity

- **Dynamics.** Differential tests compare all 25 state components and motor outputs against CPU Flightlib at each control step, including saturation, model changes and partial resets. The randomized suite exercises seven aircraft models and control periods from 6.125 to 50 ms. [Trajectory tests](flightlib-cuda/tests/dynamics_test.cpp)
- **Contact.** In a 32,768-transition CPU/CUDA replay on Split-S, both paths agreed on all 231 contacts and every terminal flag. Separate tests compare moving-aircraft contacts with an independent CPU solver. [Collision replay](tools/rl_comparison/README.md)
- **Sensors.** Against held-out stationary NavChip IMU data, the fitted model had 6.55% median absolute relative error in Allan deviation across axes and averaging times. Cube Orange pressure replay had 24.73 Pa held-out RMSE with GNSS altitude as the reference. These measurements cover short-term sensor behavior. [Simulation interfaces and limits](docs/simulation.md)

## Throughput

Measured on an RTX 5070 Ti with a Ryzen 7 9800X3D. Each transition advances one aircraft by one control interval.

| Workload | Measured throughput |
| --- | --- |
| Split-S, state observations and course collisions, 131,072 aircraft | **53.33 million transitions/s** on CUDA versus **4.31 million/s** on 16 CPU threads. Including PPO updates on the same GPU, **8.07 million/s** versus **2.75 million/s**. |
| Independent dynamics, 1,000 aircraft | **5.90 million transitions/s** in one world versus **5.89 million/s** in 1,000 worlds, including policy collection. Peer interactions add work within a world. |
| One formation of 1,000 aircraft, course and peer collision queries | **53× real time** for physics and contact detection. Each completed step advanced the whole formation by 10 ms, with 250 candidate pairs in this formation. [Measured workload](https://github.com/andrew-olvera-research/flightmare-CUDA/blob/v0.2.0/flightlib-cuda/docs/cuda12-swarm-collision.md#qualification) |
| Split-S, 256 independent 256×192 RGB cameras | **83,037 image-conditioned transitions/s** in warmed collection with tiled rendering. |

![Measured CUDA and CPU scaling and the effect of environment grouping](docs/figures/headless-scaling.png)

[CPU comparison and chart data](tools/rl_comparison/README.md) · [Camera workload](tools/newton/README.md)

## Install

From this checkout, start Docker Compose on a machine with an NVIDIA GPU visible to Linux containers.

```sh
docker compose up -d --build dev
docker compose exec -T dev bash scripts/install-headless.sh --with-rl
docker compose exec -T dev /build/flightlib-cuda-python/bin/python -m flightrl_cuda.custom_collector
```

Omit `--with-rl` for the standalone simulator. [Core Python example](flightlib-cuda/python/examples/core.py) · [GPU targets and API](docs/simulation.md) · [Newton camera](tools/newton/README.md) · [ROS 2](tools/ros2/README.md) · [PX4 SITL](tools/px4/README.md)

Contributions are welcome on bounded neighbor observations, delayed swarm communication, replayable peer aircraft and sensor models with held-out recordings. Include a reproducible fixture and the quantity it checks. [Build and test guide](docs/validation.md)

Derived from **Flightmare: A Flexible Quadrotor Simulator** (Song et al., CoRL 2020). The original CPU simulator remains the numerical reference. [Upstream Flightmare](https://github.com/uzh-rpg/flightmare) · [MIT license](LICENSE)
