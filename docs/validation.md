# Verifying a change

Start the dev service from the repository root. Run these commands on the same checkout and build volume.

```sh
docker compose exec -T dev bash scripts/setup-cuda-python.sh
docker compose exec -T dev python3 scripts/local_ci.py verify
```

The harness checks the frozen CPU source manifest, then compares CUDA state and motor trajectories against the original and optimized CPU implementations. It tests graph replay, mixed models, selected resets, observation and reward continuity, and the standalone C++ and Python installations. Reports with source fingerprints and per-stage results are written to the ignored `.local-ci/` directory. Use `release` on a clean candidate before tagging. Multi-GPU binding checks require two devices. The separate `sanitize` profile requires a working NVIDIA debugger interface.

For changes to optional features, exercise the actual consumer. The Newton camera tests compare depth with independently computed ray/box intersections and check image and episode identity across resets. The ROS subscriber tests check wire units, frames and acquisition timestamps. PX4 qualification matches actuator replies to simulation ticks and compares its estimator against simulator truth. Installation and commands are in the [camera](../tools/newton/README.md), [ROS 2](../tools/ros2/README.md) and [PX4](../tools/px4/README.md) guides.

For performance work the [comparison harness](../tools/rl_comparison/README.md) measures OpenMP and CUDA on matched trajectories. Run the developer profiler inside the dev container with `PYTHONPATH=/workspace/tools/profile python3 -m flightmare_profile --help`.
