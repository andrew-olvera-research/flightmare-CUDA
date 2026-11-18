# Reproduce the Split-S CPU and CUDA PPO comparison

Run from the repository root after the [CUDA-15 setup](../../flightlib-cuda/docs/cuda15-isaac.md). The dev container hosts both simulation backends. The FPV comparison uses Windows Kit with Direct3D 12. Both producers send poses and transition flags through the same benchmark stdio transport. RGB, inference and PPO stay on the RTX GPU.

## Build and check the CPU reference

The target links the optimized Flightlib CPU library and uses its original RK integrator and task implementation. At build time it inserts observation hooks into a copy of those sources to call the CUDA runtime's swept collision math at the same substeps. The CPU and CUDA backends use the same compiled Split-S BVH, a 0.125 m collision radius, deterministic resets, stock tracking reward and one aircraft per environment. The CPU code shares the BVH across OpenMP workers.

```sh
docker compose exec -T dev cmake -S tools/rl_comparison -B /build/course-profile -G Ninja
docker compose exec -T dev cmake --build /build/course-profile
docker compose exec -T dev bash
# Inside the dev container
export PYTHONPATH=/build/flightmare-isaac
export OMP_PROC_BIND=spread OMP_PLACES=cores
mkdir -p /workspace/.local-ci/rl-course-comparison
cd /workspace
out=.local-ci/rl-course-comparison
py=/build/flightlib-cuda-python/bin/python
"$py" tools/rl_comparison/bench.py parity --output "$out/parity.jsonl"
```

Parity replays each action batch on both backends, starting with a forced gate-post crossing. It compares hit and terminal flags, captured episode IDs, pre-reset poses, post-reset observations and rewards for 256 ticks with queries enabled and another 256 with queries disabled.

## Time the simulator and state PPO

```sh
"$py" tools/rl_comparison/bench.py raw --envs 64 1024 4096 16384 --workers 0 8 16 --rollouts 5 --output "$out/raw.jsonl"
"$py" tools/rl_comparison/bench.py ppo --envs 64 256 1024 4096 8192 16384 --workers 0 8 16 --queries 1 --output "$out/ppo-on.jsonl"
"$py" tools/rl_comparison/bench.py ppo --envs 64 1024 4096 8192 --workers 0 8 --queries 0 --output "$out/ppo-off.jsonl"
"$py" tools/rl_comparison/bench.py ppo --envs 16384 32768 65536 --workers 0 16 --queries 1 --rollouts 8 --output "$out/scale.jsonl"
"$py" tools/rl_comparison/bench.py ppo --envs 65536 131072 262144 --workers 0 --queries 1 --rollouts 6 --output "$out/upper.jsonl"
"$py" tools/rl_comparison/bench.py ppo --envs 131072 --workers 0 16 --queries 1 --rollouts 24 --output "$out/100m.jsonl"
```

Worker count zero selects CUDA graph replay. Positive counts select the OpenMP backend. Raw timings use resident zero commands, ground termination, and a synchronization at the end of each CUDA timing block. Course hits leave the trajectory unchanged in this test so query-on and query-off timings cover the same flight. The CPU wrapper also validates the transition and sums hit flags each tick. Use PPO mode for sampled actions, course-hit terminations, autoresets and feedback to the policy. On CPU, a blocking download supplies actions and pinned memory supplies observation, reward and done buffers for upload.

State PPO uses all 12 state observations, two 304-unit hidden layers and the same training loop as FPV. The FPV model consumes 768 pooled RGB features through two 112-unit hidden layers. Both use BF16 autocast, FP32 GAE, fused Adam, 32-step rollouts, four epochs and four minibatches per epoch. Counts exclude the forced crossing and 24 warm-up steps. Timings include the first optimizer update. Throughput counts one new transition per aircraft per 20 ms of simulated time. Optimization reuses those transitions.

## Time camera-controlled PPO

In Windows PowerShell with the renderer package on Kit's Python path, run each case in a fresh process. The 16-thread CPU example measures with course queries enabled.

```powershell
C:\isaacsim\python.bat tools/isaac/bench_fpv.py --docker-producer --cpu-workers 16 --course flightlib-cuda/python/flightlib_cuda/assets/split_s.usda --env-counts 64 --warmup 24 --steps 32 --rollouts 4 --results .local-ci/rl-course-comparison/fpv-cpu-on.jsonl --/app/vulkan=false
```

Use `--cpu-workers 0` for CUDA and add `--no-course-queries` to either producer to disable swept scene queries and their terminal response. Ground termination remains enabled. Change the results filename for each case. Every environment requests a fresh 256×192 RGB image at every simulated tick. IMU, odometry and depth are disabled. The collection clock covers rendering, policy inference, action feedback, transition transfer and storage. The training clock also covers GAE and updates. Start benchmarks sequentially so the cases have exclusive use of the GPU workload.
