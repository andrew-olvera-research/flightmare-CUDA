# Reproduce the Split-S CPU and CUDA PPO comparison

Run the [validation profile](../../docs/validation.md) to build the optimized CPU reference in the dev volume. Install the optional RL package for the Python benchmarks.

## Build and check the CPU reference

The target links the optimized Flightlib CPU library and uses its original RK integrator and task implementation. At build time it inserts observation hooks into a copy of those sources to call the CUDA runtime's swept collision math at the same substeps. The CPU and CUDA backends use the same compiled Split-S BVH, a 0.125 m collision radius, deterministic resets, stock tracking reward and one aircraft per environment. The CPU code shares the BVH across OpenMP workers.

```sh
docker compose exec -T dev cmake -S tools/rl_comparison -B /build/course-profile -G Ninja
docker compose exec -T dev cmake --build /build/course-profile
docker compose exec -T dev bash
# Inside the dev container
export PYTHONPATH=/build/flightlib-cuda-python-native/python
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

State PPO uses all 12 state observations, two 304-unit hidden layers and the same training loop as FPV. Both use BF16 autocast, FP32 GAE, fused Adam, 32-step rollouts, four epochs and four minibatches per epoch. Counts exclude the forced crossing and 24 warm-up steps. Timings include the first optimizer update. Throughput counts one new transition per aircraft per 20 ms of simulated time. Optimization reuses those transitions.


## Recorded measurements

The chart uses the matched state-observation workload recorded on the RTX 5070 Ti and Ryzen 7 9800X3D. Each iteration collects 32 steps, computes GAE and runs two PPO epochs. For each batch size the comparator is the best measured parallel CPU configuration, including CPU physics with a GPU learner. Grouping measurements use independent physics lanes.

| Lanes | Broad sweep: GPU collection | Collection speedup | GPU complete iteration | Complete iteration speedup | GPU transitions/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 33 | 5.242 ms | 0.71x | 8.609 ms | 0.94x | 0.123 million |
| 1,024 | 5.453 ms | 4.84x | 8.897 ms | 3.38x | 3.68 million |
| 16,384 | 7.984 ms | 14.55x | 35.021 ms | 4.09x | 14.97 million |

| E | A | GPU collection ms | GPU total ms | Collection speedup | Total speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 100 | 5.125 | 8.580 | 1.72x | 1.93x |
| 100 | 1 | 5.566 | 8.977 | 1.57x | 1.97x |
| 1 | 1,000 | 5.420 | 9.122 | 5.39x | 3.63x |
| 1,000 | 1 | 5.433 | 8.789 | 5.30x | 3.75x |
| 10 | 100 | 5.465 | 8.940 | 5.10x | 3.56x |

The Split-S workload below uses a separate BF16 policy, four PPO epochs and continuous course collision checks. Rates include 100,663,296 transitions at 131,072 environments. The raw append-only [trial records](results.jsonl) include collision parity and training timings.

| Backend | Environments | Course queries | Collection, M transitions/s | Full PPO, M transitions/s | Collection | GAE + updates | Timed training | Including setup and warm-up¹ |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA | 131,072 | On | 53.33 | 8.07 | 1.89 s | 10.59 s | 12.48 s | 13.35 s |
| OpenMP, 16 threads | 131,072 | On | 4.31 | 2.75 | 23.35 s | 13.22 s | 36.57 s | 47.83 s |
| OpenMP, 16 threads | 65,536 | On | 4.41 | 2.99 | 22.81 s | 10.80 s | 33.61 s | 39.33 s |
| CUDA | 131,072 | Off | 85.66 | 8.67 | 1.18 s | 10.44 s | 11.61 s | 11.70 s |

The final column includes setup and warm-up. For historical flags and sample counts see the [original benchmark record](https://github.com/andrew-olvera-research/flightmare-CUDA/blob/v0.2.0/flightlib-cuda/docs/cuda15-ppo-comparison.md).
