# Split-S PPO throughput on the RTX 5070 Ti and Ryzen 7 9800X3D

This benchmark compares parallel Flightlib CPU simulation and CUDA simulation on one workstation. Both feed a BF16 actor-critic and fused Adam on the same GPU. Each independent environment contains one aircraft, runs at a 20 ms simulated control period and queries the floor, walls and gates. The timed policy samples actions, consumes terminal flags, bootstraps values and trains for four epochs per 32-step rollout. Sensors are disabled. The rendered path adds one fresh 256×192 RGB exposure per environment per tick.

## Collision validation

Stock parallel CPU Flightlib has no Split-S query. The benchmark-only CPU adapter calls the same swept collision code on each original RK substep and checks the final world-boundary correction. Both simulators use the same scene BVH. Replaying identical actions for 32,768 transitions produced identical terminal flags and captured episode IDs. The collision-enabled half recorded 231 hits, with agreement on every lane and tick. Maximum absolute discrepancies were 0.0000401 in captured pose components, 0.000133 in the 12 state observation components, and 0.00000193 in rewards.

## Cost of the query in isolation

These are microseconds per **whole batch tick**. Each entry is the median of five blocks of 96 completed steps, with seven blocks for the 131,072-environment case. Actions remain resident and course hits do not terminate these trials. The final observations were bitwise identical for each query-on/off pair. State PPO below also exercises the narrow phase and actual termination under sampled actions.

| Independent environments | CUDA without course queries | CUDA with course queries | Added per batch |
| ---: | ---: | ---: | ---: |
| 64 | 17.2 µs | 42.6 µs | 25.5 µs |
| 1,024 | 21.1 µs | 48.4 µs | 27.3 µs |
| 4,096 | 19.1 µs | 45.6 µs | 26.6 µs |
| 16,384 | 27.2 µs | 70.2 µs | 42.9 µs |
| 131,072 | 165.0 µs | 487.0 µs | 321.9 µs |

At 64 environments the extra 0.025 ms per batch can be significant to a physics microbenchmark. The CUDA collision-enabled FPV renderer took a median 440 ms to produce the same batch of images.

## State-observation PPO

The large-batch sweep reached 8.39M full PPO transitions/s at 131,072 environments and 8.31M at 262,144. Doubling beyond 131,072 gave little extra training throughput. A separate run at 131,072 collected **100,663,296** transitions in 24 successive rollouts with real optimizer updates.

| Backend | Environments | Course queries | Collection, M transitions/s | Full PPO, M transitions/s | Collection | GAE + updates | Timed training | Including setup and warm-up¹ |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA | 131,072 | On | 53.33 | 8.07 | 1.89 s | 10.59 s | 12.48 s | 13.35 s |
| OpenMP, 16 threads | 131,072 | On | 4.31 | 2.75 | 23.35 s | 13.22 s | 36.57 s | 47.83 s |
| OpenMP, 16 threads | 65,536 | On | 4.41 | 2.99 | 22.81 s | 10.80 s | 33.61 s | 39.33 s |
| CUDA | 131,072 | Off | 85.66 | 8.67 | 1.18 s | 10.44 s | 11.61 s | 11.70 s |

¹ Case wall time starts inside the Python process and includes world construction, warm-up, training, final statistics and cleanup. The CUDA on/off cases ran consecutively in one process with queries enabled first. All rows time the same number of new transitions. At 65,536 environments this takes 48 rollouts. At 131,072 environments each rollout contains 4,194,304 transitions followed by 16 optimizer steps.

At the same lane count, collision-enabled CUDA collected 12.4× faster and finished PPO 2.93× faster. Against the quickest CPU long run the full-training advantage was 2.69×. GAE and optimization accounted for 85% of CUDA time. Disabling the query also changes which episodes terminate. The raw same-trajectory measurements above isolate the cost of geometry queries themselves.

The speedup depends on batch size. At 1,024 environments the four-rollout sweep measured 708k full PPO transitions/s on CUDA and 494k on eight CPU threads. At 8,192 environments those rates were 4.66M on CUDA and 1.79M on 16 CPU threads. Short trials include the first optimizer update, which can disproportionately affect small batches. All per-rollout timings and the 8- and 16-thread sweep are in the results file.

## Isaac Sim camera-controlled PPO

| Environments | Simulator | Course queries | Collection, transitions/s | Full PPO, transitions/s | Median render per tick |
| ---: | --- | --- | ---: | ---: | ---: |
| 16 | CUDA | On | 147.2 | 144.2 | 88 ms |
| 32 | CUDA | On | 155.3 | 153.7 | 183 ms |
| 32 | OpenMP, 16 threads | On | 155.6 | 154.0 | 185 ms |
| 64 | CUDA | On | 137.1 | 136.5 | 440 ms |
| 64 | CUDA | Off | 138.0 | 137.3 | 448 ms |
| 64 | OpenMP, 16 threads | On | 140.6 | 139.8 | 431 ms |
| 64 | OpenMP, 16 threads | Off | 137.1 | 136.5 | 449 ms |

The 64-camera CUDA run recorded 124 course hits in 8,192 timed transitions. It spent 58.51 s in the renderer, 1.23 s on the remainder of collection and 0.28 s on GAE and PPO. At 32 cameras both simulation backends delivered about 154 full PPO transitions/s. The renderer used 91% of collection time at 32 and 98% at 64 on CUDA. Turning off course collision queries at 64 changed full CUDA throughput by less than 1%. The initial CPU on/off pair used an O3 build with assertions and measured 140.7/136.5 transitions/s. The CPU collision-enabled runs in the table use the final Release build.

For a fresh process, the fastest collision-enabled CUDA configuration among 16, 32 and 64 cameras was 32. At its measured rate, **100M FPV transitions would take 7.53 days** of sustained training. This is an extrapolation from 4,096 timed transitions. The earlier [CUDA-15 sweep](cuda15-isaac.md) peaked at 203.6 transitions/s at 64 cameras, equivalent to 5.68 days. That point followed smaller configurations in the same Kit process. This rerun did not reproduce the earlier peak.

The current renderer sets one pose per camera and requests a separate Replicator render product and RGB annotator for each. All products advance in one synchronous Kit step. The adapter then materializes an owned CUDA RGB tensor per camera. A rendering optimization can be evaluated directly against the 32- and 64-camera measurements while keeping the same observations and collision semantics.

## Method and reproduction

The GPU is a 16 GB RTX 5070 Ti with driver 610.88. The CPU has eight cores and 16 hardware threads. Headless runs used the Linux dev container and PyTorch 2.9.1+cu130. Rendered runs used Windows Isaac Sim 6.2.0, Direct3D 12 and PyTorch 2.11.0+cu128. Both rendered producers used the same Docker stdio path. Each measured camera point had a fresh Kit process and 24 warm-up exposures. The graphics timer covers pose submission, RTX, acquisition of the annotator output, per-camera copies into owned GPU tensors, and synchronization. These measurements do not subdivide that interval into driver, shading and adapter time.

The state MLP has 98,201 parameters and uses all 12 state features. The FPV MLP has 99,353 parameters and uses 768 pooled RGB features. The learner, rollout horizon, epoch count and number of minibatches per epoch are identical. The larger lane counts also make each minibatch larger. These runs establish throughput for the stock reward. Evaluating a learned racing policy requires a separate task-quality measurement.

[Build instructions and commands](../../tools/rl_comparison/README.md) · [Per-rollout measurements](cuda15-ppo-comparison.jsonl)
