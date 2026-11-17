# CUDA-7 validation and performance

Base commit: c32de1d. Version 0.1.0, frozen tensor schema v1 and CPU oracle
unchanged. Hardware: RTX 5070 Ti, Ryzen 7 9800X3D, Docker/WSL, CUDA 13.3,
PyTorch 2.9.1+cu130. See [support and numerical contract](flightlib-support.md)
for what is qualified and what is deferred.

## Functional evidence

`docker compose exec -T dev python3 scripts/local_ci.py verify` passes frozen
CPU tests (36), native contracts/runtime/lifecycle/dynamics/environment checks,
the new optimized CPU dynamics/environment checks, Python tests (8 passed,
1 multi-GPU skip), and the legacy binding smoke checks. Initial complete report:
`.local-ci/20261117T025000.510524Z-verify/report.json`. The later CPU comparison
also rebuilt and passed both optimized differential suites with contraction
disabled in the test driver. Profiler analysis tests: 10 passed.

The new YAML saturation regression fails at step 0 when the allocation fix is
removed (yaw GPU 0, CPU pi); the corrected build passes actual observation,
reward, terminal and lifecycle comparisons. The negative control was restored
before final verification. No frozen source, test or hash manifest was changed.

The final optimized randomization preflight reports the following maxima across
17 lanes, seven models and 160 control intervals at each timestep. Normalized
error is abs(GPU-CPU)/max(1,abs(CPU)). All 25 fields, motors and time are checked
at every interval, including model update and setState/reset transitions.

| dt (seconds) | State excluding acceleration | Acceleration | Motor speed (rad/s) |
| ---: | ---: | ---: | ---: |
| .006125 | 2.825e-6 | 1.144e-5 | .000733 |
| .02 | 5.923e-6 | 7.134e-6 | .000977 |
| .05 | 1.431e-4 | 1.461e-4 | .001282 |

CPU packet versus scalar inverse-motor conversion differs by up to .000489rad/s
in this fixture. This is why the numerical contract explicitly distinguishes
CPU SIMD approximations from CUDA and gives the new longer-step acceleration
trajectory a 3e-4 budget. Original trajectory/observation/reward thresholds are
unchanged. Passive bias fields, clocks and discrete lifecycle results are exact.

## Residency and matched wall time

The CUDA-6 24-case matrix was repeated: native environment, native graph, Python,
Python graph, policy and policy graph at 1, 33, 16,384 and 65,536 lanes. dt=.01,
50% initial reset density, ten warmups, five samples of 100 steps. Independent
Nsight traces cover every case. Baseline:
`.local-ci/profile-20261117T013741.581504Z/report.json`.

All 24 cases pass residency/launch/functional checks: no simulator hot-loop
host/device copies, allocations or frees; three environment kernel nodes per
step, or one graph API launch with those same three nodes. Python allocator
deltas are zero and completion synchronization remains at measurement boundaries.
Policy library scratch clears retain their existing separately bounded budget.
D uses 254 registers with zero spill loads/stores and zero stack bytes, unchanged;
there is no new arena storage, global intermediate, launch or data transfer.

Two full sweeps **failed** the historical wall-time gate and remain recorded:

- `.local-ci/profile-20261117T025312.722427Z/report.json`: all trace checks passed;
  one 65,536-lane policy-graph timing was 1.422x baseline, with elevated host
  submission time. All other rows stayed below 1.15x.
- `.local-ci/profile-20261117T025526.087123Z/report.json`: timing repeat without
  tracing; policy-graph passed, but the 65,536-lane policy row was 1.176x. All
  16 simulation-only rows were within .871–1.061x baseline.

Do not label either entire report a passing performance run. To investigate the
large policy cases, c32de1d was rebuilt from `git archive` in a separate `/build`
directory, then before/after binaries were alternated (AB/BA order, four trials,
five 100-step samples per trial, identical warmup/shapes/configuration). Neither
the 1.15x budget nor the simulator workloads were relaxed. Median of trial medians:

| 65,536-lane workload | Before (us/step) | After (us/step) | Ratio |
| --- | ---: | ---: | ---: |
| Policy | 152.897 | 169.977 | 1.112x |
| Policy + environment graph | 148.149 | 164.569 | 1.111x |

Paired evidence and raw samples: `.local-ci/cuda7-ab-report.json`; local reproducer
`.local-ci/cuda7-ab.py`. Both meet the unchanged 1.15x threshold. The approximately
11% closed-loop policy cost is measurable and is **not** dismissed as noise.
Correcting state arithmetic can change Euler branches and subsequent policy
inputs; this comparison measures the resulting complete workload, not merely
instruction cost. The historical outliers remain visible. CUDA-9 should repeat
the full qualification under controlled load; unlocked clocks/WSL scheduling
are not latency guarantees.

Both full sweeps used native driver SHA256
`4c9187c450e02c6c5b338e594d08871a97d8968a5a22faa9a103cea12e456ab6`
and Python extension SHA256
`20b634be575feff771d5019ad1c9250d5ba7994dbac58dc6fdcdd6018a89e69a`.
The repeat/paired/CPU comparison source fingerprint is
`083d0bfbb3e6c7d2aaab8fd39ba94dba5a6649f1673b222051c740d3fe715d01`;
final evidence-document additions occur after measurement. Raw reports retain
their own full source identities and hardware details.

## Focused threaded CPU comparison

The production graph path was also rechecked against unchanged optimized
Flightlib with 8 and 16 OpenMP threads, dt=.05, YAML model and flight actions
[1.1,.9,1.05,.95]. Ten warmups, five 100-step samples; all final observations,
transition observations, rewards, done and reset totals match. Each lane undergoes
eight automatic resets over 510 intervals. Best measured thread count is 16 in
each row:

| Lanes | Completion cadence | CPU us/step | GPU us/step | CPU/GPU |
| ---: | --- | ---: | ---: | ---: |
| 1,024 | every step | 413.29 | 58.49 | 7.07x |
| 1,024 | every 100 steps | 398.75 | 27.13 | 14.70x |
| 16,384 | every step | 7,300.46 | 59.05 | 123.63x |
| 16,384 | every 100 steps | 7,410.61 | 40.53 | 182.85x |

Evidence: `.local-ci/cpu-parallel-20261117T025822.897934Z/report.json` (12 cases).
This is a focused threaded retest, **not** a replacement for the historical
[full thread/process sweep](cpu-parallel-performance.md), nor a claim about the
fastest conceivable CPU implementation. Setup/readback remain outside timing.

## Reproduction and limits

```sh
docker compose exec -T dev python3 scripts/local_ci.py fast
docker compose exec -T dev python3 scripts/local_ci.py verify
docker compose exec -T -e PYTHONPATH=/workspace/flightmare-profile dev python3 -m unittest discover -s flightmare-profile/tests -v
docker compose exec -T -e PYTHONPATH=/workspace/flightmare-profile dev python3 -m flightmare_profile --workloads environment,environment-graph,python,python-graph,policy,policy-graph --cases 1x1,3x11,1024x16,4096x16 --reset-percent 50 --iterations 100 --samples 5 --warmup 10 --trace-all --baseline /workspace/.local-ci/profile-20261117T013741.581504Z/report.json --max-wall-ratio 1.15
docker compose exec -T -e OMP_PROC_BIND=spread -e OMP_PLACES=cores -e OPENBLAS_NUM_THREADS=1 -e PYTHONPATH=/workspace/flightmare-profile dev /build/flightlib-cuda-python/bin/python -m flightmare_profile.cpu_parallel --lanes 1024,16384 --workers 8,16 --chunks 1,100 --actions=1 --modes thread --iterations 100 --samples 5 --warmup 10 --control-dt .05
```

Performance runs are separate from routine verification and retain a build lock.
Hardware performance counters remain unavailable; WDDM sanitizer and multi-GPU
qualification gaps are unchanged. These results qualify the stated Flightlib
behavior envelope, not physical calibration, unbounded numerical equivalence,
rendering fidelity or real-world transfer. CUDA-8 standalone API/package work
and CUDA-9 integrated qualification remain open.
