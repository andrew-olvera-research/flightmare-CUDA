# Parallel CPU versus resident GPU environment stepping

Checkpoint before this work: ef242b2 (CUDA-6). The matched benchmark uses actual
Flightlib QuadrotorEnv code compiled with O3, native architecture vectorization,
no fast math and FP contraction disabled. The frozen oracle is unchanged.
GPU timing uses the production Python Environment with captured environment steps.

## Main comparison

Ryzen 7 9800X3D, 8 physical cores/16 hardware threads, versus RTX 5070 Ti;
Docker/WSL, no cgroup CPU quota. CPU workers: 1,2,4,8,16, both in-process OpenMP
and persistent spawned processes. OpenMP uses static scheduling and a persistent
team within each rollout chunk. Processes receive explicit physical-core-first
CPU assignments; all worker affinities are recorded. OpenMP uses
OMP_PROC_BIND=spread and OMP_PLACES=cores. GPU PyTorch version is 2.9.1+cu130.

Both sides use the original YAML model (mass .73, arm .17), the same world box,
initial ground reset, actions, timestep, warmup and total intervals. Actions are
[a+.1,a-.1,a+.05,a-.05], with a=-1 for reset stress and a=1 for flight. The main
run uses dt=.05: this permits deterministic lift-off through the legacy public
ground-reset API without changing its terminal/reset behavior. It is not the
earlier .01-second dynamics benchmark or a universal speedup estimate.

Five samples of 100 intervals follow ten warmup intervals. Chunk=1 returns after
every completed step; chunk=100 amortizes coordination/completion across a rollout.
GPU has one explicit completion wait per chunk. CPU threads execute directly in
the coordinator, without process IPC; worker processes exchange only small
control/completion messages. Each process keeps its state and outputs locally.
No observations are serialized through pipes, and no CPU-to-GPU policy transfer
is charged to CPU. Neither side performs a policy or history copy in this test.

The fastest median across both CPU modes and all tested worker counts is selected
for each row. Completed batch-step time in microseconds:

| Lanes | Workload | Completion cadence | Best CPU | CPU us | GPU us | CPU/GPU |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 33 | reset stress | every step | 16 threads | 21.46 | 64.24 | 0.33x |
| 33 | flight | every step | 8 threads | 21.59 | 58.72 | 0.37x |
| 1,024 | reset stress | every step | 16 threads | 493.44 | 63.52 | 7.77x |
| 1,024 | flight | every step | 16 threads | 395.60 | 57.97 | 6.82x |
| 16,384 | reset stress | every step | 16 threads | 6,494.43 | 84.37 | 76.97x |
| 16,384 | flight | every step | 16 threads | 6,500.34 | 81.20 | 80.06x |
| 33 | reset stress | 100 steps | 8 threads | 21.60 | 29.98 | 0.72x |
| 33 | flight | 100 steps | 16 threads | 15.06 | 23.76 | 0.63x |
| 1,024 | reset stress | 100 steps | 16 processes | 342.44 | 38.80 | 8.83x |
| 1,024 | flight | 100 steps | 16 processes | 335.47 | 26.43 | 12.69x |
| 16,384 | reset stress | 100 steps | 16 processes | 5,440.58 | 42.03 | 129.46x |
| 16,384 | flight | 100 steps | 16 processes | 5,437.23 | 49.67 | 109.46x |

At 16,384 lanes, flight rollout throughput is approximately 3.01 million lane
steps/s on the best tested CPU configuration versus 329.82 million on GPU.
The CPU wins at 33 lanes. Large-batch acceleration remains substantial after
using all CPU cores; the earlier sequential dynamics ratio should not be reused
as the parallel full-environment speedup.

Evidence: `.local-ci/cpu-parallel-20261117T015712.177977Z/report.json` contains
132 cases (12 GPU plus 120 CPU), all samples, setup times, reset counts, worker
outputs, affinity, CPU/compiler/GPU identity, source fingerprint and CPU binary
hash. All matched output/reset checks passed. Setup and final readback are outside
timing. Reset stress resets every step; flight records eight automatic resets per
lane across 510 intervals, proving this is not merely the same ground-reset loop.

## Interpretation and limitations

A supplemental run at the original dt=.02, a=-1, swept 1/2/4/8/16 OpenMP threads
with the same sample counts and completion cadences. At 16,384 lanes, the best
threaded CPU took 3,755.84 us versus GPU 54.20 us for per-step completion (69.29x),
and 2,985.38 us versus 43.03 us for rollout completion (69.39x). At 1,024 lanes
the corresponding ratios were 3.56x and 6.38x; CPU again won at 33 lanes.
Evidence: `cpu-parallel-20261117T022353.467304Z/report.json` under `.local-ci`.
This supplemental check covers threads only; do not label its winner the best
of a full multiprocessing sweep. It confirms the benefit is not exclusive to
the longer integration interval. The harness now reuses CPU thread-batch objects
between cases, resets them outside timing and records cold/reused setup cost;
the timed stepping algorithm is unchanged.

This is a strong tested baseline for unchanged Flightlib scalar physics/task
code, with an optimized benchmark dispatcher. It is not the untouched VecEnv
wrapper: that wrapper uses dynamic OpenMP scheduling and random automatic resets;
the adapter uses static scheduling and explicit resets for matched behavior,
preserves transition observations, and omits empty extra-info/render dispatch.
It is not a claim to the fastest conceivable CPU implementation. A CPU SoA/SIMD
rewrite or temporally fused per-lane rollout could improve CPU throughput, but
would be a new implementation rather than merely configuring base Flightmare.

The optimized dynamics differential suite runs before benchmarking. Each measured
case checks all lanes for finite/equal outputs, then compares final observation,
transition observation, reward, done and aggregate reset count against GPU.
These final-state checks complement rather than replace the existing trajectory
parity suites. Constant actions, one model, independent lanes and two reset
patterns are measured; diverse policies and geometry are not.

A rejected exploratory symmetric-action case exposed legacy Euler branch
sensitivity: tiny CPU attitude differences selected a pi/pi/pi observation
instead of zero and changed reward. That mismatch was not accepted or hidden by
an angle-equivalence tolerance. CUDA-7 subsequently reproduced it against both
frozen and optimized CPU, identified Eigen packet FMA accumulation, fixed the
CUDA ordering and added a saturated symmetry regression. See the
[numerical contract](flightlib-support.md) for its disposition and remaining
cross-platform approximation limits. The historical table above is unchanged.
An interrupted affinity-contaminated run was also rejected. Only the successful
report above supports the table.

The benchmark checks warmed GPU allocation counters; CUDA-6's separate trace
suite establishes residency of the production binding. This new comparison does
not claim a new independent Nsight or hardware-bandwidth measurement. CPU/GPU
clocks are unlocked and WSL scheduling can perturb results. Selecting the best
median is deliberately favorable to CPU; it is not a latency guarantee.
