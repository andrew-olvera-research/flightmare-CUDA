# CUDA-6 systems evidence

CUDA-6 adds the production PyTorch binding, retained CUDA tensor ownership,
producer/consumer stream ordering, A=1 compatibility views, explicit CPU
snapshots, and native environment graph capture/replay. The v1 contracts and
Flightmare physics/task kernels retain their shapes and numerical behavior.
Graph execution uses one host graph launch for preparation, D and fused T/R.

## Measured results

RTX 5070 Ti / driver 610.88 / CUDA toolkit 13.3 / 120-real; Python 3.12,
PyTorch 2.9.1+cu130. Control interval .01 seconds; 100 intervals/sample,
five samples, ten warmup intervals, 50% nominal initial terminal density.
Clocks are unlocked. Median completed-step wall time, microseconds:

| Lanes | Native | Native graph | Python | Python graph | Policy | Policy graph |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 24.750 | 13.700 | 27.608 | 13.638 | 78.837 | 86.747 |
| 33 | 29.653 | 16.767 | 28.997 | 17.494 | 85.411 | 83.612 |
| 16,384 | 33.286 | 27.493 | 36.957 | 25.645 | 114.214 | 112.705 |
| 65,536 | 78.084 | 73.129 | 81.675 | 76.298 | 153.250 | 147.455 |

Evidence: `.local-ci/profile-20261117T013741.581504Z/report.json`.
All 24 workload/shape combinations have independent Nsight node-level traces,
including intermediate shapes. Timings above are untraced samples; instrumented
trace times are not mixed into them. Reports retain samples, versions, source
fingerprint, binary hashes, capture/setup time and API/kernel counts.

Python environment graph replay reduced completed wall time by about 40% at
33 lanes, 31% at 16,384 lanes and 7% at 65,536 lanes in this run. Native graph
submission was roughly 2.4-2.7 us/iteration versus 8.3-9.3 us without graphs;
Python graph submission was 6.3-6.7 us versus 11.8-12.6 us. The larger case is
increasingly limited by actual simulation work rather than submission.
Small native/Python differences within one mode are measurement variability,
not evidence that adding Python accelerates a kernel.

The policy workload performs preallocated observation[N,12] x weights[12,4]
matmul into actions, environment stepping, and reward aggregation across slots
into persistent output[E]. It is a small GPU integration probe, not training or
a representative neural policy. Its feedback changes trajectories/reset density;
compare policy against policy-graph, not against fixed-action environment timing.
Workload version 2 records this per-environment reward aggregation and rejects
comparison to earlier global-sum policy reports.

Policy graph gains are modest at larger sizes (about 4% at 65,536 lanes here),
and graph replay is slower at one lane despite lower host submission time.
Do not claim universal end-to-end policy speedup. CPU scheduling, two GPU stream
handoffs, and policy kernel launches still matter. The graph currently captures
the environment only. Keep graph replay optional; measure complete target policy
loops before pursuing whole-loop capture or new fusion. There is no evidence
requiring a state-layout or physics redesign at this layer.

## Residency and launch gates

The warmed simulator loops contain no application CUDA allocation/free, copies,
host state reads, or host completion waits. Native counters and independent
traces agree on three kernels/step, or one graph launch with three kernel nodes.
Python tensor addresses remain stable; trace windows include exactly one event
synchronization at the measurement boundary. Stepping bridges streams with
persistent event record/wait operations, without allocating staging tensors.

Python allocator gates include cumulative allocation/free requests, device
allocation/free calls, current allocated bytes and reserved bytes. This catches
temporary cached tensor allocations that a net-byte check would miss. Nsight
separately checks copies, CUDA errors, kernel nodes and synchronization APIs.
Untraced allocator evidence alone cannot prove absence of transfers or allocations
outside the Torch allocator. Setup, capture, first replay/upload, explicit
snapshots and destruction are outside the warmed loop and may allocate or wait.

Tracing every shape found one library detail the original smallest/largest-only
traces missed: cuBLAS clears 1 KiB of persistent scratch before each skinny matmul
at 16,384 lanes. CUDA API ordering places this memset before the matmul and before
the environment graph, not in the reward reduction or simulator. It creates no
host/device copy or allocator churn. The harness initially failed on it; the
failed reports are retained (`profile-20261117T013347.791127Z` and
`profile-20261117T013552.555484Z`). The first run's attribution to reward summation
was disproved by correlating the CUDA calls.

The reviewed policy-only budget now permits at most one zero-fill, at most
1,024 bytes per iteration. Actual counts/bytes are reported; nonzero-pattern
clears, larger traffic, transfers, allocations or host synchronization still
fail. Native and Python environment-only workloads continue to require zero
memcpy/memset events. This is explicit GPU library bookkeeping, not a host-data
movement exemption. Use `--trace-all` when library algorithm choices can vary
by shape. Interval-union accounting handles overlapping streams without calling
idle gaps occupancy or measured bandwidth.

No frozen buffers were added. Arena sizes are about 8.62 MiB at 16,384 lanes and
34.46 MiB at 65,536 lanes. Python bound inputs add 121 bytes/lane; policy tensors,
framework/context memory and graph internals are separate from arena accounting.
Registration owners live until arena teardown, so repeated rebinding would retain
memory and is a setup-only API. Borrowed outputs preserve allocation, not history.

## Validation and reproduction

Local verify runs frozen CPU behavior, CPU/GPU lifecycle/dynamics/environment
differentials, exact native graph comparisons, and Python graph/native parity.
Python tests exercise multi-slot and legacy shapes, explicit/random resets,
invalid inputs/aliases, storage reallocation rejection, setup invalidation,
recapture, stream handoffs, exported lifetime, GPU policy feedback and rejection
of enclosing framework capture. Eight Python cases pass; cross-device binding
is explicitly skipped on this one-GPU host. Nine profiler analysis tests cover
failure gates, graph/node distinctions, interval overlap and scratch budgets.
The CMake-installed package also completes the resident rollout example.

The unchanged native environment passed a matched 1.15x wall-time gate against
CUDA-5: `profile-20261117T013052.578123Z` versus
`profile-20261117T004627.014698Z`. This compares the same shapes, density and
sampling settings. Sparse activity (10%), remainder timestep .006125, 33 and
16,384 lanes passed all eight workloads/traces in
`profile-20261117T013920.830704Z` (native graph, Python, Python graph and policy
graph at each shape). Failed exploratory traces are not presented as passes.

```sh
docker compose exec -T dev bash scripts/setup-cuda-python.sh
docker compose exec -T dev python3 scripts/local_ci.py verify
docker compose exec -T -e PYTHONPATH=/workspace/flightmare-profile dev python3 -m flightmare_profile --workloads environment,environment-graph,python,python-graph,policy,policy-graph --cases 1x1,3x11,1024x16,4096x16 --reset-percent 50 --iterations 100 --samples 5 --warmup 10 --trace-all
```

Hardware occupancy/bandwidth counters and Compute Sanitizer remain unavailable
under the previously documented host restrictions; these results claim neither
counter-level efficiency nor a sanitizer pass. Geometry, rendering, multi-GPU
scaling and full training-loop graph capture are outside CUDA-6 validation.
