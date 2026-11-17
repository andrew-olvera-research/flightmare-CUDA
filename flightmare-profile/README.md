# Flightmare systems profiling

An independent Python package for measuring the CUDA rewrite before and during
dynamics implementation. The orchestrator uses Python's standard library; CUDA-6
Python/policy probes use the pinned PyTorch environment. Every probe uses the
production library or binding, not an alternative simulator.
Run locally in the existing Linux CUDA container, from the repository root:

```sh
docker compose exec -T -e PYTHONPATH=/workspace/flightmare-profile dev \
  python3 -m flightmare_profile --iterations 500 --samples 7 --trace
```

The package can also be installed with `pip install -e flightmare-profile` in a
development environment with setuptools. The CLI requires this source checkout
(`--repo`), the CUDA container's dependencies, CMake/Ninja, and an NVIDIA GPU.
Nsight Systems is required only for `--trace`. A failed trace is not a green run.
Results live in ignored `.local-ci/profile-<timestamp>/`: JSON, Markdown summary,
per-case samples/logs, binary/source identity, and optional Nsight/SQLite files.
No cloud infrastructure is required. Native probes do not require PyTorch.

## Measurement design

| Workload | One iteration |
| --- | --- |
| dispatch | Runtime callback with no kernel; measures event/control submission floor |
| direct | One minimal touch kernel launched directly on a nonblocking stream |
| runtime | The identical touch kernel through Simulation::enqueue |
| state | One actual setState operation |
| command | One actual setCommand operation |
| mixed | setState, setCommand, resetState: three operations, not a physics timestep |
| dynamics | One real fused control interval; constant rotor commands, configurable control_dt |
| environment | Raw actions through dynamics, observations/reward/terminal, explicit auto-reset; three launches |
| environment-split | Same workload, separate T/R reference; four launches |
| environment-random | Three-launch environment with seeded device random auto-reset |
| environment-graph | Native explicit-reset environment through one graph launch / three kernel nodes |
| python / python-graph | Production Python binding, ordinary submission / environment graph replay |
| policy / policy-graph | Python binding plus preallocated GPU observation matmul -> actions and per-environment reward reduction |

For CUDA-6, install the pinned binding dependencies once with
`docker compose exec -T dev bash scripts/setup-cuda-python.sh`, then run:

```sh
docker compose exec -T -e PYTHONPATH=/workspace/flightmare-profile dev python3 -m flightmare_profile --workloads environment,environment-graph,python,python-graph,policy,policy-graph --cases 1x1,3x11,1024x16,4096x16 --reset-percent 50 --iterations 100 --samples 5 --warmup 10 --trace
```

The harness incrementally builds the actual extension. Override
`--python-interpreter` and `--python-build-dir` when needed. Graph capture and
first replay/upload happen outside measurement; capture_ms reports host capture/
instantiation time, not upload completion latency. Python inputs/outputs and
policy workspaces are persistent. The policy is a 12x4 linear matmul and reward
sum over slots, not a trained network or optimizer; its feedback changes trajectories and
reset density, so compare policy against policy-graph, not fixed-action latency.
The graph captures the environment, not the whole policy/learning iteration.
Use `--trace-all` to trace intermediate shapes too: framework algorithms can
switch by shape, so tracing only the smallest/largest cases can miss temporary
workspaces. Policy workload version 2 uses per-environment aggregation; matched
baseline checks reject older global-sum workloads.

For CUDA-5 fusion and reset-density comparisons:

```sh
docker compose exec -T -e PYTHONPATH=/workspace/flightmare-profile dev python3 -m flightmare_profile --workloads environment,environment-split --reset-percent 50 --iterations 100 --samples 5 --warmup 10 --trace
```

Use reset-percent 0/50/100 and sparse active-percent runs. Environment probes start
selected terminal lanes at ground with raw action -1; other lanes start at altitude
50 with action 0. Explicit auto-reset repeats those samples; random auto-reset
instead samples the legacy reset distribution and changes subsequent reset density.
The repeated 100-lane mask is approximate for small batches. Native reports include actual
auto-reset counts (including warmup), random draws and final done count. Functional
readback checks active validity/status, finite outputs and inactive output isolation.
The differential suite remains responsible for complete transition parity.

An optional local wall-time gate compares against a prior matched report:

```sh
docker compose exec -T -e PYTHONPATH=/workspace/flightmare-profile dev python3 -m flightmare_profile --workloads environment,environment-split --reset-percent 50 --iterations 100 --samples 5 --warmup 10 --baseline /workspace/.local-ci/profile-<baseline>/report.json --max-wall-ratio 1.15
```

Keep the same cases, toolchain, GPU/driver, sample settings and densities. Failed,
missing or mismatched evidence cannot pass. A wall-time regression fails the command
and retains ratios in report.json. This opt-in gate stays outside ordinary verify;
residency and launch budgets are mandatory on every run. The CPU comparison below
currently covers dynamics only, not the environment workloads.

CUDA-4 adds actual stepping and an optional **sequential optimized CPU** baseline:

```sh
docker compose exec -T -e PYTHONPATH=/workspace/flightmare-profile dev python3 -m flightmare_profile --workloads dynamics --iterations 100 --samples 5 --warmup 10 --trace --cpu-baseline --compute
```

The CPU comparison builds unchanged Flightlib sources separately with `-O3
-fno-fast-math -ffp-contract=off -march=native`, first runs trajectory checks,
then uses the same initial state, rotor commands, 0.01-second interval, warmup,
sample lengths and lane counts as GPU. The frozen correctness oracle remains
unchanged at -O0. This is not an optimized multicore CPU comparison. Initialization,
command upload and final readback are excluded from both steady-state loops.
CPU metadata and samples are in report.json; its binary hash is recorded.

Use `--control-dt .006125` for remainder/substep studies and `--active-percent 10`
for sparse dynamics. CPU comparison currently requires dt=.01 and all lanes active.
Zero duration still launches once to validate lanes and apply legacy world-box
semantics. Dynamics completion checks status and exact accumulated simulation time;
the differential suite owns full trajectory comparisons. `--compute` attempts a
separate Nsight Compute capture for the largest dynamics case. Unavailable counters
are explicitly reported and are not a bandwidth/occupancy pass; inspect its CSV/log.

Direct/runtime touch workloads update every lane regardless of active-percent;
state/command/mixed use the requested active mask. The mask repeats across groups
of 100 lanes; actual active-lane count is recorded for small nonmultiple batches.
All input allocations and registrations happen once before warmup. Every case
has a fresh process/context; setup time includes context/JIT/initialization costs
and must not be compared to steady-state step time. Jobs are shuffled with seed 0.

Native workloads use C++ loops; python/policy workloads execute the production
Python API on every iteration, including tensor-owner checks and stream handoffs.
All timings exclude setup, warmup and final functional readback. Dynamics covers
physics only; environment includes task/reset; policy adds the small GPU consumer/
producer workload. No current workload measures geometry.

Three clocks are kept distinct:

- Host submission: wall clock around the C++ or Python loop, excluding completion wait.
- End-to-end wall: timing markers through one explicit completion wait per sample.
- GPU timeline: CUDA event elapsed time, including idle/queue gaps between work.

Samples retain median/min/max and nearest-rank p95. Seven samples are an early
diagnostic, not a reliable tail-latency distribution. No clocks are locked; GPU
temperature/clocks/power and hardware/toolchain identity are recorded. Other GPU
processes and WDDM scheduling can still perturb measurements. Do not conclude
one path is faster from a small difference or a single case.

Link-time wrappers count selected application CUDA calls only while the loop
runs. They include CUDA 13's internal launch entry point and fail if launch
counts differ from the workload. They do not capture every driver-internal call
or allocation. Timing includes counter overhead. Nsight traces are separate
processes with one short measured sample, so traced durations do not contaminate
the untraced sample distribution. Capture covers boundary markers and the final
explicit synchronization as well as the loop. Runtime workloads therefore add
two marker submissions to trace event counts; native hot-loop counters exclude
them. Trace kernel count and CUDA API errors are checked. Graph host API launch
counts and GPU kernel-node counts are checked separately. Nsight runs with
`--cuda-graph-trace=node`; its default graph-level view would hide those nodes.

Python probes report PyTorch allocator deltas rather than pretending native link
wrappers observe framework calls. Deltas include cumulative allocation/free
requests as well as device allocations and current bytes, catching temporary
cached workspace churn. They do not rule out transfers or non-Torch allocation.
Use `--trace` (or `--trace-all`) to check the measured Python path:
it rejects malloc/free/memcpy and device/stream synchronization APIs, requires
one event completion wait at the measurement boundary, checks CUDA errors and
recorded memory events, and counts all three environment kernel nodes per
iteration. Environment traces require zero memcpy/memset events. Policy workloads
also require zero copies, but permit at most one 1-KiB device zero-fill per
iteration: the measured cuBLAS skinny-matmul scratch initialization at intermediate
shapes. Counts and bytes remain in the report; larger or nonzero-pattern clears
fail. This allowance does not permit allocation or synchronization. Policy kernels
are reported separately. This is evidence about the
warmed measured window, not all possible user policy implementations.

For multi-stream traces, the union of kernel intervals and first-to-last kernel
span give an idle-gap fraction; summed kernel durations are also retained. The
union avoids double-counting concurrent kernels. It is not occupancy or GPU utilization. Local
memory and register metadata are reported but do not substitute for Nsight
Compute's spill/occupancy metrics. Missing memcpy/memset tables mean no recorded
events of those types only after runtime and kernel trace data are verified.

## Parallel CPU comparison

Use the production Python GPU graph path against optimized unchanged Flightlib
environments, with both in-process OpenMP and persistent spawned workers:

```sh
docker compose exec -T -e OMP_PROC_BIND=spread -e OMP_PLACES=cores -e OPENBLAS_NUM_THREADS=1 -e PYTHONPATH=/workspace/flightmare-profile dev /build/flightlib-cuda-python/bin/python -m flightmare_profile.cpu_parallel --lanes 33,1024,16384 --workers 1,2,4,8,16 --chunks 1,100 --iterations 100 --samples 5 --control-dt .05
```

Requires `scripts/setup-cuda-python.sh`. Builds are requested under the common
local build lock. The benchmark runs optimized dynamics and environment parity
(including randomized models and saturated symmetric Euler/reward cases), then a one-lane
matched-output preflight before the sweep. Each case also checks output/reset
agreement. `--modes thread` selects only OpenMP; `--actions=-1` selects reset
stress, and `--control-dt .02` matches the original environment timestep.
The default second action is an asymmetric flight workload at dt=.05; at shorter
intervals the legacy ground-reset terminal condition may prevent lift-off.

Chunk=1 measures completed-step latency; chunk=100 amortizes completion across
100 steps. Actions/state/outputs stay local on both sides. There is no CPU policy
upload or worker-output serialization penalty. Pick the best measured CPU median
across worker counts, not the sequential CPU or a fixed process count. Initialization
can be expensive at large batches and is reported outside timing. Worker affinity
is explicitly restored to avoid inheriting an OpenMP-pinned coordinator core.

Reports and partial progress live in `.local-ci/cpu-parallel-<timestamp>/`.
A failed/interrupted report is not performance evidence. See
[the measured comparison and limitations](../flightlib-cuda/docs/cpu-parallel-performance.md).

## Invariants and extension points

The harness fails on functional result mismatch, wrong launch counts, hot-loop
application copies/allocations/frees/host synchronizations, trace errors, or a
changed source fingerprint during the run. Python untraced runs enforce allocator
deltas; independent trace gates provide the broader residency evidence above.
Optional matched baselines enforce the configured wall-time ratio. Arena payload,
alignment padding, and external input bytes are separate;
driver/context overhead and peak VRAM are not measured by those byte counts.

Builds are incremental but always requested so stale binaries are not silently
used. The harness uses the same build-volume lock as local verification; profiling
is deliberately separate from normal tests. Profiling instrumentation stays in the probe; the frozen CPU oracle is unchanged.
To validate the analysis logic:

```sh
docker compose exec -T -e PYTHONPATH=/workspace/flightmare-profile dev \
  python3 -m unittest discover -s flightmare-profile/tests -v
```

Future layers should add representative production workloads here, including
model-diversity and command-mode sweeps, reset bursts, query density/capacity
overflow, and representative trained policies. Preserve source/schema,
Torch/CUDA versions and hardware identity when adding machine-specific baselines.
