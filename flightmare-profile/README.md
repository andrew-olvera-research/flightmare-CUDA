# Flightmare systems profiling

An independent Python package for measuring the CUDA rewrite before and during
dynamics implementation. Uses only Python's standard library. The native probe
links the actual runtime/lifecycle library; it is not an alternative simulator.
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
No cloud infrastructure or Python CUDA framework is required.

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

For CUDA-5 fusion and reset-density comparisons:

```sh
docker compose exec -T -e PYTHONPATH=/workspace/flightmare-profile dev python3 -m flightmare_profile --workloads environment,environment-split --reset-percent 50 --iterations 100 --samples 5 --warmup 10 --trace
```

Use reset-percent 0/50/100 and sparse active-percent runs. Environment probes start
selected terminal lanes at ground with raw action -1; other lanes start at altitude
50 with action 0. Explicit auto-reset repeats those samples; random auto-reset
instead samples the legacy reset distribution and changes subsequent reset density.
The repeated 100-lane mask is approximate for small batches. Reports include actual
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

Python orchestrates native loops. **These are not Python-per-step latency or
Torch interop measurements.** That overhead belongs in the harness when CUDA-6
adds the real binding. Do not add a synthetic Python binding and treat its cost
as the future production API. Only the dynamics workload measures physics stepping;
it does not measure task logic, policy integration, or geometry.

Three clocks are kept distinct:

- Host submission: wall clock around the native loop, excluding completion wait.
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
them. Trace kernel count and CUDA API errors are checked.

For these single-stream traces, summed kernel durations and first-to-last kernel
span give an idle-gap fraction. It is not occupancy or GPU utilization. Local
memory and register metadata are reported but do not substitute for Nsight
Compute's spill/occupancy metrics. Missing memcpy/memset tables mean no recorded
events of those types only after runtime and kernel trace data are verified.

## Invariants and extension points

The harness fails on functional result mismatch, wrong launch counts, hot-loop
application copies/allocations/frees/host synchronizations, trace errors, or a
changed source fingerprint during the run. No performance thresholds are frozen
yet. Arena payload, alignment padding, and external input bytes are separate;
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
model-diversity and command-mode sweeps, reset bursts, query density/capacity overflow, graph
replay, stream handoffs, and the real policy binding. Preserve source/schema and
hardware identity when adding machine-specific baselines. For multi-stream
workloads replace the single-stream idle formula with interval-union accounting.
