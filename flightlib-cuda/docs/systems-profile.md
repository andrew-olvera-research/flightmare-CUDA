# Systems profile before CUDA-4

The current evidence supports keeping the frozen component-major tensors and
persistent arena. It also shows that small operations are launch limited: CUDA-4
must preserve the planned fused control-interval boundary. This is validation of
the runtime/lifecycle foundation, not a dynamics throughput or trajectory-parity
result. No production runtime or tensor contracts were changed for this profile.

## Reproducible local harness

[flightmare-profile](../../flightmare-profile/README.md) is a separate v0.1.0
Python package. Python builds and orchestrates a native probe linked against the
actual runtime. It records host submission, synchronized wall time, CUDA event
timeline time, selected CUDA API counts, arena categories, external input bytes,
and optional independent Nsight Systems traces. Production Python call overhead
will be measured when the actual binding exists.

Each case gets a fresh process, setup and binding registration, 100 warmup
iterations, then repeated samples. Functional readback occurs after timing.
Copies, allocations, frees, and host completion waits inside the loop fail the
run; launch counts must match the workload. Traces independently check kernel
counts, API errors, and recorded memcpy/memset activity. Source and binary hashes
are retained, and source changes during measurement invalidate the run.

The timing sweep is separate from ordinary verify. Performance thresholds are
not yet justified; functional/residency failures are hard failures today.

## Local evidence, 2026-11-16

RTX 5070 Ti, NVIDIA driver 610.88, CUDA 13.3 container, Release build targeting
120-real, Docker Desktop/WSL on Windows. Clocks were not locked. These are local
diagnostics subject to scheduling and clock variation, not portable budgets.

Main run: 500 iterations/sample, 7 samples, 5 shapes (1x1, 3x11, 1024x1,
1024x16, 4096x16), 6 workloads = 30 timing cases. Six separate traces cover
direct/runtime/mixed at the smallest and largest sizes, 256 iterations each.
All cases and traces passed. Local raw evidence is retained in
`.local-ci/profile-20261116T211929.510294Z/` (ignored build artifacts).
The measured library revision is `367c753`; report.json also identifies the
uncommitted harness source fingerprint and binary hash.

Selected medians in microseconds per iteration:

| Lanes | Workload | Host submission | Wall completion | GPU timeline |
| ---: | --- | ---: | ---: | ---: |
| 1 | Empty runtime submission | 0.543 | 0.711 | 0.636 |
| 1 | Direct touch kernel | 2.494 | 6.173 | 6.118 |
| 1 | Same touch through runtime | 3.296 | 6.705 | 6.653 |
| 1 | setState + setCommand + resetState | 17.084 | 23.674 | 23.660 |
| 65,536 | Direct touch kernel | 2.433 | 6.474 | 6.438 |
| 65,536 | Same touch through runtime | 3.276 | 6.539 | 6.210 |
| 65,536 | setState | 3.529 | 13.361 | 13.276 |
| 65,536 | setCommand | 3.423 | 8.967 | 8.882 |
| 65,536 | setState + setCommand + resetState | 20.983 | 36.105 | 36.114 |

Each column is summarized independently; medians need not add up or preserve
clock ordering. GPU event elapsed time includes idle gaps and is not kernel
execution time. Submission can include driver queue backpressure. The mixed
workload is a lifecycle stress sequence, not a proposed production timestep.

A second run exercised state/command/mixed with a 10% repeating active mask,
3x11 and 1024x16 capacities, 500 iterations and 5 samples: all six cases passed.
Actual active counts are recorded because small batches do not contain an exact
percentage. Evidence: `.local-ci/profile-20261116T212129.452985Z/`.
Sparse activity retains the same launch count and capacity-based storage.

### Residency and dispatch

All measured loops had zero application allocations, frees, copies, and host
synchronizations. The traced windows also contained no recorded memcpy/memset
events. Setup transfers and one completion wait per sample are intentional and
outside the loop counters. Wrappers count selected application runtime APIs;
they do not prove the absence of every driver-internal allocation.

Each lifecycle operation currently makes these measured calls:

| API group | Calls per operation |
| --- | ---: |
| Kernel launch | 1 |
| Stream event wait / event record | 1 / 1 |
| Get device / set device | 1 / 2 |
| Stream device / flags / capture query | 1 / 1 / 1 |
| Last-error check | 1 |

The same-stream completion chain and metadata checks repeat for every operation.
At one lane, traced direct/runtime touch kernels occupy only about 9%/7% of the
first-to-last-kernel span. The mixed lifecycle sequence occupies about 27%; at
65,536 lanes it occupies about 60%. These are single-stream timeline fractions,
not occupancy or device utilization. Both driver/platform launch overhead and
runtime submission work contribute; this does not isolate a causal cost for
each API call. The direct/runtime submission comparison suggests about 0.8 us
additional runtime cost on this host, but wall-time differences are noisy.

Nsight reports 64 registers/thread for the state kernel and 34 for command,
with zero reported local bytes/thread. This is a reason to inspect register
pressure as dynamics grows, not evidence of a proven occupancy or spill problem.

### Storage budget

At 65,536 lanes, the persistent arena is 36,133,888 bytes (34.460 MiB), plus
8,257,536 bytes (7.875 MiB) of this probe's registered external inputs. At 16,384
lanes the arena is 8.617 MiB. These cases use one model and no schedule or query
capacity. The arena includes eagerly reserved task/reset/output buffers, not
just physical state. At large capacity it is roughly 551 bytes/lane before
external inputs. Driver/context overhead and peak process VRAM are not included.

This is a reasonable initial budget, with no observed steady-state growth in
application allocation calls. Large query capacities, many models, and million-
lane deployments still need separate memory-budget cases. Binding registrations
must stay outside the loop: retained external owners survive until teardown.

## Decisions for the next layers

### Cost categories and issue ownership

Data movement minimization is a design priority, but the current measured
problem is dispatch overhead, not CPU-to-VRAM transfers. Separate three costs:
host/device transfers (zero in these measured loops), device-memory traffic
(not yet quantified), and launch/ordering gaps (observed). Residency does not
prove bandwidth efficiency, and timeline gaps do not prove bandwidth saturation.

The first algorithmic response is to group useful work: one dynamics submission
per nonzero control interval, with controller, motor update and RK substeps
inside the kernel. Retain intermediates locally and write final telemetry where
semantics require it. Avoid a host loop launching individual integration stages.
Check register pressure and spills: excessive fusion can increase device-memory
traffic and lose performance. Graph replay reduces host dispatch across remaining
stages but does not eliminate their intermediate global-memory traffic.

The corresponding GitHub issue bodies now carry explicit acceptance items:

| Layer | Optimization and required evidence |
| --- | --- |
| CUDA-4 / #4 | Fused dynamics; launch budget independent of lanes/substeps; logical bytes per interval; measured DRAM/L2 traffic, bandwidth, registers, local-memory traffic and occupancy where tooling permits; parity and wall-time comparison |
| CUDA-5 / #5 | Minimize D/T/R passes and output copies; device reset decisions; reset-density sweep; preserve transition observations |
| CUDA-6 / #6 | Measure same-stream wait elision and metadata caching safely; graph replay; bind-once policy tensors; count device-to-device copies; measure actual Python/policy overhead |
| CUDA-7 / #7 | Shared scene assets, bounded scratch, transform/update/query traffic; qualify refit/rebuild costs |
| CUDA-8 / #8 | Bounded candidate traffic; avoid unbounded all-pairs materialization; device overflow/validity handling |
| CUDA-9 / #9 | Integrated launch, transfer, bandwidth and capacity evidence; reviewed machine-specific baselines |

Parent #1 records this ownership without changing layer dependencies. Reuse
flightmare-profile throughout. Keep logical byte estimates separate from hardware
traffic counters, and report unavailable counters explicitly. Reassess after the
first parity-passing CUDA-4 workload before CUDA-5. Runtime dispatch optimization
belongs to CUDA-6 unless CUDA-4 measurements establish an earlier dependency;
the fused dynamics algorithm belongs to CUDA-4 immediately.

1. Keep the tensor layout, stable lanes, persistent arena, and setup/upload
   boundary. There is no current measurement supporting a contract redesign.
2. Implement CUDA-4's dynamics as the planned fused control-interval operation.
   Keep controller, motor response, integration stages, and final state handling
   inside that boundary; do not issue host submissions per lane or RK stage.
   Extend this harness with actual control intervals and assert the intended
   launch count plus zero hot-loop transfers/allocations/synchronizations.
3. Profile register pressure, spills, wall time, and parity before enlarging or
   splitting fused kernels. Use narrow internal kernel views where useful while
   preserving the public frozen contract. Fusion alone is not a performance pass.
4. Evaluate same-stream wait elision and caching stable stream/device metadata
   as measured runtime optimizations. Preserve cross-stream ordering, device
   restoration, error handling, and resource lifetimes. The current observations
   do not justify removing those protections globally.
5. Keep graph replay and production Python/policy interop in CUDA-6. Internal
   batched work should pay the runtime ownership boundary once, not once per
   internal node. Re-measure with the real binding before setting latency targets.
6. Add realistic reset bursts and geometry capacities as their layers arrive.
   Compare useful active work as well as allocated lanes; a sparse mask does not
   itself eliminate dispatch cost or shrink persistent memory.

Proceed with CUDA-4 using these constraints. Reassess after its first parity-
passing control interval; the present workload cannot establish physics speedup,
memory-bandwidth saturation, multi-stream scaling, or policy-loop throughput.
