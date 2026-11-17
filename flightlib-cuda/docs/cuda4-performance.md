# CUDA-4 stepping and scaling

The fused algorithm passes trajectory parity and benefits batched simulation.
Small batches remain launch limited. Keep this boundary for CUDA-5; do not infer
whole-environment or policy throughput from these dynamics-only measurements.

Measured on RTX 5070 Ti / driver 610.88 / CUDA 13.3, 120-real. Local evidence:
`.local-ci/profile-20261117T002200.598847Z/report.json`. Five samples of 100
intervals after 10 warmup intervals, dt=.01, constant rotor commands, all lanes
active, one default model. CPU and GPU start with identical state and follow
identical interval counts. CPU is sequential, unchanged Flightlib source built
separately at O3 with strict floating point; its trajectory suite also passed.
The frozen O0 oracle is still used by verify and was not used for these speedups.

| Lanes | GPU completed interval us | CPU completed interval us | CPU/GPU wall |
| ---: | ---: | ---: | ---: |
| 1 | 8.679 | 0.574 | 0.066x |
| 1,024 | 9.833 | 601.989 | 61.2x |
| 16,384 | 10.831 | 9,970.34 | 920.5x |
| 65,536 | 31.787 | 38,841.4 | 1,221.9x |

These are amortized interval wall times, not individual-step latency percentiles.
No policy, task, Python-per-step, geometry or host-adapter costs are included.
This is not a comparison with an optimized parallel CPU simulator. Clocks are
unlocked and figures are machine/workload-specific. One lane is slower on GPU.

The loop uses exactly one dynamics launch per interval, with zero measured
application GPU allocations/frees/copies/host synchronizations. Nsight traces
independently confirm launches and no recorded memory-transfer events. Kernel
time is distinct from CUDA-event timeline and synchronized wall completion.

RK stages reuse slope/sum/temporary storage. Compilation reports 254 registers
per thread, zero stack and zero spill loads/stores. A 128-register variant
introduced 168 bytes of spill loads/stores and failed to improve wall time
consistently: 16,384-lane wall time rose from roughly 10.0 to 11.6 us; 65,536
lanes stayed roughly 29.7-29.9 us in that comparison. It was rejected. Evidence:
`profile-20261117T001855.999839Z` and `profile-20261117T001951.676255Z` under
`.local-ci`. High register use remains a tuning opportunity, not a reason to
split RK stages into separately launched global-memory passes.

Logical traffic accounting per active lane: state/time/motors read+write 272
bytes, accepted command 21 bytes, active/model ID 5 bytes, bounds 24 bytes, status
write 4 bytes: 326 bytes, excluding model and schedule reads. Unique referenced
model data is 136 bytes for rotor mode, plus 76 for rate mode; schedule is 4*S
bytes. These are logical access budgets, not actual DRAM traffic or a bound on
cache transactions/reloads. Model data is shared and can benefit from caches.
There are no global RK-stage arrays or per-substep state writebacks.

Nsight Compute was attempted but returned ERR_NVGPUCTRPERM. Hardware DRAM/L2
traffic, bandwidth and achieved occupancy remain unmeasured on this host. The
profiler's --compute option retains this limitation explicitly. Compiler spill
statistics do not substitute for those counters. Sanitizer host limitations
from CUDA-2 also remain unresolved.

Final local verify passed in `.local-ci/20261117T002412.839677Z-verify/`.
Sparse-lane/remainder profiles passed in `profile-20261117T002434.530938Z`
(including explicit unavailable Compute counters); zero-duration and all seven
workload paths passed in `profile-20261117T002501.049715Z`. These artifacts are
local and ignored by Git; their report files retain source fingerprints.

Next: keep the one-launch dynamics boundary, add T/R traffic accounting under
CUDA-5, then evaluate runtime submission/graph replay under CUDA-6. Revisit
register/model-loading choices with hardware counters when available. Do not
change frozen tensor semantics based solely on synthetic small-kernel gaps.
