# CUDA-5 systems evidence

The native environment path now includes raw actions, dynamics, observations,
reward, terminal detection and automatic resets. One runtime submission launches
three kernels: preparation, D, fused T/R. The reference path launches four with
separate T and R. This is not yet Python-per-step, policy, graph or geometry timing.

The profiler measures both paths and explicit/random reset workloads. All measured
loops retained zero application allocation/free/copy/host-sync counts. Nsight
independently checked kernel counts and absence of recorded memcpy/memset events.
Ordering wait/record and stream checks occur once per environment step, not per
kernel. Dynamics remains one kernel regardless of substep count.

## Measured decisions

RTX 5070 Ti / driver 610.88 / CUDA 13.3 / 120-real; dt=.01, 100 intervals per
sample, five samples, ten warmup intervals. Clocks are unlocked. At 50% nominal
initial terminal density, median completed-step wall time in microseconds:

| Lanes | Fused T/R | Separate T/R |
| ---: | ---: | ---: |
| 1 | 26.510 | 34.679 |
| 33 | 32.140 | 36.086 |
| 16,384 | 33.569 | 41.308 |
| 65,536 | 78.967 | 93.912 |

The mask repeats every 100 lanes, so small batches do not have exactly 50%
terminal lanes. Actual episode resets and final done count are recorded. Evidence:
`.local-ci/profile-20261117T004627.014698Z/`. This run passed a 1.15x wall-time gate
against the previous matched run `profile-20261117T004313.730360Z`.

Always using double trig for Euler branches initially cost about 151 us at 65,536
lanes. Limiting the accurate calculation to branch-sensitive cases reduced that
to about 87 us while preserving parity. Reading only the 13 observation fields
and avoiding a transient terminal observation write reduced the latest result
to about 79 us. These are local diagnostic comparisons, not universal budgets.
The fused path is retained because it beats the reference schedule in measured
completed-step wall time while retaining the same lifecycle semantics.

Zero- and 100%-initial-terminal sweeps passed in
`profile-20261117T004412.848568Z` and `profile-20261117T004418.328746Z`; the latter
also exercises random resets. Sparse active masks, remainder schedules and random
resets passed with traces in `profile-20261117T004734.891829Z`. Random reset moves
terminal lanes into new states, so its subsequent reset density differs from the
explicit ground-reset stress workload. Compare workloads by their actual reset
counts; do not claim the random case is faster for equivalent reset work.

The dynamics-only workload also passed the 1.15x matched wall-time regression gate
against the CUDA-4 report: `profile-20261117T004424.316062Z` versus
`profile-20261117T002200.598847Z`. This prevents task integration from silently
regressing the existing layer.

## Traffic and kernel budget

No host round trips are used for terminal/reset decisions. Commands and all
outputs stay resident. Useful logical traffic accounting (not measured DRAM):

| Stage | Main per-active-lane traffic |
| --- | --- |
| Action preparation | Read 16 bytes raw action, 32 scaling, 104 physical state/time validation; write 21 command bytes plus clock/validity |
| D | CUDA-4's state/model/schedule budget, with one prepared-validity read |
| T | Read 52 state bytes, 16 raw action, 48 goal, 20 reward coefficients; emit 48 transition-observation bytes and reward/done/valid |
| Nonterminal output | Write 48 returned-observation bytes |
| Explicit terminal R | Read 100 reset-state bytes + time/mask; write physical state/motors, commands/counters and 48 returned reset-observation bytes |

These exclude additional diagnostics/topology/counter accesses, caching effects
and repeated transactions. The fused terminal path avoids writing and then
overwriting 48 bytes of returned observation per terminal lane. Random R generates
state locally instead of reading explicit reset-state/time arrays. There is no
global state copy, layout conversion buffer or global RNG-sample pass.

Current trace register counts: preparation 38, D 254, fused T/R 90, separate T 48,
separate R 64. Compiler reports no spill loads/stores; math routines can still
have stack frames. This does not prove achieved occupancy or bandwidth efficiency.
Hardware counters remain blocked by ERR_NVGPUCTRPERM; the existing sanitizer host
limitation also remains. No new arena buffers or frozen contract changes were
needed. The probe's extra external action array is 16 bytes/lane.

## Local performance gate

Use `flightmare-profile --baseline <report.json> --max-wall-ratio 1.15` with
matching GPU/driver/toolchain, control interval, active/reset densities, sample
counts and workload shapes. A missing/failed/mismatched baseline is rejected.
The report retains the baseline hash, per-case ratios and gate result. Changes
in timings are evaluated separately from mandatory parity, residency and launch
count checks. The 15% allowance is a configurable local diagnostic budget, not a
portable latency guarantee or a test added to ordinary verify. Investigate load
and rerun a failed timing gate before considering a reviewed budget change.

CUDA-6 should use this native path as its baseline, measure actual Python/policy
overhead, and evaluate graph submission and input-preparation fusion without
weakening ownership, error or transition semantics.
