# Directed kernel diagnosis — 2026-11-17

Production simulator unchanged from v0.1.2. This replaces the earlier broad
experiment ordering with a measured next hypothesis. No driver settings changed.

## Harness

`python -m flightmare_profile.diagnose` orchestrates randomized timestep/reset
workloads through the existing native profiler. Each case has untraced completed
wall samples, an independent Nsight kernel trace, functional/residency gates,
observed reset counts and per-stage cost/share. Reports include binary hashes,
source fingerprints, underlying report paths/hashes and compiled resource queries
against the exact measured binary. One identical anchor repeats at campaign end.

Wall drift beyond symmetric 1.15x fails the campaign. Per-stage drift is reported
separately: stable wall time does not prove stable kernel attribution. Failed or
incomplete traces are rejected. Theoretical half-stage Amdahl estimates are marked
as estimates; no achieved speedup, bandwidth or utilization is inferred from them.

First campaign: `.local-ci/diagnose-20261117T073116.665093Z/report.json`, N=65,536
and 1,048,576, dt=0/.0025/.01/.02, reset fixture 0/100%, five samples of 40
intervals after 10 warmups. Eighteen measurements including repeated anchors.
Both wall anchors passed. Postprocessed `stage-stability.json` flags the smaller
case's dynamics trace ratio 1.219 as unstable; it is not used for fine attribution.

Confirmation: `.local-ci/diagnose-20261117T073404.049214Z/report.json`, N=1,048,576,
dt=0/.01, reset fixture 0/100%, seven samples of 100 intervals after 20 warmups.
Five measurements including anchor. Wall ratio 1.00034; stage ratios
prepare=.98994, dynamics=.99847, task/reset=1.00662. All gates passed. No copies,
allocations/frees or per-step host waits were introduced into measurement.

## Confirmation measurements

| dt | Reset fixture | Completed wall ms | Prepare us | Dynamics us | Task/reset us |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0% | 1.172 | 238.65 | 460.67 | 462.15 |
| .01 | 0% | 1.283 | 245.99 | 542.98 | 491.34 |
| 0 | 100% | 1.492 | 252.34 | 443.74 | 786.40 |
| .01 | 100% | 1.615 | 256.91 | 561.48 | 793.91 |

Kernel columns are separate instrumented-window averages, not components to
subtract from the untraced wall median. These fixtures contain independent agents,
one model, rotor commands and explicit resets; no neural network or learning.
Observed reset totals confirm zero resets in the 0% cases and one reset per agent
per interval in 100% cases (including warmup). This is not a trained-policy mix.

At no reset, removing integration changes completed time by only 8.7%, and the
zero-substep dynamics trace is still about 85% of the four-substep trace. Therefore
raw RK arithmetic is not the first thing to rewrite for this large-batch workload.
Zero-substep dynamics still validates, loads/stores state, constrains world bounds,
updates diagnostics/time and uses the same high-register compiled kernel. This
is evidence for substantial fixed cost, not proof of DRAM saturation.

At dt=.01, resetting every lane raises task/reset time from 491 to 794 us, about
62%, and completed time by 26%. Already at dt=0 the reset epilogue is expensive.
Reset state, dual observation materialization and output/counter handling deserve
attention independently of integration. The intervention changes trajectories
and terminal behavior, so the difference is not a pure cost per reset instruction.

Schedule inspection through the native `prepare_schedule` API confirms counts
0,1,4,9 for dt=0,.0025,.01,.02. The .02 case includes a final
1.862645149230957e-9-second remainder. Do not treat it as eight substeps, remove
the remainder, or assume motor-decay factors are identical. This preserves the
frozen CPU float-subtraction semantics.

## Next directed change

**Hypothesis: duplicate state traversal and intermediate command traffic are a
material fixed cost at large N.** Action preparation validates all 25 state
components and time; dynamics immediately loads and validates them again. The
preparation kernel itself costs about 246 us in the no-reset confirmation.

The next implementation should fuse preparation and dynamics around a shared
per-agent state load, retaining the existing dynamics equations and task/reset
kernel. Keep accepted command/clock visibility and failure ordering identical.
Make this one coherent change: do not simultaneously change RK formulation,
tensor layout, thread-per-agent mapping or Torch capture. Compare it to the pinned
baseline with the same campaign, inspect registers/local storage and then run
full parity/usage gates. Reject it if extra register pressure or spills erase the
saved work. This is a testable hypothesis, not a promised speedup.

The reset findings identify the following investigation, not another concurrent
change: determine which output/state writes can be avoided when integration is
immediately followed by reset, while retaining terminal observations and failed
reset semantics. That may eventually justify dynamics/task fusion, but its
register cost and numerical branches need evidence of their own.

We still cannot distinguish DRAM bandwidth, cache behavior, occupancy-limited
latency hiding and instruction issue quantitatively without hardware counters.
The harness does not request privileges or change that limitation. A focused
counter capture on another suitably configured GPU host remains an optional way
to resolve it. Current data supports testing redundant traversal before a more
invasive cooperative-agent or integrator redesign.

## Reproduction

Inside the existing container, with `PYTHONPATH=/workspace/flightmare-profile`:

```sh
python3 -m flightmare_profile.diagnose
python3 -m flightmare_profile.diagnose --lanes 1048576 --timesteps 0,.01 --samples 7 --iterations 100 --warmup 20
python3 -m unittest discover -s flightmare-profile/tests -v
```

Generated JSON and Markdown reports live in `.local-ci/diagnose-*/`. All 14
analysis tests pass, including wrong trace-window normalization, missing/ambiguous
kernel evidence and per-stage drift hidden by stable wall time. Production source,
the CPU oracle, contracts, `main` and `v0.1.2` remain unchanged.
