# Repository correctness audit — 2026-11-17

Scope: accelerated native core, Python interoperability, RL usage, local build/test
and profiling harnesses, and retained upstream headless behavior. Version remains
0.1.2. This audit found and fixed three correctness risks; passing covered workloads
does not establish that no bugs remain.

## Findings and repairs

1. **Large-batch address overflow.** Lifecycle state/command inputs multiplied
   component indices by a 32-bit lane count. Model world-box indexing and initial
   reset quaternion indexing had the same problem; the state launch ceiling also
   added in 32 bits. Promote before arithmetic, following the existing 64-bit
   contract. The state problem appears beyond roughly 172 million lanes; this was
   found by source inspection, not by allocating a batch of that size on this GPU.
   Existing host contract checks cover addressing beyond 4 Gi elements. Native
   boundary/parity workloads validate the changed kernels at practical sizes.
2. **Eigen alignment mismatch.** UBSan reported a 64-byte-aligned Matrix4f access
   to a less-aligned stack address in model preparation, called by lifecycle
   parity. The runtime and optimized CPU oracle instantiate inline Eigen code
   using different SIMD flags; the linker can coalesce incompatible definitions.
   Use a distinct `Eigen::DontAlign` allocation-matrix type inside model
   preparation. No Eigen types cross the CUDA public API. The original alignment
   diagnostic is retained in `.local-ci/ubsan-before.log`; the same instrumented
   suite passes after the repair without disabling alignment checks. CPU model,
   dynamics and downstream task comparisons remain mandatory.
3. **Unchecked RL launch extents.** Bound tensors were validated, but scalar
   dimensions passed through ctypes were not checked against them. An oversized
   horizon/lane count could submit out-of-bounds work; ctypes can also truncate
   integers to C int. Validate the actual stats/GAE schemas and positive int32
   capacities at binding, then validate launch dimensions before native entry.
   GAE requires its bound stride and horizon; stats allows a prefix for lane-zero
   execution. Regression tests replace the FFI callable with a recorder so invalid
   dimensions are tested without deliberately launching unsafe GPU work.

## Additional coverage

`guarded_test` adds a diagnostic-only layout with gaps surrounding every nonempty
arena field. Default production offsets, tensor shapes and allocations are
unchanged. Tests fill/check gaps explicitly, outside the stepping path, and guard
external input/output allocations as well. A deliberately corrupted byte must be
detected before any workload can pass.

The matrix covers N=1/31/32/33/127/128/129/257, represented as E=1,A=N and E=N,A=1
(15 unique cases), inactive tail lanes, partial resets, lifecycle, both command
and dynamics paths, all nine optional core operations, every task execution mode,
explicit/random reset, seed changes and graph replay. Checks include guards,
inactive outputs, nonselected state preservation and read-only input preservation.
RL GAE fixtures now cover 127/128/129 lanes with terminal/invalid/horizon behavior
and independent expected returns across the whole batch.

Canaries detect writes into gaps, not invalid reads or every erroneous write
within a valid tensor. Allocation-level GPU memcheck alone cannot distinguish
adjacent tensor fields inside the arena. Differential tests and both mechanisms
are complementary. No guard scans, device reads or synchronization were added to
production stepping; RL scalar checks run on host metadata only.

## Evidence and scope

- Full verify: `.local-ci/20261117T062028.853719Z-verify/report.json`. Frozen oracle
  integrity, 36 CPU reference cases, native contracts/runtime/guards/differentials,
  optimized CPU checks, 15 Python core/interop tests, 33 RL cases, fresh installed
  native/Python examples and train/save/load/evaluate passed. One Python test is
  explicitly skipped because cross-device binding needs a second GPU.
- Upstream headless suite: `.local-ci/20261117T062205.915262Z-legacy/report.json`,
  passed. The aggregate includes demonstration tests and is not itself a parity
  claim. ROS/Unity/rendering/sensors and obsolete TensorFlow training integration
  are not executed or certified by this audit.
- Host UBSan: six native targets passed, including the previously failing
  lifecycle target. `.local-ci/ubsan.log` and `ubsan.xml` contain the manual
  instrumented run. The repeatable `ubsan` profile records subsequent runs with
  source/device/toolchain identity; the first profile run passed at
  `.local-ci/20261117T062407.839089Z-ubsan/report.json`.
  It instruments host code in C++/CUDA sources;
  GPU instructions and the frozen oracle static library are not instrumented.
- Profiling-analysis harness: all 11 failure/trace/measurement tests passed.
- Python static scan: Ruff 0.17.5, rules
  `F821,F822,F823,E9,B006,B008,B012,B018,B023,B904` over all three accelerated
  packages, scripts and retained flightlib/flightrl. No findings in accelerated
  code; one B904 exception-chaining diagnostic in legacy `flightlib/setup.py:49`
  remains. It is packaging diagnostics quality, not demonstrated CUDA corruption.
  `.local-ci/repo-ruff.json` retains the finding. Compiler warnings were reviewed;
  those observed concern test formatting/signed comparisons and upstream headers.
- Targeted source review covered offset arithmetic, bound pointer ownership,
  arena packing, stream/event ordering, capture invalidation, failed submissions,
  reset isolation, Python storage mutation and RL native submission. Static
  inspection does not replace executing concurrent workloads under instrumentation.

Fresh grouped RL benchmark and Nsight traces passed at
`.local-ci/rl-20261117T062532.479977Z/report.json`: 1,024 agents/environment,
N=1,024/16,384, 32 steps, two PPO epochs, nine samples/five warmups, CPU eight
threads with the same GPU learner. GPU collection medians were 5.477/7.880 ms;
whole iterations were 10.391/37.074 ms. Both downstream differential and unchanged
1.15x timing-bracket gates passed. Relative to this CPU configuration, whole-loop
speedups were 2.99x/5.32x; this was not another best-CPU sweep or a claim of a new
performance improvement. Collection traces had zero host/device transfers,
allocations/frees or CUDA API errors. Resident history/policy device-to-device
copies remain: 322 events, 4,149,248/66,387,968 bytes, within the existing budget.

The earlier `.local-ci/rl-20261117T062507.635757Z` run failed the timing-bracket
gate (ratio 0.665) and is not performance evidence. It used two warmups/five
samples and explicit OMP binding; the passing run used five/nine and the default
thread environment recorded in its report. No failure threshold was relaxed.

## Blocked checks

Compute Sanitizer fails to initialize NVIDIA's WDDM debugger interface on this
host. A normal runtime test passing inside that invocation is **not** a sanitizer
pass. The local `sanitize` profile fails immediately on instrumentation preflight;
after support is enabled it runs verify and all four sanitizer modes over runtime,
guarded arena, lifecycle, dynamics, task and selected RL workloads.
The recorded preflight failure is
`.local-ci/20261117T062449.162178Z-sanitize/report.json`; independent racecheck,
initcheck and synccheck probes also returned exit 1 with the same WDDM error
(`.local-ci/sanitizer-{racecheck,initcheck,synccheck}.log`).

NVIDIA documents the Windows `GPUDebugger/EnableInterface` prerequisite in its
[Compute Sanitizer guide](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/).
This process lacks administrator rights; the administrator setup request is
pending. No GPU timeout setting was changed. Racecheck focuses on shared-memory
hazards, not arbitrary global-memory or cross-stream races; the current production
kernels have no shared-memory/barrier algorithms to stress.

Host AddressSanitizer plus CUDA fails at `cudaGetDeviceCount` with out-of-memory
and driver-allocation leak reports. A standalone program calling only this API
reproduces the failure, while its unsanitized build succeeds. This is a host/tool
blockage, not a passing scan or an established simulator leak. Logs:
`.local-ci/host-sanitizers.log` and `.local-ci/driver-asan.log`. CPU-only contracts
passed under ASan; the GPU suite did not execute under it.

Multi-GPU, every future batch/model/policy distribution, hardware-counter analysis
and sim-to-real transfer remain outside this qualification. Do not describe this
audit as exhaustive or as complete GPU memory-safety certification.

## Reproduction

```sh
docker compose exec -T dev python3 scripts/local_ci.py verify
docker compose exec -T dev python3 scripts/local_ci.py legacy
docker compose exec -T dev python3 scripts/local_ci.py ubsan
docker compose exec -T dev python3 scripts/local_ci.py sanitize
docker compose exec -T -e PYTHONPATH=/workspace/flightmare-profile dev python3 -m unittest discover -s flightmare-profile/tests -v
```

Run profiles sequentially; they share a build-volume lock. Device sanitizer runs
are deliberately separate from routine checks and from performance timing.
