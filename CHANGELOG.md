# Changelog

## 0.1.0 — development baseline

- CUDA development container and frozen CPU behavior oracle.
- Flightlib CUDA v1 tensor contracts and implementation architecture.
- Local validation profiles, source fingerprints, oracle integrity guard,
  and saved test reports.
- [CUDA-2] Persistent GPU arena, validated external bindings, asynchronous launch
  ordering, shared ownership, and runtime regression/profiling checks.
- [CUDA-3] Model/YAML preparation, exact float substep schedules, state and command
  lifecycle kernels, and CPU/GPU differential lifecycle validation.
- Separate Python systems profiling package with native runtime/lifecycle probes,
  CUDA call counters, independent Nsight traces, and a pre-CUDA-4 assessment.

- [CUDA-4] Fused dynamics, trajectory differential gate, and systems profiler
  stepping/scaling workloads with an optimized strict-FP CPU baseline.

- [CUDA-5] Native environment stepping, legacy observation/reward/terminal behavior,
  explicit and seeded device resets, episode lifecycle parity, reset-density
  profiling and matched wall-time regression gates.

- [CUDA-6] Resident PyTorch bindings with retained tensor ownership and explicit
  stream handoffs, native environment graph capture/replay and invalidation,
  legacy A=1 views, GPU rollout example, and Python/policy profiling workloads.

Geometry remains a subsequent layer; this
version does not imply complete integrated Flightmare parity.

- Post-CUDA-6: matched optimized CPU OpenMP/multiprocessing versus GPU environment
  benchmark, worker-affinity safeguards, behavior gates and collision-first
  CUDA-7 planning. No simulation equations or frozen tensor contracts changed.
- Scope correction: CUDA-7/8/9 now cover Flightlib state fidelity, resident core
  API/package completion and qualification. Collision geometry, rendering/sensors
  and the RL wrapper are deferred; the earlier collision-first plan is historical.
- [CUDA-7] Source-anchored core support matrix, targeted Eigen packet arithmetic
  parity fixes, saturated YAML Euler/reward regression, randomized model/update
  trajectories, explicit numerical budgets, and optimized CPU differential gates
  in local verification and profiling. Frozen oracle and tensor contracts unchanged.
- [CUDA-8] Standalone resident Core API, stream bridges and graph execution for
  accepted-command and command-plus-step paths, nine optional batched CPU helpers,
  complete public-core-method audit, external scene metadata, installed Python/C++
  consumers and core systems/CPU telemetry probes. Frozen contracts unchanged.
