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

Policy integration, graph execution and geometry remain subsequent layers; this
version does not imply complete integrated Flightmare parity.
