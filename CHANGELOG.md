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

Dynamics kernels are not implemented yet. This version identifies the rewrite
baseline, not completed CPU/GPU physics parity.
