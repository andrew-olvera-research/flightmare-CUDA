# Flightlib CUDA

Version **0.1.1** completes the scoped headless core and usage implementation.
Start with the [repository quick start](../README.md),
[implemented usage contracts](docs/cuda9-usage-contract.md), and
[current scaling evidence](docs/v0.1.1-scaling.md). The historical layer notes
below record development chronology, not unfinished implementation requirements.

CUDA rewrite of Flightlib with persistent tensor state and unchanged Flightmare
quadrotor dynamics. No PhysX backend.

The v1 data contracts are frozen. CUDA-2 provides persistent device storage,
validated external bindings, and ordered asynchronous launches. CUDA-3 adds
model/YAML preparation and state lifecycle. CUDA-4 provides fused dynamics,
CUDA-5 adds environment/task/episode behavior, and CUDA-6 adds resident PyTorch
integration and explicit graph execution. Geometry is deferred to a future
extension; [remaining work](docs/flightlib-scope.md) closes core parity/API gaps
and qualifies the standalone accelerated package.

- [Architecture, tensor shapes, kernel boundaries, and risks](docs/architecture.md)
- [C++17 and CUDA compatible data contracts](include/flightlib_cuda/contracts.hpp)
- [CUDA runtime API and ownership rules](include/flightlib_cuda/runtime.hpp)
- [Model setup and state lifecycle API](include/flightlib_cuda/lifecycle.hpp)
- [Resident Python API, installation and stream ownership](python/README.md)
- [CUDA-6 performance and validation evidence](docs/cuda6-performance.md)
- [CUDA-7 core support and numerical fidelity contract](docs/flightlib-support.md)
- [CUDA-7 validation and performance evidence](docs/cuda7-validation.md)
- [CUDA-8 standalone Core API and public-method checklist](docs/core-api.md)
- [CUDA-8 validation and performance evidence](docs/cuda8-validation.md)

Run host and device contract checks inside the existing dev container:

```sh
docker compose exec dev bash docker/check-cuda-contracts.sh
```

The interface target is `flightlib_cuda::contracts`; host-only consumers can
configure without CUDA. `FLIGHTLIB_CUDA_BUILD_RUNTIME=ON` builds
`flightlib_cuda::runtime` with CUDA 13 or later. `FLIGHTLIB_CUDA_TEST_DEVICE=ON`
also builds the runtime and its GPU validation test, replacing the redundant
standalone device-contract launch check. These tests validate infrastructure,
not physics parity. Run `./scripts/check.ps1 verify` for the full local gate.
