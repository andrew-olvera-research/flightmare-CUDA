# Flightlib CUDA

CUDA rewrite of Flightlib with persistent tensor state and unchanged Flightmare
quadrotor dynamics. No PhysX backend.

The v1 data contracts are frozen. CUDA-2 provides persistent device storage,
validated external bindings, and ordered asynchronous launches. Dynamics and
geometry kernels are not implemented yet. CUDA-3 adds model/YAML preparation,
float timestep schedules, and asynchronous reset/setState/setCommand operations.

- [Architecture, tensor shapes, kernel boundaries, and risks](docs/architecture.md)
- [C++17 and CUDA compatible data contracts](include/flightlib_cuda/contracts.hpp)
- [CUDA runtime API and ownership rules](include/flightlib_cuda/runtime.hpp)
- [Model setup and state lifecycle API](include/flightlib_cuda/lifecycle.hpp)

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
