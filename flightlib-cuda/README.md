# Flightlib CUDA

Flightlib CUDA provides the physical simulation, task runtime and PyTorch bindings. Start with the [build instructions and runnable examples](../README.md). Use [Core](docs/core-api.md) for physical state, commands and motors, or [Environment](python/README.md) for the quadrotor task, collisions and sensor packets.

The [support matrix](docs/flightlib-support.md) describes CPU/CUDA numerical comparisons. The [architecture notes](docs/architecture.md) record kernel boundaries and tensor ownership. Earlier design proposals are collected in [development history](docs/history/).

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
