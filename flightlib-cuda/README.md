# Flightlib CUDA

CUDA rewrite of Flightlib with persistent tensor state and unchanged Flightmare
quadrotor dynamics. No PhysX backend.

This initial package freezes the v1 data contracts; dynamics kernels and runtime
allocation are not implemented yet.

- [Architecture, tensor shapes, kernel boundaries, and risks](docs/architecture.md)
- [C++17 and CUDA compatible data contracts](include/flightlib_cuda/contracts.hpp)

Run host and device contract checks inside the existing dev container:

```sh
docker compose exec dev bash docker/check-cuda-contracts.sh
```

The CMake interface target is `flightlib_cuda::contracts`. Host-only consumers
can configure this directory without a CUDA toolkit. Device tests are enabled
with `FLIGHTLIB_CUDA_TEST_DEVICE=ON`; they check descriptors on a GPU, not physics.
