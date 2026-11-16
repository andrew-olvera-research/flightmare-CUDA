#!/usr/bin/env bash
set -euo pipefail
cd "${FLIGHTMARE_PATH:-/workspace}"
cmake -S flightlib-cuda -B /build/flightlib-cuda-contracts -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DFLIGHTLIB_CUDA_TEST_DEVICE=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${FLIGHTMARE_CUDA_ARCHITECTURES:-120-real}"
cmake --build /build/flightlib-cuda-contracts --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
ctest --test-dir /build/flightlib-cuda-contracts --output-on-failure
