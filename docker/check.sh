#!/usr/bin/env bash
set -euo pipefail

cd "${FLIGHTMARE_PATH:-/workspace}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"

nvcc --version
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
cmake -S docker/smoke -B /build/cuda-smoke -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${FLIGHTMARE_CUDA_ARCHITECTURES:-120-real}"
cmake --build /build/cuda-smoke
cuobjdump --list-elf /build/cuda-smoke/cuda_smoke
ctest --test-dir /build/cuda-smoke --output-on-failure

cmake -S flightlib -B /build/flightlib -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DFLIGHTLIB_USE_SYSTEM_DEPS=ON \
    -DBUILD_TESTS=ON -DBUILD_UNITY_BRIDGE_TESTS=OFF -DENABLE_FAST=OFF
cmake --build /build/flightlib
ctest --test-dir /build/flightlib --output-on-failure
PYTHONPATH=/build/flightlib python3 -m pytest -q docker/smoke/test_flightgym.py
