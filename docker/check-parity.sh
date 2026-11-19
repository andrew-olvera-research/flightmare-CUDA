#!/usr/bin/env bash
set -euo pipefail

cd "${FLIGHTMARE_PATH:-/workspace}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"

cmake -S tests/parity -B /build/v3-parity -G Ninja \
    -DCMAKE_BUILD_TYPE=Release
cmake --build /build/v3-parity
ctest --test-dir /build/v3-parity --output-on-failure "$@"
