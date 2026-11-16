#!/usr/bin/env bash
set -euo pipefail

cd "${FLIGHTMARE_PATH:-/workspace}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"

cmake -S flightmare-parity-tests -B /build/flightmare-parity-tests -G Ninja \
    -DCMAKE_BUILD_TYPE=Release
cmake --build /build/flightmare-parity-tests
ctest --test-dir /build/flightmare-parity-tests --output-on-failure "$@"
