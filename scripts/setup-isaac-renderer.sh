#!/usr/bin/env bash
# Run in the Flightlib dev container after local_ci.py verify configures the native build.
set -euo pipefail
cd /workspace
target=/build/flightmare-isaac
cmake --build /build/flightlib-cuda-python-native --parallel
cmake --install /build/flightlib-cuda-python-native --prefix "$target"
/build/flightlib-cuda-python/bin/python -m pip install --no-deps --no-build-isolation \
    --upgrade --target "$target" ./flightmare-isaac
# Kit ships CUDA 12 Torch. Supply the CUDA 13 runtime and YAML ABI used by
# the installed Flightlib extension; the NVIDIA driver comes from the host.
mkdir -p "$target/deps"
cp -L /usr/lib/x86_64-linux-gnu/libyaml-cpp.so.0.8 "$target/deps/"
cp -L /usr/local/cuda/lib64/libcudart.so.13 "$target/deps/"
