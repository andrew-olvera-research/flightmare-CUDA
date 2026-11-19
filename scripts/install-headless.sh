#!/usr/bin/env bash
# Install into the persistent Compose build volume. Run from inside the dev service.
set -euo pipefail
case "$#:$*" in
  0:|1:--with-rl) ;;
  *) echo 'Usage: bash scripts/install-headless.sh [--with-rl]' >&2; exit 2 ;;
esac

cd /workspace
bash scripts/setup-cuda-python.sh --headless
venv=/build/flightlib-cuda-python
python="$venv/bin/python"
build=/build/flightmare-headless
site_dir=$("$python" -c 'import os, sys, sysconfig; print(os.path.relpath(sysconfig.get_path("platlib"), sys.prefix))')
cmake -S flightlib-cuda -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DFLIGHTLIB_CUDA_BUILD_RUNTIME=ON -DFLIGHTLIB_CUDA_BUILD_PYTHON=ON \
  -DPython_EXECUTABLE="$python" -DCMAKE_INSTALL_PREFIX="$venv" \
  -DFLIGHTLIB_CUDA_PYTHON_INSTALL_DIR="$site_dir" \
  -DCMAKE_CUDA_ARCHITECTURES="${FLIGHTMARE_CUDA_ARCHITECTURES:-native}"
cmake --build "$build" --parallel 4
cmake --install "$build"
if [[ ${1:-} == --with-rl ]]; then
  "$python" -m pip install --no-deps --no-build-isolation ./flightrl-cuda
fi
echo "Flightlib installed. Python: $python. CMake prefix: $venv."
