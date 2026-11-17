#!/usr/bin/env bash
set -euo pipefail
cd /workspace
python3 -m venv --system-site-packages /build/flightlib-cuda-python
/build/flightlib-cuda-python/bin/pip install -r flightlib-cuda/python/requirements.txt
/build/flightlib-cuda-python/bin/pip install PyYAML==6.0.3 gymnasium==1.2.2
