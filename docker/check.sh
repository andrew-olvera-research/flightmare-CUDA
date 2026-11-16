#!/usr/bin/env bash
set -euo pipefail

cd "${FLIGHTMARE_PATH:-/workspace}"
exec python3 scripts/local_ci.py verify
