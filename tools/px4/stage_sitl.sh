#!/usr/bin/env bash
# Stage a trusted, already-built Ubuntu 24.04 x86_64 px4_sitl_default for the ROS service.
set -euo pipefail
src=$(realpath "${1:?pass the PX4 build directory}")
dest=${2:?pass a new output directory on the shared build volume}
test -x "$src/bin/px4"
test -f "$src/etc/init.d-posix/airframes/10016_none_iris"
deps=$(ldd "$src/bin/px4")
if [[ "$deps" == *"not found"* ]]; then
  echo "$deps" >&2
  echo "Source the toolchain environment used for this PX4 build and retry." >&2
  exit 1
fi
mkdir "$dest"
mkdir "$dest/lib"
cp -a "$src/bin" "$dest/bin"
cp -RL "$src/etc" "$dest/etc"
while read -r name path; do
  # The runtime image supplies glibc and its loader as a matched set.
  case "$name" in libc.so.*|libdl.so.*|libm.so.*|libpthread.so.*|librt.so.*|libresolv.so.*) continue ;; esac
  cp -L "$path" "$dest/lib/$name"
done < <(awk '$2 == "=>" && $3 ~ /^\// {print $1, $3}' <<<"$deps")
