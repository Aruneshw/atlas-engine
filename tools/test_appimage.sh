#!/bin/bash
# Run the built AppImage on clean distro containers under a real X server (xcb), then real training on one.
cd "$(dirname "$0")/.."
A=/src/dist/Atlas_Engine-x86_64.AppImage
RUN="xvfb-run -a -s '-screen 0 1400x900x24' $A --appimage-extract-and-run"
inst_deb='export DEBIAN_FRONTEND=noninteractive; apt-get update -qq; apt-get install -y -qq --no-install-recommends xvfb xauth libgl1 libegl1 ca-certificates >/dev/null'
inst_rpm='dnf install -y -q xorg-x11-server-Xvfb xorg-x11-xauth mesa-libGL mesa-libEGL ca-certificates >/dev/null'
for spec in "ubuntu:22.04|$inst_deb" "ubuntu:24.04|$inst_deb" "debian:13|$inst_deb" "fedora:42|$inst_rpm"; do
  img=${spec%%|*}; setup=${spec#*|}
  echo "=== $img quick"
  docker run --rm -v "$PWD/dist:/src/dist:ro" "$img" bash -c "$setup; timeout 120 $RUN --selftest 2>&1 | grep -E 'SELFTEST|error|Error|cannot open' | cut -c1-300"
done
echo "=== ubuntu:22.04 real setup + training + export"
docker run --rm -v "$PWD/dist:/src/dist:ro" ubuntu:22.04 bash -c "$inst_deb; timeout 520 $RUN --selftest-train 2>&1 | grep -E 'SELFTEST' | cut -c1-1500"
