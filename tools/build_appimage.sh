#!/bin/bash
# Builds the AppImage in an Ubuntu 22.04 container, so it runs on that glibc and anything newer.
set -euo pipefail
cd "$(dirname "$0")/.."
docker run --rm -v "$PWD:/src" -w /src ubuntu:22.04 bash tools/build_linux_inner.sh
