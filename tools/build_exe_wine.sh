#!/bin/bash
# Builds the Windows .exe on Linux with Wine in a container.
set -euo pipefail
cd "$(dirname "$0")/.."
docker build --load -q -t atlas-wine-builder -f packaging/Dockerfile.wine packaging
docker run --rm -v "$PWD:/src" -w /src -e EXTRA_SELFTEST="${EXTRA_SELFTEST:-}" atlas-wine-builder bash tools/build_exe_wine_inner.sh
