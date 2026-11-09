#!/bin/bash
# Build both packages, then test them. Output: dist/. Stops at the first step that fails.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf dist/AppDir dist/atlas-engine dist/win
echo "=== build AppImage"; tools/build_appimage.sh > build-appimage.log 2>&1 || { tail -30 build-appimage.log; exit 1; }
rm -rf dist/AppDir dist/atlas-engine
echo "=== test AppImage"; tools/test_appimage.sh 2>&1 | grep -E '^===|SELFTEST'
if [ "${WITH_WINDOWS:-0}" = "1" ]; then          # Linux only by default
  echo "=== build exe"; EXTRA_SELFTEST="${EXTRA_SELFTEST:-}" tools/build_exe_wine.sh 2>&1 | grep -E 'SELFTEST|exit code|no result|Error' | cut -c1-400
fi
rm -rf dist/win
ls -la dist
echo RELEASE_DONE
