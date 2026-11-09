#!/bin/bash
# Runs inside the ubuntu:22.04 build container. Output: /src/dist/Atlas_Engine-x86_64.AppImage
set -euo pipefail
trap 'chmod -R a+rwX /src/dist 2>/dev/null || true' EXIT       # never leave locked files behind, even on failure
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends curl ca-certificates file binutils libegl1 libgl1 libglib2.0-0 libdbus-1-3 libfontconfig1 \
  libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
  libxcb-shape0 libxcb-xinerama0 libxcb-xfixes0 libxcb-sync1 libxcb-render0 libxcb-shm0 libwayland-client0 libwayland-cursor0 libwayland-egl1 > /dev/null

UV_VERSION=0.12.23
curl -fsSL "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-musl.tar.gz" | tar xz -C /usr/local/bin --strip-components=1
curl -fsSL -o /usr/local/bin/appimagetool "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
chmod +x /usr/local/bin/appimagetool

cd /src
BUILD=/tmp/build
rm -rf "$BUILD" dist/atlas-engine dist/AppDir
uv venv --python 3.12 "$BUILD/venv"
uv pip install -q --python "$BUILD/venv/bin/python" "PySide6-Essentials==6.8.2.1" "pyinstaller==6.16.0"
"$BUILD/venv/bin/pyinstaller" $(cat tools/pyi_args.txt) --distpath dist --workpath "$BUILD/work" --specpath "$BUILD" \
  --paths /src \
  --add-data "/src/atlas_engine/data:atlas_engine/data" \
  --add-data "/src/atlas_worker:worker_src/atlas_worker" \
  --add-binary "/usr/local/bin/uv:bin" \
  --hidden-import=atlas_worker.registry --hidden-import=atlas_worker.runtime_pins \
  tools/launcher.py

find dist/atlas-engine -name '__pycache__' -type d -prune -exec rm -rf {} +
APP=dist/AppDir
mkdir -p "$APP/usr/lib"
cp -a dist/atlas-engine "$APP/usr/lib/atlas-engine"
cp packaging/AppRun packaging/atlas-engine.desktop packaging/atlas-engine.png "$APP/"
ARCH=x86_64 appimagetool --appimage-extract-and-run --no-appstream "$APP" dist/Atlas_Engine-x86_64.AppImage
chmod -R a+rwX dist
ls -la dist/*.AppImage
