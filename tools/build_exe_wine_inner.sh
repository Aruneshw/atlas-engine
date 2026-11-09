#!/bin/bash
# Runs inside the Wine builder. Output: /src/dist/Atlas_Engine-windows-x64.zip
set -euo pipefail
cd /src
rm -rf /tmp/wb dist/win
mkdir -p /tmp/wb dist/win
wine /opt/py/python.exe -m PyInstaller $(cat tools/pyi_args.txt | tr -d '\r') \
  --distpath 'Z:\tmp\wb\dist' --workpath 'Z:\tmp\wb\work' --specpath 'Z:\tmp\wb' --paths 'Z:\src' \
  --icon 'Z:\src\packaging\atlas-engine.ico' \
  --add-data 'Z:\src\atlas_engine\data;atlas_engine\data' \
  --add-data 'Z:\src\atlas_worker;worker_src\atlas_worker' \
  --add-binary 'Z:\opt\uv\uv.exe;bin' \
  --hidden-import=atlas_worker.registry --hidden-import=atlas_worker.runtime_pins \
  'Z:\src\tools\launcher.py' 2>&1 | tail -4
wineserver -w || true
find /tmp/wb/dist -name '__pycache__' -type d -prune -exec rm -rf {} +
(cd /tmp/wb/dist && zip -qr9 /src/dist/Atlas_Engine-windows-x64.zip atlas-engine)
chmod a+rw /src/dist/Atlas_Engine-windows-x64.zip
ls -la /src/dist/Atlas_Engine-windows-x64.zip

for mode in --selftest ${EXTRA_SELFTEST:-}; do
  echo "== wine $mode"
  rm -f "/wine/drive_c/users/root/AppData/Local/AtlasEngine/selftest_result.txt"
  QT_QPA_PLATFORM=offscreen timeout 1500 wine /tmp/wb/dist/atlas-engine/atlas-engine.exe "$mode" >/dev/null 2>&1 || echo "exit code $?"
  cat "/wine/drive_c/users/root/AppData/Local/AtlasEngine/selftest_result.txt" || echo "no result file"
done
