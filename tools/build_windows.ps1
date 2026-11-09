# Builds dist\atlas-engine\atlas-engine.exe and dist\Atlas_Engine-windows-x64.zip. Run on Windows with uv installed.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$uv = (Get-Command uv).Source
uv venv --python 3.12 .build-venv
uv pip install -q --python .build-venv\Scripts\python.exe "PySide6-Essentials==6.8.2.1" "pyinstaller==6.16.0"
$args = (Get-Content tools\pyi_args.txt) + @(
  "--distpath", "dist", "--workpath", "build-win", "--specpath", "build-win", "--paths", (Get-Location).Path,
  "--icon", (Resolve-Path packaging\atlas-engine.ico).Path,
  "--add-data", "$((Resolve-Path atlas_engine\data).Path);atlas_engine\data",
  "--add-data", "$((Resolve-Path atlas_worker).Path);worker_src\atlas_worker",
  "--add-binary", "$uv;bin",
  "--hidden-import=atlas_worker.registry", "--hidden-import=atlas_worker.runtime_pins",
  "tools\launcher.py")
& .build-venv\Scripts\pyinstaller.exe @args
Compress-Archive -Force -Path dist\atlas-engine -DestinationPath dist\Atlas_Engine-windows-x64.zip
$p = Start-Process -FilePath dist\atlas-engine\atlas-engine.exe -ArgumentList "--selftest" -Wait -PassThru
Get-Content "$env:LOCALAPPDATA\AtlasEngine\selftest_result.txt"
if ($p.ExitCode -ne 0) { throw "selftest failed" }
