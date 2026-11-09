Train, compare and export machine-learning models from an 830-entry algorithms atlas, in a desktop app.

## Download

- **Linux:** `Atlas_Engine-x86_64.AppImage`. Run `chmod +x` on it, then open it.
- **Windows:** `Atlas_Engine-windows-x64.zip`. Unzip it and run `atlas-engine.exe`.

The app downloads only the libraries an algorithm needs, the first time you train it.

## In this version

- 228 trainable algorithms, with ONNX and web API export
- Live CPU, RAM and GPU limits, and a GPU accelerator button (Linux)
- Leak scan, column picker, clean time-ordered splits, a score to beat and a 95% range for each table model

## Known limits

- Each package passed real training on CPU in this build. GPU training was tested only from source, on Linux with an NVIDIA RTX 3050.
- On Windows the CPU limit becomes a thread count, and the RAM and GPU limits are unavailable.
- Without swap, the RAM limit rises when a job truly needs more, so it is not a hard cap.
- Ultralytics YOLO models are AGPL-3.0, which matters if you ship a product built on them.
- Version 0.x: validate any model on your own real data before relying on it.
