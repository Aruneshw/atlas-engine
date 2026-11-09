"""Exact package versions for each runtime group. Run tools/smoke_test.py after changing any."""

PINS = {
    "core": ['numpy==2.5.4', 'pandas==3.1.1', 'scipy==1.18.1', 'scikit-learn==1.9.1', 'joblib==1.6.0', 'psutil==7.2.2', 'pyarrow==26.0.1', 'onnx==1.23.2', 'skl2onnx==1.20.0', 'onnxruntime==1.31.1', 'fastapi==0.162.2', 'uvicorn==0.54.0'],
    "boost": ['xgboost==3.4.1', 'lightgbm==4.7.0', 'onnxmltools==1.16.0'],
    "catboost": ['catboost==1.2.10'],
    "stats": ['statsmodels==0.15.0'],
    "pyod": ['pyod==3.6.7'],
    "umap": ['umap-learn==0.5.12'],
    "text": ['transformers==5.22.2', 'accelerate==1.15.0', 'sentencepiece==0.2.3', 'protobuf==7.36.2'],
    "peft": ['peft==0.21.2'],
    "timm": ['timm==1.0.31'],
    "ultra": ['ultralytics==8.4.208'],
    "smp": ['segmentation-models-pytorch==0.6.0'],
    "audio": ['soundfile==0.14.0'],
    "video": ['opencv-python==5.0.0.93'],
    "rl": ['stable-baselines3==2.9.2', 'sb3-contrib==2.9.2', 'gymnasium==1.4.0'],
    "graph": ['torch-geometric==2.8.1'],
    "formats": ['openpyxl==3.1.5', 'xlrd==2.0.2', 'pyxlsb==1.0.10', 'odfpy==1.4.1', 'lxml==6.1.3', 'tables==3.11.1', 'pyreadstat==1.3.7'],
    "torch": ['torch==2.15.0'],
    "vision": ['torchvision==0.29.2', 'pillow==13.0.0'],
}
# XGBoost 3.4 is built for CUDA 13.3 and silently runs on the CPU under a CUDA 12 driver. 3.3 is the newest that reaches the GPU there.
XGBOOST_CUDA12 = "xgboost==3.3.0"
PYTHON = "3.12"
