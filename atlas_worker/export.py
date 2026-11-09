"""Export a trained run into a deployable bundle, then prove the export matches the original.

An ONNX file is kept only if it reproduces the original model's output on held-out rows.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np

from .data import coerce
from .predictor import Predictor
from .protocol import emit, log
from .runtime_pins import PINS

OPSET = {"": 17, "ai.onnx.ml": 3}
HERE = Path(__file__).resolve().parent


def _column_inputs(features):
    from skl2onnx.common.data_types import FloatTensorType, StringTensorType

    return [(f["name"], FloatTensorType([None, 1]) if f["type"] == "num" else StringTensorType([None, 1])) for f in features]


def _feed(X, features):
    return {f["name"]: (X[f["name"]].to_numpy(np.float32) if f["type"] == "num" else X[f["name"]].to_numpy(dtype=object).astype(str)).reshape(-1, 1)
            for f in features}


def _register_gbm():
    from skl2onnx import update_registered_converter
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes as clf
    from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes as reg

    opts = {"nocl": [True, False], "zipmap": [True, False, "columns"]}
    try:
        import xgboost
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost as cx

        update_registered_converter(xgboost.XGBClassifier, "XGBClassifier", clf, cx, options=opts)
        update_registered_converter(xgboost.XGBRegressor, "XGBRegressor", reg, cx)
    except ImportError:
        pass
    try:
        import lightgbm
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm as cl

        update_registered_converter(lightgbm.LGBMClassifier, "LGBMClassifier", clf, cl, options=opts)
        update_registered_converter(lightgbm.LGBMRegressor, "LGBMRegressor", reg, cl)
    except ImportError:
        pass


def _to_onnx(pipe, features, path, classifier):
    from skl2onnx import convert_sklearn

    _register_gbm()
    inputs = _column_inputs(features)
    onx = None
    if classifier:
        try:
            onx = convert_sklearn(pipe, initial_types=inputs, target_opset=OPSET, options={"zipmap": False})
        except Exception:
            onx = None
    if onx is None:
        onx = convert_sklearn(pipe, initial_types=inputs, target_opset=OPSET)
    path.write_bytes(onx.SerializeToString())


def _session(path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _close(a, b):
    rows = len(a)
    a, b = np.ravel(np.asarray(a, float)), np.ravel(np.asarray(b, float))
    scale = max(float(np.max(np.abs(a))), 1e-9)
    diff = float(np.max(np.abs(a - b)))
    return diff / scale < 5e-3, dict(max_difference=diff, relative=diff / scale, rows=rows)


def _agree(a, b):
    a, b = np.ravel(a).astype(int), np.ravel(b).astype(int)
    rate = float(np.mean(a == b))
    return rate >= 0.99, dict(agreement=rate, rows=len(a))


def export_sklearn(p, sample, out):
    """Pipeline models. Returns a verification dict; raises if the model has no converter."""
    from sklearn.pipeline import Pipeline

    meta, path = p.meta, out / "model.onnx"
    features = meta["features"]
    X = coerce(sample, features)
    pipe = p._pipe()
    if p.kind == "anomaly":
        if meta.get("adapter") != "sk" or meta.get("decision_threshold") is None:
            raise NotImplementedError("this detector has no ONNX converter")
        est = pipe.named_steps["model"].est
        conv = Pipeline([("pre", pipe.named_steps["pre"]), ("model", est)])
        _to_onnx(conv, features, path, False)
        scores = np.ravel(_session(path).run(None, _feed(X, features))[-1])
        ok, info = _close(conv.decision_function(X), scores)
        info["output"] = "scores: rows below decision_threshold are abnormal"
        return ok, info
    _to_onnx(pipe, features, path, p.kind == "supervised" and bool(meta.get("classes")))
    got = _session(path).run(None, _feed(X, features))[0]
    if p.kind == "supervised" and meta.get("classes"):
        return _agree(pipe.predict(X), got)
    if p.kind == "supervised":
        return _close(pipe.predict(X), got)
    if p.kind == "cluster":
        return _agree(pipe.predict(X), got)
    return _close(pipe.transform(X), got)


def export_torch(p, sample, out):
    import torch

    if p.kind == "forecaster":
        raise NotImplementedError("statistical forecasters have no ONNX form")
    meta, path = p.meta, out / "model.onnx"
    if p.kind == "image":
        from .vision import _logits, make_arch

        net = make_arch(meta["arch"], len(meta["classes"]))
        net.load_state_dict(torch.load(p.dir / "net.pt", map_location="cpu"))
        net.eval()
        x = torch.as_tensor(sample[:4]).float()
        reference = _logits(net(x)).detach().numpy()
    else:
        net = p._net().eval()
        if p.kind == "seq_forecast":
            x = torch.tensor(np.array(meta["tail"], np.float32)[None, :, None])
        elif p.kind == "tab_ae":
            pre = joblib.load(p.dir / "pre.joblib")
            x = torch.as_tensor(pre.transform(coerce(sample, meta["features"])).astype(np.float32))
            _to_onnx(pre, meta["features"], out / "preprocess.onnx", False)
        else:
            x = torch.as_tensor(sample)
            if meta["kind"] == "seq_anomaly" and meta["spec"]["net"] == "lstm_pred":
                x = x[:, :-1]
        reference = net(x).detach().numpy()
    torch.onnx.export(net, (x[:1],), str(path), input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}, opset_version=17, dynamo=False)
    session = _session(path)
    ok, info = _close(reference, session.run(None, {"input": x.numpy()})[0])
    single, _ = _close(reference[:1], session.run(None, {"input": x[:1].numpy()})[0])    # batch size must really be free
    return ok and single, info


def export_text_sk(p, sample, out):
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType

    path = out / "model.onnx"
    pipe = p._load_text()
    onx = convert_sklearn(pipe, initial_types=[("text", StringTensorType([None, 1]))], target_opset=OPSET,
                          options={"zipmap": False})
    path.write_bytes(onx.SerializeToString())
    texts = np.array(sample, dtype=object).astype(str).reshape(-1, 1)
    return _agree(pipe.predict(list(sample)), _session(path).run(None, {"text": texts})[0])


ONNX_PARTS = {"hf_text": "nlp", "segment": "segment", "audio": "media", "video": "media", "gen": "gen", "selfsup": "gen", "rl_sb3": "rl", "rl_table": "rl", "bandit": "rl",
              "recsys": "relational"}


def export_parts(p, sample, out):
    """Any network described by a module's onnx_parts(): export it, then compare at two batch sizes."""
    import importlib

    import torch

    net, args, names = getattr(importlib.import_module(f"{__package__}.{ONNX_PARTS[p.kind]}"), "onnx_parts")(p, sample)
    path = out / "model.onnx"
    with torch.no_grad():
        reference = net(*args).detach().numpy()
    first = tuple(a[:1] for a in args)
    axes = {n: {0: "batch"} for n in names}
    axes["output"] = {0: "batch"}
    torch.onnx.export(net, first, str(path), input_names=names, output_names=["output"], dynamic_axes=axes, opset_version=17, dynamo=False)
    session = _session(path)
    ok, info = _close(reference, session.run(None, {n: a.numpy() for n, a in zip(names, args)})[0])
    single, _ = _close(reference[:1], session.run(None, {n: a.numpy() for n, a in zip(names, first)})[0])
    return ok and single, info


def schema_for(p):
    m = p.meta
    keep = ("kind", "task", "target", "features", "classes", "threshold", "decision_threshold", "seq_len", "mean", "std",
            "median", "image_size", "arch", "tail", "horizon", "names", "values", "audio", "frames", "text_column", "max_length", "columns",
            "env", "user_column", "item_column", "base", "block_size", "sampler")
    return dict({k: m[k] for k in keep if k in m}, input=p.needs)


SERVE = '''"""Exported model server. Run: pip install -r requirements.txt && python serve.py

Listens on 127.0.0.1:8000. Set HOST and PORT to change that; the Dockerfile sets HOST=0.0.0.0.
"""
import os
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException

from atlas_worker.data import DataError
from atlas_worker.predictor import Predictor

here = Path(__file__).resolve().parent
model = Predictor(here / "model")
app = FastAPI(title="Atlas Engine model")


@app.get("/health")
def health():
    return {"status": "ok", "kind": model.kind}


@app.get("/schema")
def schema():
    return model.meta | {"input": model.needs}


@app.post("/predict")
def predict(body: dict = Body(...)):
    try:
        return {"predictions": model.predict_json(body)}
    except (DataError, KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
'''

DOCKER = """FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENV HOST=0.0.0.0
EXPOSE 8000
CMD ["python", "serve.py"]
"""

WORKER_FILES = ("__init__.py", "audit.py", "data.py", "fileinfo.py", "protocol.py", "predictor.py", "simple.py", "tabular.py", "metrics.py", "registry.py",
                "forecast.py", "deep.py", "vision.py", "runtime_pins.py", "nlp.py", "detect.py", "segment.py", "media.py", "gen.py", "rl.py", "relational.py")


def _requirements(groups):
    wanted = ["fastapi", "uvicorn", "numpy", "pandas", "scipy", "scikit-learn", "joblib", "psutil"]
    for g in groups:
        if g != "core":
            wanted += PINS[g]
    wanted += [p for p in PINS["core"] if p.split("==")[0] in {"numpy", "pandas", "scipy", "scikit-learn", "joblib", "psutil", "fastapi", "uvicorn"}]
    pinned = {p.split("==")[0]: p for g in PINS.values() for p in g}
    out = sorted({pinned.get(w.split("==")[0], w) for w in wanted})
    if any(p.startswith("xgboost==") for p in out):      # GPU runtimes on CUDA 12 drivers use a different XGBoost from the pin
        from importlib.metadata import version

        out = [f"xgboost=={version('xgboost')}" if p.startswith("xgboost==") else p for p in out]
    return out


def export_api(p, out, groups):
    w = out / "atlas_worker"
    w.mkdir(exist_ok=True)
    for f in WORKER_FILES:
        shutil.copy2(HERE / f, w / f)
    (out / "serve.py").write_text(SERVE)
    (out / "Dockerfile").write_text(DOCKER)
    (out / "requirements.txt").write_text("\n".join(_requirements(groups)) + "\n")


def _readme(p, report):
    media = '{"files_b64": [{"name": "input.jpg", "data": "<base64>"}]}'
    ex = {"rows": '{"rows": [{"column_a": 1.5, "column_b": "x"}]}', "steps": '{"steps": 24}', "samples": '{"steps": 8}', "episodes": '{"obs": [0.1, 0.2, 0.3, 0.4]}',
          "images": media, "audio": media, "video": media, "prompt": '{"prompt": "Hello", "max_new_tokens": 60}', "graph": '{"nodes": [...], "edges": [...]}'}[p.needs]
    lines = ["# Exported model", "", f"Kind: {p.kind}  ", f"Task: {p.meta.get('task')}", "", "`schema.json` lists the exact inputs and outputs.", ""]
    v = report.get("onnx") or {}
    if v.get("status") == "ok":
        lines += ["## model.onnx", "", f"Checked against the original model: `{json.dumps(v.get('check'))}`", "",
                  "Runs through ONNX Runtime from Python, C++, C#, Java and JavaScript, including ONNX Runtime Mobile (Android, iOS)",
                  "and ONNX Runtime Web (browser). OpenVINO reads this file directly on Intel hardware.", ""]
        if p.kind not in ("supervised", "cluster", "reducer", "anomaly", "text_sk", "detect_ultra", "detect_tv"):
            lines += ["This file holds the network only: apply the scaling in `schema.json` to inputs yourself.", ""]
    if "api" in report:
        lines += ["## REST API", "", "```", "pip install -r requirements.txt", "python serve.py", "```", "",
                  "Or with Docker: `docker build -t my-model . && docker run -p 8000:8000 my-model`", "",
                  "```", f"curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{ex}'", "```", ""]
    return "\n".join(lines)


def run_export(run_dir, out_dir, formats):
    run_dir, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    job = json.loads((run_dir / "job.json").read_text())
    p = Predictor(run_dir / "model")
    report = {}
    if not p.can_predict:
        emit("export", status="blocked", msg="This algorithm cannot process new data, so there is nothing to deploy.")
        return dict(blocked=True)
    shutil.copytree(run_dir / "model", out / "model", dirs_exist_ok=True)
    if "onnx" in formats:
        log("converting to ONNX...")
        target = out / "model.onnx"
        try:
            sample = joblib.load(run_dir / "model" / "sample.joblib") if (run_dir / "model" / "sample.joblib").exists() else None
            if p.kind in ("supervised", "cluster", "reducer", "anomaly"):
                fn = export_sklearn
            elif p.kind == "text_sk":
                fn = export_text_sk
            elif p.kind in ("detect_tv", "detect_ultra"):
                from .detect import export_onnx as fn
            elif p.kind in ONNX_PARTS:
                fn = export_parts
            elif p.kind in ("llm", "graph"):
                raise NotImplementedError("this model type is delivered through the REST API bundle")
            else:
                fn = export_torch
            ok, info = fn(p, sample, out)
            if ok:
                report["onnx"] = dict(status="ok", check=info, size_kb=round(target.stat().st_size / 1024))
                log(f"ONNX verified against the original model: {info}")
            else:
                target.unlink(missing_ok=True)
                report["onnx"] = dict(status="mismatch", check=info)
                log(f"ONNX output did not match the original, so it was discarded: {info}", "warn")
        except Exception as e:
            target.unlink(missing_ok=True)
            report["onnx"] = dict(status="unsupported", reason=f"{type(e).__name__}: {str(e)[:300]}")
            log(f"ONNX export is not available for this model: {type(e).__name__}", "warn")
    if "api" in formats:
        from . import registry

        export_api(p, out, registry.get(job["entry"])["groups"])
        report["api"] = dict(status="ok")
        log("REST API bundle written: serve.py, Dockerfile, requirements.txt")
    schema = schema_for(p)
    (out / "schema.json").write_text(json.dumps(schema, indent=1, default=float))
    (out / "README.md").write_text(_readme(p, report))
    (out / "export_report.json").write_text(json.dumps(report, indent=1, default=float))
    emit("export", status="done", out=str(out), report=report)
    return report
