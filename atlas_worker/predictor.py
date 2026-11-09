"""Load any trained run and predict. Also the code behind the exported API server."""
from __future__ import annotations

import base64
import importlib
import io
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import DataError, coerce
from .registry import NEEDS


DELEGATED = {"text_sk": ("nlp", "predict"), "hf_text": ("nlp", "predict"), "llm": ("nlp", "predict"), "detect_tv": ("detect", "predict"), "detect_ultra": ("detect", "predict"),
             "segment": ("segment", "predict"), "audio": ("media", "predict"), "video": ("media", "predict"), "gen": ("gen", "predict"), "selfsup": ("gen", "predict_selfsup"),
             "rl_sb3": ("rl", "predict"), "rl_table": ("rl", "predict"), "bandit": ("rl", "predict"), "graph": ("relational", "predict_graph"), "recsys": ("relational", "predict_recsys")}


class Predictor:
    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.kind = self.meta["kind"]
        self._model = None

    @property
    def needs(self):
        """What input this model takes."""
        return NEEDS.get(self.kind, "rows")

    def cached(self, load):
        """One expensive load per Predictor, shared by repeated requests."""
        if self._model is None:
            self._model = load()
        return self._model

    @property
    def can_predict(self):
        return self.meta.get("inductive", True)

    def _net(self):
        import torch

        from .deep import make_net

        if self._model is None:
            net = make_net(self.meta["spec"])
            net.load_state_dict(torch.load(self.dir / "net.pt", map_location="cpu"))
            self._model = net.eval()
        return self._model

    def predict(self, df=None, steps=None, images=None, prompt=None, path=None, out_dir=None, nodes=None, edges=None) -> pd.DataFrame:
        if not self.can_predict:
            raise DataError("This algorithm only labels the data it was fitted on; it cannot score new rows.")
        if self.kind in DELEGATED:
            module, fn = DELEGATED[self.kind]
            return getattr(importlib.import_module(f"{__package__}.{module}"), fn)(self, df=df, steps=steps, images=images, prompt=prompt, path=path, out_dir=out_dir, nodes=nodes, edges=edges)
        return getattr(self, "_p_" + self.kind)(df=df, steps=steps, images=images)

    def _load_text(self):
        return joblib.load(self.dir / "pipeline.joblib")

    def _pipe(self):
        if self._model is None:
            self._model = joblib.load(self.dir / "pipeline.joblib")
        return self._model

    def _p_supervised(self, df, **_):
        X = coerce(df, self.meta["features"])
        pipe, classes = self._pipe(), self.meta.get("classes")
        pred = np.ravel(pipe.predict(X))
        if not classes:
            return pd.DataFrame({"prediction": pred})
        out = pd.DataFrame({"prediction": [classes[int(p)] for p in pred]})
        if hasattr(pipe, "predict_proba"):
            try:
                p = pipe.predict_proba(X)
                for i, c in enumerate(classes):
                    out[f"proba_{c}"] = p[:, i]
            except Exception:
                pass
        return out

    def _p_cluster(self, df, **_):
        return pd.DataFrame({"cluster": self._pipe().predict(coerce(df, self.meta["features"]))})

    def _p_reducer(self, df, **_):
        Z = self._pipe().transform(coerce(df, self.meta["features"]))
        return pd.DataFrame(Z, columns=[f"dim_{i + 1}" for i in range(Z.shape[1])])

    def _p_anomaly(self, df, **_):
        pipe = self._pipe()
        s = pipe.named_steps["model"].score(pipe.named_steps["pre"].transform(coerce(df, self.meta["features"])))
        return pd.DataFrame({"anomaly_score": s, "anomaly": (s > self.meta["threshold"]).astype(int)})

    def _p_forecaster(self, steps, **_):
        if self._model is None:
            self._model = joblib.load(self.dir / "forecaster.joblib")
        return _steps(self._model.forecast(_n(steps, self.meta)))

    def _p_seq_forecast(self, steps, **_):
        from .deep import forecast_recursive

        m = self.meta
        return _steps(forecast_recursive(self._net(), np.array(m["tail"]), _n(steps, m), "cpu") * m["std"] + m["mean"])

    def _windows(self, df):
        from .deep import windows

        m = self.meta
        X = coerce(df, m["features"]).to_numpy(float, copy=True)
        mask = np.isnan(X)
        X[mask] = np.take(np.array(m["median"]), np.nonzero(mask)[1])
        return windows((X - np.array(m["mean"])) / np.array(m["std"]), m["seq_len"])

    def _p_seq_class(self, df, **_):
        from .deep import _softmax, batched

        import torch

        logits = batched(self._net(), torch.as_tensor(self._windows(df)).numpy(), "cpu")
        classes, p = self.meta["classes"], _softmax(logits)
        out = pd.DataFrame({"prediction": [classes[i] for i in logits.argmax(1)]})
        for i, c in enumerate(classes):
            out[f"proba_{c}"] = p[:, i]
        return out

    def _p_seq_anomaly(self, df, **_):
        from .deep import batched

        net, W = self._net(), self._windows(df)
        if self.meta["spec"]["net"] == "lstm_ae":
            err = ((batched(net, W, "cpu") - W) ** 2).mean((1, 2))
        else:
            err = ((batched(net, W[:, :-1], "cpu") - W[:, -1]) ** 2).mean(1)
        return pd.DataFrame({"anomaly_score": err, "anomaly": (err > self.meta["threshold"]).astype(int)})

    def _p_tab_ae(self, df, **_):
        from .deep import loader, tab_error

        import torch

        pre = joblib.load(self.dir / "pre.joblib")
        X = pre.transform(coerce(df, self.meta["features"])).astype(np.float32)
        net = self._net()
        if self.meta.get("reducer"):
            with torch.no_grad():
                Z = torch.cat([net.enc(b[0]) for b in loader([X], 512)]).numpy()
            return pd.DataFrame(Z, columns=[f"dim_{i + 1}" for i in range(Z.shape[1])])
        err = tab_error(net, self.meta["spec"]["net"], X, "cpu")
        return pd.DataFrame({"anomaly_score": err, "anomaly": (err > self.meta["threshold"]).astype(int)})

    def _p_image(self, images, **_):
        import torch
        from PIL import Image

        from .vision import _logits, eval_tf, make_arch

        m = self.meta
        if self._model is None:
            net = make_arch(m["arch"], len(m["classes"]))
            net.load_state_dict(torch.load(self.dir / "net.pt", map_location="cpu"))
            self._model = net.eval()
        tf = eval_tf(m["image_size"])
        opened = [Image.open(i if not isinstance(i, bytes) else io.BytesIO(i)).convert("RGB") for i in images]
        with torch.no_grad():
            p = torch.softmax(_logits(self._model(torch.stack([tf(i) for i in opened]))), 1).numpy()
        out = pd.DataFrame({"prediction": [m["classes"][i] for i in p.argmax(1)]})
        for i, c in enumerate(m["classes"]):
            out[f"proba_{c}"] = p[:, i]
        return out

    def predict_json(self, body: dict) -> list[dict]:
        """The request format used by the exported API."""
        if "obs" in body:
            from . import rl

            return [{"action": rl.act(self, body["obs"])}]
        if self.needs in ("steps", "samples", "episodes"):
            res = self.predict(steps=body.get("steps"))
        elif self.needs == "prompt":
            res = self.predict(prompt=body.get("prompt"), steps=body.get("max_new_tokens"))
        elif self.needs == "graph":
            res = self.predict(nodes=pd.DataFrame(body["nodes"]), edges=pd.DataFrame(body["edges"]))
        elif self.needs in ("images", "audio", "video"):
            with tempfile.TemporaryDirectory() as td:
                paths = []
                for i, item in enumerate(body["files_b64"]):
                    target = Path(td) / f"{i}{Path(str(item.get('name', 'file.bin'))).suffix}"      # only the suffix is trusted
                    target.write_bytes(base64.b64decode(item["data"]))
                    paths.append(str(target))
                res = self.predict(images=paths)
        else:
            res = self.predict(df=pd.DataFrame(body["rows"]))
        return json.loads(res.to_json(orient="records"))


def _n(steps, meta):
    return int(steps or meta.get("horizon", 12))


def _steps(values):
    return pd.DataFrame({"step": range(1, len(values) + 1), "forecast": values})
