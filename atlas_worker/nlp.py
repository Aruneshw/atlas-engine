"""Text classification (classic and transformer) and LLM fine-tuning."""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import DataError, split_indices
from .metrics import from_labels
from .protocol import log, stage
from .tabular import warn_user


def _text_column(df, job):
    col = job["dataset"].get("text_column")
    if not col:
        raise DataError("Choose the column that holds the text.")
    if col not in df.columns:
        raise DataError(f"Text column '{col}' is not in this file.")
    return col


def _prepare_classes(df, job):
    col, target = _text_column(df, job), job["dataset"].get("target")
    if not target:
        raise DataError("Choose the column with the category to predict.")
    df = df[df[col].notna() & df[target].notna()].reset_index(drop=True)
    if len(df) < 20:
        raise DataError(f"Only {len(df)} rows have both text and a label: at least 20 are needed.")
    classes = sorted(df[target].astype(str).unique().tolist())
    if len(classes) < 2:
        raise DataError("The label column has only one value.")
    if len(classes) > 0.5 * len(df):
        raise DataError(f"'{target}' has {len(classes)} distinct values in {len(df)} rows, which is too many to be categories.")
    y = df[target].astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy()
    texts = df[col].astype(str).tolist()
    return df, texts, y, classes, col


def _check_test(y, classes, ctx):
    for name, n in zip(classes, np.bincount(y, minlength=len(classes))):
        if n < 5:
            warn_user(ctx, f"The test split has only {n} texts of class '{name}', so its metrics are unreliable.")


def run_text_sk(df, job, algo, impl, params, ctx):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    stage("prepare")
    df, texts, y, classes, col = _prepare_classes(df, job)
    tr, va, te, _ = split_indices(len(y), y, dict(job, task="classification"))
    log(f"split: train {len(tr)}, validation {len(va)}, test {len(te)}")
    _check_test(y[te], classes, ctx)
    stage("build")
    vec = (TfidfVectorizer if impl["vec"] == "tfidf" else CountVectorizer)(max_features=params["max_features"], ngram_range=(1, params["ngram"]))
    pipe = Pipeline([("vec", vec), ("model", LogisticRegression(max_iter=2000, C=params["C"]))])
    stage("train")
    pick = lambda idx: [texts[i] for i in idx]
    pipe.fit(pick(tr), y[tr])
    log(f"vocabulary: {len(pipe['vec'].vocabulary_):,} terms")
    stage("evaluate")
    charts = []
    probs = lambda idx: pipe.predict_proba(pick(idx))
    val = from_labels(pipe.predict(pick(va)), y[va], classes, probs(va))
    test = from_labels(pipe.predict(pick(te)), y[te], classes, probs(te), charts)
    stage("save")
    d = Path(job["run_dir"]) / "model"
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, d / "pipeline.joblib")
    (d / "meta.json").write_text(json.dumps(dict(kind="text_sk", task=job["task"], classes=classes, text_column=col), indent=1))
    return dict(primary=("f1_macro", test["f1_macro"], True), val=val, test=test, charts=charts, sample=pick(te)[:32], params=params)


def run_hf_text(df, job, algo, impl, params, ctx):
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    from .deep import _setup, fit, loss_chart

    device, batch, epochs, lr = _setup(job, params)
    name = params["model"]
    stage("prepare")
    df, texts, y, classes, col = _prepare_classes(df, job)
    tr, va, te, _ = split_indices(len(y), y, dict(job, task="classification"))
    _check_test(y[te], classes, ctx)
    log(f"loading tokenizer and weights for {name} (downloads on first use)")
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    max_len = params["max_length"]
    enc = tok(texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="np")
    ids, mask = enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)
    stage("build")
    cfg = AutoConfig.from_pretrained(name)
    cfg.num_labels = len(classes)           # also rebuilds id2label, which some base models set to a different size
    cfg.id2label = {i: c for i, c in enumerate(classes)}
    cfg.label2id = {c: i for i, c in enumerate(classes)}
    net = AutoModelForSequenceClassification.from_pretrained(name, config=cfg, ignore_mismatched_sizes=True)
    if net.config.pad_token_id is None:
        net.config.pad_token_id = tok.pad_token_id
    log(f"{name}: {sum(p.numel() for p in net.parameters()):,} parameters, batch {batch}, device {device}")
    stage("train")
    loss = lambda m, b: m(input_ids=b[0], attention_mask=b[1], labels=b[2]).loss
    arrays = lambda idx: (ids[idx], mask[idx], y[idx].astype(np.int64))
    history, device, batch = fit(net, loss, arrays(tr), arrays(va), epochs, lr, device, batch, patience=3)
    stage("evaluate")

    @torch.no_grad()
    def logits(idx):
        net.eval().to(device)
        out = [net(input_ids=torch.as_tensor(ids[idx[i:i + batch]]).to(device), attention_mask=torch.as_tensor(mask[idx[i:i + batch]]).to(device)).logits.cpu()
               for i in range(0, len(idx), batch)]
        return torch.softmax(torch.cat(out), 1).numpy()

    charts = []
    pv, pt = logits(va), logits(te)
    val = from_labels(pv.argmax(1), y[va], classes, pv)
    test = from_labels(pt.argmax(1), y[te], classes, pt, charts)
    charts.append(loss_chart(history))
    stage("save")
    d = Path(job["run_dir"]) / "model"
    net.cpu().save_pretrained(d / "hf")
    tok.save_pretrained(d / "hf")
    (d / "meta.json").write_text(json.dumps(dict(kind="hf_text", task=job["task"], classes=classes, text_column=col, max_length=max_len, base=name), indent=1))
    return dict(primary=("f1_macro", test["f1_macro"], True), val=val, test=test, charts=charts, sample=[texts[i] for i in te[:8]], params=params)


def _build_documents(df, job):
    col = _text_column(df, job)
    response = job["dataset"].get("target")
    df = df[df[col].notna()]
    if response:
        df = df[df[response].notna()]
        log(f"instruction tuning: '{col}' is the instruction, '{response}' the response")
        return [f"### Instruction:\n{a}\n\n### Response:\n{b}" for a, b in zip(df[col].astype(str), df[response].astype(str))]
    return df[col].astype(str).tolist()


def run_llm(df, job, algo, impl, params, ctx):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .deep import _setup, fit, loss_chart

    device, batch, epochs, lr = _setup(job, params)
    name, block = params["model"], params["block_size"]
    stage("prepare")
    docs = _build_documents(df, job)
    if len(docs) < 8:
        raise DataError(f"Only {len(docs)} texts found: at least 8 are needed.")
    log(f"loading tokenizer and weights for {name} (downloads on first use)")
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos = tok.eos_token or ""
    enc = tok([d + eos for d in docs], truncation=True, padding="max_length", max_length=block, return_tensors="np")
    ids, mask = enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)
    labels = np.where(mask == 1, ids, -100)
    order = np.random.RandomState(42).permutation(len(docs))
    cut = max(1, int(0.1 * len(docs)))
    va, tr = order[:cut], order[cut:]
    log(f"{len(tr)} training texts, {len(va)} held out")
    stage("build")
    net = AutoModelForCausalLM.from_pretrained(name)
    if impl.get("lora"):
        from peft import LoraConfig, TaskType, get_peft_model

        cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=params["rank"], lora_alpha=params["rank"] * 2, lora_dropout=0.05, use_dora=bool(impl.get("dora")))
        net = get_peft_model(net, cfg)
        trainable, total = net.get_nb_trainable_parameters()
        log(f"LoRA: training {trainable:,} of {total:,} parameters")
    else:
        log(f"{name}: {sum(p.numel() for p in net.parameters()):,} parameters")
    stage("train")
    loss = lambda m, b: m(input_ids=b[0], attention_mask=b[1], labels=b[2]).loss
    arrays = lambda idx: (ids[idx], mask[idx], labels[idx])
    history, device, batch = fit(net, loss, arrays(tr), arrays(va), epochs, lr, device, batch, patience=3)
    stage("evaluate")
    val_loss = float(history[-1][1])
    m = dict(val_loss=val_loss, perplexity=float(math.exp(min(val_loss, 20))))
    if impl.get("lora"):
        net = net.merge_and_unload()
    prompt = docs[0][:40]
    sample = _generate(net.to(device).eval(), tok, prompt, 40, device)
    stage("save")
    d = Path(job["run_dir"]) / "model"
    net.cpu().save_pretrained(d / "hf")
    tok.save_pretrained(d / "hf")
    (d / "meta.json").write_text(json.dumps(dict(kind="llm", task=job["task"], base=name, block_size=block), indent=1))
    charts = [dict(type="text", title="Sample from the fine-tuned model", text=sample), loss_chart(history)]
    return dict(primary=("perplexity", m["perplexity"], False), val=None, test=m, charts=charts, sample=None, params=params)


def _generate(net, tok, prompt, new_tokens, device):
    import torch

    x = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = net.generate(**x, max_new_tokens=new_tokens, do_sample=True, top_p=0.9, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


# ---- prediction ---------------------------------------------------------------------
def predict(p, df=None, prompt=None, steps=None, **_):
    import torch

    m = p.meta
    if p.kind == "text_sk":
        pipe = p.cached(lambda: joblib.load(p.dir / "pipeline.joblib"))
        texts = _texts(df, m)
        proba = pipe.predict_proba(texts)
        return _labelled(texts, proba, m["classes"])
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    if p.kind == "hf_text":
        tok, net = p.cached(lambda: (AutoTokenizer.from_pretrained(p.dir / "hf"), AutoModelForSequenceClassification.from_pretrained(p.dir / "hf").eval()))
        texts = _texts(df, m)
        enc = tok(texts, truncation=True, padding=True, max_length=m["max_length"], return_tensors="pt")
        with torch.no_grad():
            proba = torch.softmax(net(**enc).logits, 1).numpy()
        return _labelled(texts, proba, m["classes"])
    tok, net = p.cached(lambda: (AutoTokenizer.from_pretrained(p.dir / "hf"), AutoModelForCausalLM.from_pretrained(p.dir / "hf").eval()))
    if not prompt:
        raise DataError("Type a prompt first.")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return pd.DataFrame({"prompt": [prompt], "generated": [_generate(net, tok, prompt, int(steps or 80), "cpu")]})


def _texts(df, m):
    if m["text_column"] not in df.columns:
        raise DataError(f"Missing required column: {m['text_column']}")
    return df[m["text_column"]].fillna("").astype(str).tolist()


def _labelled(texts, proba, classes):
    out = pd.DataFrame({"prediction": [classes[i] for i in proba.argmax(1)]})
    for i, c in enumerate(classes):
        out[f"proba_{c}"] = proba[:, i]
    return out


def onnx_parts(p, sample):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if p.kind != "hf_text":
        raise NotImplementedError("large language models are exported as a Hugging Face folder, not ONNX")

    class Wrap(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, input_ids, attention_mask):
            return self.net(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]

    tok = AutoTokenizer.from_pretrained(p.dir / "hf")
    enc = tok(sample[:4], truncation=True, padding="max_length", max_length=p.meta["max_length"], return_tensors="pt")
    net = AutoModelForSequenceClassification.from_pretrained(p.dir / "hf", attn_implementation="eager").eval()
    return Wrap(net), (enc["input_ids"], enc["attention_mask"]), ["input_ids", "attention_mask"]
