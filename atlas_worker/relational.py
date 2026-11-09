"""Graph node classification (nodes.csv + edges.csv) and recommender systems (user, item, rating)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import DataError, load_table
from .metrics import from_labels
from .protocol import emit, log, stage


# ---- graph --------------------------------------------------------------------------
def _read_graph(nodes, edges, meta=None, target=None):
    id_col = "id" if "id" in nodes.columns else nodes.columns[0]
    index = {v: i for i, v in enumerate(nodes[id_col].tolist())}
    src, dst = ("source", "target") if {"source", "target"} <= set(edges.columns) else (edges.columns[0], edges.columns[1])
    known = edges[src].isin(index) & edges[dst].isin(index)
    if not known.all():
        log(f"ignoring {int((~known).sum())} edges that mention unknown nodes", "warn")
    a = edges.loc[known, src].map(index).to_numpy()
    b = edges.loc[known, dst].map(index).to_numpy()
    edge_index = np.stack([np.r_[a, b], np.r_[b, a]]).astype(np.int64)      # undirected
    if meta:
        cols = meta["columns"]
        missing = [c for c in cols if c not in nodes.columns]
        if missing:
            raise DataError(f"nodes.csv is missing columns: {missing}")
    else:
        skip = {id_col, target}
        cols = [c for c in nodes.columns if c not in skip and pd.api.types.is_numeric_dtype(nodes[c])]
        if not cols:
            raise DataError("nodes.csv needs at least one numeric feature column besides the id and label.")
    X = nodes[cols].to_numpy(float)
    return nodes, id_col, edge_index, X, cols


class GNN:
    """Built lazily so torch_geometric is only imported when needed."""

    @staticmethod
    def make(name, n_in, hidden, n_out):
        import torch.nn as nn
        import torch.nn.functional as F
        import torch_geometric.nn as G

        conv = {"gcn": lambda a, b: G.GCNConv(a, b), "sage": lambda a, b: G.SAGEConv(a, b), "gat": lambda a, b: G.GATConv(a, b, heads=2, concat=False),
                "cheb": lambda a, b: G.ChebConv(a, b, K=3), "transformer": lambda a, b: G.TransformerConv(a, b, heads=2, concat=False),
                "gin": lambda a, b: G.GINConv(nn.Sequential(nn.Linear(a, b), nn.ReLU(), nn.Linear(b, b)))}[name]

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.a, self.b, self.head = conv(n_in, hidden), conv(hidden, hidden), nn.Linear(hidden, n_out)

            def forward(self, x, edge_index):
                h = F.dropout(F.relu(self.a(x, edge_index)), 0.3, self.training)
                return self.head(F.relu(self.b(h, edge_index)))

        return Net()


def run_graph(job, algo, impl, params, ctx):
    import torch
    import torch.nn.functional as F

    from .deep import _setup, loss_chart

    device, _, epochs, lr = _setup(job, params)
    root = Path(job["dataset"]["path"])
    target = job["dataset"].get("target")
    stage("validate")
    nodes, edges = load_table(root / "nodes.csv"), load_table(root / "edges.csv")
    if not target or target not in nodes.columns:
        raise DataError("Choose the column in nodes.csv that holds the node label.")
    nodes, id_col, ei, X, cols = _read_graph(nodes, edges, target=target)
    labelled = nodes[target].notna().to_numpy()
    classes = sorted(nodes.loc[labelled, target].astype(str).unique().tolist())
    if len(classes) < 2:
        raise DataError("The node label needs at least two classes.")
    if labelled.sum() < 20:
        raise DataError(f"Only {int(labelled.sum())} labelled nodes: at least 20 are needed.")
    y = np.full(len(nodes), -1, np.int64)
    y[labelled] = nodes.loc[labelled, target].astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy()
    log(f"{len(nodes)} nodes, {ei.shape[1] // 2} edges, {len(cols)} features, {len(classes)} classes")
    stage("prepare")
    rs = np.random.RandomState(42)
    idx = rs.permutation(np.flatnonzero(labelled))
    s = job.get("split") or {}
    a, b = int(len(idx) * s.get("val", 0.15)) or 1, int(len(idx) * s.get("test", 0.15)) or 1
    va, te, tr = idx[:a], idx[a:a + b], idx[a + b:]
    med = np.nanmedian(X[tr], 0)
    med = np.where(np.isnan(med), 0, med)
    mask = np.isnan(X)
    X = X.copy()
    X[mask] = np.take(med, np.nonzero(mask)[1])
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xs = torch.tensor((X - mean) / std, dtype=torch.float32, device=device)
    edge = torch.tensor(ei, device=device)
    labels = torch.tensor(y, device=device)
    log(f"labelled nodes: train {len(tr)}, validation {len(va)}, test {len(te)}; all nodes pass messages")
    stage("build")
    net = GNN.make(impl["net"], X.shape[1], params["hidden"], len(classes)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    stage("train")
    tr_t, va_t = torch.tensor(tr, device=device), torch.tensor(va, device=device)
    history, best, state, wait = [], float("inf"), None, 0
    for e in range(1, epochs + 1):
        net.train()
        loss = F.cross_entropy(net(Xs, edge)[tr_t], labels[tr_t])
        opt.zero_grad(), loss.backward(), opt.step()
        net.eval()
        with torch.no_grad():
            vl = F.cross_entropy(net(Xs, edge)[va_t], labels[va_t]).item()
        history.append((loss.item(), vl))
        if e % max(1, epochs // 40) == 0 or e == epochs:
            emit("epoch", epoch=e, total=epochs, train_loss=loss.item(), val_loss=vl)
        if vl < best - 1e-5:
            best, state, wait = vl, {k: v.cpu().clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 40:
                log(f"early stop at epoch {e}")
                break
    net.load_state_dict(state)
    stage("evaluate")
    net.eval()
    with torch.no_grad():
        p = torch.softmax(net(Xs, edge), 1).cpu().numpy()
    charts = []
    val = from_labels(p[va].argmax(1), y[va], classes, p[va])
    test = from_labels(p[te].argmax(1), y[te], classes, p[te], charts)
    charts.append(loss_chart(history[:: max(1, len(history) // 200)]))
    stage("save")
    d = Path(job["run_dir"]) / "model"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(net.cpu().state_dict(), d / "net.pt")
    (d / "meta.json").write_text(json.dumps(dict(kind="graph", task=job["task"], net=impl["net"], hidden=params["hidden"], classes=classes, columns=cols,
                                                 median=med.tolist(), mean=mean.tolist(), std=std.tolist()), indent=1))
    return dict(primary=("f1_macro", test["f1_macro"], True), val=val, test=test, charts=charts, sample=None, params=params)


def predict_graph(p, path=None, nodes=None, edges=None, **_):
    import torch

    m = p.meta
    if path is not None:
        root = Path(path)
        for f in ("nodes.csv", "edges.csv"):
            if not (root / f).exists():
                raise DataError(f"The folder must contain {f}.")
        nodes, edges = load_table(root / "nodes.csv"), load_table(root / "edges.csv")
    nodes, id_col, ei, X, _ = _read_graph(nodes, edges, meta=m)
    X = X.copy()
    mask = np.isnan(X)
    X[mask] = np.take(np.array(m["median"]), np.nonzero(mask)[1])

    def load():
        net = GNN.make(m["net"], len(m["columns"]), m["hidden"], len(m["classes"]))
        net.load_state_dict(torch.load(p.dir / "net.pt", map_location="cpu"))
        return net.eval()

    net = p.cached(load)
    with torch.no_grad():
        proba = torch.softmax(net(torch.tensor((X - np.array(m["mean"])) / np.array(m["std"]), dtype=torch.float32), torch.tensor(ei)), 1).numpy()
    out = pd.DataFrame({"node": nodes[id_col].tolist(), "prediction": [m["classes"][i] for i in proba.argmax(1)]})
    for i, c in enumerate(m["classes"]):
        out[f"proba_{c}"] = proba[:, i]
    return out


# ---- recommender --------------------------------------------------------------------
class ItemKNN:
    def __init__(self, k):
        self.k = k

    def fit(self, u, i, r, n_users, n_items):
        from scipy.sparse import csr_matrix
        from sklearn.metrics.pairwise import cosine_similarity

        self.mean = r.mean()
        self.R = csr_matrix((r - self.mean, (u, i)), shape=(n_users, n_items))
        self.sim = cosine_similarity(self.R.T, dense_output=False).tolil()
        self.sim.setdiag(0)
        self.sim = self.sim.tocsr()
        return self

    def predict(self, u, i):
        out = np.zeros(len(u))
        for n, (a, b) in enumerate(zip(u, i)):
            s = self.sim[b].toarray().ravel()
            rated = self.R[a].toarray().ravel()
            mask = rated != 0
            idx = np.flatnonzero(mask)
            idx = idx[np.argsort(-np.abs(s[idx]))[: self.k]]
            denom = np.abs(s[idx]).sum()
            out[n] = s[idx] @ rated[idx] / denom if denom > 0 else 0.0
        return out + self.mean


class ALS:
    def __init__(self, rank, reg, iters):
        self.rank, self.reg, self.iters = rank, reg, iters

    def fit(self, u, i, r, n_users, n_items, report=None):
        self.mean = r.mean()
        y = r - self.mean
        rs = np.random.RandomState(42)
        self.U, self.V = rs.normal(0, 0.1, (n_users, self.rank)), rs.normal(0, 0.1, (n_items, self.rank))
        by_u = [np.flatnonzero(u == a) for a in range(n_users)]
        by_i = [np.flatnonzero(i == b) for b in range(n_items)]
        eye = self.reg * np.eye(self.rank)
        for it in range(1, self.iters + 1):
            for a, idx in enumerate(by_u):
                if len(idx):
                    v = self.V[i[idx]]
                    self.U[a] = np.linalg.solve(v.T @ v + eye, v.T @ y[idx])
            for b, idx in enumerate(by_i):
                if len(idx):
                    w = self.U[u[idx]]
                    self.V[b] = np.linalg.solve(w.T @ w + eye, w.T @ y[idx])
            if report:
                report(it, self.iters, float(np.mean((y - (self.U[u] * self.V[i]).sum(1)) ** 2)))
        return self

    def predict(self, u, i):
        return (self.U[u] * self.V[i]).sum(1) + self.mean


def _rec_net(name, n_users, n_items, dim):
    import torch
    import torch.nn as nn

    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.u, self.i = nn.Embedding(n_users, dim), nn.Embedding(n_items, dim)
            self.bu, self.bi = nn.Embedding(n_users, 1), nn.Embedding(n_items, 1)
            for e in (self.u, self.i):
                nn.init.normal_(e.weight, std=0.05)
            for e in (self.bu, self.bi):
                nn.init.zeros_(e.weight)
            if name == "ncf":
                self.mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, 1))
            elif name == "fm":
                self.w0 = nn.Parameter(torch.zeros(1))
            elif name == "two_tower":
                self.tu, self.ti = nn.Sequential(nn.Linear(dim, dim), nn.ReLU()), nn.Sequential(nn.Linear(dim, dim), nn.ReLU())

        def forward(self, a, b):
            bias = self.bu(a).squeeze(1) + self.bi(b).squeeze(1)
            u, v = self.u(a), self.i(b)
            if name == "ncf":
                return self.mlp(torch.cat([u, v], 1)).squeeze(1) + bias
            if name == "two_tower":
                return (self.tu(u) * self.ti(v)).sum(1) + bias
            if name == "fm":
                return self.w0 + bias + (u * v).sum(1)
            return (u * v).sum(1) + bias

    return Base()


def _hit_rate(score, u, i, r, seen, n_items, mean, rng, k=10):
    good = np.flatnonzero(r >= mean)
    if not len(good):
        return float("nan")
    good = rng.choice(good, min(500, len(good)), replace=False)
    hits = 0
    for t in good:
        pool = rng.randint(0, n_items, 150)
        negs = np.array([x for x in pool if x not in seen[u[t]]][:99], dtype=np.int64)
        items = np.r_[i[t], negs]
        rank = (score(np.full(len(items), u[t]), items) > score(np.array([u[t]]), np.array([i[t]]))[0]).sum()
        hits += rank < k
    return float(hits / len(good))


def run_recommender(df, job, algo, impl, params, ctx):
    from .deep import _setup, fit, loss_chart
    import torch

    ds = job["dataset"]
    ucol, icol, rcol = ds.get("user_column"), ds.get("item_column"), ds.get("target")
    stage("validate")
    for label, c in (("User", ucol), ("Item", icol), ("Rating", rcol)):
        if not c or c not in df.columns:
            raise DataError(f"{label} column is not set or not in this file.")
    df = df[[ucol, icol, rcol]].dropna()
    ratings = pd.to_numeric(df[rcol], errors="coerce")
    if ratings.isna().mean() > 0.05:
        raise DataError(f"Rating column '{rcol}' is not numeric.")
    df = df[ratings.notna()]
    if len(df) < 50:
        raise DataError(f"Only {len(df)} interactions: at least 50 are needed.")
    stage("prepare")
    users, items = sorted(df[ucol].astype(str).unique()), sorted(df[icol].astype(str).unique())
    um, im = {v: k for k, v in enumerate(users)}, {v: k for k, v in enumerate(items)}
    u, i, r = df[ucol].astype(str).map(um).to_numpy(np.int64), df[icol].astype(str).map(im).to_numpy(np.int64), df[rcol].astype(float).to_numpy()
    rs = np.random.RandomState(42)
    order = rs.permutation(len(r))
    s = job.get("split") or {}
    a, b = int(len(r) * s.get("val", 0.15)) or 1, int(len(r) * s.get("test", 0.15)) or 1
    va, te, tr = order[:a], order[a:a + b], order[a + b:]
    log(f"{len(users)} users, {len(items)} items; train {len(tr)}, validation {len(va)}, test {len(te)} interactions")
    mean = float(r[tr].mean())
    stage("build")
    d = Path(job["run_dir"]) / "model"
    d.mkdir(parents=True, exist_ok=True)
    history, sample = [], None
    kind_extra = {}
    stage("train")
    if impl["net"] in ("item_knn", "als"):
        if impl["net"] == "item_knn":
            model = ItemKNN(params["neighbours"]).fit(u[tr], i[tr], r[tr], len(users), len(items))
        else:
            def rep(it, total, loss):
                history.append((loss, loss))
                emit("epoch", epoch=it, total=total, train_loss=loss, val_loss=loss)

            model = ALS(params["rank"], params["reg"], params["iterations"]).fit(u[tr], i[tr], r[tr], len(users), len(items), rep)
        score = lambda x, y: model.predict(x, y)
        joblib.dump(model, d / "rec.joblib")
    else:
        device, batch, epochs, lr = _setup(job, params)
        spec = dict(net=impl["net"], users=len(users), items=len(items), dim=params["dim"])
        net = _rec_net(impl["net"], len(users), len(items), params["dim"])
        loss = lambda m, b: torch.nn.functional.mse_loss(m(b[0], b[1]) + mean, b[2])
        arrays = lambda idx: (u[idx], i[idx], r[idx].astype(np.float32))
        history, device, batch = fit(net, loss, arrays(tr), arrays(va), epochs, lr, device, max(batch, 256), patience=5)
        net.eval().to(device)

        @torch.no_grad()
        def score(x, y):
            return np.concatenate([(net(torch.as_tensor(x[k:k + 4096]).to(device), torch.as_tensor(y[k:k + 4096]).to(device)).cpu().numpy() + mean) for k in range(0, len(x), 4096)])

        torch.save({k: v.cpu() for k, v in net.state_dict().items()}, d / "net.pt")
        kind_extra = dict(spec=spec)
        sample = (u[te[:4]], i[te[:4]])
    stage("evaluate")
    clip = lambda x: np.clip(x, r.min(), r.max())
    metrics = lambda idx: dict(rmse=float(np.sqrt(np.mean((clip(score(u[idx], i[idx])) - r[idx]) ** 2))), mae=float(np.mean(np.abs(clip(score(u[idx], i[idx])) - r[idx]))))
    val, test = metrics(va), metrics(te)
    seen = [set() for _ in users]
    for a_, b_ in zip(u[tr], i[tr]):
        seen[a_].add(b_)
    test["hit_rate_at_10"] = _hit_rate(score, u[te], i[te], r[te], seen, len(items), mean, rs)
    test["global_mean_rmse"] = float(np.sqrt(np.mean((mean - r[te]) ** 2)))
    charts = [loss_chart(history)] if history else []
    stage("save")
    (d / "meta.json").write_text(json.dumps(dict(kind="recsys", task=job["task"], users=users, items=items, user_column=ucol, item_column=icol, mean=mean, low=float(r.min()),
                                                 high=float(r.max()), net=impl["net"], **kind_extra), default=float))
    if sample is not None:
        joblib.dump(sample, d / "sample.joblib")
    return dict(primary=("rmse", test["rmse"], False), val=val, test=test, charts=charts, sample=sample, params=params)


def predict_recsys(p, df=None, **_):
    import torch

    m = p.meta
    for c in (m["user_column"], m["item_column"]):
        if c not in df.columns:
            raise DataError(f"Missing required column: {c}")
    um, im = p.cached(lambda: ({v: k for k, v in enumerate(m["users"])}, {v: k for k, v in enumerate(m["items"])}))
    u = df[m["user_column"]].astype(str).map(um)
    i = df[m["item_column"]].astype(str).map(im)
    known = u.notna() & i.notna()
    pred = np.full(len(df), m["mean"], float)
    if known.any():
        a, b = u[known].to_numpy(np.int64), i[known].to_numpy(np.int64)
        if m["net"] in ("item_knn", "als"):
            model = joblib.load(p.dir / "rec.joblib")
            pred[known.to_numpy()] = model.predict(a, b)
        else:
            net = _rec_net(m["spec"]["net"], m["spec"]["users"], m["spec"]["items"], m["spec"]["dim"])
            net.load_state_dict(torch.load(p.dir / "net.pt", map_location="cpu"))
            with torch.no_grad():
                pred[known.to_numpy()] = net.eval()(torch.as_tensor(a), torch.as_tensor(b)).numpy() + m["mean"]
    return pd.DataFrame({"prediction": np.clip(pred, m["low"], m["high"]), "known_user_and_item": known.to_numpy().astype(int)})


def onnx_parts(p, sample):
    import torch

    if p.kind != "recsys" or "spec" not in p.meta:
        raise NotImplementedError("this model has no network to export")
    s = p.meta["spec"]
    net = _rec_net(s["net"], s["users"], s["items"], s["dim"])
    net.load_state_dict(torch.load(p.dir / "net.pt", map_location="cpu"))
    u, i = sample
    return net.eval(), (torch.as_tensor(u), torch.as_tensor(i)), ["user_index", "item_index"]
