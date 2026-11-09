"""Image generation (VAE, GAN, diffusion, flow matching) and self-supervised representation learning."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from . import fileinfo
from .data import DataError
from .deep import _setup, loss_chart
from .protocol import emit, log, stage

LATENT = 64


def image_files(job, minimum=16):
    root = Path(job["dataset"]["path"])
    files = sorted(f for f in root.rglob("*") if f.suffix.lower() in fileinfo.IMAGE_EXT and f.is_file())
    if len(files) < minimum:
        raise DataError(f"Only {len(files)} images found: at least {minimum} are needed.")
    return files


class Pictures(Dataset):
    def __init__(self, files, tf):
        self.files, self.tf = files, tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("RGB"))


def _tf(size):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])


def _steps(size):
    return int(math.log2(size)) - 2        # doublings from 4x4


class Decoder(nn.Module):
    def __init__(self, size, ch=48, latent=LATENT):
        super().__init__()
        n = _steps(size)
        c = ch * 2 ** (n - 1)
        layers = [nn.ConvTranspose2d(latent, c, 4), nn.BatchNorm2d(c), nn.ReLU()]
        for _ in range(n - 1):
            layers += [nn.ConvTranspose2d(c, c // 2, 4, 2, 1), nn.BatchNorm2d(c // 2), nn.ReLU()]
            c //= 2
        layers += [nn.ConvTranspose2d(c, 3, 4, 2, 1), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z[:, :, None, None])


class Encoder(nn.Module):
    def __init__(self, size, out, ch=48, norm=True):
        super().__init__()
        n, c, layers = _steps(size), ch, [nn.Conv2d(3, ch, 4, 2, 1), nn.LeakyReLU(0.2)]
        for _ in range(n - 1):
            layers += [nn.Conv2d(c, c * 2, 4, 2, 1)] + ([nn.BatchNorm2d(c * 2)] if norm else []) + [nn.LeakyReLU(0.2)]
            c *= 2
        self.net, self.out = nn.Sequential(*layers), nn.Conv2d(c, out, 4)

    def forward(self, x):
        return self.out(self.net(x)).flatten(1)


class MLPGen(nn.Module):
    def __init__(self, size, latent=LATENT):
        super().__init__()
        self.size = size
        self.net = nn.Sequential(nn.Linear(latent, 512), nn.ReLU(), nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 3 * size * size), nn.Tanh())

    def forward(self, z):
        return self.net(z).reshape(-1, 3, self.size, self.size)


class MLPDisc(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * size * size, 512), nn.LeakyReLU(0.2), nn.Linear(512, 1))

    def forward(self, x):
        return self.net(x)


class TimeUNet(nn.Module):
    """Small U-Net with a sinusoidal time embedding, used by every diffusion-style method."""

    def __init__(self, ch=48):
        super().__init__()
        self.ch = ch
        self.t = nn.Sequential(nn.Linear(ch, ch * 2), nn.SiLU(), nn.Linear(ch * 2, ch * 2))
        self.d1, self.d2, self.mid = self._b(3, ch), self._b(ch, ch * 2), self._b(ch * 2, ch * 2)
        self.t1, self.t2, self.t3 = nn.Linear(ch * 2, ch), nn.Linear(ch * 2, ch * 2), nn.Linear(ch * 2, ch * 2)
        self.u2, self.u1 = self._b(ch * 4, ch), self._b(ch * 2, ch)
        self.out = nn.Conv2d(ch, 3, 1)

    @staticmethod
    def _b(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU(), nn.Conv2d(b, b, 3, padding=1), nn.SiLU())

    def embed(self, t):
        half = self.ch // 2
        f = torch.exp(-math.log(1000) * torch.arange(half, device=t.device) / half)
        a = t[:, None].float() * f[None] * 1000
        return self.t(torch.cat([a.sin(), a.cos()], 1))

    def forward(self, x, t):
        e = self.embed(t)
        add = lambda h, lin: h + lin(e)[:, :, None, None]
        h1 = add(self.d1(x), self.t1)
        h2 = add(self.d2(F.avg_pool2d(h1, 2)), self.t2)
        m = add(self.mid(F.avg_pool2d(h2, 2)), self.t3)
        up = self.u2(torch.cat([F.interpolate(m, scale_factor=2), h2], 1))
        return self.out(self.u1(torch.cat([F.interpolate(up, scale_factor=2), h1], 1)))


def sliced_wasserstein(a, b, projections=128):
    """Distance between two image sets at low resolution. Lower means closer."""
    x = F.adaptive_avg_pool2d(a, 8).flatten(1)
    y = F.adaptive_avg_pool2d(b, 8).flatten(1)
    n = min(len(x), len(y))
    dirs = F.normalize(torch.randn(projections, x.shape[1], generator=torch.Generator().manual_seed(0)), dim=1)
    return float((torch.sort(x[:n] @ dirs.T, 0)[0] - torch.sort(y[:n] @ dirs.T, 0)[0]).abs().mean())


def _grid_png(images, path, cols=8):
    x = ((images.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
    h, w = x.shape[1:3]
    rows = len(x) // cols
    canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, im in enumerate(x[:rows * cols]):
        canvas[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = im
    Image.fromarray(canvas).resize((cols * 64, rows * 64), Image.NEAREST).save(path)


BETA = torch.linspace(1e-4, 0.02, 200)


@torch.no_grad()
def sample(kind, net, n, size, device, steps=None):
    """Generate n images from a trained model."""
    net.eval()
    if kind in ("vae", "gan"):
        return net(torch.randn(n, LATENT, device=device))
    x = torch.randn(n, 3, size, size, device=device)
    if kind == "flow":
        steps = steps or 50
        for i in range(steps):
            t = torch.full((n,), i / steps, device=device)
            x = x + net(x, t) / steps
        return x
    beta = BETA.to(device)
    alpha = 1 - beta
    bar = torch.cumprod(alpha, 0)
    if kind == "ddim":
        seq = list(range(0, len(beta), len(beta) // (steps or 40)))[::-1]
        for j, i in enumerate(seq):
            t = torch.full((n,), i / len(beta), device=device)
            eps = net(x, t)
            x0 = (x - (1 - bar[i]).sqrt() * eps) / bar[i].sqrt()
            prev = bar[seq[j + 1]] if j + 1 < len(seq) else torch.tensor(1.0, device=device)
            x = prev.sqrt() * x0 + (1 - prev).sqrt() * eps
        return x
    for i in reversed(range(len(beta))):
        t = torch.full((n,), i / len(beta), device=device)
        eps = net(x, t)
        x = (x - beta[i] / (1 - bar[i]).sqrt() * eps) / alpha[i].sqrt()
        if i:
            x += beta[i].sqrt() * torch.randn_like(x)
    return x


def run_generative(job, algo, impl, params, ctx):
    device, batch, epochs, lr = _setup(job, params)
    method, size = impl["method"], int(params["image_size"])
    stage("validate")
    files = image_files(job)
    log(f"{len(files)} images at {size}px; no labels are used")
    stage("prepare")
    order = np.random.RandomState(42).permutation(len(files))
    held = max(8, int(0.1 * len(files)))
    te, tr = [files[i] for i in order[:held]], [files[i] for i in order[held:]]
    tf = _tf(size)
    train_dl = DataLoader(Pictures(tr, tf), batch_size=batch, shuffle=True, drop_last=len(tr) > batch)
    stage("build")
    history = []

    def report(e, a, b):
        history.append((a, b))
        emit("epoch", epoch=e, total=epochs, train_loss=a, val_loss=b)

    stage("train")
    if method in ("vae", "beta_vae"):
        kind, gen = "vae", _train_vae(tr, train_dl, size, epochs, lr, device, params.get("beta", 1.0), report)
    elif method in ("gan", "dcgan", "wgan_gp"):
        kind, gen = "gan", _train_gan(method, train_dl, size, epochs, lr, device, report)
    else:
        kind = {"ddpm": "ddpm", "ddim": "ddim", "flow": "flow"}[method]
        gen = _train_diffusion(kind, train_dl, epochs, lr, device, report)
    stage("evaluate")
    real = torch.stack([tf(Image.open(f).convert("RGB")) for f in te])
    fake = sample(kind, gen, len(real), size, device).cpu()
    swd = sliced_wasserstein(real, fake)
    baseline = sliced_wasserstein(real, torch.randn_like(real).clamp(-1, 1))
    m = dict(sliced_wasserstein=swd, noise_baseline=baseline, images_trained=len(tr))
    stage("save")
    run = Path(job["run_dir"])
    d = run / "model"
    d.mkdir(parents=True, exist_ok=True)
    grid = sample(kind, gen, 32, size, device).cpu()
    _grid_png(grid, run / "samples.png")
    torch.save(gen.cpu().state_dict(), d / "net.pt")
    (d / "meta.json").write_text(json.dumps(dict(kind="gen", task=job["task"], sampler=kind, arch=method, image_size=size), indent=1))
    charts = [dict(type="image", title="Generated samples", path="samples.png"), loss_chart(history)]
    return dict(primary=("sliced_wasserstein", swd, False), val=None, test=m, charts=charts, sample=None, params=params)


def _train_vae(tr, loader_, size, epochs, lr, device, beta, report):
    enc, dec = Encoder(size, LATENT * 2).to(device), Decoder(size).to(device)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=lr)
    for e in range(1, epochs + 1):
        enc.train(), dec.train()
        total, n = 0.0, 0
        for x in loader_:
            x = x.to(device)
            mu, lv = enc(x).chunk(2, 1)
            lv = lv.clamp(-10, 10)
            z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
            rec = F.mse_loss(dec(z), x, reduction="sum") / len(x)
            kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1).mean()
            loss = rec + beta * kl
            opt.zero_grad(), loss.backward(), opt.step()
            total += loss.item()
            n += 1
        report(e, total / max(n, 1), total / max(n, 1))
    return dec


def _train_gan(method, loader_, size, epochs, lr, device, report):
    if method == "gan":
        g, d = MLPGen(size).to(device), MLPDisc(size).to(device)
    else:
        g, d = Decoder(size).to(device), Encoder(size, 1, norm=method == "dcgan").to(device)
    betas = (0.5, 0.9) if method == "wgan_gp" else (0.5, 0.999)
    og, od = torch.optim.Adam(g.parameters(), lr=lr, betas=betas), torch.optim.Adam(d.parameters(), lr=lr, betas=betas)
    bce = F.binary_cross_entropy_with_logits
    for e in range(1, epochs + 1):
        g.train(), d.train()
        ld, lg, n = 0.0, 0.0, 0
        for x in loader_:
            x = x.to(device)
            b = len(x)
            if b < 2:
                continue
            fake = g(torch.randn(b, LATENT, device=device))
            if method == "wgan_gp":
                eps = torch.rand(b, 1, 1, 1, device=device)
                mix = (eps * x + (1 - eps) * fake.detach()).requires_grad_(True)
                grad = torch.autograd.grad(d(mix).sum(), mix, create_graph=True)[0].flatten(1)
                loss_d = d(fake.detach()).mean() - d(x).mean() + 10 * ((grad.norm(2, 1) - 1) ** 2).mean()
            else:
                loss_d = bce(d(x), torch.ones(b, 1, device=device)) + bce(d(fake.detach()), torch.zeros(b, 1, device=device))
            od.zero_grad(), loss_d.backward(), od.step()
            out = d(fake)
            loss_g = -out.mean() if method == "wgan_gp" else bce(out, torch.ones(b, 1, device=device))
            og.zero_grad(), loss_g.backward(), og.step()
            ld += loss_d.item()
            lg += loss_g.item()
            n += 1
        report(e, lg / max(n, 1), ld / max(n, 1))
    return g


def _train_diffusion(kind, loader_, epochs, lr, device, report):
    net = TimeUNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bar = torch.cumprod(1 - BETA, 0).to(device)
    for e in range(1, epochs + 1):
        net.train()
        total, n = 0.0, 0
        for x in loader_:
            x = x.to(device)
            noise = torch.randn_like(x)
            if kind == "flow":
                t = torch.rand(len(x), device=device)
                w = t[:, None, None, None]
                loss = F.mse_loss(net((1 - w) * noise + w * x, t), x - noise)
            else:
                i = torch.randint(0, len(BETA), (len(x),), device=device)
                a = bar[i][:, None, None, None]
                loss = F.mse_loss(net(a.sqrt() * x + (1 - a).sqrt() * noise, i.float() / len(BETA)), noise)
            opt.zero_grad(), loss.backward(), opt.step()
            total += loss.item()
            n += 1
        report(e, total / max(n, 1), total / max(n, 1))
    return net


def _load(p):
    m = p.meta
    size = m["image_size"]
    if m["sampler"] == "vae":
        net = Decoder(size)
    elif m["sampler"] == "gan":
        net = MLPGen(size) if m["arch"] == "gan" else Decoder(size)
    else:
        net = TimeUNet()
    net.load_state_dict(torch.load(p.dir / "net.pt", map_location="cpu"))
    return net.eval()


def predict(p, steps=None, out_dir=None, **_):
    import base64
    import io

    net = p.cached(lambda: _load(p))
    n = int(steps or 8)
    imgs = sample(p.meta["sampler"], net, n, p.meta["image_size"], "cpu")
    x = ((imgs.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy()
    rows = []
    for i, arr in enumerate(x):
        im = Image.fromarray(arr)
        if out_dir:
            f = Path(out_dir) / f"sample_{i:04d}.png"
            f.parent.mkdir(parents=True, exist_ok=True)
            im.save(f)
            rows.append(dict(sample=i, file=str(f)))
        else:
            buf = io.BytesIO()
            im.save(buf, "PNG")
            rows.append(dict(sample=i, image_b64=base64.b64encode(buf.getvalue()).decode()))
    return pd.DataFrame(rows)


def onnx_parts(p, sample):
    if p.kind == "selfsup":
        net = _backbone_loaded(p)
        return net, (torch.as_tensor(sample).float(),), ["input"]
    if p.meta["sampler"] not in ("vae", "gan"):
        raise NotImplementedError("a diffusion sampler is a loop over a network, not one exportable graph")
    net = _load(p)
    return net, (torch.randn(4, LATENT, generator=torch.Generator().manual_seed(0)),), ["latent"]


# ---- self-supervised ----------------------------------------------------------------
def _backbone():
    import torchvision

    net = torchvision.models.resnet18(weights=None)
    net.fc = nn.Identity()
    return net


def _backbone_loaded(p):
    def load():
        net = _backbone()
        net.load_state_dict(torch.load(p.dir / "net.pt", map_location="cpu"))
        return net.eval()

    return p.cached(load)


class TwoViews(Dataset):
    def __init__(self, files, size):
        self.files = files
        self.aug = T.Compose([T.RandomResizedCrop(size, scale=(0.3, 1.0)), T.RandomHorizontalFlip(), T.ColorJitter(0.4, 0.4, 0.4, 0.1), T.RandomGrayscale(0.2),
                              T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        im = Image.open(self.files[i]).convert("RGB")
        return self.aug(im), self.aug(im)


def _nt_xent(a, b, tau=0.2):
    z = F.normalize(torch.cat([a, b]), dim=1)
    s = z @ z.T / tau
    s.fill_diagonal_(-1e9)
    n = len(a)
    return F.cross_entropy(s, torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(s.device))


def _barlow(a, b, lam=5e-3):
    za, zb = (a - a.mean(0)) / (a.std(0) + 1e-6), (b - b.mean(0)) / (b.std(0) + 1e-6)
    c = za.T @ zb / len(a)
    off = c - torch.diag_embed(c.diagonal())
    return ((c.diagonal() - 1) ** 2).sum() + lam * (off ** 2).sum()


def run_selfsup(job, algo, impl, params, ctx):
    import copy

    device, batch, epochs, lr = _setup(job, params)
    method, size = impl["method"], int(params["image_size"])
    stage("validate")
    files = image_files(job, 16)
    stage("build")
    net = _backbone().to(device)
    log(f"resnet18 backbone, {len(files)} images, method {method}, batch {batch}, device {device}")
    proj = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128)).to(device)
    params_ = [*net.parameters(), *proj.parameters()]
    target_net, target_proj, predictor = None, None, None
    if method == "byol":
        target_net, target_proj = copy.deepcopy(net), copy.deepcopy(proj)
        for q in [*target_net.parameters(), *target_proj.parameters()]:
            q.requires_grad_(False)
        predictor = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128)).to(device)
        params_ += list(predictor.parameters())
    if method == "rotation":
        rot = nn.Linear(512, 4).to(device)
        params_ += list(rot.parameters())
    opt = torch.optim.Adam(params_, lr=lr)
    data = DataLoader(TwoViews(files, size), batch_size=batch, shuffle=True, drop_last=len(files) > batch)
    stage("train")
    history = []
    for e in range(1, epochs + 1):
        net.train()
        total, n = 0.0, 0
        for a, b in data:
            if len(a) < 2:
                continue
            a, b = a.to(device), b.to(device)
            if method == "simclr":
                loss = _nt_xent(proj(net(a)), proj(net(b)))
            elif method == "barlow":
                loss = _barlow(proj(net(a)), proj(net(b)))
            elif method == "rotation":
                k = torch.randint(0, 4, (len(a),), device=device)
                loss = F.cross_entropy(rot(net(torch.stack([torch.rot90(x, int(r), (1, 2)) for x, r in zip(a, k)]))), k)
            else:
                pa, pb = predictor(proj(net(a))), predictor(proj(net(b)))
                with torch.no_grad():
                    ta, tb = target_proj(target_net(a)), target_proj(target_net(b))
                loss = (2 - 2 * F.cosine_similarity(pa, tb).mean()) + (2 - 2 * F.cosine_similarity(pb, ta).mean())
            opt.zero_grad(), loss.backward(), opt.step()
            if method == "byol":
                with torch.no_grad():
                    for q, w in zip([*target_net.parameters(), *target_proj.parameters()], [*net.parameters(), *proj.parameters()]):
                        q.mul_(0.99).add_(0.01 * w)
            total += loss.item()
            n += 1
        history.append((total / max(n, 1), total / max(n, 1)))
        emit("epoch", epoch=e, total=epochs, train_loss=total / max(n, 1), val_loss=total / max(n, 1))
    stage("evaluate")
    eval_tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    net.eval()
    with torch.no_grad():
        z = torch.cat([net(torch.stack([eval_tf(Image.open(f).convert("RGB")) for f in files[i:i + batch]]).to(device)).cpu() for i in range(0, len(files), batch)]).numpy()
    m = dict(embedding_std=float(z.std(0).mean()), images=len(files))
    names = [f.parent.name for f in files]
    primary = ("embedding_std", m["embedding_std"], True)
    if len(set(names)) >= 2 and min(names.count(c) for c in set(names)) >= 2:
        from sklearn.model_selection import cross_val_score
        from sklearn.neighbors import KNeighborsClassifier

        folds = min(3, min(names.count(c) for c in set(names)))
        if folds >= 2:
            m["knn_accuracy"] = float(cross_val_score(KNeighborsClassifier(3), z, names, cv=folds).mean())
            primary = ("knn_accuracy", m["knn_accuracy"], True)
            log(f"folder names were used only to measure quality: kNN accuracy {m['knn_accuracy']:.3f}")
    from .tabular import _projection_chart

    charts = _projection_chart(z, None, "Learned embeddings") + [loss_chart(history)]
    stage("save")
    d = Path(job["run_dir"]) / "model"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(net.cpu().state_dict(), d / "net.pt")
    (d / "meta.json").write_text(json.dumps(dict(kind="selfsup", task=job["task"], image_size=size), indent=1))
    sample_x = torch.stack([eval_tf(Image.open(f).convert("RGB")) for f in files[:4]]).numpy()
    return dict(primary=primary, val=None, test=m, charts=charts, sample=sample_x, params=params)


def predict_selfsup(p, images=None, **_):
    net = _backbone_loaded(p)
    size = p.meta["image_size"]
    tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with torch.no_grad():
        z = net(torch.stack([tf(Image.open(f).convert("RGB")) for f in images])).numpy()
    out = pd.DataFrame(z, columns=[f"dim_{i}" for i in range(z.shape[1])])
    out.insert(0, "file", [str(f) for f in images])
    return out
