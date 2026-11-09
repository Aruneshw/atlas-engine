"""Reinforcement learning on built-in simulated environments: bandits, tabular methods, Stable-Baselines3."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DataError
from .protocol import emit, log, stage
from .registry import ENVIRONMENTS

DISCRETE_STATE = {"FrozenLake-v1", "CliffWalking-v1", "Taxi-v3"}


def make(name):
    import gymnasium as gym

    return gym.make(name)


def _env_name(job):
    name = job["dataset"].get("env")
    if name not in ENVIRONMENTS:
        raise DataError(f"Unknown environment: {name}")
    return name


def _save(job, meta):
    d = Path(job["run_dir"]) / "model"
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(dict(task=job["task"], **meta), indent=1))
    return d


def _curve(rewards, title="Reward per episode"):
    r = np.asarray(rewards, float)
    k = max(1, len(r) // 50)
    smooth = np.convolve(r, np.ones(k) / k, mode="valid")
    idx = np.linspace(0, len(smooth) - 1, min(len(smooth), 300)).astype(int)
    return dict(type="line", title=title, xlabel="episode", x=idx.tolist(), series=[dict(name="reward", y=smooth[idx])])


# ---- bandits -------------------------------------------------------------------------
def run_bandit(job, algo, impl, params, ctx):
    if _env_name(job) != "Bandit-10":
        raise DataError("Bandit methods run on the Bandit-10 environment.")
    method, steps = impl["method"], params["steps"]
    stage("build")
    rng = np.random.RandomState(42)
    means = rng.uniform(0.1, 0.9, 10)
    counts, values, succ, fail = np.zeros(10), np.zeros(10), np.ones(10), np.ones(10)
    stage("train")
    rewards = np.zeros(steps)
    for t in range(steps):
        if method == "epsilon":
            a = rng.randint(10) if rng.random_sample() < params["epsilon"] else int(values.argmax())
        elif method == "ucb":
            a = int(np.argmin(counts)) if (counts == 0).any() else int((values + params["c"] * np.sqrt(np.log(t + 1) / counts)).argmax())
        else:
            a = int(rng.beta(succ, fail).argmax())
        r = float(rng.random_sample() < means[a])
        counts[a] += 1
        values[a] += (r - values[a]) / counts[a]
        succ[a] += r
        fail[a] += 1 - r
        rewards[t] = r
        if (t + 1) % max(1, steps // 20) == 0:
            emit("epoch", epoch=t + 1, total=steps, train_loss=float(rewards[: t + 1].mean()), val_loss=float(means.max() - rewards[: t + 1].mean()))
    stage("evaluate")
    regret = float(means.max() * steps - rewards.sum())
    m = dict(mean_reward=float(rewards.mean()), best_arm_found=int(values.argmax() == means.argmax()), total_regret=regret, best_possible=float(means.max()))
    stage("save")
    d = _save(job, dict(kind="bandit", env="Bandit-10"))
    np.savez(d / "bandit.npz", values=values, means=means)
    charts = [_curve(rewards, "Reward per pull"), dict(type="bars", title="Estimated arm values", labels=[f"arm {i}" for i in range(10)], values=values.tolist())]
    return dict(primary=("mean_reward", m["mean_reward"], True), val=None, test=m, charts=charts, sample=None, params=params)


# ---- tabular -------------------------------------------------------------------------
def run_tabular(job, algo, impl, params, ctx):
    name = _env_name(job)
    if name not in DISCRETE_STATE:
        raise DataError("Tabular methods need a grid environment: FrozenLake, CliffWalking or Taxi.")
    env = make(name)
    method = impl["method"]
    ns, na = env.observation_space.n, env.action_space.n
    Q = np.zeros((ns, na))
    gamma, alpha, episodes = params["gamma"], params["alpha"], params["episodes"]
    rng = np.random.RandomState(42)
    eps_start, eps_end = 1.0, 0.02
    stage("train")
    rewards = []
    if method == "value_iteration":
        P = env.unwrapped.P
        for sweep in range(1, 2001):
            delta = 0.0
            for s in range(ns):
                q = [sum(pr * (rew + gamma * (0 if done else Q[n].max())) for pr, n, rew, done in P[s][a]) for a in range(na)]
                delta = max(delta, abs(max(q) - Q[s].max()))
                Q[s] = q
            if sweep % 10 == 0:
                emit("epoch", epoch=sweep, total=2000, train_loss=delta, val_loss=delta)
            if delta < 1e-6:
                log(f"converged after {sweep} sweeps")
                break
    else:
        model = {}
        policy = lambda s, eps: rng.randint(na) if rng.random_sample() < eps else int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
        for ep in range(1, episodes + 1):
            eps = max(eps_end, eps_start * (1 - ep / (0.8 * episodes)))
            s, _ = env.reset(seed=int(rng.randint(1 << 30)))
            a = policy(s, eps)
            trail, total, done = [], 0.0, False
            for _ in range(params.get("max_steps", 500)):
                s2, r, term, trunc, _ = env.step(int(a))
                total += r
                done = term or trunc
                a2 = policy(s2, eps)
                if method == "mc":
                    trail.append((s, a, r))
                else:
                    if method == "sarsa":
                        tgt = Q[s2, a2]
                    elif method == "expected_sarsa":
                        best = Q[s2].argmax()
                        probs = np.full(na, eps / na)
                        probs[best] += 1 - eps
                        tgt = float(probs @ Q[s2])
                    else:
                        tgt = Q[s2].max()
                    Q[s, a] += alpha * (r + gamma * (0 if term else tgt) - Q[s, a])
                    if method == "dyna_q":
                        model[(s, a)] = (r, s2, term)
                        keys = list(model)
                        for k in rng.choice(len(keys), min(10, len(keys)), replace=False):
                            ps, pa = keys[k]
                            pr, pn, pt = model[(ps, pa)]
                            Q[ps, pa] += alpha * (pr + gamma * (0 if pt else Q[pn].max()) - Q[ps, pa])
                s, a = s2, a2
                if done:
                    break
            if method == "mc":
                g = 0.0
                for st, ac, rw in reversed(trail):
                    g = rw + gamma * g
                    Q[st, ac] += alpha * (g - Q[st, ac])
            rewards.append(total)
            if ep % max(1, episodes // 20) == 0:
                avg = float(np.mean(rewards[-max(1, episodes // 20):]))
                emit("epoch", epoch=ep, total=episodes, train_loss=-avg, val_loss=-avg)
    stage("evaluate")
    scores = _rollouts(lambda o, st: (int(Q[o].argmax()), None), env, 50)
    m = dict(mean_reward=float(np.mean(scores)), std_reward=float(np.std(scores)), states=ns, actions=na)
    stage("save")
    d = _save(job, dict(kind="rl_table", env=name))
    np.save(d / "q_table.npy", Q)
    charts = [_curve(rewards)] if rewards else []
    return dict(primary=("mean_reward", m["mean_reward"], True), val=None, test=m, charts=charts, sample=None, params=params)


def _rollouts(act, env, episodes, limit=2000):
    out = []
    for i in range(episodes):
        o, _ = env.reset(seed=1000 + i)
        total, state, done, n = 0.0, None, False, 0
        while not done and n < limit:
            a, state = act(o, state)
            o, r, term, trunc, _ = env.step(a)
            total += float(r)
            done = term or trunc
            n += 1
        out.append(total)
    return out


# ---- Stable-Baselines3 ---------------------------------------------------------------
def _sb3_class(path):
    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)


def _sb3_act(model, recurrent):
    def act(o, state):
        if recurrent:
            prev, starts = state if state is not None else (None, np.ones(1, bool))
            a, prev = model.predict(o, state=prev, episode_start=starts, deterministic=True)
            a = a.item() if a.ndim == 0 else a
            return (int(a) if np.isscalar(a) else a), (prev, np.zeros(1, bool))
        a, _ = model.predict(o, deterministic=True)
        return (int(a) if np.ndim(a) == 0 else a), None

    return act


def run_sb3(job, algo, impl, params, ctx):
    import gymnasium as gym
    from stable_baselines3.common.callbacks import BaseCallback

    name = _env_name(job)
    if name == "Bandit-10":
        raise DataError("Deep RL methods need a control environment such as CartPole or Pendulum.")
    env = make(name)
    continuous = isinstance(env.action_space, gym.spaces.Box)
    if impl["actions"] == "discrete" and continuous:
        raise DataError(f"This algorithm needs discrete actions; {name} has continuous ones. Try CartPole or Acrobot.")
    if impl["actions"] == "continuous" and not continuous:
        raise DataError(f"This algorithm needs continuous actions; {name} has discrete ones. Try Pendulum.")
    stage("build")
    device = "cuda" if job.get("device") == "cuda" else "cpu"
    total = params["timesteps"]
    cls = _sb3_class(impl["cls"])
    kw = dict(learning_rate=params["lr"], gamma=params["gamma"], seed=42, device=device, verbose=0)
    kw.update(impl.get("fixed", {}))
    model = cls(impl.get("policy", "MlpPolicy"), env, **kw)
    log(f"{cls.__name__} on {name}, {total:,} steps, device {device}")
    curve = []

    class Progress(BaseCallback):
        def _on_step(self):
            buf = self.model.ep_info_buffer
            if buf and self.num_timesteps % max(1, total // 40) == 0:
                avg = float(np.mean([x["r"] for x in buf]))
                curve.append((self.num_timesteps, avg))
                emit("epoch", epoch=self.num_timesteps, total=total, train_loss=-avg, val_loss=-avg)
            return True

    stage("train")
    model.learn(total_timesteps=total, callback=Progress())
    stage("evaluate")
    recurrent = impl.get("policy") == "MlpLstmPolicy"
    scores = _rollouts(_sb3_act(model, recurrent), make(name), 20)
    m = dict(mean_reward=float(np.mean(scores)), std_reward=float(np.std(scores)), timesteps=total)
    stage("save")
    d = _save(job, dict(kind="rl_sb3", env=name, cls=impl["cls"], recurrent=recurrent))
    model.save(d / "policy.zip")
    charts = []
    if curve:
        charts.append(dict(type="line", title="Average episode reward", xlabel="steps", x=[c[0] for c in curve], series=[dict(name="reward", y=[c[1] for c in curve])]))
    obs = np.stack([env.reset(seed=i)[0] for i in range(4)]).astype(np.float32)
    return dict(primary=("mean_reward", m["mean_reward"], True), val=None, test=m, charts=charts, sample=obs, params=params)


# ---- prediction and export -------------------------------------------------------------
def _policy(p):
    def load():
        m = p.meta
        if p.kind == "rl_table":
            q = np.load(p.dir / "q_table.npy")
            return lambda o, st: (int(q[o].argmax()), None)
        if p.kind == "bandit":
            arm = int(np.load(p.dir / "bandit.npz")["values"].argmax())
            return lambda o, st: (arm, None)
        return _sb3_act(_sb3_class(m["cls"]).load(p.dir / "policy.zip", device="cpu"), m["recurrent"])

    return p.cached(load)


def predict(p, steps=None, **_):
    if p.kind == "bandit":
        return pd.DataFrame({"best_arm": [_policy(p)(None, None)[0]]})
    env = make(p.meta["env"])
    scores = _rollouts(_policy(p), env, int(steps or 10))
    return pd.DataFrame({"episode": range(1, len(scores) + 1), "reward": scores})


def act(p, obs):
    if p.kind == "rl_table":
        action, _ = _policy(p)(int(obs), None)
    else:
        action, _ = _policy(p)(np.asarray(obs, np.float32), None)
    return action.tolist() if hasattr(action, "tolist") else action


def onnx_parts(p, sample):
    import torch

    if p.kind != "rl_sb3" or p.meta["recurrent"]:
        raise NotImplementedError("this policy has no single-network form")
    model = _sb3_class(p.meta["cls"]).load(p.dir / "policy.zip", device="cpu")
    policy = model.policy.eval()

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            if hasattr(self.policy, "actor"):
                return self.policy.actor(obs, deterministic=True) if hasattr(self.policy.actor, "action_dist") else self.policy.actor(obs)
            return self.policy._predict(obs, deterministic=True)

    return Wrap().eval(), (torch.as_tensor(sample),), ["observation"]
