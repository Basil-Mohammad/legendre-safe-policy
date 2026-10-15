"""Train and evaluate one (env, algo, method, scale, seed) cell.   python -m lsp2.train --env navsi --algo sac ..."""
from __future__ import annotations

import argparse, json, os, time

import numpy as np
import torch

from . import envs as E                                   # noqa: F401  (import side effect: lsp sets float64 default)
torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from .faults import FaultWrapper, FAULTS
from .sb3_ext import (InfoStats, MapSACPolicy, MapVecEnv, PenaltyReward, RowFeaturesExtractor, PATHWISE)

ENVS = dict(navsi=E.NavSI, navdi=E.NavDI, unicycle=E.Unicycle, fetch=E.FetchSafe, panda=E.PandaSafe)
CONDS = ("A", "B", "C")

SAC_KW = dict(learning_rate=3e-4, batch_size=256, buffer_size=200_000, tau=0.005, gamma=0.99, train_freq=2,
              gradient_steps=1, learning_starts=2_000)
PPO_KW = dict(learning_rate=3e-4, n_steps=256, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
              ent_coef=0.0, clip_range=0.2)
NET = dict(net_arch=[128, 128])


def parse_method(m: str):
    """'sp' | 'gauge' | 'legendre' | 'legendre_st' | 'se' | 'pen0.5' | 'pen2' -> (kind, lam)"""
    if m.startswith("pen"):
        return "pen", float(m[3:])
    return m, 0.0


def make_vec(env, cond, seed, algo, method, scale, n=1, fault=None):
    kind, lam = parse_method(method)
    thunks = []
    for i in range(n):
        def f(i=i):
            e = ENVS[env](cond, seed * 100 + i)
            if kind == "pen":
                e = PenaltyReward(e, lam)
            if fault is not None:
                e = FaultWrapper(e, fault, seed=seed * 100 + i + 7)
            e = InfoStats(e)
            return Monitor(e)
        thunks.append(f)
    v = DummyVecEnv(thunks)
    spec = v.envs[0].unwrapped.spec
    if kind == "pen":
        v = MapVecEnv(v, spec, "identity", 1.0)
    elif kind in ("se",) or algo == "ppo":
        v = MapVecEnv(v, spec, kind, scale)
    return v, spec


def build(env, algo, method, scale, seed, log_dir=None):
    kind, _ = parse_method(method)
    if algo == "ppo" and kind in ("legendre_st", "se"):
        raise ValueError("legendre_st / se are not defined for PPO")
    v, spec = make_vec(env, "A", seed, algo, method, scale, n=8 if algo == "ppo" else 1)
    pk = dict(features_extractor_class=RowFeaturesExtractor, features_extractor_kwargs=dict(spec=spec), **NET)
    if algo == "ppo":
        model = PPO("MlpPolicy", v, policy_kwargs=pk, seed=seed, verbose=0, device="cpu", **PPO_KW)
    elif kind in PATHWISE:
        pk.update(spec=spec, kind=kind, scale=scale)
        model = SAC(MapSACPolicy, v, policy_kwargs=pk, seed=seed, verbose=0, device="cpu", **SAC_KW)
    else:
        model = SAC("MlpPolicy", v, policy_kwargs=pk, seed=seed, verbose=0, device="cpu", **SAC_KW)
    return model, spec


def evaluate(model, env, algo, method, scale, seed, cond, episodes, fault=None):
    v, spec = make_vec(env, cond, 10_000 + seed, algo, method, scale, n=1, fault=fault)
    obs = v.reset()
    stats = []
    steps = 0
    while len(stats) < episodes:
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, done, infos = v.step(act)
        steps += 1
        if done[0]:
            stats.append(infos[0]["episode_stats"])
    keys = stats[0].keys()
    out = {k: float(np.mean([s[k] for s in stats])) for k in keys}
    out["episodes"] = len(stats)
    out["coll_total"] = int(sum(s["coll_steps"] for s in stats))
    return out


def run(args):
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.env}_{args.algo}_{args.method}_s{args.scale:g}_seed{args.seed}"
    path = os.path.join(args.out, tag + ".json")
    if os.path.exists(path):
        return path
    model, spec = build(args.env, args.algo, args.method, args.scale, args.seed)
    t0 = time.time()
    model.learn(total_timesteps=args.steps)
    train_time = time.time() - t0
    res = dict(env=args.env, algo=args.algo, method=args.method, scale=args.scale, seed=args.seed, steps=args.steps,
               train_time=train_time, eval={})
    for c in CONDS:
        res["eval"][c] = evaluate(model, args.env, args.algo, args.method, args.scale, args.seed, c, args.eval_eps)
    json.dump(res, open(path + ".partial", "w"), indent=1)       # keep the (expensive) main evaluation even if a fault run fails
    res["faults"] = {}
    if args.fault_eps > 0:
        for f in FAULTS:
            res["faults"][f] = evaluate(model, args.env, args.algo, args.method, args.scale, args.seed, "A",
                                        args.fault_eps, fault=f)
    res["total_time"] = time.time() - t0
    json.dump(res, open(path, "w"), indent=1)
    if args.save_model:
        model.save(os.path.join(args.out, tag + ".zip"))
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=list(ENVS))
    p.add_argument("--algo", required=True, choices=["sac", "ppo"])
    p.add_argument("--method", required=True)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--eval_eps", type=int, default=100)
    p.add_argument("--fault_eps", type=int, default=0)
    p.add_argument("--out", default="results_v6")
    p.add_argument("--save_model", action="store_true")
    print(run(p.parse_args()))
