"""Train one (method, seed) on the double integrator and evaluate on conditions A, B, C."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lsp.di_agent import evaluate, train  # noqa: E402
from lsp.di_env import DI_CONDITIONS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--penalty", type=float, default=2.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--eval_eps", type=int, default=500)
    ap.add_argument("--out", default="results_v3")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag
    if tag is None:
        tag = f"{a.method}_s{a.seed}"
        if a.method == "penalty":
            tag = f"penalty_l{a.penalty}_s{a.seed}"
        elif a.method == "legendre" and a.mu != 0.05:
            tag = f"legendre_mu{a.mu}_s{a.seed}"
        if a.scale != 1.0:
            tag = tag.replace(f"_s{a.seed}", f"_sc{a.scale}_s{a.seed}")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, tag + ".json")
    if os.path.exists(path):
        print(f"[{tag}] exists, skipping", flush=True)
        return
    t0 = time.time()
    agent, curve = train(a.method, a.seed, steps=a.steps, mu=a.mu, penalty=a.penalty, scale=a.scale)
    final = {}
    for name, cfg in DI_CONDITIONS.items():
        final[name] = evaluate(agent, cfg, n_episodes=a.eval_eps, seed=2024 + a.seed, jac_samples=4000)
        r = final[name]
        print(f"[{tag}] cond {name}: succ={r['success']:.3f} coll={r['collision_rate']:.3f} "
              f"fb={r['fallback_step_rate']:.4f} ttg={r['time_to_goal']:.1f}", flush=True)
    json.dump(dict(method=a.method, seed=a.seed, mu=a.mu, penalty=a.penalty, scale=a.scale, steps=a.steps, curve=curve,
                   final=final, wall_s=time.time() - t0), open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
