"""Train one (method, seed) and evaluate on conditions A, B, C.  Saves results/<tag>.json"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lsp.agent import evaluate, train  # noqa: E402
from lsp.env import CONDITIONS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--penalty", type=float, default=2.0)
    ap.add_argument("--eval_eps", type=int, default=500)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or (f"{a.method}_mu{a.mu}_s{a.seed}" if a.method == "legendre" else f"{a.method}_s{a.seed}")
    if a.method == "penalty" and a.tag is None:
        tag = f"penalty_l{a.penalty}_s{a.seed}"
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, tag + ".json")):
        print(f"[{tag}] exists, skipping", flush=True)
        return
    t0 = time.time()
    agent, curve = train(a.method, a.seed, steps=a.steps, mu=a.mu, penalty=a.penalty)
    final = {}
    for name, cfg in CONDITIONS.items():
        final[name] = evaluate(agent, cfg, n_episodes=a.eval_eps, seed=2024 + a.seed, jac_samples=4000)
        r = final[name]
        print(f"[{tag}] cond {name}: succ={r['success']:.3f} coll={r['collision_rate']:.3f} "
              f"ttg={r['time_to_goal']:.1f} rankdef={r['jacobian']['frac_rank_deficient']:.3f}", flush=True)
    json.dump(dict(method=a.method, seed=a.seed, mu=a.mu, penalty=a.penalty, steps=a.steps, curve=curve, final=final,
                   wall_s=time.time() - t0), open(os.path.join(a.out, tag + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
