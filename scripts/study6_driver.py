"""Study-6 driver: tuning phase -> scale selection -> confirmatory phase, for one environment.
Resume-safe (finished cells are skipped).  Used identically in the sandbox and in the Kaggle notebooks.

  python scripts/study6_driver.py --env navsi --workers 2 --out results_v6
"""
import argparse, itertools, json, os, subprocess, sys, glob
from concurrent.futures import ThreadPoolExecutor
import numpy as np

SAC_STEPS = dict(navsi=50_000, navdi=60_000, unicycle=80_000, fetch=15_000, panda=60_000)
PPO_STEPS = dict(navsi=200_000, navdi=200_000, unicycle=200_000, fetch=60_000, panda=200_000)
GRID = {"sp": [0.5, 1, 2, 4], "gauge": [0.5, 1, 2, 4], "se": [0.5, 1, 2, 4], "legendre": [5, 10, 20, 40]}
CENTRE = {"sp": 1.0, "gauge": 1.0, "se": 1.0, "legendre": 15.0}   # tie-break anchor (geometric centre of the grid)
TUNE_SEEDS = [100, 101]
CONF_SEEDS = list(range(10))
ALGO_MAPS = {"sac": ["sp", "gauge", "se", "legendre"], "ppo": ["sp", "gauge", "legendre"]}
ALGO_CONF = {"sac": ["sp", "gauge", "se", "legendre", "legendre_st", "pen2"], "ppo": ["sp", "gauge", "legendre", "pen2"]}


def tag(env, algo, method, scale, seed):
    return f"{env}_{algo}_{method}_s{scale:g}_seed{seed}"


def run_cell(cell, out, tune):
    env, algo, method, scale, seed = cell
    path = os.path.join(out, tag(env, algo, method, scale, seed) + ".json")
    if os.path.exists(path):
        return
    steps = (SAC_STEPS if algo == "sac" else PPO_STEPS)[env]
    cmd = [sys.executable, "-m", "lsp2.train", "--env", env, "--algo", algo, "--method", method, "--scale", str(scale),
           "--seed", str(seed), "--steps", str(steps), "--out", out,
           "--eval_eps", "50" if tune else "100", "--fault_eps", "0" if tune else "40"]
    if not tune:
        cmd.append("--save_model")
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=open(os.path.join(out, "err.log"), "a"))


def pool(cells, out, tune, workers):
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda c: run_cell(c, out, tune), cells))


def select(env, algo, method, out):
    """highest mean condition-A success over tuning seeds; <0.5 point differences are ties -> closest to the centre (log)."""
    sc = {}
    for s in GRID[method]:
        v = []
        for sd in TUNE_SEEDS:
            p = os.path.join(out, tag(env, algo, method, s, sd) + ".json")
            if os.path.exists(p):
                v.append(json.load(open(p))["eval"]["A"]["success"])
        sc[s] = float(np.mean(v)) if len(v) == len(TUNE_SEEDS) else -1.0
    best = max(sc.values())
    tied = [s for s in sc if best - sc[s] < 0.005]
    return min(tied, key=lambda s: abs(np.log(s / CENTRE[method]))), sc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results_v6")
    ap.add_argument("--algos", default="sac,ppo")
    ap.add_argument("--methods", default="", help="comma list restricting the maps (sharding); legendre_st needs legendre")
    a = ap.parse_args()
    out = os.path.join(a.out, a.env)
    os.makedirs(out, exist_ok=True)
    algos = a.algos.split(",")
    only = set(a.methods.split(",")) if a.methods else None
    keep = lambda m: only is None or m in only
    tune = [(a.env, al, m, s, sd) for al in algos for m in ALGO_MAPS[al] if keep(m) or (m == "legendre" and keep("legendre_st"))
            for s in GRID[m] for sd in TUNE_SEEDS]
    # cheapest first so that early progress is visible
    pool(sorted(tune, key=lambda c: (c[1] != "ppo", c[2] != "se")), out, True, a.workers)
    sel = {}
    for al in algos:
        for m in ALGO_MAPS[al]:
            if not (keep(m) or (m == "legendre" and keep("legendre_st"))):
                continue
            s, sc = select(a.env, al, m, out)
            sel[f"{al}/{m}"] = dict(scale=s, tuning_success=sc)
    json.dump(sel, open(os.path.join(out, "selected%s.json" % ("_" + a.methods.replace(",", "-") if a.methods else "")), "w"), indent=1)
    conf = []
    for al in algos:
        for m in ALGO_CONF[al]:
            if not keep(m):
                continue
            s = 1.0 if m == "pen2" else sel[f"{al}/{'legendre' if m == 'legendre_st' else m}"]["scale"]
            for sd in CONF_SEEDS:
                conf.append((a.env, al, m, s, sd))
    pool(conf, out, False, a.workers)
    print("done", a.env)
