"""Diagnostic: is the gauge baseline's failure caused by polytope collapse near obstacles?
Trains gauge (seed 0, 100k) and legendre (seed 0, 100k) and, on cond A rollouts, records for each
episode the minimum CBF clearance D_min and the mean executed speed over the last 20 steps of
non-successful episodes."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from lsp.agent import train
from lsp.env import CONDITIONS, NavEnv

out = {}
for m in ("gauge", "legendre"):
    ag, _ = train(m, 0, steps=100_000, log=lambda *a: None)
    cfg = CONDITIONS["A"]; env = NavEnv(500, cfg, seed=2024)
    alive = np.ones(500, bool); reached = np.zeros(500, bool)
    dmin = np.full(500, np.inf); speed_tail = [[] for _ in range(500)]
    near_obstacle_stuck = np.zeros(500, bool)
    for t in range(cfg.horizon):
        o = env.obs(); res = ag.act(o, env.pos.copy(), env.cons(), explore=False)
        sp = np.linalg.norm(res["u_exec"], axis=1)
        d = env.clearance().min(1)
        dmin = np.where(alive, np.minimum(dmin, d), dmin)
        if t >= cfg.horizon - 20:
            for i in np.where(alive)[0]: speed_tail[i].append(sp[i])
        _, _, term, _, _ = env.step(res["u_exec"])
        reached |= term & alive; alive &= ~term
    fail = ~reached
    tail = np.array([np.mean(s) if s else np.nan for s in speed_tail])
    out[m] = dict(success=float(reached.mean()),
                  median_Dmin_failed=float(np.median(dmin[fail])) if fail.any() else None,
                  median_Dmin_success=float(np.median(dmin[reached])) if reached.any() else None,
                  frac_failed_Dmin_lt_1e6=float((dmin[fail] < 1e-6).mean()) if fail.any() else None,
                  mean_tail_speed_failed=float(np.nanmean(tail[fail])) if fail.any() else None)
    print(m, out[m], flush=True)
json.dump(out, open("results/diag_gauge_pinch.json", "w"), indent=1)
