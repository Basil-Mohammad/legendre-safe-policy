"""EXPLORATORY probe for Study 3 (double integrator). Not a confirmatory result.

Questions (decided before looking at the numbers):
 Q1. Among states inside the safe set C, how often is the polytope empty / without strict interior?
 Q2. Does C stay invariant, and are there collisions, when actions are projected into the polytope
     (random actions), with the emergency brake as fallback when the polytope is empty?
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from scipy.optimize import linprog

from lsp.di import AMAX, VMAX, brake_action, collided, di_polytope, in_safe_set, step
from lsp.env import KSLOTS
from lsp.layers import project_polytope_2d

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)


def sample_scene(n, kmin=1, kmax=4, rmin=0.10, rmax=0.25, margin=0.01):
    """States inside the safe set C (rejection sampling)."""
    out = []
    while sum(len(o[0]) for o in out) < n:
        m = 4 * n
        K = rng.integers(kmin, kmax + 1, m)
        cen = rng.uniform(-0.9, 0.9, (m, KSLOTS, 2))
        rad = rng.uniform(rmin, rmax, (m, KSLOTS))
        mask = np.arange(KSLOTS)[None, :] < K[:, None]
        p = rng.uniform(-0.95, 0.95, (m, 2))
        v = rng.uniform(-VMAX, VMAX, (m, 2)) * rng.choice([0.0, 0.5, 1.0], (m, 1))
        t = torch.as_tensor
        ok = in_safe_set(t(p), t(v), t(cen), t(rad), t(mask), margin).numpy()
        out.append((p[ok], v[ok], cen[ok], rad[ok], mask[ok]))
    return [np.concatenate([o[i] for o in out])[:n] for i in range(5)]


def chebyshev(A, b, mask):
    """max t s.t. A a + t |A_i| <= b, a in R^2 (valid rows only). Returns t (<=0: no strict interior)."""
    Ai, bi = A[mask], b[mask]
    nrm = np.linalg.norm(Ai, axis=1)
    keep = nrm > 1e-12
    Ai, bi, nrm = Ai[keep], bi[keep], nrm[keep]
    c = np.array([0, 0, -1.0])
    Aub = np.hstack([Ai, nrm[:, None]])
    res = linprog(c, A_ub=Aub, b_ub=bi, bounds=[(None, None), (None, None), (None, 10)], method="highs")
    return res.x[2] if res.status == 0 else -1.0


def q1(n=4000, **kw):
    p, v, cen, rad, mask = sample_scene(n, **kw)
    t = torch.as_tensor
    A, b, m = di_polytope(t(p), t(v), t(cen), t(rad), t(mask))
    A, b, m = A.numpy(), b.numpy(), m.numpy()
    R = np.array([chebyshev(A[i], b[i], m[i]) for i in range(len(p))])
    speed = np.linalg.norm(v, axis=1)
    return dict(n=int(len(R)), frac_empty_or_flat=float((R <= 1e-9).mean()), frac_radius_lt_1e3=float((R < 1e-3).mean()),
                median_radius=float(np.median(R)),
                frac_empty_fast=float((R[speed > 0.5] <= 1e-9).mean()) if (speed > 0.5).any() else None,
                frac_empty_slow=float((R[speed <= 0.1] <= 1e-9).mean()) if (speed <= 0.1).any() else None)


def q2(n=2000, T=80, **kw):
    p, v, cen, rad, mask = [torch.as_tensor(x) for x in sample_scene(n, **kw)]
    n_coll = torch.zeros(n, dtype=torch.bool)
    left_C = torch.zeros(n, dtype=torch.bool)
    n_fallback = 0
    n_steps = 0
    for t in range(T):
        A, b, m = di_polytope(p, v, cen, rad, mask)
        u0 = torch.as_tensor(rng.uniform(-AMAX, AMAX, (n, 2)))
        a = project_polytope_2d(u0, A, b, m)
        feas = (((A * a.unsqueeze(1)).sum(-1) <= b + 1e-9) | ~m).all(-1)
        a = torch.where(feas.unsqueeze(-1), a, brake_action(v))
        n_fallback += int((~feas).sum())
        n_steps += n
        p, v = step(p, v, a)
        n_coll |= collided(p, cen, rad, mask)
        left_C |= ~in_safe_set(p, v, cen, rad, mask, margin=-1e-9)
    return dict(n=n, T=T, frac_episodes_with_collision=float(n_coll.float().mean()),
                frac_episodes_leaving_C=float(left_C.float().mean()), fallback_rate=n_fallback / n_steps)


if __name__ == "__main__":
    res = {"q1_default": q1(), "q1_dense": q1(kmin=5, kmax=8, rmin=0.08, rmax=0.18),
           "q2_default": q2(), "q2_dense": q2(kmin=5, kmax=8, rmin=0.08, rmax=0.18)}
    print(json.dumps(res, indent=1))
    os.makedirs("results_v3_probe", exist_ok=True)
    json.dump(res, open("results_v3_probe/probe_di.json", "w"), indent=1)
