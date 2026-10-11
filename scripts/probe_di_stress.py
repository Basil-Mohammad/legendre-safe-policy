"""EXPLORATORY stress probe for Study 3 (double integrator).  Not a confirmatory result.

Scenarios (all rejection-sampled to lie inside the safe set C with clearance margin 0.002):
  single : agent 0..0.15 above an obstacle surface, velocity pointing at it
  pinch  : agent between two obstacles with a gap of 0.05..0.30, velocity along/across the gap
  corner : agent near a wall AND an obstacle
Rollouts use ADVERSARIAL actions: full-acceleration towards the nearest obstacle centre/wall,
projected onto the polytope; the emergency brake is used when the polytope is empty.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
from lsp.di import AMAX, VMAX, brake_action, collided, di_polytope, in_safe_set, step
from lsp.env import KSLOTS
from lsp.layers import project_polytope_2d
from probe_di import chebyshev

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(1)
t = torch.as_tensor


def unit(th):
    return np.stack([np.cos(th), np.sin(th)], -1)


def propose(kind, m):
    cen = rng.uniform(-0.9, 0.9, (m, KSLOTS, 2)); rad = rng.uniform(0.10, 0.25, (m, KSLOTS))
    K = np.full(m, 3); mask = np.arange(KSLOTS)[None, :] < K[:, None]
    th = rng.uniform(0, 2 * np.pi, m); s = rng.uniform(0.0, 0.15, m)
    if kind == "single":
        p = cen[:, 0] + (rad[:, 0] + s)[:, None] * unit(th)
        dirn = -unit(th + rng.normal(0, 0.4, m))
        v = dirn * rng.uniform(0, VMAX, m)[:, None]
    elif kind == "pinch":
        g = rng.uniform(0.05, 0.30, m)
        axis = unit(th); perp = np.stack([-axis[:, 1], axis[:, 0]], -1)
        mid = rng.uniform(-0.6, 0.6, (m, 2))
        cen[:, 0] = mid + perp * (rad[:, 0] + g / 2)[:, None]
        cen[:, 1] = mid - perp * (rad[:, 1] + g / 2)[:, None]
        p = mid + axis * rng.uniform(-0.3, 0.0, m)[:, None]
        v = axis * rng.uniform(0, VMAX, m)[:, None] + perp * rng.normal(0, 0.1, (m, 1))
    else:  # corner
        p = np.stack([rng.uniform(0.7, 0.98, m), rng.uniform(0.7, 0.98, m)], -1) * rng.choice([-1, 1], (m, 2))
        cen[:, 0] = p + unit(rng.uniform(0, 2 * np.pi, m)) * (rad[:, 0] + s)[:, None]
        v = rng.uniform(-VMAX, VMAX, (m, 2))
    v = np.clip(v, -VMAX, VMAX)
    return p, v, cen, rad, mask


def sample(kind, n, margin=0.002):
    out = []
    while sum(len(o[0]) for o in out) < n:
        p, v, cen, rad, mask = propose(kind, 20 * n)
        ok = in_safe_set(t(p), t(v), t(cen), t(rad), t(mask), margin).numpy() & (np.abs(p) < 0.99).all(1)
        out.append((p[ok], v[ok], cen[ok], rad[ok], mask[ok]))
    return [np.concatenate([o[i] for o in out])[:n] for i in range(5)]


def run(kind, n=1500, T=100):
    p, v, cen, rad, mask = [t(x) for x in sample(kind, n)]
    A, b, m = di_polytope(p, v, cen, rad, mask)
    R = np.array([chebyshev(A[i].numpy(), b[i].numpy(), m[i].numpy()) for i in range(n)])
    q1 = dict(frac_no_strict_interior=float((R <= 1e-9).mean()), frac_radius_lt_1e3=float((R < 1e-3).mean()))
    coll = torch.zeros(n, dtype=torch.bool); leftC = torch.zeros(n, dtype=torch.bool)
    fb = 0; steps = 0; fb_ep = torch.zeros(n, dtype=torch.bool)
    for _ in range(T):
        A, b, m = di_polytope(p, v, cen, rad, mask)
        d = cen - p.unsqueeze(1); dist = d.norm(dim=-1) - rad
        dist = torch.where(mask, dist, torch.full_like(dist, 1e9))
        j = dist.argmin(1); tgt = d[torch.arange(n), j]
        tgt = tgt / tgt.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        u0 = AMAX * tgt + 0.3 * t(rng.normal(size=(n, 2)))
        a = project_polytope_2d(u0.clamp(-AMAX, AMAX), A, b, m)
        feas = (((A * a.unsqueeze(1)).sum(-1) <= b + 1e-9) | ~m).all(-1)
        a = torch.where(feas.unsqueeze(-1), a, brake_action(v))
        fb += int((~feas).sum()); steps += n; fb_ep |= ~feas
        p, v = step(p, v, a)
        coll |= collided(p, cen, rad, mask)
        leftC |= ~in_safe_set(p, v, cen, rad, mask, margin=-1e-9)
    return dict(**q1, frac_episodes_with_collision=float(coll.float().mean()),
                frac_episodes_leaving_C=float(leftC.float().mean()),
                fallback_step_rate=fb / steps, frac_episodes_using_fallback=float(fb_ep.float().mean()))


if __name__ == "__main__":
    res = {k: run(k) for k in ("single", "pinch", "corner")}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results_v3_probe", "probe_di_stress.json"), "w"), indent=1)
