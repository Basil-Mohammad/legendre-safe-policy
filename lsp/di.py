"""Double-integrator (relative degree 2) safety rows.  EXPLORATORY (Study 3 probe).

State (p, v) in R^2 x R^2, action a = acceleration, |a|_inf <= AMAX, |v|_inf <= VMAX,
exact zero-order-hold step  p' = p + dt v + dt^2/2 a,  v' = v + dt a.

Braking-aware barrier per circular obstacle (clearance s = |p-c| - r, unit normal n = (p-c)/|p-c|):
    C_i = { x : s >= phi(V) },   V = max(0, -v.n),   phi(V) = V^2 / (2 beta),  beta = THETA * AMAX.
Affine-in-a sufficient condition for  x' in C_i  (derived in the module docstring of probe_di.py):
  * s' >= s + dt v.n + dt^2/2 n.a                       (convexity of the norm: tangent lower bound, exact)
  * closing speed z(a) = -(v + dt a).n + eps  with eps covering the rotation of n (state-dependent, a
    conservative margin, NOT a proof); phi(max(0,z)) is convex in z, so it is bounded above by its
    secant on [z_lo, z_hi] = range of z over the actuation box -> affine in a.
Walls are half-spaces (fixed n, eps = 0), for which the rows are exact.
The resulting guarantee for circular obstacles is therefore EMPIRICAL (rotation margin), exact for walls.
"""
from __future__ import annotations

import torch

DT = 0.1
AMAX = 1.0
VMAX = 0.6
THETA = 0.5
BETA = THETA * AMAX
EPS0 = 0.01
ARENA = 1.0


def phi(V):
    return V.clamp_min(0.0) ** 2 / (2.0 * BETA)


def _row(s, vn, nA, eps):
    """Row for one (half-)space.  s: clearance, vn: v.n, nA: |n|_1 (max of n.a over box / AMAX).

    Returns coefficient c (for A = -c * n) and rhs b such that  -c n.a <= b.
    """
    z_lo = -vn - DT * AMAX * nA + eps
    z_hi = -vn + DT * AMAX * nA + eps
    f_lo, f_hi = phi(z_lo), phi(z_hi)
    width = (z_hi - z_lo).clamp_min(1e-12)
    sigma = torch.where(z_hi > 0, (f_hi - f_lo) / width, torch.zeros_like(width))
    # (0.5 dt^2 + sigma dt) n.a >= f_lo + sigma(-vn + eps - z_lo) - s - dt vn
    rhs = f_lo + sigma * (-vn + eps - z_lo) - s - DT * vn
    coef = 0.5 * DT**2 + sigma * DT
    return coef, rhs


def di_polytope(p, v, centers, radii, omask):
    """A (B,K+8,2), b (B,K+8), mask (B,K+8): obstacle rows, 4 wall rows, 4 box rows."""
    B, K, _ = centers.shape
    diff = p.unsqueeze(1) - centers
    rho = diff.norm(dim=-1).clamp_min(1e-9)
    n = diff / rho.unsqueeze(-1)
    s = rho - radii
    vn = (n * v.unsqueeze(1)).sum(-1)
    vperp2 = (v.unsqueeze(1) ** 2).sum(-1) - vn**2
    rho_lb = torch.maximum(radii, rho - DT * v.norm(dim=-1, keepdim=True) - 0.5 * DT**2 * AMAX * 1.4143)
    eps = DT * vperp2.clamp_min(0) / (8.0 * rho_lb.clamp_min(1e-3)) + EPS0
    coef, rhs = _row(s, vn, n.abs().sum(-1), eps)
    A_obs = -(coef.unsqueeze(-1) * n)
    b_obs = -rhs
    A_obs = torch.where(omask.unsqueeze(-1), A_obs, torch.zeros_like(A_obs))
    b_obs = torch.where(omask, b_obs, torch.ones_like(b_obs))

    # walls: n points into the arena
    wn = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]], dtype=p.dtype, device=p.device)
    sw = torch.stack([ARENA - p[:, 0], ARENA + p[:, 0], ARENA - p[:, 1], ARENA + p[:, 1]], 1)
    vnw = (v.unsqueeze(1) * wn.unsqueeze(0)).sum(-1)
    coef_w, rhs_w = _row(sw, vnw, torch.ones_like(sw), torch.zeros_like(sw))
    A_w = -(coef_w.unsqueeze(-1) * wn.unsqueeze(0))
    b_w = -rhs_w

    # box: |a_j| <= AMAX and |v_j + dt a_j| <= VMAX
    eye = torch.eye(2, dtype=p.dtype, device=p.device)
    A_box = torch.cat([eye, -eye], 0).unsqueeze(0).expand(B, 4, 2)
    up = torch.minimum(torch.full_like(v, AMAX), (VMAX - v) / DT)
    lo = torch.minimum(torch.full_like(v, AMAX), (VMAX + v) / DT)
    b_box = torch.cat([up, lo], 1)  # +x,+y,-x,-y

    A = torch.cat([A_obs, A_w, A_box], 1)
    b = torch.cat([b_obs, b_w, b_box], 1)
    mask = torch.cat([omask, torch.ones(B, 8, dtype=torch.bool, device=p.device)], 1)
    # amendment 1 to prereg v2.1 (before any run): rows normalised to unit length so that the slack
    # b_i - a_i.u is a Euclidean distance to the row's hyperplane (padded rows keep a = 0, b = 1)
    nrm = A.norm(dim=-1, keepdim=True)
    real = nrm > 1e-12
    A = torch.where(real, A / nrm.clamp_min(1e-12), A)
    b = torch.where(real.squeeze(-1), b / nrm.squeeze(-1).clamp_min(1e-12), b)
    return A, b, mask


def in_safe_set(p, v, centers, radii, omask, margin=0.0):
    """x in C_i for every obstacle and wall (with an optional clearance margin)."""
    diff = p.unsqueeze(1) - centers
    rho = diff.norm(dim=-1).clamp_min(1e-9)
    n = diff / rho.unsqueeze(-1)
    s = rho - radii
    V = (-(n * v.unsqueeze(1)).sum(-1)).clamp_min(0)
    ok_obs = ((s >= phi(V) + margin) | ~omask).all(-1)
    wn = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]], dtype=p.dtype, device=p.device)
    sw = torch.stack([ARENA - p[:, 0], ARENA + p[:, 0], ARENA - p[:, 1], ARENA + p[:, 1]], 1)
    Vw = (-(v.unsqueeze(1) * wn.unsqueeze(0)).sum(-1)).clamp_min(0)
    ok_w = (sw >= phi(Vw) + margin).all(-1)
    return ok_obs & ok_w


def step(p, v, a):
    return p + DT * v + 0.5 * DT**2 * a, v + DT * a


def collided(p, centers, radii, omask, tol=1e-6):
    D = ((p.unsqueeze(1) - centers) ** 2).sum(-1) - radii**2
    out_wall = (p.abs() > ARENA + 1e-9).any(-1)
    return ((D < -tol) & omask).any(-1) | out_wall


def brake_action(v):
    """Certified emergency action: stop as fast as the actuation box allows."""
    return (-v / DT).clamp(-AMAX, AMAX)
