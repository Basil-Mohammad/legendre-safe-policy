"""Safe-action layers.

All layers map an unconstrained / box-bounded network output ``a`` to an action ``u`` inside the
state-dependent safe polytope  P(x) = {u : A(x) u <= b(x)}.

Rows are padded to a fixed number of slots; ``mask`` marks valid rows.  Padded rows have a = 0 and
b = 1 and never influence any layer.

Maps
----
legendre_map : u*(nu) = argmin_u  -mu * sum_i log(b_i - a_i.u) - nu.u      (proposed)
               Jacobian du/dnu = (1/mu) * H^{-1},  H = sum_i a_i a_i^T / s_i^2  (SPD)  -> no aliasing.
project_polytope_2d : exact Euclidean projection (m = 2) by candidate enumeration.
gauge_map    : u = ||nu||_inf * nu / gamma_P(nu)   (Tabas & Zhang style bijection).
"""
from __future__ import annotations

import itertools

import torch


B_FLOOR = 1e-10


def build_polytope(pos, centers, radii, omask, gamma, dt, umax, gamma_wall=0.5, arena=1.0):
    """Return A (B,K+4,2), b (B,K+4), mask (B,K+4).

    Obstacle i:  a_i = -2 dt (p - c_i),  b_i = gamma * (|p-c_i|^2 - r_i^2)
        Because |p+dt u - c|^2 >= |p-c|^2 + 2 dt (p-c).u, the row implies
        D_next >= (1-gamma) D with D = |p-c|^2 - r^2  (exact discrete-time CBF, gamma in (0,1]).
    Box / wall rows:  +-u_j <= min(umax, gamma_wall * (arena -+ p_j) / dt).
    """
    B, K, _ = centers.shape
    diff = pos.unsqueeze(1) - centers
    D = (diff**2).sum(-1) - radii**2
    a_obs = -2.0 * dt * diff
    # Numerical floor (all methods): a state that sits on the CBF boundary for many steps makes D
    # decay geometrically to ~0 (b_i = 0 -> 0/0 in the gauge map).  1e-10 << collision_tol (1e-6).
    b_obs = (gamma.unsqueeze(1) * D).clamp_min(B_FLOOR)
    a_obs = torch.where(omask.unsqueeze(-1), a_obs, torch.zeros_like(a_obs))
    b_obs = torch.where(omask, b_obs, torch.ones_like(b_obs))

    eye = torch.eye(2, dtype=pos.dtype, device=pos.device)
    a_box = torch.cat([eye, -eye], 0).unsqueeze(0).expand(B, 4, 2)  # +e1,+e2,-e1,-e2
    up = gamma_wall * (arena - pos) / dt
    lo = gamma_wall * (arena + pos) / dt
    b_box = torch.cat([up, lo], 1).clamp(min=B_FLOOR, max=umax)  # (B,4): +x,+y,-x,-y
    A = torch.cat([a_obs, a_box], 1)
    b = torch.cat([b_obs, b_box], 1)
    mask = torch.cat([omask, torch.ones(B, 4, dtype=torch.bool, device=pos.device)], 1)
    return A, b, mask


def _newton_solve(nu, A, b, mask, mu, iters=60, tol=1e-9):
    """Damped Newton on f(u) = -sum log s_i - (nu/mu).u, s_i = b_i - a_i.u  (self-concordant).

    Step size t = 1/(1+lambda) keeps iterates strictly feasible for ANY start in the interior.
    u = 0 is strictly interior by construction of the polytope (b > 0).
    """
    z = nu / mu
    u = torch.zeros_like(nu)
    maskf = mask.to(nu.dtype)
    for it in range(iters):
        s = b - (A * u.unsqueeze(1)).sum(-1)
        s = torch.where(mask, s, torch.ones_like(s))
        inv = maskf / s
        g = (A * inv.unsqueeze(-1)).sum(1) - z
        H = torch.einsum("bk,bki,bkj->bij", inv * inv, A, A)
        d = -torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)
        lam = (-(g * d).sum(-1)).clamp_min(0.0).sqrt()
        u = u + (1.0 / (1.0 + lam)).unsqueeze(-1) * d
        if it >= 4 and it % 3 == 0 and float(lam.max()) < tol:
            break
    return u


def legendre_hessian(u, A, b, mask):
    """H = sum_i a_i a_i^T / s_i^2 evaluated at u (not scaled by mu)."""
    s = b - (A * u.unsqueeze(1)).sum(-1)
    s = torch.where(mask, s, torch.ones_like(s))
    inv = mask.to(u.dtype) / s
    return torch.einsum("bk,bki,bkj->bij", inv * inv, A, A)


class _LegendreFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nu, A, b, mask, mu):
        with torch.no_grad():
            u = _newton_solve(nu, A, b, mask, mu)
            H = legendre_hessian(u, A, b, mask)
        ctx.save_for_backward(H)
        ctx.mu = mu
        return u

    @staticmethod
    def backward(ctx, grad_u):
        (H,) = ctx.saved_tensors
        g = torch.linalg.solve(H, grad_u.unsqueeze(-1)).squeeze(-1) / ctx.mu
        return g, None, None, None, None


def legendre_map(nu, A, b, mask, mu):
    """Differentiable, strictly-interior action  u*(nu)  (see module docstring)."""
    return _LegendreFn.apply(nu, A, b, mask, mu)


def legendre_jacobian(u, A, b, mask, mu):
    H = legendre_hessian(u, A, b, mask)
    return torch.linalg.inv(H) / mu


_PAIR_CACHE: dict = {}


def _pairs(K, device):
    if K not in _PAIR_CACHE:
        ij = torch.tensor(list(itertools.combinations(range(K), 2)), dtype=torch.long)
        _PAIR_CACHE[K] = (ij[:, 0], ij[:, 1])
    I, J = _PAIR_CACHE[K]
    return I.to(device), J.to(device)


def project_polytope_2d(u0, A, b, mask, feas_tol=1e-9):
    """argmin_u |u-u0|^2  s.t. A u <= b   (rows with mask=False ignored).

    The minimiser is u0 itself, a projection onto ONE face hyperplane, or a vertex (two active
    rows).  We enumerate all candidates, keep the feasible ones and return the closest.
    Autograd through the chosen candidate yields the exact a.e. Jacobian:
    I (interior), I - a a^T/|a|^2 (face), 0 (vertex: rank-deficient -> action aliasing).
    """
    B, K, _ = A.shape
    dtype = u0.dtype
    big = torch.tensor(1e30, dtype=dtype, device=u0.device)

    a2 = (A * A).sum(-1).clamp_min(1e-18)
    viol = (A * u0.unsqueeze(1)).sum(-1) - b
    face = u0.unsqueeze(1) - A * (viol / a2).unsqueeze(-1)
    face_ok = mask & (A * A).sum(-1).gt(1e-18)

    I, J = _pairs(K, u0.device)
    a_i, a_j = A[:, I], A[:, J]
    b_i, b_j = b[:, I], b[:, J]
    det = a_i[..., 0] * a_j[..., 1] - a_i[..., 1] * a_j[..., 0]
    safe = torch.where(det.abs() > 1e-12, det, torch.ones_like(det))
    vx = (b_i * a_j[..., 1] - b_j * a_i[..., 1]) / safe
    vy = (a_i[..., 0] * b_j - a_j[..., 0] * b_i) / safe
    vert = torch.stack([vx, vy], -1)
    vert_ok = mask[:, I] & mask[:, J] & (det.abs() > 1e-12)

    cand = torch.cat([u0.unsqueeze(1), face, vert], 1)
    ok = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=u0.device), face_ok, vert_ok], 1)
    lhs = torch.einsum("bki,bci->bck", A, cand)
    feas = ((lhs <= b.unsqueeze(1) + feas_tol) | (~mask).unsqueeze(1)).all(-1)
    valid = ok & feas
    dist = ((cand - u0.unsqueeze(1)) ** 2).sum(-1)
    dist = torch.where(valid, dist, big)
    idx = dist.argmin(1)
    return cand[torch.arange(B), idx]


def gauge_map(nu, A, b, mask):
    """u = ||nu||_inf * nu / gamma_P(nu),  gamma_P(v) = max_i a_i.v / b_i  (b_i > 0)."""
    ratio = (A * nu.unsqueeze(1)).sum(-1) / b
    ratio = torch.where(mask, ratio, torch.full_like(ratio, -1e30))
    g = ratio.max(-1).values.clamp_min(1e-9)
    scale = nu.abs().max(-1).values / g
    return nu * scale.unsqueeze(-1)
