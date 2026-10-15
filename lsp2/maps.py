"""Batched, m-dimensional safe action maps for polytopes P = {u : A u <= b} with u = 0 strictly interior (b > 0).

legendre : u*(nu) = argmax nu.u + mu * sum log(b_i - a_i.u)           (Newton; exact implicit Jacobian)
project  : Euclidean projection of y onto P (dual FISTA + active-set polish; Jacobian I - A_S^+ A_S)
gauge    : u = ||nu||_inf nu / gamma_P(nu)                              (interior point 0)
All maps return strictly feasible points up to 1e-12 (restoration towards 0 after projection).
"""
from __future__ import annotations

import torch

from lsp.layers import legendre_map  # damped Newton, any m (u0 = 0 is strictly interior)

__all__ = ["legendre_map", "legendre_robust", "project_nd", "gauge_nd", "restore_feasible"]


def restore_feasible(u, A, b, mask, margin=0.0):
    """Scale u towards the interior point 0 so that A u <= (1-margin) b exactly (differentiable)."""
    au = (A * u.unsqueeze(1)).sum(-1)
    ratio = torch.where(mask & (au > 1e-14), (1.0 - margin) * b / au.clamp_min(1e-14), torch.full_like(au, 1e30))
    theta = ratio.min(-1).values.clamp(max=1.0)
    return u * theta.unsqueeze(-1)


def _lh_dual(G, r, mask, tol=1e-11, max_iter=None):
    """Lawson-Hanson active-set solution of  min_{lam>=0} 0.5 lam^T G lam - r^T lam  (batched; finite convergence)."""
    B, K, _ = G.shape
    dtype, dev = G.dtype, G.device
    eye = torch.eye(K, dtype=dtype, device=dev).expand(B, K, K)
    lam = torch.zeros(B, K, dtype=dtype, device=dev)
    S = torch.zeros(B, K, dtype=torch.bool, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    max_iter = max_iter or 6 * K + 20
    ar = torch.arange(B, device=dev)
    for _ in range(max_iter):
        Gs = torch.where(S.unsqueeze(1) & S.unsqueeze(2), G, torch.zeros_like(G)) + torch.where(
            S.unsqueeze(-1) & eye.bool(), torch.zeros_like(G), eye) + 1e-13 * eye
        rhs = torch.where(S, r, torch.zeros_like(r))
        s = torch.linalg.solve(Gs, rhs.unsqueeze(-1)).squeeze(-1)
        s = torch.where(S, s, torch.zeros_like(s))
        neg = S & (s <= tol)
        has_neg = neg.any(-1) & ~done
        # infeasible: step towards s until the first multiplier hits zero, drop it
        ratio = torch.where(neg, lam / (lam - s).clamp_min(1e-300), torch.full_like(lam, float("inf")))
        alpha = ratio.min(-1).values.clamp(0.0, 1.0)
        lam_step = lam + alpha.unsqueeze(-1) * (s - lam)
        S_step = S & (lam_step > tol)
        # feasible: accept s, then add the most violated dual constraint
        w = torch.where(~S & mask, r - torch.einsum("bkl,bl->bk", G, s), torch.full_like(r, -1e30))
        wmax, j = w.max(-1)
        add = (wmax > tol) & ~has_neg & ~done
        S_add = S.clone(); S_add[ar, j] = S_add[ar, j] | add
        finished = ~has_neg & ~(wmax > tol)
        lam = torch.where(has_neg.unsqueeze(-1), lam_step, torch.where(done.unsqueeze(-1), lam, s))
        S = torch.where(has_neg.unsqueeze(-1), S_step, torch.where(done.unsqueeze(-1), S, S_add))
        done = done | finished
        if bool(done.all()):
            break
    return S, done


def project_nd(y, A, b, mask, tol=1e-11):
    """argmin_u 0.5|u-y|^2  s.t.  A u <= b (rows with mask False ignored); batched, exact (dual active set),
    differentiable a.e. (Jacobian I - A_S^+ A_S on the active face).  Residual infeasibility from rounding is removed
    by scaling towards the interior point 0 (restore_feasible)."""
    B, K, m = A.shape
    maskf = mask.to(y.dtype)
    Am = A * maskf.unsqueeze(-1)
    G = torch.einsum("bki,bli->bkl", Am, Am)
    r = (Am * y.unsqueeze(1)).sum(-1) - b * maskf
    with torch.no_grad():
        S, done = _lh_dual(G, r, mask, tol)
    eye = torch.eye(K, dtype=y.dtype, device=y.device).expand(B, K, K)
    Gs = torch.where(S.unsqueeze(1) & S.unsqueeze(2), G, torch.zeros_like(G)) + torch.where(
        S.unsqueeze(-1) & eye.bool(), torch.zeros_like(G), eye) + 1e-13 * eye
    rhs = torch.where(S, r, torch.zeros_like(r))
    lam_s = torch.linalg.solve(Gs, rhs.unsqueeze(-1)).squeeze(-1)    # differentiable path (S fixed)
    lam_s = torch.where(S, lam_s, torch.zeros_like(lam_s)).clamp_min(0.0)
    u = y - torch.einsum("bk,bki->bi", lam_s, Am)
    return restore_feasible(u, A, b, mask)


def gauge_nd(nu, A, b, mask, rmax=1.0):
    """u = min(||nu||_inf, rmax) nu / gamma_P(nu), gamma_P(v) = max_i a_i.v / b_i  (b_i > 0).  rmax = 1 keeps the image
    inside P when the output scale makes ||nu||_inf > 1 (gauge of the image is min(||nu||_inf, 1) <= 1)."""
    ratio = (A * nu.unsqueeze(1)).sum(-1) / b.clamp_min(1e-12)
    ratio = torch.where(mask, ratio, torch.full_like(ratio, -1e30))
    g = ratio.max(-1).values.clamp_min(1e-9)
    return nu * (nu.abs().max(-1).values.clamp_max(rmax) / g).unsqueeze(-1)


# ---------------------------------------------------------------------------------------------
# Robust Legendre map (fraction-to-boundary safeguard).  Mathematically identical to lsp.layers.legendre_map;
# the safeguard only matters when slacks are tiny (cond(H) ~ 1e14), where a plain damped Newton step computed from
# an inaccurate linear solve can leave the polytope (observed in 7-D with active joint-limit rows).
def _newton_robust(nu, A, b, mask, mu, iters=100, tol=1e-10):
    z = nu / mu
    u = torch.zeros_like(nu)
    maskf = mask.to(nu.dtype)
    eye = torch.eye(nu.shape[-1], dtype=nu.dtype, device=nu.device)
    failed = torch.zeros(nu.shape[0], dtype=torch.bool, device=nu.device)
    for it in range(iters):
        s = torch.where(mask, b - (A * u.unsqueeze(1)).sum(-1), torch.ones_like(b))
        inv = maskf / s
        g = (A * inv.unsqueeze(-1)).sum(1) - z
        H = torch.einsum("bk,bki,bkj->bij", inv * inv, A, A)
        H = H + 1e-14 * H.diagonal(dim1=-2, dim2=-1).amax(-1)[:, None, None] * eye
        sol, info = torch.linalg.solve_ex(H, g.unsqueeze(-1))
        d = -sol.squeeze(-1)
        failed = failed | (info > 0) | ~torch.isfinite(d).all(-1)      # degenerate row sets (e.g. hidden rows -> unbounded polytope)
        d = torch.where(failed.unsqueeze(-1), torch.zeros_like(d), d)
        lam = (-(g * d).sum(-1)).clamp_min(0.0).sqrt()
        t = 1.0 / (1.0 + lam)
        ad = (A * d.unsqueeze(1)).sum(-1)
        tmax = torch.where(mask & (ad > 0), s / ad.clamp_min(1e-300), torch.full_like(s, float("inf"))).min(-1).values
        t = torch.minimum(t, 0.99 * tmax)
        u = u + t.unsqueeze(-1) * d
        if it >= 4 and it % 3 == 0 and float(lam.max()) < tol:
            break
    return torch.where(failed.unsqueeze(-1) | ~torch.isfinite(u).all(-1, keepdim=True), torch.zeros_like(u), u)


class _LegendreRobust(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nu, A, b, mask, mu, straight_through=False):
        with torch.no_grad():
            u = _newton_robust(nu, A, b, mask, mu)
            s = torch.where(mask, b - (A * u.unsqueeze(1)).sum(-1), torch.ones_like(b))
            inv = mask.to(nu.dtype) / s
            H = torch.einsum("bk,bki,bkj->bij", inv * inv, A, A)
            H = H + 1e-14 * H.diagonal(dim1=-2, dim2=-1).amax(-1)[:, None, None] * torch.eye(nu.shape[-1], dtype=nu.dtype)
        ctx.save_for_backward(H)
        ctx.mu, ctx.st = mu, straight_through
        return u

    @staticmethod
    def backward(ctx, grad_u):
        (H,) = ctx.saved_tensors
        if ctx.st:
            return grad_u, None, None, None, None, None
        g = torch.linalg.solve(H, grad_u.unsqueeze(-1)).squeeze(-1) / ctx.mu
        return g, None, None, None, None, None


def legendre_robust(nu, A, b, mask, mu, straight_through=False):
    return _LegendreRobust.apply(nu, A, b, mask, mu, straight_through)
