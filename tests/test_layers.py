import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from scipy.optimize import minimize

from lsp.env import CONDITIONS, NavEnv
from lsp.layers import (build_polytope, gauge_map, legendre_jacobian, legendre_map,
                        project_polytope_2d)

torch.set_default_dtype(torch.float64)


def random_polytope(B=64, seed=0, cond="A"):
    env = NavEnv(B, CONDITIONS[cond], seed=seed)
    t = lambda x: torch.as_tensor(x)
    A, b, m = build_polytope(t(env.pos), t(env.centers), t(env.radii), t(env.omask), t(env.gamma),
                             env.cfg.dt, env.cfg.umax)
    return env, A, b, m


def test_origin_strictly_interior():
    for cond in "ABC":
        _, A, b, m = random_polytope(200, 1, cond)
        s = torch.where(m, b, torch.ones_like(b))
        assert (s > 0).all()


def test_legendre_kkt_and_interior():
    _, A, b, m = random_polytope()
    mu = 0.05
    nu = torch.empty(64, 2).uniform_(-1, 1)
    u = legendre_map(nu, A, b, m, mu)
    s = torch.where(m, b - (A * u.unsqueeze(1)).sum(-1), torch.ones_like(b))
    assert (s > 0).all(), "must be strictly interior"
    grad = mu * (A * (m.double() / s).unsqueeze(-1)).sum(1)
    assert torch.allclose(grad, nu, atol=1e-6), float((grad - nu).abs().max())


def test_legendre_gradcheck():
    _, A, b, m = random_polytope(8, 3)
    nu = torch.empty(8, 2).uniform_(-0.9, 0.9).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: legendre_map(x, A, b, m, 0.1), (nu,), atol=1e-5, rtol=1e-4)


def test_legendre_jacobian_spd():
    _, A, b, m = random_polytope(128, 4)
    nu = torch.empty(128, 2).uniform_(-1, 1)
    u = legendre_map(nu, A, b, m, 0.05)
    J = legendre_jacobian(u, A, b, m, 0.05)
    ev = torch.linalg.eigvalsh(0.5 * (J + J.transpose(1, 2)))
    assert (ev > 0).all()


def test_projection_matches_scipy_and_gradient():
    _, A, b, m = random_polytope(16, 5)
    u0 = torch.empty(16, 2).uniform_(-1.5, 1.5)
    u = project_polytope_2d(u0, A, b, m)
    for i in range(16):
        rows = m[i].numpy()
        Ai, bi = A[i].numpy()[rows], b[i].numpy()[rows]
        res = minimize(lambda x: ((x - u0[i].numpy()) ** 2).sum(), np.zeros(2),
                       constraints=[{"type": "ineq", "fun": lambda x: bi - Ai @ x}], method="SLSQP",
                       options=dict(ftol=1e-14, maxiter=200))
        assert np.allclose(res.x, u[i].numpy(), atol=1e-5), (i, res.x, u[i].numpy())
    assert ((A * u.unsqueeze(1)).sum(-1) <= b + 1e-8)[m].all()


def test_projection_jacobian_rank_deficient_at_vertex():
    A = torch.tensor([[[1.0, 0], [-1, 0], [0, 1], [0, -1]]])
    b = torch.ones(1, 4)
    m = torch.ones(1, 4, dtype=torch.bool)
    u0 = torch.tensor([[3.0, 3.0]], requires_grad=True)
    u = project_polytope_2d(u0, A, b, m)
    g, = torch.autograd.grad(u[:, 0].sum(), u0)
    assert torch.allclose(g, torch.zeros_like(g))


def test_gauge_in_polytope():
    _, A, b, m = random_polytope(128, 6)
    nu = torch.empty(128, 2).uniform_(-1, 1)
    u = gauge_map(nu, A, b, m)
    assert ((A * u.unsqueeze(1)).sum(-1) <= b + 1e-9)[m].all()


def test_end_to_end_no_collisions_all_safe_layers():
    """Random actions through each layer for 150 steps: zero collisions."""
    t = lambda x: torch.as_tensor(x)
    rng = np.random.default_rng(0)
    for cond in "ABC":
        for layer in ("sp", "gauge", "legendre"):
            env = NavEnv(64, CONDITIONS[cond], seed=11)
            worst = np.inf
            for step in range(150):
                A, b, m = build_polytope(t(env.pos), t(env.centers), t(env.radii), t(env.omask),
                                         t(env.gamma), env.cfg.dt, env.cfg.umax)
                a = torch.as_tensor(rng.uniform(-1, 1, (64, 2)))
                if layer == "sp":
                    u = project_polytope_2d(env.cfg.umax * a, A, b, m)
                elif layer == "gauge":
                    u = gauge_map(a, A, b, m)
                else:
                    u = legendre_map(a, A, b, m, 0.05)
                _, _, _, _, info = env.step(u.numpy())
                assert not info["collided"].any(), (cond, layer, step)
                worst = min(worst, float(np.min(np.where(np.isfinite(env.clearance()), env.clearance(), 9))))
            assert worst > -1e-6


def test_margin_controls_are_inside_safe_polytope_and_safe():
    """Shrunken-polytope controls (kappa<1) keep u=0 interior and never collide."""
    from lsp.agent import Agent
    t = lambda x: torch.as_tensor(x)
    rng = np.random.default_rng(1)
    for name in ("sp_m09", "sp_m07", "gauge_m09"):
        ag = Agent(name, CONDITIONS["A"], seed=0)
        assert ag.kappa < 1.0 and ag.base in ("sp", "gauge")
        for cond in "ABC":
            env = NavEnv(64, CONDITIONS[cond], seed=21)
            for _ in range(100):
                a = t(rng.uniform(-1, 1, (64, 2)))
                cons = {k: t(v) for k, v in env.cons().items()}
                u = ag.layer(a, t(env.pos), cons)
                A, b, m = build_polytope(t(env.pos), cons["centers"], cons["radii"], cons["omask"],
                                         cons["gamma"], env.cfg.dt, env.cfg.umax)
                assert ((A * u.unsqueeze(1)).sum(-1) <= ag.kappa * b + 1e-8)[m].all()
                _, _, _, _, info = env.step(u.numpy())
                assert not info["collided"].any(), (name, cond)
