import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from lsp import di as D
from lsp.di_agent import (METHODS3, DIAgent, env_filter_projection, interior_info, polytope_and_interior,
                          safe_action)
from lsp.di_env import DI_CONDITIONS, DIEnv
from lsp.layers import legendre_map

torch.set_default_dtype(torch.float64)
t = lambda x: torch.as_tensor(x)
SAFE = [m for m in METHODS3 if m != "penalty"]


def _cons(env):
    return {k: t(v) for k, v in env.cons().items() if k != "gamma"}


def test_interior_point_is_strictly_interior_when_flagged():
    env = DIEnv(200, DI_CONDITIONS["A"], seed=3)
    rng = np.random.default_rng(0)
    for _ in range(30):
        c = _cons(env)
        A, b, m, cen, has, strict = polytope_and_interior(t(env.pos), t(env.vel), c)
        slack = torch.where(m, b - torch.einsum("bri,bi->br", A, cen), torch.full_like(b, 9.0)).min(1).values
        assert (slack[strict] > 1e-9).all()
        env.step(rng.uniform(-1, 1, (200, 2)) * 0.5)


def test_straight_through_forward_equal_backward_identity():
    env = DIEnv(32, DI_CONDITIONS["A"], seed=4)
    A, b, m, cen, has, strict = polytope_and_interior(t(env.pos), t(env.vel), _cons(env))
    nu = torch.empty(32, 2).uniform_(-1, 1).requires_grad_(True)
    u = legendre_map(nu, A, b, m, 0.05, cen, False)
    ust = legendre_map(nu, A, b, m, 0.05, cen, True)
    assert torch.allclose(u, ust)
    w = torch.randn(32, 2)
    g, = torch.autograd.grad((ust * w).sum(), nu)
    assert torch.allclose(g[strict], w[strict])


def test_fallback_is_brake_and_constant():
    env = DIEnv(8, DI_CONDITIONS["A"], seed=5)
    env.vel[:] = np.array([0.5, -0.3])
    c = _cons(env)
    # force "no strict interior" by emptying the polytope: obstacle on top of the agent
    c["centers"][:, 0] = t(env.pos)
    c["radii"][:, 0] = 0.2
    c["omask"][:, 0] = True
    nu = torch.zeros(8, 2, requires_grad=True)
    a, fb = safe_action("legendre", nu, t(env.pos), t(env.vel), c, 0.05)
    assert fb.all()
    assert torch.allclose(a, D.brake_action(t(env.vel)))
    g, = torch.autograd.grad(a.sum(), nu, allow_unused=True)
    assert g is None or torch.all(g == 0)


def test_all_safe_methods_zero_collisions_under_adversarial_actions():
    """Property test of the preregistered safety claim (EMPIRICAL for circular obstacles)."""
    rng = np.random.default_rng(1)
    for cond in "ABC":
        for method in SAFE:
            env = DIEnv(64, DI_CONDITIONS[cond], seed=17)
            ag = DIAgent(method, seed=0)
            ncoll = 0
            nfb = 0
            for step in range(150):
                c = _cons(env)
                # adversarial nominal: push towards the nearest obstacle centre half of the time, random otherwise
                d = env.centers - env.pos[:, None, :]
                dist = np.where(env.omask, np.linalg.norm(d, axis=-1), np.inf)
                j = dist.argmin(1)
                towards = d[np.arange(64), j]
                towards = towards / (np.linalg.norm(towards, axis=1, keepdims=True) + 1e-9)
                nu = np.where(rng.random((64, 1)) < 0.5, towards, rng.uniform(-1, 1, (64, 2)))
                nu = np.clip(nu, -1, 1)
                o = ag.act(env.obs(), env.pos.copy(), env.vel.copy(), env.cons(), explore=False,
                           rng=rng, random_actions=False)
                # act() uses the actor; override with the adversarial nu through the same safe map
                p, v = t(env.pos), t(env.vel)
                if ag.base == "se":
                    a_exec, fb = env_filter_projection(D.AMAX * t(nu), p, v, c)
                else:
                    a_exec, fb = safe_action(method, t(nu), p, v, c, 0.05)
                _, _, _, _, info = env.step(a_exec.numpy())
                ncoll += int(info["collided"].sum())
                nfb += int(fb.sum())
            assert ncoll == 0, (cond, method, ncoll, nfb)
