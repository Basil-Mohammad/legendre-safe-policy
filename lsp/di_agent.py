"""Study 3 agent: TD3 on the double integrator with pluggable safe action maps and a certified fallback.

Every safe method shares one rule (docs/PREREGISTRATION_v2.1.md, "Fallback rule"): if the method's own
construction needs a strictly interior point and none exists (max slack <= 1e-9), or the projection is
infeasible (empty polytope), the executed action is the emergency brake a = clip(-v/dt, +-AMAX).
Fallback steps are counted and reported.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn

from .agent import mlp, to_t  # importing lsp.agent sets the float64 default dtype
from .di import AMAX, brake_action, di_polytope
from .di_env import DI_CONDITIONS, DI_OBS_DIM, DIEnv
from .env import KSLOTS
from .layers import _pairs, legendre_map, project_polytope_2d

METHODS3 = ("penalty", "se", "sp", "gauge", "legendre", "legendre_st", "sp_m09", "sp_m07", "gauge_m09")
KAPPA = {"m09": 0.9, "m07": 0.7}
TOL = 1e-9


def parse(method):
    st = method.endswith("_st")
    base = method.replace("_st", "").split("_m")[0]
    kappa = KAPPA[method.split("_")[-1]] if "_m" in method else 1.0
    return base, kappa, st


# ----------------------------------------------------------------------------------------
# Polytope helpers
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def interior_info(A, b, mask, tol=TOL):
    """Feasible-vertex centroid (a strictly interior point when the polytope is full-dimensional),
    has_vertex (non-empty), strict (max slack > tol) and the slack of the centroid."""
    B, R, _ = A.shape
    I, J = _pairs(R, A.device)
    ai, aj, bi, bj = A[:, I], A[:, J], b[:, I], b[:, J]
    det = ai[..., 0] * aj[..., 1] - ai[..., 1] * aj[..., 0]
    ok = mask[:, I] & mask[:, J] & (det.abs() > 1e-12)
    safe = torch.where(ok, det, torch.ones_like(det))
    vx = (bi * aj[..., 1] - bj * ai[..., 1]) / safe
    vy = (ai[..., 0] * bj - aj[..., 0] * bi) / safe
    V = torch.stack([vx, vy], -1)
    lhs = torch.einsum("bri,bpi->bpr", A, V)
    feas = (((lhs <= b.unsqueeze(1) + tol) | (~mask).unsqueeze(1)).all(-1)) & ok
    cnt = feas.sum(1)
    cen = (V * feas.unsqueeze(-1)).sum(1) / cnt.clamp_min(1).unsqueeze(-1)
    s = b - torch.einsum("bri,bi->br", A, cen)
    s = torch.where(mask, s, torch.full_like(s, float("inf"))).min(1).values
    has = cnt > 0
    return cen, has, has & (s > tol), s


def _dummy(B, R, dtype, device):
    """Harmless polytope (the box [-1,1]^2) used in place of fallback samples to avoid NaN."""
    A = torch.zeros(B, R, 2, dtype=dtype, device=device)
    b = torch.ones(B, R, dtype=dtype, device=device)
    m = torch.zeros(B, R, dtype=torch.bool, device=device)
    A[:, 0, 0], A[:, 1, 0], A[:, 2, 1], A[:, 3, 1] = 1.0, -1.0, 1.0, -1.0
    m[:, :4] = True
    return A, b, m


def gauge_about(nu, A, b, mask, u0):
    """Gauge map relative to the interior point u0: u = u0 + ||nu||_inf * nu / gamma(nu)."""
    s = (b - torch.einsum("bri,bi->br", A, u0)).clamp_min(1e-12)
    ratio = (A * nu.unsqueeze(1)).sum(-1) / s
    ratio = torch.where(mask, ratio, torch.full_like(ratio, -1e30))
    g = ratio.max(-1).values.clamp_min(1e-9)
    return u0 + nu * (nu.abs().max(-1).values / g).unsqueeze(-1)


def polytope_and_interior(p, v, cons):
    A, b, m = di_polytope(p, v, cons["centers"], cons["radii"], cons["omask"])
    cen, has, strict, slack = interior_info(A.detach(), b.detach(), m)
    return A, b, m, cen, has, strict


def safe_action(method, nu, p, v, cons, mu=0.05):
    """Executed acceleration for the safe methods.  Returns (a_exec, fallback_mask).
    Differentiable w.r.t. nu except on fallback samples (constant brake)."""
    base, kappa, st = parse(method)
    A, b, m, cen, has, strict = polytope_and_interior(p, v, cons)
    need_interior = base in ("gauge", "legendre") or kappa < 1.0
    fb = ~(strict if need_interior else has)
    B, R, _ = A.shape
    Ad, bd, md = _dummy(B, R, A.dtype, A.device)
    A = torch.where(fb.view(-1, 1, 1), Ad, A)
    b = torch.where(fb.view(-1, 1), bd, b)
    m = torch.where(fb.view(-1, 1), md, m)
    u0 = torch.where(fb.view(-1, 1), torch.zeros_like(cen), cen)
    if kappa < 1.0:  # shrink the polytope towards the interior point: a.u <= (1-k) a.u0 + k b
        au0 = torch.einsum("bri,bi->br", A, u0)
        b = torch.where(m, (1 - kappa) * au0 + kappa * b, b)
    if base == "sp":
        u = project_polytope_2d(AMAX * nu, A, b, m)
    elif base == "gauge":
        u = gauge_about(nu, A, b, m, u0)
    elif base == "legendre":
        u = legendre_map(nu, A, b, m, mu, u0, st)
    else:
        raise ValueError(method)
    return torch.where(fb.unsqueeze(-1), brake_action(v), u), fb


@torch.no_grad()
def env_filter_projection(a_nom, p, v, cons):
    """Method 'se': the environment projects the nominal action onto the safe polytope."""
    A, b, m, cen, has, strict = polytope_and_interior(p, v, cons)
    B, R, _ = A.shape
    Ad, bd, md = _dummy(B, R, A.dtype, A.device)
    fb = ~has
    A = torch.where(fb.view(-1, 1, 1), Ad, A)
    b = torch.where(fb.view(-1, 1), bd, b)
    m = torch.where(fb.view(-1, 1), md, m)
    u = project_polytope_2d(a_nom, A, b, m)
    return torch.where(fb.unsqueeze(-1), brake_action(v), u), fb


# ----------------------------------------------------------------------------------------
# Learner
# ----------------------------------------------------------------------------------------
class ActorD(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(DI_OBS_DIM, 2)

    def forward(self, o):
        return torch.tanh(self.net(o))


class CriticD(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1, self.q2 = mlp(DI_OBS_DIM + 2, 1), mlp(DI_OBS_DIM + 2, 1)

    def forward(self, o, a):
        x = torch.cat([o, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(self, o, a):
        return self.q1(torch.cat([o, a], -1)).squeeze(-1)


class DIReplay:
    def __init__(self, cap):
        self.cap, self.n, self.p = cap, 0, 0
        f = np.float64
        self.d = dict(
            obs=np.zeros((cap, DI_OBS_DIM), f), nobs=np.zeros((cap, DI_OBS_DIM), f),
            act=np.zeros((cap, 2), f), rew=np.zeros(cap, f), term=np.zeros(cap, f),
            pos=np.zeros((cap, 2), f), vel=np.zeros((cap, 2), f),
            npos=np.zeros((cap, 2), f), nvel=np.zeros((cap, 2), f),
            centers=np.zeros((cap, KSLOTS, 2), f), radii=np.zeros((cap, KSLOTS), f),
            omask=np.zeros((cap, KSLOTS), bool),
        )

    def add(self, **kw):
        n = len(kw["rew"])
        idx = (self.p + np.arange(n)) % self.cap
        for k, v in kw.items():
            self.d[k][idx] = v
        self.p = (self.p + n) % self.cap
        self.n = min(self.n + n, self.cap)

    def sample(self, bs, rng):
        i = rng.integers(0, self.n, bs)
        return {k: to_t(v[i]) for k, v in self.d.items()}


class DIAgent:
    def __init__(self, method, mu=0.05, lr=3e-4, tau=0.005, disc=0.99, noise=0.2, tnoise=0.2,
                 tclip=0.5, delay=2, seed=0):
        assert method in METHODS3, method
        self.method, self.mu = method, mu
        self.base, self.kappa, self.st = parse(method)
        self.tau, self.disc, self.noise, self.tnoise, self.tclip, self.delay = tau, disc, noise, tnoise, tclip, delay
        torch.manual_seed(seed)
        self.actor, self.critic = ActorD(), CriticD()
        self.actor_t, self.critic_t = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.oa = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.oc = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.upd = 0

    def critic_action(self, nu, p, v, cons):
        """Differentiable action as seen by the critic (normalised by AMAX); also the fallback mask."""
        if self.base in ("penalty", "se"):
            return nu, torch.zeros(len(nu), dtype=torch.bool)
        a, fb = safe_action(self.method, nu, p, v, cons, self.mu)
        return a / AMAX, fb

    @torch.no_grad()
    def act(self, obs, pos, vel, cons, explore, rng=None, random_actions=False):
        o, p, v = to_t(obs), to_t(pos), to_t(vel)
        c = {k: to_t(x) for k, x in cons.items()}
        if random_actions:
            nu = to_t(rng.uniform(-1, 1, (len(obs), 2)))
        else:
            nu = self.actor(o)
            if explore:
                nu = (nu + self.noise * to_t(rng.standard_normal(nu.shape))).clamp(-1, 1)
        if self.base == "penalty":
            a_exec, fb, a_nom = AMAX * nu, torch.zeros(len(nu), dtype=torch.bool), AMAX * nu
        elif self.base == "se":
            a_nom = AMAX * nu
            a_exec, fb = env_filter_projection(a_nom, p, v, c)
        else:
            a_exec, fb = safe_action(self.method, nu, p, v, c, self.mu)
            a_nom = AMAX * nu if self.base == "sp" else torch.full_like(a_exec, float("nan"))
        stored = nu if self.base in ("penalty", "se") else a_exec / AMAX
        return dict(nu=nu.numpy(), a_exec=a_exec.numpy(), a_nom=a_nom.numpy(), stored=stored.numpy(),
                    fallback=fb.numpy())

    def update(self, B):
        o, no, rew, term = B["obs"], B["nobs"], B["rew"], B["term"]
        cons = dict(centers=B["centers"], radii=B["radii"], omask=B["omask"])
        with torch.no_grad():
            n2 = self.actor_t(no)
            n2 = (n2 + (self.tnoise * torch.randn_like(n2)).clamp(-self.tclip, self.tclip)).clamp(-1, 1)
            ta, _ = self.critic_action(n2, B["npos"], B["nvel"], cons)
            q1t, q2t = self.critic_t(no, ta)
            y = rew + self.disc * (1 - term) * torch.minimum(q1t, q2t)
        q1, q2 = self.critic(o, B["act"])
        lc = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
        self.oc.zero_grad(); lc.backward(); self.oc.step()
        self.upd += 1
        if self.upd % self.delay == 0:
            nu = self.actor(o)
            a, _ = self.critic_action(nu, B["pos"], B["vel"], cons)
            la = -self.critic.q1_only(o, a).mean()
            self.oa.zero_grad(); la.backward(); self.oa.step()
            with torch.no_grad():
                for n_, t_ in ((self.actor, self.actor_t), (self.critic, self.critic_t)):
                    for p_, pt in zip(n_.parameters(), t_.parameters()):
                        pt.mul_(1 - self.tau).add_(self.tau * p_)
        return float(lc)


# ----------------------------------------------------------------------------------------
# Evaluation and diagnostics
# ----------------------------------------------------------------------------------------
def jacobian_diag(agent, visited, max_samples):
    """sigma_min of d a_exec / d nu (in units of AMAX) over visited NON-fallback states."""
    P, V, O = [], [], []
    C = {k: [] for k in ("centers", "radii", "omask")}
    for pos, vel, obs, cons, alive in visited:
        P.append(pos[alive]); V.append(vel[alive]); O.append(obs[alive])
        for k in C:
            C[k].append(cons[k][alive])
    pos, vel, obs = to_t(np.concatenate(P)), to_t(np.concatenate(V)), to_t(np.concatenate(O))
    cons = {k: to_t(np.concatenate(x)) for k, x in C.items()}
    if len(pos) > max_samples:
        sel = np.random.default_rng(0).choice(len(pos), max_samples, replace=False)
        pos, vel, obs = pos[sel], vel[sel], obs[sel]
        cons = {k: x[sel] for k, x in cons.items()}
    with torch.no_grad():
        nu = agent.actor(obs)
    nu = nu.clone().requires_grad_(True)
    a, fb = agent.critic_action(nu, pos, vel, cons)  # identity for penalty/se, as in Study 1/2
    J = torch.stack([torch.autograd.grad(a[:, j].sum(), nu, retain_graph=True)[0] for j in range(2)], 1)
    keep = ~fb
    if not keep.any():
        return dict(n=0, frac_rank_deficient=float("nan"), median_log10_smin=float("nan"))
    sv = torch.linalg.svdvals(J[keep])
    smin = sv[:, 1]
    return dict(n=int(keep.sum()), n_all=int(len(fb)),
                frac_rank_deficient=float((smin < 1e-3).double().mean()),
                frac_smin_lt_1e6=float((smin < 1e-6).double().mean()),
                frac_smin_lt_1e2=float((smin < 1e-2).double().mean()),
                median_log10_smin=float(torch.log10(smin.clamp_min(1e-12)).median()))


def evaluate(agent, cfg, n_episodes=500, seed=12345, jac_samples=0):
    env = DIEnv(n_episodes, cfg, seed=seed)
    n = n_episodes
    alive = np.ones(n, bool)
    reached = np.zeros(n, bool)
    collided = np.zeros(n, bool)
    t_goal = np.full(n, np.nan)
    steps = np.zeros(n)
    fb_steps = np.zeros(n)
    interv = np.zeros(n)
    visited = []
    for t in range(cfg.horizon):
        obs = env.obs()
        out = agent.act(obs, env.pos.copy(), env.vel.copy(), env.cons(), explore=False)
        if jac_samples and t % 5 == 0:
            visited.append((env.pos.copy(), env.vel.copy(), obs, env.cons(), alive.copy()))
        du = np.linalg.norm(out["a_exec"] - out["a_nom"], axis=1) / AMAX
        _, _, term, _, info = env.step(out["a_exec"])
        a = alive
        steps[a] += 1
        fb_steps[a] += out["fallback"][a]
        interv[a] += du[a]
        collided |= info["collided"] & a
        newly = term & a
        reached |= newly
        t_goal[newly] = t + 1
        alive = alive & ~term
        if not alive.any():
            break
    res = dict(
        success=float(reached.mean()),
        collision_rate=float(collided.mean()),
        time_to_goal=float(np.nanmean(t_goal)) if reached.any() else float("nan"),
        deadlock_rate=float((~reached & ~collided).mean()),
        fallback_step_rate=float(fb_steps.sum() / max(steps.sum(), 1)),
        frac_episodes_using_fallback=float((fb_steps > 0).mean()),
        intervention=float((interv / np.maximum(steps, 1)).mean()) if agent.base in ("se", "sp") else float("nan"),
    )
    if jac_samples:
        res["jacobian"] = jacobian_diag(agent, visited, jac_samples)
    return res


def train(method, seed, steps=100_000, n_envs=16, utd=0.25, batch=256, warmup=5000, mu=0.05,
          penalty=2.0, eval_every=10_000, eval_eps=100, log=print):
    rng = np.random.default_rng(seed)
    cfg = DI_CONDITIONS["A"]
    env = DIEnv(n_envs, cfg, seed=seed)
    ag = DIAgent(method, mu=mu, seed=seed)
    buf = DIReplay(steps)
    pen = penalty if method == "penalty" else 0.0
    curve, t0, total, upd_budget = [], time.time(), 0, 0.0
    while total < steps:
        obs, pos, vel, cons = env.obs(), env.pos.copy(), env.vel.copy(), env.cons()
        out = ag.act(obs, pos, vel, cons, explore=True, rng=rng, random_actions=total < warmup)
        nobs, rew, term, _, info = env.step(out["a_exec"], collision_penalty=pen)
        buf.add(obs=obs, nobs=nobs, act=out["stored"], rew=rew, term=term.astype(float), pos=pos, vel=vel,
                npos=info["next_pos"], nvel=info["next_vel"],
                centers=cons["centers"], radii=cons["radii"], omask=cons["omask"])
        total += n_envs
        if total >= warmup:
            upd_budget += utd * n_envs
            while upd_budget >= 1.0:
                ag.update(buf.sample(batch, rng))
                upd_budget -= 1.0
        if total % eval_every < n_envs and total >= eval_every:
            r = evaluate(ag, cfg, n_episodes=eval_eps, seed=999 + total)
            r["steps"] = total
            curve.append(r)
            log(f"[{method} s{seed}] {total:>7d} succ={r['success']:.2f} coll={r['collision_rate']:.2f} "
                f"fb={r['fallback_step_rate']:.4f} ttg={r['time_to_goal']:.1f} ({time.time()-t0:.0f}s)")
    return ag, curve
