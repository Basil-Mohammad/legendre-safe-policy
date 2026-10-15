"""TD3 with a pluggable action map.  Only the map a -> u_exec differs between methods."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn

from .env import CONDITIONS, KSLOTS, OBS_DIM, EnvCfg, NavEnv
from .layers import build_polytope, gauge_map, legendre_map, project_polytope_2d

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

METHODS = ("penalty", "se", "sp", "gauge", "legendre", "sp_m09", "sp_m07", "gauge_m09")
KAPPA = {"m09": 0.9, "m07": 0.7}  # margin controls: polytope shrunk to {A u <= kappa b}


def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(OBS_DIM, 2)

    def forward(self, o):
        return torch.tanh(self.net(o))


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1, self.q2 = mlp(OBS_DIM + 2, 1), mlp(OBS_DIM + 2, 1)

    def forward(self, o, a):
        x = torch.cat([o, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(self, o, a):
        return self.q1(torch.cat([o, a], -1)).squeeze(-1)


def to_t(x):
    return torch.as_tensor(x)


class Replay:
    def __init__(self, cap):
        self.cap, self.n, self.p = cap, 0, 0
        f = np.float64
        self.d = dict(
            obs=np.zeros((cap, OBS_DIM), f), nobs=np.zeros((cap, OBS_DIM), f),
            act=np.zeros((cap, 2), f), rew=np.zeros(cap, f), term=np.zeros(cap, f),
            pos=np.zeros((cap, 2), f), npos=np.zeros((cap, 2), f),
            centers=np.zeros((cap, KSLOTS, 2), f), radii=np.zeros((cap, KSLOTS), f),
            omask=np.zeros((cap, KSLOTS), bool), gamma=np.zeros(cap, f),
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


class Agent:
    def __init__(self, method, cfg: EnvCfg, mu=0.05, lr=3e-4, tau=0.005, disc=0.99,
                 noise=0.2, tnoise=0.2, tclip=0.5, delay=2, seed=0, scale=1.0):
        assert method in METHODS
        self.method, self.cfg, self.mu, self.scale = method, cfg, mu, scale
        assert scale == 1.0 or method.split("_m")[0] in ("se", "sp", "gauge"), "scale is a baseline knob"
        self.base = method.split("_m")[0]
        self.kappa = KAPPA[method.split("_")[-1]] if "_m" in method else 1.0
        self.lr, self.tau, self.disc = lr, tau, disc
        self.noise, self.tnoise, self.tclip, self.delay = noise, tnoise, tclip, delay
        torch.manual_seed(seed)
        self.actor, self.critic = Actor(), Critic()
        self.actor_t, self.critic_t = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.oa = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.oc = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.upd = 0

    def poly(self, pos, cons):
        A, b, m = build_polytope(pos, cons["centers"], cons["radii"], cons["omask"], cons["gamma"],
                                 self.cfg.dt, self.cfg.umax)
        if self.kappa < 1.0:
            b = b * self.kappa  # b > 0, so u = 0 stays strictly interior
        return A, b, m

    def layer(self, a, pos, cons):
        """Differentiable map a in [-1,1]^2 -> executed (or nominal for penalty/se) action u."""
        um = self.cfg.umax
        if self.base == "penalty":
            return um * a
        if self.base == "se":
            return um * self.scale * a
        A, b, m = self.poly(pos, cons)
        if self.base == "sp":
            return project_polytope_2d(um * self.scale * a, A, b, m)
        if self.base == "gauge":
            return gauge_map(self.scale * a, A, b, m)
        return legendre_map(a, A, b, m, self.mu)

    def env_filter(self, u_nom, pos, cons):
        """Safeguard applied by the ENVIRONMENT (method 'se' only)."""
        if self.base != "se":
            return u_nom
        A, b, m = self.poly(pos, cons)
        return project_polytope_2d(u_nom, A, b, m)

    def stored(self, a, u_exec):
        """What the replay buffer stores as the action seen by the critic (normalised)."""
        if self.base in ("penalty", "se"):
            return a
        return u_exec / self.cfg.umax

    @torch.no_grad()
    def act(self, obs, pos, cons, explore, rng=None, random_actions=False):
        o, p = to_t(obs), to_t(pos)
        c = {k: to_t(v) for k, v in cons.items()}
        if random_actions:
            a = to_t(rng.uniform(-1, 1, (len(obs), 2)))
        else:
            a = self.actor(o)
            if explore:
                a = (a + self.noise * to_t(rng.standard_normal(a.shape))).clamp(-1, 1)
        u_layer = self.layer(a, p, c)
        u_exec = self.env_filter(u_layer, p, c)
        if self.base in ("gauge", "legendre"):
            u_nom = torch.full_like(u_exec, float("nan"))  # no nominal action defined
        elif self.base == "sp":
            u_nom = self.cfg.umax * self.scale * a
        else:
            u_nom = u_layer
        return dict(a=a.numpy(), u_exec=u_exec.numpy(), u_nom=u_nom.numpy(),
                    stored=self.stored(a, u_exec).numpy())

    def update(self, B):
        o, no, rew, term = B["obs"], B["nobs"], B["rew"], B["term"]
        cons_n = dict(centers=B["centers"], radii=B["radii"], omask=B["omask"], gamma=B["gamma"])
        cons_c = cons_n
        with torch.no_grad():
            a2 = self.actor_t(no)
            a2 = (a2 + (self.tnoise * torch.randn_like(a2)).clamp(-self.tclip, self.tclip)).clamp(-1, 1)
            u2 = self.layer(a2, B["npos"], cons_n)
            ta = self.stored(a2, u2)
            q1t, q2t = self.critic_t(no, ta)
            y = rew + self.disc * (1 - term) * torch.minimum(q1t, q2t)
        q1, q2 = self.critic(o, B["act"])
        lc = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
        self.oc.zero_grad(); lc.backward(); self.oc.step()
        self.upd += 1
        info = dict(lc=float(lc))
        if self.upd % self.delay == 0:
            a = self.actor(o)
            u = self.layer(a, B["pos"], cons_c)
            la = -self.critic.q1_only(o, self.stored(a, u)).mean()
            self.oa.zero_grad(); la.backward(); self.oa.step()
            with torch.no_grad():
                for n_, t_ in ((self.actor, self.actor_t), (self.critic, self.critic_t)):
                    for p_, pt in zip(n_.parameters(), t_.parameters()):
                        pt.mul_(1 - self.tau).add_(self.tau * p_)
            info["la"] = float(la)
        return info


def evaluate(agent: Agent, cfg: EnvCfg, n_episodes=500, seed=12345, jac_samples=0):
    """Deterministic evaluation on ``n_episodes`` independent episodes (run in parallel)."""
    env = NavEnv(n_episodes, cfg, seed=seed)
    n = n_episodes
    alive = np.ones(n, bool)
    reached = np.zeros(n, bool)
    collided = np.zeros(n, bool)
    ncoll = np.zeros(n)
    t_goal = np.full(n, np.nan)
    ret = np.zeros(n)
    interv = np.zeros(n)
    steps = np.zeros(n)
    visited = []
    for t in range(cfg.horizon):
        o = env.obs()
        out = agent.act(o, env.pos.copy(), env.cons(), explore=False)
        if jac_samples and t % 4 == 0:
            visited.append((env.pos.copy(), env.obs(), env.cons(), alive.copy()))
        du = np.linalg.norm(out["u_exec"] - out["u_nom"], axis=1) / cfg.umax
        _, _, term, _, info = env.step(out["u_exec"])
        a = alive
        ret[a] += info["task_reward"][a]
        interv[a] += du[a]
        steps[a] += 1
        coll_now = info["collided"] & a
        collided |= coll_now
        ncoll += coll_now
        newly = term & a
        reached |= newly
        t_goal[newly] = t + 1
        alive = alive & ~term
        if not alive.any():
            break
    res = dict(
        success=float(reached.mean()),
        collision_rate=float(collided.mean()),
        collisions_per_ep=float(ncoll.mean()),
        time_to_goal=float(np.nanmean(t_goal)) if reached.any() else float("nan"),
        task_return=float(ret.mean()),
        intervention=float((interv / np.maximum(steps, 1)).mean()) if agent.base in ("penalty", "se", "sp") else float("nan"),
        deadlock_rate=float((~reached & ~collided).mean()),
    )
    if jac_samples:
        res["jacobian"] = jacobian_diagnostics(agent, visited, jac_samples)
    return res


def jacobian_diagnostics(agent: Agent, visited, max_samples):
    """sigma_min of d u_exec / d a over visited states, in units of umax (unconstrained map = I)."""
    P, O, C = [], [], {k: [] for k in ("centers", "radii", "omask", "gamma")}
    for pos, obs, cons, alive in visited:
        P.append(pos[alive]); O.append(obs[alive])
        for k in C:
            C[k].append(cons[k][alive])
    pos = to_t(np.concatenate(P)); obs = to_t(np.concatenate(O))
    cons = {k: to_t(np.concatenate(v)) for k, v in C.items()}
    if len(pos) > max_samples:
        sel = np.random.default_rng(0).choice(len(pos), max_samples, replace=False)
        pos, obs = pos[sel], obs[sel]
        cons = {k: v[sel] for k, v in cons.items()}
    with torch.no_grad():
        a = agent.actor(obs)
    a = a.clone().requires_grad_(True)
    u = agent.layer(a, pos, cons)
    J = []
    for j in range(2):
        g, = torch.autograd.grad(u[:, j].sum(), a, retain_graph=True)
        J.append(g)
    J = torch.stack(J, 1) / agent.cfg.umax  # (N,2,2)
    sv = torch.linalg.svdvals(J)
    smin, smax = sv[:, 1], sv[:, 0]
    dist = ((pos.unsqueeze(1) - cons["centers"]).pow(2).sum(-1).sqrt() - cons["radii"])
    dist = torch.where(cons["omask"], dist, torch.full_like(dist, 1e9)).min(1).values
    near = dist < 0.15
    return dict(
        n=int(len(smin)),
        frac_rank_deficient=float((smin < 1e-3).double().mean()),
        frac_smin_lt_1e6=float((smin < 1e-6).double().mean()),
        frac_smin_lt_1e2=float((smin < 1e-2).double().mean()),
        frac_rank_deficient_near=float((smin[near] < 1e-3).double().mean()) if near.any() else float("nan"),
        median_log10_smin=float(torch.log10(smin.clamp_min(1e-12)).median()),
        frac_near=float(near.double().mean()),
        median_cond=float((smax / smin.clamp_min(1e-12)).median()),
    )


def train(method, seed, steps=100_000, n_envs=16, utd=0.25, batch=256, warmup=5000, mu=0.05,
          penalty=2.0, scale=1.0, eval_every=10_000, eval_eps=100, log=print):
    rng = np.random.default_rng(seed)
    cfg = CONDITIONS["A"]
    env = NavEnv(n_envs, cfg, seed=seed)
    ag = Agent(method, cfg, mu=mu, seed=seed, scale=scale)
    buf = Replay(steps)
    pen = penalty if method == "penalty" else 0.0  # penalty only for the unfiltered baseline
    curve, t0, total, upd_budget = [], time.time(), 0, 0.0
    ep_coll = 0.0
    while total < steps:
        obs, pos, cons = env.obs(), env.pos.copy(), env.cons()
        out = ag.act(obs, pos, cons, explore=True, rng=rng, random_actions=total < warmup)
        nobs, rew, term, _, info = env.step(out["u_exec"], collision_penalty=pen)
        buf.add(obs=obs, nobs=nobs, act=out["stored"], rew=rew, term=term.astype(float),
                pos=pos, npos=info["next_pos"],
                centers=cons["centers"], radii=cons["radii"], omask=cons["omask"], gamma=cons["gamma"])
        total += n_envs
        ep_coll += float(info["collided"].sum())
        if total >= warmup:
            upd_budget += utd * n_envs
            while upd_budget >= 1.0:
                ag.update(buf.sample(batch, rng))
                upd_budget -= 1.0
        if total % eval_every < n_envs and total >= eval_every:
            r = evaluate(ag, cfg, n_episodes=eval_eps, seed=999 + total)
            r["steps"] = total
            r["train_collision_steps"] = ep_coll
            curve.append(r)
            log(f"[{method} s{seed}] {total:>7d} succ={r['success']:.2f} coll={r['collision_rate']:.2f} "
                f"ttg={r['time_to_goal']:.1f} ({time.time()-t0:.0f}s)")
    return ag, curve
