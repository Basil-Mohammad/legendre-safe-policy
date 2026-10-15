"""Stable-Baselines3 integration of the action maps (Study 6).

Policy parametrisation (identical for every method):  the network emits a in [-1,1]^m (tanh for SAC; clipped Gaussian
sample for PPO).  A map M_s turns it into an executed normalised action  w = u0 + M(s * a)  that lies in the safe
polytope {A w <= b} (A, b, u0 are read from the observation, double precision, see lsp2.rows).

  sp         Euclidean projection of u0 + s a onto the polytope                (pathwise gradient: exact Jacobian)
  gauge      gauge ("radial") map of s a about u0
  legendre   analytic-centre/Legendre map T_mu(s a), mu = 1 (the output scale s = 1/mu_paper)
  legendre_st  Legendre forward pass, identity backward pass (ablation of the Jacobian)
  se         environment-side projection of the nominal action (SAC: gradient does not see the projection)
  pen        no map, no executor restoration; reward penalty on collisions (lambda in {0.5, 2})

SAC: pathwise maps (sp, gauge, legendre, legendre_st) live inside the actor, so the critic evaluates the *executed*
action and the actor gradient flows through the map.  The entropy term is that of the pre-map tanh-Gaussian (a
documented, method-independent approximation).  se / pen / all PPO variants are environment wrappers (score-function
learners never differentiate the map; consequently sp == se for PPO and `se` is not run with PPO).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.sac.policies import Actor, SACPolicy

from .maps import gauge_nd, legendre_robust, project_nd
from .rows import RowSpec

PATHWISE = ("sp", "gauge", "legendre", "legendre_st")
ENV_SIDE = ("se", "pen")


class MapLayer:
    """Batched map on the interior-shifted polytope.  kind in {sp, gauge, legendre, legendre_st, se, identity}."""

    def __init__(self, spec: RowSpec, kind: str, scale: float):
        self.spec, self.kind, self.scale = spec, kind, float(scale)

    def __call__(self, a: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            return self._call(a, obs)
        finally:
            torch.set_default_dtype(old)

    def _call(self, a, obs):
        dt = a.dtype
        A, b, mask, u0, fb = self.spec.unpack_t(obs.detach())
        a64 = a.to(torch.float64)
        bs = (b - (A * u0.unsqueeze(1)).sum(-1))              # slacks at u0 (shifted polytope contains 0 strictly)
        # invalid polytopes (certified-fallback states, or stale/dropped rows that leave no strict interior at u0): the map is
        # not defined; the batch entry is replaced by a trivial polytope and the executed command is u0.
        bad = fb | ((bs <= 1e-12) & mask).any(-1)
        if bool(bad.any()):
            m_ = A.shape[-1]
            box = torch.cat([torch.eye(m_), -torch.eye(m_)], 0).to(A.dtype)          # trivial bounded polytope |w_i| <= 1
            Ab = torch.zeros_like(A[0]); Ab[: 2 * m_] = box
            mb = torch.zeros_like(mask[0]); mb[: 2 * m_] = True
            A = torch.where(bad[:, None, None], Ab.expand_as(A), A)
            mask = torch.where(bad[:, None], mb.expand_as(mask), mask)
            bs = torch.where(bad[:, None], torch.ones_like(bs), bs)
        bs = torch.where(mask, bs, torch.ones_like(bs))
        nu = self.scale * a64
        k = self.kind
        if k in ("sp", "se"):
            y = project_nd(nu, A, bs, mask)
        elif k == "gauge":
            y = gauge_nd(nu, A, bs, mask)
        elif k in ("legendre", "legendre_st"):
            y = legendre_robust(nu, A, bs, mask, 1.0, straight_through=(k == "legendre_st"))
        elif k in ("identity", "pen"):
            y = nu
        else:
            raise ValueError(k)
        y = torch.where(bad.unsqueeze(-1), torch.zeros_like(y), y)
        w = u0 + y
        return w.to(dt)


class RowFeaturesExtractor(BaseFeaturesExtractor):
    """Bounded policy/critic input: [features | A*mask | slack(u0) clipped | mask | u0 | fb].  Uses only the hi part of the
    row block (float32 is ample for learning; the maps use the exact rows)."""

    def __init__(self, observation_space, spec: RowSpec):
        K, m = spec.K, spec.m
        super().__init__(observation_space, spec.nf + K * m + K + K + m + 1)
        self.spec = spec

    def forward(self, obs):
        s = self.spec
        K, m = s.K, s.m
        o = obs
        feat = o[:, : s.nf]
        h = o[:, s.nf : s.olo]
        A = h[:, : K * m].reshape(-1, K, m)
        b = h[:, K * m : K * m + K]
        mask = h[:, K * m + K : K * m + 2 * K]
        u0 = h[:, K * m + 2 * K : K * m + 2 * K + m]
        fb = h[:, -1:]
        slack = (b - (A * u0.unsqueeze(1)).sum(-1)).clamp(0.0, 4.0) * mask
        return torch.cat([feat, (A * mask.unsqueeze(-1)).reshape(-1, K * m), slack, mask, u0, fb], dim=1)


class MapActor(Actor):
    """SAC actor whose squashed output is passed through a differentiable map (pathwise gradient)."""

    def __init__(self, *args, spec=None, kind="sp", scale=1.0, **kw):
        super().__init__(*args, **kw)
        self._map = MapLayer(spec, kind, scale)
        self._spec, self._kind, self._scale = spec, kind, scale

    def _constructor_args(self):
        d = super()._constructor_args()
        d.update(dict(spec=self._spec, kind=self._kind, scale=self._scale))
        return d

    def forward(self, obs, deterministic: bool = False):
        a = super().forward(obs, deterministic)
        return self._map(a, obs)

    def action_log_prob(self, obs):
        a, logp = super().action_log_prob(obs)
        return self._map(a, obs), logp


class MapSACPolicy(SACPolicy):
    def __init__(self, *args, spec=None, kind="sp", scale=1.0, **kw):
        self._mp = dict(spec=spec, kind=kind, scale=scale)
        super().__init__(*args, **kw)

    def make_actor(self, features_extractor=None):
        kw = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return MapActor(**kw, **self._mp).to(self.device)

    def _get_constructor_parameters(self):
        d = super()._get_constructor_parameters()
        d.update(self._mp)
        return d


class MapVecEnv(VecEnvWrapper):
    """Batched environment-side map for PPO (and SAC-se / penalty).  Receives a in [-1,1]^m, executes w = u0 + M(s a)."""

    def __init__(self, venv, spec: RowSpec, kind: str, scale: float):
        super().__init__(venv)
        self.layer = MapLayer(spec, kind, scale)
        self.last = None

    def reset(self):
        self.last = self.venv.reset()
        return self.last

    def step_async(self, actions):
        o = torch.as_tensor(self.last)
        a = torch.as_tensor(np.clip(actions, -1, 1), dtype=torch.float32)
        w = self.layer(a, o).numpy().astype(np.float64)
        self.venv.step_async(np.clip(w, -1, 1).astype(np.float32))

    def step_wait(self):
        obs, rew, done, infos = self.venv.step_wait()
        self.last = obs
        return obs, rew, done, infos


class PenaltyReward(gym.Wrapper):
    """Penalty baseline: executed w = clip(a) with no feasibility restoration; reward -= lam * 1[collision]."""

    def __init__(self, env, lam):
        super().__init__(env)
        self.lam = lam
        env.unwrapped.enforce = False

    def step(self, a):
        o, r, te, tr, info = self.env.step(a)
        return o, r - self.lam * float(info.get("collided", False)), te, tr, info


class InfoStats(gym.Wrapper):
    """Episode-level safety/performance statistics (collisions, success, fallback steps, restoration counts)."""

    def reset(self, **kw):
        self.c = self.fb = self.n = 0
        self.succ = False
        self.ret = 0.0
        self.viol = 0.0
        u = self.env.unwrapped
        self._r0, self._m0 = u.restorations, u.restoration_mag
        return self.env.reset(**kw)

    def step(self, a):
        o, r, te, tr, info = self.env.step(a)
        self.c += int(info["collided"]); self.fb += int(info["fallback"]); self.n += 1
        self.succ |= bool(info["success"]); self.ret += r
        self.viol = max(self.viol, float(info.get("viol", 0.0)))
        if te or tr:
            u = self.env.unwrapped
            info["episode_stats"] = dict(coll_steps=self.c, collided=self.c > 0, success=self.succ, fb_steps=self.fb,
                                         steps=self.n, ret=self.ret, restorations=u.restorations - self._r0,
                                         restoration_mag=u.restoration_mag - self._m0, viol=self.viol)
        return o, r, te, tr, info
