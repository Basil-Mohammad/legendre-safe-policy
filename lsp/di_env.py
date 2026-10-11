"""Study 3 environment: exact zero-order-hold double integrator with run-time circular keep-out zones.

Same arena, start/goal/obstacle sampling and conditions A/B/C as the single integrator (lsp.env),
but the agent controls ACCELERATION and starts at rest.  See docs/PREREGISTRATION_v2.1.md and
docs/AMENDMENTS_v2.1.md for the exact rules (arena clipping, observation, collision definition).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import di as D
from .env import CONDITIONS, OBS_DIM, NavEnv

DI_OBS_DIM = OBS_DIM + 2
DI_CONDITIONS = {k: replace(v, horizon=150) for k, v in CONDITIONS.items()}


class DIEnv(NavEnv):
    def __init__(self, n_envs, cfg, seed=0):
        self.vel = np.zeros((n_envs, 2))
        super().__init__(n_envs, cfg, seed)

    def reset_idx(self, idx):
        super().reset_idx(idx)
        self.vel[idx] = 0.0

    def obs(self):
        o = super().obs()
        o[:, 4] = 0.0  # the CBF decay rate gamma of Study 1/2 is unused in Study 3
        return np.concatenate([o, self.vel], 1)

    def step(self, a, collision_penalty=0.0):
        """Advance with acceleration a (n,2).  Returns next_obs, reward, terminated, truncated, info."""
        c = self.cfg
        a = np.clip(a, -D.AMAX, D.AMAX)
        v_new = np.clip(self.vel + D.DT * a, -D.VMAX, D.VMAX)
        a_eff = (v_new - self.vel) / D.DT  # consistent with velocity clipping (no-op for safe actions)
        d_prev = np.linalg.norm(self.goal - self.pos, axis=1)
        p_new = self.pos + D.DT * self.vel + 0.5 * D.DT**2 * a_eff
        out_wall = (np.abs(p_new) > D.ARENA + 1e-9).any(axis=1)
        coll = (self.clearance(p_new) < -c.collision_tol).any(axis=1) | out_wall
        clipped = np.abs(p_new) > D.ARENA
        self.pos = np.clip(p_new, -D.ARENA, D.ARENA)
        self.vel = np.where(clipped, 0.0, v_new)
        self.t += 1
        d_new = np.linalg.norm(self.goal - self.pos, axis=1)
        reached = d_new < c.goal_radius
        task_r = 5.0 * (d_prev - d_new) + 10.0 * reached
        reward = task_r - collision_penalty * coll
        truncated = (self.t >= c.horizon) & ~reached
        next_obs = self.obs()
        info = dict(reached=reached.copy(), collided=coll.copy(), task_reward=task_r, t=self.t.copy(),
                    next_pos=self.pos.copy(), next_vel=self.vel.copy())
        idx = np.where(reached | truncated)[0]
        info["done_idx"] = idx
        if len(idx):
            self.reset_idx(idx)
        return next_obs, reward, reached, truncated, info
