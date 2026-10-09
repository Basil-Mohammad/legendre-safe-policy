"""Batched single-integrator navigation with run-time-specified circular keep-out zones.

State p in (-1,1)^2, action u (velocity, |u|_inf <= umax), dt = 0.1.
Constraints (obstacle centres/radii and CBF decay rate gamma) are re-sampled every episode and are
visible to the policy ONLY through the observation -> "constraint-conditioned" policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KSLOTS = 8  # maximum number of obstacles representable
NEAR = 3  # nearest obstacles shown to the policy
OBS_DIM = 5 + 4 * NEAR


@dataclass
class EnvCfg:
    kmin: int = 1
    kmax: int = 4
    rmin: float = 0.10
    rmax: float = 0.25
    gmin: float = 0.2
    gmax: float = 0.9
    radius_scale: float = 1.0
    horizon: int = 120
    dt: float = 0.1
    umax: float = 0.5
    goal_radius: float = 0.08
    block_prob: float = 0.5  # prob. that one obstacle is placed between start and goal
    start_margin: float = 0.08
    goal_margin: float = 0.12
    min_goal_dist: float = 0.9
    collision_tol: float = 1e-6  # D < -tol counts as a collision


CONDITIONS = {
    "A": EnvCfg(),
    "B": EnvCfg(kmin=5, kmax=8, rmin=0.08, rmax=0.18),
    "C": EnvCfg(radius_scale=1.5, gmin=0.1, gmax=0.1),
}


class NavEnv:
    def __init__(self, n_envs: int, cfg: EnvCfg, seed: int = 0):
        self.n = n_envs
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        n = n_envs
        self.pos = np.zeros((n, 2))
        self.goal = np.zeros((n, 2))
        self.centers = np.zeros((n, KSLOTS, 2))
        self.radii = np.zeros((n, KSLOTS))
        self.omask = np.zeros((n, KSLOTS), dtype=bool)
        self.gamma = np.zeros(n)
        self.t = np.zeros(n, dtype=int)
        self.reset_idx(np.arange(n))

    def reset_idx(self, idx):
        c = self.cfg
        rng = self.rng
        for i in idx:
            while True:
                K = int(rng.integers(c.kmin, c.kmax + 1))
                start = rng.uniform(-0.95, 0.95, 2)
                goal = rng.uniform(-0.95, 0.95, 2)
                if np.linalg.norm(goal - start) < c.min_goal_dist:
                    continue
                r = rng.uniform(c.rmin, c.rmax, K) * c.radius_scale
                cen = rng.uniform(-0.9, 0.9, (K, 2))
                if rng.random() < c.block_prob:
                    mid = 0.5 * (start + goal)
                    perp = np.array([-(goal - start)[1], (goal - start)[0]])
                    perp /= np.linalg.norm(perp)
                    cen[0] = mid + perp * rng.uniform(-0.08, 0.08)
                ds = np.linalg.norm(cen - start, axis=1) - r
                dg = np.linalg.norm(cen - goal, axis=1) - r
                if (ds > c.start_margin).all() and (dg > c.goal_margin).all():
                    break
            self.pos[i], self.goal[i] = start, goal
            self.centers[i] = 0.0
            self.radii[i] = 0.0
            self.omask[i] = False
            self.centers[i, :K] = cen
            self.radii[i, :K] = r
            self.omask[i, :K] = True
            self.gamma[i] = rng.uniform(c.gmin, c.gmax)
            self.t[i] = 0

    def clearance(self, pos=None):
        """D_i = |p-c_i|^2 - r_i^2 ; +inf for padded slots."""
        pos = self.pos if pos is None else pos
        D = ((pos[:, None, :] - self.centers) ** 2).sum(-1) - self.radii**2
        return np.where(self.omask, D, np.inf)

    def obs(self):
        n = self.n
        o = np.zeros((n, OBS_DIM))
        o[:, 0:2] = self.pos
        o[:, 2:4] = self.goal - self.pos
        o[:, 4] = self.gamma
        dist = np.linalg.norm(self.pos[:, None, :] - self.centers, axis=-1) - self.radii
        dist = np.where(self.omask, dist, np.inf)
        order = np.argsort(dist, axis=1)[:, :NEAR]
        for k in range(NEAR):
            j = order[:, k]
            valid = np.isfinite(dist[np.arange(n), j])
            rel = self.centers[np.arange(n), j] - self.pos
            base = 5 + 4 * k
            o[:, base : base + 2] = np.where(valid[:, None], rel, 0.0)
            o[:, base + 2] = np.where(valid, self.radii[np.arange(n), j], 0.0)
            o[:, base + 3] = valid.astype(float)
        return o

    def cons(self):
        return dict(
            centers=self.centers.copy(),
            radii=self.radii.copy(),
            omask=self.omask.copy(),
            gamma=self.gamma.copy(),
        )

    def step(self, u, collision_penalty=0.0):
        """Advance all envs with executed action u (n,2).

        Returns next_obs (before auto-reset), reward, terminated, truncated, info.
        """
        c = self.cfg
        u = np.clip(u, -c.umax, c.umax)
        d_prev = np.linalg.norm(self.goal - self.pos, axis=1)
        self.pos = np.clip(self.pos + c.dt * u, -1.0, 1.0)
        self.t += 1
        d_new = np.linalg.norm(self.goal - self.pos, axis=1)
        reached = d_new < c.goal_radius
        coll = (self.clearance() < -c.collision_tol).any(axis=1)
        task_r = 5.0 * (d_prev - d_new) + 10.0 * reached
        reward = task_r - collision_penalty * coll
        truncated = (self.t >= c.horizon) & ~reached
        next_obs = self.obs()
        info = dict(reached=reached.copy(), collided=coll.copy(), task_reward=task_r, t=self.t.copy(),
                    next_pos=self.pos.copy())
        done = reached | truncated
        idx = np.where(done)[0]
        info["done_idx"] = idx
        if len(idx):
            self.reset_idx(idx)
        return next_obs, reward, reached, truncated, info
