"""Study-6 environments (gymnasium).  All expose the row-observation convention of lsp2.rows.

NavSI / NavDI : wrappers of the validated batched environments of Studies 1-5 (relative degree 1 / 2, m = 2).
Unicycle      : MuJoCo differential-drive style base (v, omega) with a look-ahead CBF point (m = 2).
FetchSafe     : Gymnasium-Robotics FetchReach-v4, Cartesian end-effector displacement with keep-out spheres (m = 3).
PandaSafe     : MuJoCo Menagerie Franka Panda, joint-velocity commands with keep-out spheres (m = 7).
"""
from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from lsp import di as DI
from lsp.di_env import DI_CONDITIONS, DIEnv
from lsp.env import CONDITIONS, KSLOTS, NavEnv
from lsp.layers import build_polytope
from lsp.di_agent import interior_info
from .rows import RowSpec, normalise_rows



def f64(fn):
    """Run fn under float64 default dtype (the env/row/map code is double precision; SB3 networks are float32)."""
    import functools

    @functools.wraps(fn)
    def w(*a, **k):
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            return fn(*a, **k)
        finally:
            torch.set_default_dtype(old)
    return w


B_FLOOR = 1e-10


class RowEnv(gym.Env):
    """Base class: action = executed normalised action w in [-1,1]^m (already mapped by the policy / wrapper)."""
    metadata = {"render_modes": []}

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        for n in ("__init__", "reset", "step", "_obs", "_rows"):
            if n in cls.__dict__:
                setattr(cls, n, f64(cls.__dict__[n]))

    def __init__(self, spec: RowSpec):
        self.spec = spec
        self.observation_space = spaces.Box(-np.inf, np.inf, (spec.dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (spec.m,), np.float32)
        self.fault = None  # optional fault injector (lsp2.faults)
        self.enforce = True  # False only for the penalty baseline (no executor-side feasibility restoration)
        self._rows64 = None   # exact double-precision rows of the current state (A, b, mask, u0)
        self.restorations = 0
        self.restoration_mag = 0.0

    def _set_rows(self, A, b, ok, u0):
        self._rows64 = (np.asarray(A, np.float64), np.asarray(b, np.float64), np.asarray(ok, bool), np.asarray(u0, np.float64))

    def restore(self, w):
        """Double-precision executor check: the policy sees float32 rows; if its action violates an exact row by
        rounding, shrink it towards the interior point u0 (counted and reported; identical for every map)."""
        if self._rows64 is None or not self.enforce:
            return w
        A, b, ok, u0 = self._rows64
        slack0 = b - A @ u0
        d = w - u0
        ad = A @ d
        bad = ok & (ad > slack0 * (1.0 - 1e-9))
        if not bad.any():
            return w
        theta = np.min(np.where(bad, slack0 * (1.0 - 1e-9) / np.maximum(ad, 1e-300), 1.0))
        theta = float(np.clip(theta, 0.0, 1.0))
        self.restorations += 1
        self.restoration_mag += 1.0 - theta
        return u0 + theta * d


# ---------------------------------------------------------------------------------------------
class NavSI(RowEnv):
    """Single integrator (velocity command), exact CBF rows (relative degree one)."""

    def __init__(self, cond="A", seed=0):
        self.cfg = CONDITIONS[cond]
        self.core = NavEnv(1, self.cfg, seed=seed)
        self.K = KSLOTS + 4
        super().__init__(RowSpec(17, self.K, 2))

    def _obs(self):
        e, c = self.core, self.cfg
        t = lambda x: torch.as_tensor(x)
        A, b, m = build_polytope(t(e.pos), t(e.centers), t(e.radii), t(e.omask), t(e.gamma), c.dt, c.umax)
        A, b, m = A[0].numpy() * c.umax, b[0].numpy(), m[0].numpy()   # normalised action space w = u / umax
        A, b, ok = normalise_rows(A, b, m)
        self._set_rows(A, b, ok, np.zeros(2))
        return self.spec.pack(e.obs()[0], A, b, ok, np.zeros(2), 0.0)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.core.rng = np.random.default_rng(seed)
        self.core.reset_idx(np.array([0]))
        return self._obs(), {}

    def step(self, w):
        e, c = self.core, self.cfg
        u = np.clip(self.restore(np.clip(np.asarray(w, dtype=np.float64), -1, 1)), -1, 1)[None] * c.umax
        _, rew, term, trunc, info = e.step(u)           # the core auto-resets; the returned obs is of the next state
        return self._obs(), float(rew[0]), bool(term[0]), bool(trunc[0]), dict(
            collided=bool(info["collided"][0]), success=bool(info["reached"][0]), fallback=False)


class NavDI(RowEnv):
    """Double integrator (acceleration command), braking-aware rows, certified brake fallback (relative degree two)."""

    def __init__(self, cond="A", seed=0):
        self.cfg = DI_CONDITIONS[cond]
        self.core = DIEnv(1, self.cfg, seed=seed)
        self.K = KSLOTS + 8
        super().__init__(RowSpec(19, self.K, 2))

    def _obs(self):
        e = self.core
        t = lambda x: torch.as_tensor(x)
        A, b, m = DI.di_polytope(t(e.pos), t(e.vel), t(e.centers), t(e.radii), t(e.omask))
        cen, has, strict, _ = interior_info(A, b, m)
        A, b, m = A[0].numpy() * DI.AMAX, b[0].numpy(), m[0].numpy()        # w = a / AMAX
        A, b, ok = normalise_rows(A, b, m, floor=None)
        u0 = cen[0].numpy() / DI.AMAX
        u0 = u0 if bool(strict[0]) else np.zeros(2)
        self._set_rows(A, b, ok, u0)
        return self.spec.pack(e.obs()[0], A, b, ok, u0, 0.0 if bool(strict[0]) else 1.0)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.core.rng = np.random.default_rng(seed)
        self.core.reset_idx(np.array([0]))
        o = self._obs()
        self._cur_fb = bool(o[self.spec.ofb_index] > 0.5)
        return o, {}

    def step(self, w):
        e = self.core
        if self._cur_fb:   # certified fallback: emergency brake (policy output ignored)
            a = DI.brake_action(torch.as_tensor(e.vel)).numpy()
        else:
            a = np.clip(self.restore(np.clip(np.asarray(w, dtype=np.float64), -1, 1)), -1, 1)[None] * DI.AMAX
        fb_used = self._cur_fb
        _, rew, term, trunc, info = e.step(a)
        o = self._obs()
        self._cur_fb = bool(o[self.spec.ofb_index] > 0.5)
        return o, float(rew[0]), bool(term[0]), bool(trunc[0]), dict(
            collided=bool(info["collided"][0]), success=bool(info["reached"][0]), fallback=fb_used)


# ---------------------------------------------------------------------------------------------
UNICYCLE_XML = """
<mujoco model="unicycle">
  <option timestep="0.005" gravity="0 0 0" integrator="implicitfast"/>
  <worldbody>
    <body name="base" pos="0 0 0.05">
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="y" type="slide" axis="0 1 0"/>
      <joint name="th" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.03" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="vx" joint="x" kv="200"/>
    <velocity name="vy" joint="y" kv="200"/>
    <velocity name="wz" joint="th" kv="20"/>
  </actuator>
</mujoco>
"""


class Unicycle(RowEnv):
    """MuJoCo differential-drive style base commanded in (v, omega) as a ROS cmd_vel.

    Safety rows act on a look-ahead point p_l = p + l (cos th, sin th):  p_l_dot = B(th) (v, omega),
    B = [[cos th, -l sin th], [sin th, l cos th]].  Keep-out discs are inflated by the body radius rho and the
    offset l, so that |p_l - c| >= r + rho + l implies |p - c| >= r + rho.  The discrete-time CBF row
    -2 dt (p_l - c)^T B u <= gamma (|p_l - c|^2 - R^2) is exact for a frozen heading and an ideal velocity servo;
    heading rotation and servo lag are covered by a margin DELTA, so safety is empirical (reported).
    """
    DT, V_MAX, W_MAX, RHO, L, DELTA, GW = 0.05, 0.4, 2.0, 0.05, 0.05, 0.02, 0.5
    NEAR = 3

    def __init__(self, cond="A", seed=0):
        import mujoco
        from dataclasses import replace
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(UNICYCLE_XML)
        self.data = mujoco.MjData(self.model)
        self.nsub = int(round(self.DT / self.model.opt.timestep))
        self.cfg = replace(CONDITIONS[cond], horizon=150, start_margin=0.2, goal_margin=0.25)
        self.core = NavEnv(1, self.cfg, seed=seed)       # used only to sample start/goal/obstacles
        self.K = KSLOTS + 8
        super().__init__(RowSpec(5 + 2 + 4 * self.NEAR + 2, self.K, 2))
        self.collisions = 0

    # -- helpers ------------------------------------------------------------------------------
    def _state(self):
        d = self.data
        return d.qpos[0], d.qpos[1], d.qpos[2]

    def _sample(self):
        e = self.core
        lim = 1.0 - self.RHO - self.L - 0.05
        while True:
            e.reset_idx(np.array([0]))
            if (np.abs(e.pos[0]) < lim).all() and (np.abs(e.goal[0]) < lim).all():
                return

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.core.rng = np.random.default_rng(seed)
        self._sample()
        e = self.core
        self.mj.mj_resetData(self.model, self.data)
        th0 = float(self.core.rng.uniform(-np.pi, np.pi))
        self.data.qpos[:] = [e.pos[0, 0], e.pos[0, 1], th0]
        self.mj.mj_forward(self.model, self.data)
        self.t = 0
        return self._obs(), {}

    def _rows(self):
        e, l = self.core, self.L
        x, y, th = self._state()
        c_, s_ = np.cos(th), np.sin(th)
        pl = np.array([x + l * c_, y + l * s_])
        Bm = np.array([[c_, -l * s_], [s_, l * c_]])
        diff = pl[None] - e.centers[0]
        R = e.radii[0] + self.RHO + l + self.DELTA
        h = (diff ** 2).sum(-1) - R ** 2
        gam = e.gamma[0]
        Au = -2.0 * self.DT * (diff @ Bm)                      # (K, 2) rows on u = (v, omega)
        bu = gam * h
        # walls on p_l (box |p_l,j| <= 1 - rho - l - delta)
        lim = 1.0 - self.RHO - l - self.DELTA
        hw = np.array([lim - pl[0], lim + pl[0], lim - pl[1], lim + pl[1]])
        sgn = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float64)
        Aw = self.DT * (sgn @ Bm)
        bw = self.GW * hw
        # actuation box
        Ab = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float64)
        bb = np.array([self.V_MAX, self.V_MAX, self.W_MAX, self.W_MAX])
        A = np.concatenate([np.where(e.omask[0][:, None], Au, 0.0), Aw, Ab], 0)
        b = np.concatenate([np.where(e.omask[0], bu, 1.0), bw, bb], 0)
        m = np.concatenate([e.omask[0], np.ones(8, bool)], 0)
        A = A * np.array([self.V_MAX, self.W_MAX])[None]        # normalised action space w = u / (V_MAX, W_MAX)
        A, b, ok = normalise_rows(A, b, m)
        return A, b, ok

    def _obs(self):
        e = self.core
        x, y, th = self._state()
        c_, s_ = np.cos(th), np.sin(th)
        p = np.array([x, y])
        g = e.goal[0] - p
        rg = np.array([c_ * g[0] + s_ * g[1], -s_ * g[0] + c_ * g[1]])
        vel = self.data.qvel
        vf = c_ * vel[0] + s_ * vel[1]
        feat = np.zeros(self.spec.nf)
        feat[0:2] = p
        feat[2:4] = [c_, s_]
        feat[4] = e.gamma[0]
        feat[5:7] = rg
        feat[7:9] = [vf, vel[2]]
        dist = np.linalg.norm(p[None] - e.centers[0], axis=-1) - e.radii[0]
        dist = np.where(e.omask[0], dist, np.inf)
        order = np.argsort(dist)[: self.NEAR]
        for k, j in enumerate(order):
            ok = np.isfinite(dist[j])
            rel = e.centers[0, j] - p
            rr = np.array([c_ * rel[0] + s_ * rel[1], -s_ * rel[0] + c_ * rel[1]])
            base = 9 + 4 * k
            feat[base: base + 2] = rr if ok else 0.0
            feat[base + 2] = e.radii[0, j] if ok else 0.0
            feat[base + 3] = float(ok)
        feat = feat[: self.spec.nf]
        A, b, ok = self._rows()
        strict = bool(np.all(b[ok] > 0.0))
        self._set_rows(A, b, ok, np.zeros(2))
        self._cur_fb = not strict
        return self.spec.pack(feat, A, b, ok, np.zeros(2), 0.0 if strict else 1.0)

    def step(self, w):
        e, d = self.core, self.data
        w = np.clip(np.asarray(w, dtype=np.float64), -1, 1)
        fb_used = self._cur_fb
        w = np.zeros(2) if fb_used else self.restore(w)
        v, om = w[0] * self.V_MAX, w[1] * self.W_MAX
        x0, y0, _ = self._state()
        d_prev = np.linalg.norm(e.goal[0] - np.array([x0, y0]))
        for _ in range(self.nsub):
            th = d.qpos[2]
            d.ctrl[:] = [v * np.cos(th), v * np.sin(th), om]
            self.mj.mj_step(self.model, d)
        self.t += 1
        x, y, th = self._state()
        p = np.array([x, y])
        d_new = np.linalg.norm(e.goal[0] - p)
        reached = d_new < self.cfg.goal_radius
        dist = np.linalg.norm(p[None] - e.centers[0], axis=-1) - e.radii[0] - self.RHO
        coll = bool(((dist < -1e-6) & e.omask[0]).any() or (np.abs(p) > 1.0 - self.RHO + 1e-6).any())
        self.collisions += int(coll)
        rew = 5.0 * (d_prev - d_new) + 10.0 * reached
        trunc = bool(self.t >= self.cfg.horizon and not reached)
        info = dict(collided=coll, success=bool(reached), fallback=fb_used)
        return self._obs(), float(rew), bool(reached), trunc, info


# ---------------------------------------------------------------------------------------------
class FetchSafe(RowEnv):
    """Gymnasium-Robotics FetchReach-v4 (standard benchmark) with run-time spherical keep-out zones.

    The agent commands the Cartesian end-effector displacement d = 0.05 w (w in [-1,1]^3), exactly the action
    semantics of FetchReach (the gripper command is held at 0).  For a sphere i with centre c_i and inflated radius
    R_i = r_i + rho + delta the discrete-time CBF row  -2 (p - c_i)^T d <= gamma (|p-c_i|^2 - R_i^2)  is exact when
    the end effector reaches the mocap target; tracking lag is covered by delta (calibrated, see PREREGISTRATION_v5).
    Workspace walls (a box around the start pose) are linear rows.
    """
    DSTEP, RHO, DELTA, GW, HALF, NEAR = 0.05, 0.015, 0.012, 0.5, 0.22, 3
    KMAX = 6

    def __init__(self, cond="A", seed=0):
        import gymnasium_robotics  # noqa: F401
        gym.register_envs(gymnasium_robotics)
        self.env = gym.make("FetchReach-v4", reward_type="dense", max_episode_steps=10_000).unwrapped
        self.cond = cond
        self.rng = np.random.default_rng(seed)
        self.kmin, self.kmax = {"A": (1, 3), "B": (4, 6), "C": (1, 3)}[cond]
        self.rmin, self.rmax = (0.03, 0.06) if cond != "B" else (0.025, 0.05)
        self.rscale = 1.5 if cond == "C" else 1.0
        self.gmin = self.gmax = 0.1 if cond == "C" else None
        self.horizon = 80
        self.K = self.KMAX + 6 + 6        # sphere slots + 6 walls + 6 box rows
        super().__init__(RowSpec(10 + 3 + 1 + 5 * self.NEAR, self.K, 3))
        self.collisions = 0

    def _sample_obstacles(self, start, goal):
        rng = self.rng
        while True:
            K = int(rng.integers(self.kmin, self.kmax + 1))
            r = rng.uniform(self.rmin, self.rmax, K) * self.rscale
            cen = start + rng.uniform(-self.HALF + 0.03, self.HALF - 0.03, (K, 3))
            if rng.random() < 0.5:
                cen[0] = 0.5 * (start + goal) + rng.uniform(-0.02, 0.02, 3)
            ds = np.linalg.norm(cen - start, axis=1) - r - self.RHO - self.DELTA
            dg = np.linalg.norm(cen - goal, axis=1) - r - 0.05 - self.RHO
            if (ds > 0.03).all() and (dg > 0.02).all():
                break
        self.centers = np.zeros((self.KMAX, 3)); self.radii = np.zeros(self.KMAX); self.omask = np.zeros(self.KMAX, bool)
        self.centers[:K], self.radii[:K], self.omask[:K] = cen, r, True
        self.gamma = float(rng.uniform(0.2, 0.9)) if self.gmin is None else 0.1

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        o, _ = self.env.reset(seed=int(self.rng.integers(2**31)))
        self.raw = o
        self.start = o["achieved_goal"].copy()
        self.goal = o["desired_goal"].copy()
        self._sample_obstacles(self.start, self.goal)
        self.t = 0
        return self._obs(), {}

    def _rows(self):
        p = self.raw["achieved_goal"]
        diff = p[None] - self.centers
        R = self.radii + self.RHO + self.DELTA
        h = (diff ** 2).sum(-1) - R ** 2
        Au = -2.0 * self.DSTEP * diff
        bu = self.gamma * h
        lim = self.HALF - self.RHO - self.DELTA
        lo, hi = self.start - lim, self.start + lim
        sg = np.concatenate([np.eye(3), -np.eye(3)], 0)
        hw = np.concatenate([hi - p, p - lo])
        Aw = self.DSTEP * sg
        bw = self.GW * hw
        Ab = sg
        bb = np.ones(6)
        A = np.concatenate([np.where(self.omask[:, None], Au, 0.0), Aw, Ab], 0)
        b = np.concatenate([np.where(self.omask, bu, 1.0), bw, bb], 0)
        m = np.concatenate([self.omask, np.ones(12, bool)], 0)
        A = np.concatenate([A[: self.KMAX], A[self.KMAX :]], 0)
        # w-space: Au above is already per unit of d = DSTEP*w -> multiply obstacle rows by 1 (d = DSTEP w)
        A[: self.KMAX] = np.where(self.omask[:, None], -2.0 * self.DSTEP * diff, 0.0)
        return normalise_rows(A, b, m)

    def _obs(self):
        o = self.raw
        p = o["achieved_goal"]
        feat = np.zeros(self.spec.nf)
        feat[:10] = o["observation"]
        feat[10:13] = o["desired_goal"] - p
        feat[13] = self.gamma
        dist = np.linalg.norm(p[None] - self.centers, axis=-1) - self.radii
        dist = np.where(self.omask, dist, np.inf)
        for k, j in enumerate(np.argsort(dist)[: self.NEAR]):
            ok = np.isfinite(dist[j])
            base = 14 + 5 * k
            feat[base : base + 3] = (self.centers[j] - p) if ok else 0.0
            feat[base + 3] = self.radii[j] if ok else 0.0
            feat[base + 4] = float(ok)
        A, b, ok = self._rows()
        strict = bool(np.all(b[ok] > 0.0))
        self._set_rows(A, b, ok, np.zeros(3))
        self._cur_fb = not strict
        return self.spec.pack(feat, A, b, ok, np.zeros(3), 0.0 if strict else 1.0)

    def step(self, w):
        w = np.clip(np.asarray(w, dtype=np.float64), -1, 1)
        fb_used = self._cur_fb
        w = np.zeros(3) if fb_used else self.restore(w)
        p0 = self.raw["achieved_goal"].copy()
        d_prev = np.linalg.norm(self.goal - p0)
        o, _, _, _, _ = self.env.step(np.concatenate([w, [0.0]]).astype(np.float32))
        self.raw = o
        self.t += 1
        p = o["achieved_goal"]
        d_new = np.linalg.norm(self.goal - p)
        reached = d_new < 0.05
        dist = np.linalg.norm(p[None] - self.centers, axis=-1) - self.radii - self.RHO
        coll = bool(((dist < -1e-6) & self.omask).any())
        self.collisions += int(coll)
        rew = 5.0 * (d_prev - d_new) / 0.1 + 10.0 * reached
        trunc = bool(self.t >= self.horizon and not reached)
        self.last_track_err = float(np.linalg.norm(p - (p0 + self.DSTEP * w)))
        return self._obs(), float(rew), bool(reached), trunc, dict(collided=coll, success=bool(reached), fallback=fb_used,
                                                                   track_err=self.last_track_err)


# ---------------------------------------------------------------------------------------------
class PandaSafe(RowEnv):
    """MuJoCo Menagerie Franka Emika Panda, joint-velocity commands, run-time spherical keep-out zones (m = 7).

    Three control points (elbow = link4 origin, wrist = link6 origin, fingertip) are kept outside every keep-out
    sphere inflated by a per-point radius.  With x_c(q) the position of a control point and J_c its Jacobian, the
    discrete-time CBF row  -2 dt (x_c - c_i)^T J_c qdot <= gamma (|x_c - c_i|^2 - R_ci^2)  is exact to first order
    (the O(dt^2) curvature of the forward kinematics and the position-servo lag are covered by the margin DELTA);
    joint-limit and floor rows are linear in qdot.  Hence safety is empirical for obstacles, as in the other
    relative-degree-one robot environments.
    """
    DT, DELTA, GW, KV_KP, GAIN_SCALE = 0.05, 0.010, 0.5, 0.1, 10.0
    QD = np.array([1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.2])            # joint speed limits (rad/s) = action scale
    POINTS = (("link4", np.zeros(3), 0.09), ("link6", np.zeros(3), 0.08), ("hand", np.array([0, 0, 0.1034]), 0.05))
    HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    KMAX, NEAR = 3, 3

    def __init__(self, cond="A", seed=0, menagerie=None):
        import os
        import mujoco
        self.mj = mujoco
        root = menagerie or os.environ.get("MENAGERIE_PATH", "/tmp/claude-0/menagerie")
        self.model = mujoco.MjModel.from_xml_path(os.path.join(root, "franka_emika_panda", "panda.xml"))
        # stiffen the joint servos x GAIN_SCALE (kp and kv together, so kv/kp = 0.1 s is preserved): emulates the high-
        # bandwidth joint-velocity interface of a real Panda (1 kHz inner loop); documented in the paper.
        for j in range(7):
            self.model.actuator_gainprm[j, 0] *= self.GAIN_SCALE
            self.model.actuator_biasprm[j, 1] *= self.GAIN_SCALE
            self.model.actuator_biasprm[j, 2] *= self.GAIN_SCALE
        self.data = mujoco.MjData(self.model)
        self.nsub = int(round(self.DT / self.model.opt.timestep))
        self.jr = self.model.jnt_range[:7].copy()
        self.bid = [self.model.body(n).id for n, _, _ in self.POINTS]
        self.hand_id = self.model.body("hand").id
        self.rng = np.random.default_rng(seed)
        self.kmin, self.kmax = {"A": (1, 2), "B": (3, 3), "C": (1, 2)}[cond]
        self.rmin, self.rmax = (0.05, 0.10) if cond != "B" else (0.04, 0.08)
        self.rscale = 1.5 if cond == "C" else 1.0
        self.fixed_gamma = 0.1 if cond == "C" else None
        self.horizon = 120
        self.K = self.KMAX * len(self.POINTS) + 3 + 14
        super().__init__(RowSpec(7 + 7 + 3 + 3 + 1 + 5 * self.NEAR, self.K, 7))
        self.collisions = 0

    # -- kinematics ------------------------------------------------------------------------------
    def _points(self):
        d, mj = self.data, self.mj
        mj.mj_kinematics(self.model, d); mj.mj_comPos(self.model, d)   # fresh xpos/xmat/Jacobian base
        P, J = [], []
        for (name, off, _), bid in zip(self.POINTS, self.bid):
            pw = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ off
            jp = np.zeros((3, self.model.nv))
            mj.mj_jac(self.model, d, jp, None, pw, bid)
            P.append(pw.copy()); J.append(jp[:, :7].copy())
        return np.array(P), np.array(J)

    def _set_q(self, q):
        self.data.qpos[:7] = q
        self.data.qvel[:] = 0
        self.data.ctrl[:7] = q
        self.data.ctrl[7] = 0.0
        self.mj.mj_forward(self.model, self.data)

    def _ee(self):
        return self._points()[0][2]

    def _sample(self):
        rng = self.rng
        for _ in range(1000):
            q0 = np.clip(self.HOME + rng.uniform(-0.25, 0.25, 7), self.jr[:, 0] + 0.1, self.jr[:, 1] - 0.1)
            qg = np.clip(self.HOME + rng.uniform(-0.6, 0.6, 7), self.jr[:, 0] + 0.1, self.jr[:, 1] - 0.1)
            self._set_q(qg); goal = self._ee().copy()
            self._set_q(q0); P0, _ = self._points(); ee0 = P0[2].copy()
            if np.linalg.norm(goal - ee0) < 0.25 or goal[2] < 0.15:
                continue
            K = int(rng.integers(self.kmin, self.kmax + 1))
            r = rng.uniform(self.rmin, self.rmax, K) * self.rscale
            cen = np.stack([rng.uniform(0.25, 0.75, K), rng.uniform(-0.45, 0.45, K), rng.uniform(0.2, 0.75, K)], 1)
            if rng.random() < 0.5:
                cen[0] = 0.5 * (ee0 + goal) + rng.uniform(-0.04, 0.04, 3)
            ok = True
            rho = np.array([p[2] for p in self.POINTS])
            for i in range(K):                           # clearance from every control point at the start
                if (np.linalg.norm(P0 - cen[i], axis=1) - r[i] - rho - self.DELTA < 0.03).any():
                    ok = False
            if (np.linalg.norm(cen - goal, axis=1) - r - rho[2] < 0.03).any():
                ok = False
            if ok:
                self.goal = goal
                self.centers = np.zeros((self.KMAX, 3)); self.radii = np.zeros(self.KMAX); self.omask = np.zeros(self.KMAX, bool)
                self.centers[:K], self.radii[:K], self.omask[:K] = cen, r, True
                self.gamma = float(rng.uniform(0.2, 0.9)) if self.fixed_gamma is None else self.fixed_gamma
                self.q0 = q0
                return
        raise RuntimeError("could not sample a feasible Panda scene")

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._sample()
        self._set_q(self.q0)
        self.t = 0
        self.qdot = np.zeros(7)
        return self._obs(), {}

    # -- constraints -----------------------------------------------------------------------------
    def _rows(self):
        P, J = self._points()
        q = self.data.qpos[:7]
        rows, bs, ms = [], [], []
        rho = np.array([p[2] for p in self.POINTS])
        for ci in range(len(self.POINTS)):
            diff = P[ci][None] - self.centers
            R = self.radii + rho[ci] + self.DELTA
            h = (diff ** 2).sum(-1) - R ** 2
            Au = -2.0 * self.DT * diff @ J[ci]                          # (KMAX, 7) on qdot
            rows.append(np.where(self.omask[:, None], Au * self.QD[None], 0.0))   # w-space: qdot = QD * w
            bs.append(np.where(self.omask, self.gamma * h, 1.0)); ms.append(self.omask.copy())
        zmin = np.array([0.05, 0.05, 0.03])
        Af, bf = [], []
        for ci in range(len(self.POINTS)):                               # floor: z_c + dt (J qdot)_z >= zmin
            Af.append(-self.DT * J[ci][2] * self.QD); bf.append(self.GW * (P[ci][2] - zmin[ci]))
        # joint limits and velocity box merged: |w_j| <= min(1, GW * h_j / (dt * QD_j))
        lim = np.minimum(1.0, self.GW * np.concatenate([(self.jr[:, 1] - self.DELTA - q), (q - self.jr[:, 0] - self.DELTA)]) /
                         (self.DT * np.concatenate([self.QD, self.QD])))
        Ab = np.concatenate([np.eye(7), -np.eye(7)], 0)
        A = np.concatenate(rows + [np.array(Af), Ab], 0)
        b = np.concatenate(bs + [np.array(bf), lim], 0)
        m = np.concatenate(ms + [np.ones(3, bool), np.ones(14, bool)], 0)
        return normalise_rows(A, b, m)

    def _obs(self):
        d = self.data
        P, _ = self._points()
        ee = P[2]
        feat = np.zeros(self.spec.nf)
        feat[0:7], feat[7:14] = d.qpos[:7], getattr(self, 'qdot', np.zeros(7))
        feat[14:17] = ee
        feat[17:20] = self.goal - ee
        feat[20] = self.gamma
        dist = np.linalg.norm(ee[None] - self.centers, axis=-1) - self.radii
        dist = np.where(self.omask, dist, np.inf)
        for k, j in enumerate(np.argsort(dist)[: self.NEAR]):
            ok = np.isfinite(dist[j])
            base = 21 + 5 * k
            feat[base: base + 3] = (self.centers[j] - ee) if ok else 0.0
            feat[base + 3] = self.radii[j] if ok else 0.0
            feat[base + 4] = float(ok)
        A, b, ok = self._rows()
        strict = bool(np.all(b[ok] > 0.0))
        self._set_rows(A, b, ok, np.zeros(7))
        self._cur_fb = not strict
        return self.spec.pack(feat, A, b, ok, np.zeros(7), 0.0 if strict else 1.0)

    def step(self, w):
        d = self.data
        w = np.clip(np.asarray(w, dtype=np.float64), -1, 1)
        fb_used = self._cur_fb
        w = np.zeros(7) if fb_used else self.restore(w)
        ee0 = self._ee()
        d_prev = np.linalg.norm(self.goal - ee0)
        # kinematic joint-velocity interface: q <- q + dt * qd_max * w (MuJoCo supplies the exact forward kinematics and
        # geometry; the arm's actuator dynamics are NOT simulated -- see the limitations section of the paper).
        q = np.clip(d.qpos[:7] + self.DT * self.QD * w, self.jr[:, 0], self.jr[:, 1])
        d.qpos[:7] = q
        self.qdot = self.QD * w
        self.mj.mj_kinematics(self.model, d); self.mj.mj_comPos(self.model, d)
        self.t += 1
        P, _ = self._points()
        d_new = np.linalg.norm(self.goal - P[2])
        reached = d_new < 0.05
        rho = np.array([p[2] for p in self.POINTS])
        coll = False
        for ci in range(len(self.POINTS)):
            dist = np.linalg.norm(P[ci][None] - self.centers, axis=-1) - self.radii - rho[ci]
            coll |= bool(((dist < -1e-6) & self.omask).any())
        coll |= bool((P[:, 2] < np.array([0.05, 0.05, 0.03]) - 0.02).any())
        self.collisions += int(coll)
        rew = 5.0 * (d_prev - d_new) / 0.1 + 10.0 * reached
        trunc = bool(self.t >= self.horizon and not reached)
        return self._obs(), float(rew), bool(reached), trunc, dict(collided=coll, success=bool(reached), fallback=fb_used)
