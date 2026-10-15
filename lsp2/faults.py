"""Runtime faults (sudden interruptions) injected between the policy/map and the plant.

All faults are seeded, method-independent, and act on the *signals*, never on the plant itself:

  none        control
  drop        command-channel blackout bursts: with probability P_ON per step a burst of L steps starts during which the
              executor repeats the last command it received (zero-order hold; this is the default behaviour of ROS 2
              velocity controllers without a timeout)
  drop_wd     same bursts, but a watchdog replaces the held command by the certified fallback after T_WD held steps
  stale       the constraint channel is delayed: policy and map see rows (A, b, u0) that are K_STALE steps old, while the
              plant is checked against the current true rows
  rowdrop     each constraint row is independently hidden from the policy/map with probability P_ROW at every step
  latency     the state-feature channel is delayed by K_LAT steps (rows are current)

Executor feasibility restoration is OFF in fault evaluations (env.enforce = False), so that collisions and the
violation magnitude  max_i (a_i.w - b_i)_+  of the executed command w.r.t. the TRUE rows measure how much the
action map's own safety margin survives the fault.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import gymnasium as gym

P_ON, L_BURST, T_WD, K_STALE, P_ROW, K_LAT = 0.03, 6, 2, 3, 0.2, 3
FAULTS = ("none", "drop", "drop_wd", "stale", "rowdrop", "latency")


class FaultWrapper(gym.Wrapper):
    def __init__(self, env, kind: str, seed: int = 0):
        super().__init__(env)
        assert kind in FAULTS
        self.kind = kind
        self.rng = np.random.default_rng(seed)
        self.u = env.unwrapped
        self.rs = self.u.spec
        self.u.enforce = False

    # -- helpers -----------------------------------------------------------------------------------------------
    def _reset_state(self):
        self.hist = deque(maxlen=max(K_STALE, K_LAT) + 1)
        self.burst = 0
        self.held = 0
        self.last_w = np.zeros(self.rs.m)
        self.viol = 0.0
        self.drop_steps = 0

    def _view(self, o):
        s = self.rs
        self.hist.append(o.copy())
        v = o.copy()
        if self.kind == "stale" and len(self.hist) > K_STALE:
            old = self.hist[-1 - K_STALE]
            v[s.nf:] = old[s.nf:]
        elif self.kind == "latency" and len(self.hist) > K_LAT:
            old = self.hist[-1 - K_LAT]
            v[: s.nf] = old[: s.nf]
        elif self.kind == "rowdrop":
            K, m = s.K, s.m
            hid = self.rng.random(K) < P_ROW
            h0 = s.nf
            for k in np.flatnonzero(hid):
                v[h0 + k * m: h0 + (k + 1) * m] = 0.0                     # A row (hi)
                v[s.olo + k * m: s.olo + (k + 1) * m] = 0.0               # A row (lo)
                v[h0 + K * m + k] = 1.0                                   # b (hi)
                v[s.olo + K * m + k] = 0.0                                # b (lo)
                v[h0 + K * m + K + k] = 0.0                               # mask
        return v

    # -- gym API -----------------------------------------------------------------------------------------------
    def reset(self, **kw):
        self._reset_state()
        o, info = self.env.reset(**kw)
        return self._view(o), info

    def step(self, w):
        w = np.clip(np.asarray(w, dtype=np.float64), -1, 1)
        fb_force = False
        if self.kind in ("drop", "drop_wd"):
            if self.burst == 0 and self.rng.random() < P_ON:
                self.burst = L_BURST
                self.held = 0
            if self.burst > 0:
                self.burst -= 1
                self.held += 1
                self.drop_steps += 1
                w = self.last_w.copy()
                if self.kind == "drop_wd" and self.held >= T_WD:
                    fb_force = True
        # violation of the TRUE rows by the command that is about to be executed
        rows = self.u._rows64
        if rows is not None:
            A, b, ok, _ = rows
            self.viol = max(self.viol, float(np.max(np.where(ok, A @ w - b, -np.inf), initial=-np.inf).clip(min=0.0)))
        if fb_force:
            if hasattr(self.u, "_cur_fb"):
                self.u._cur_fb = True
            else:
                w = np.zeros(self.rs.m)
        self.last_w = w
        o, r, te, tr, info = self.env.step(w)
        info["viol"] = self.viol
        info["drop_steps"] = self.drop_steps
        if te or tr:
            self.viol = 0.0
        return self._view(o), r, te, tr, info
