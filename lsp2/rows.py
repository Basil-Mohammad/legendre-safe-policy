"""Observation convention shared by all Study-6 environments.

A flat observation is  [ features (nf) | A (K*m) | b (K) | mask (K) | u0 (m) | fb (1) ]  where
  A, b, mask   describe the safe polytope in NORMALISED action space w in [-1,1]^m   (A w <= b),
  u0           a strictly interior point of that polytope (0 for relative degree one),
  fb           1.0 if the polytope has no strict interior; the environment then executes the certified
               fallback and the action map output is ignored.
Constraints reach the policy only through the observation, so policies are constraint-conditioned.
"""
from __future__ import annotations

import numpy as np
import torch


class RowSpec:
    """Layout  [features | A b mask u0 fb (float32 hi parts) | A b u0 (float32 lo parts)].

    hi + lo reconstructs the double-precision rows exactly (hi = float32(x), lo = float32(x - hi)), because the
    action maps are sensitive to the slacks of nearly active rows (absolute float32 rounding of 1e-7 is not
    negligible next to slacks of 1e-5); SB3 policies otherwise receive observations in float32 only.
    """

    def __init__(self, nf: int, K: int, m: int):
        self.nf, self.K, self.m = nf, K, m
        self.oA = nf
        self.ob = self.oA + K * m
        self.om = self.ob + K
        self.ou = self.om + K
        self.ofb = self.ou + m
        self.nblk = self.ofb + 1 - nf               # size of the hi block
        self.nlo = K * m + K + m                    # lo parts of A, b, u0
        self.olo = nf + self.nblk
        self.dim = self.olo + self.nlo

    def pack(self, feat, A, b, mask, u0, fb):
        blk = np.zeros(self.nblk, dtype=np.float64)
        blk[: self.K * self.m] = np.asarray(A, np.float64).reshape(-1)
        blk[self.K * self.m : self.K * self.m + self.K] = b
        blk[self.K * self.m + self.K : self.K * self.m + 2 * self.K] = np.asarray(mask, np.float64)
        blk[self.K * self.m + 2 * self.K : self.K * self.m + 2 * self.K + self.m] = u0
        blk[-1] = float(fb)
        hi = blk.astype(np.float32)
        r = blk - hi.astype(np.float64)
        lo = np.concatenate([r[: self.K * self.m + self.K], r[self.K * self.m + 2 * self.K : self.K * self.m + 2 * self.K + self.m]]).astype(np.float32)
        out = np.zeros(self.dim, dtype=np.float32)
        out[: self.nf] = feat
        out[self.nf : self.olo] = hi
        out[self.olo :] = lo
        return out

    def unpack_t(self, obs: torch.Tensor):
        """obs (B, dim) -> A (B,K,m), b (B,K), mask (B,K) bool, u0 (B,m), fb (B,) bool   (exact float64)."""
        o = obs.to(torch.float64)
        B = o.shape[0]
        K, m = self.K, self.m
        h = o[:, self.nf : self.olo]
        lo = o[:, self.olo :]
        A = (h[:, : K * m] + lo[:, : K * m]).reshape(B, K, m)
        b = h[:, K * m : K * m + K] + lo[:, K * m : K * m + K]
        mask = h[:, K * m + K : K * m + 2 * K] > 0.5
        u0 = h[:, K * m + 2 * K : K * m + 2 * K + m] + lo[:, K * m + K : K * m + K + m]
        fb = h[:, -1] > 0.5
        return A, b, mask, u0, fb

    @property
    def ofb_index(self):
        return self.olo - 1


def normalise_rows(A, b, mask, floor=1e-10):
    # floor=None: keep b unchanged (relative degree two: b may be negative; the interior point u0 != 0)
    """Unit-length rows (slack = Euclidean distance to the row hyperplane); padded rows get a=0, b=1; b >= floor."""
    nrm = np.linalg.norm(A, axis=-1, keepdims=True)
    ok = (nrm[..., 0] > 1e-12) & mask
    A2 = np.where(ok[..., None], A / np.maximum(nrm, 1e-12), 0.0)
    b2 = np.where(ok, b / np.maximum(nrm[..., 0], 1e-12), 1.0)
    if floor is not None:
        b2 = np.where(ok, np.maximum(b2, floor), b2)
    return A2, b2, ok


def interior_shift(A, b, mask, u0):
    """Return (b' = b - A u0, valid) so that the shifted polytope {w: A w <= b'} contains 0 strictly."""
    s = b - np.einsum("...km,...m->...k", A, u0)
    return s
