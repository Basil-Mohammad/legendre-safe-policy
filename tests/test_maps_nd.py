import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from scipy.optimize import minimize
from lsp2.maps import gauge_nd, legendre_map, project_nd

torch.set_default_dtype(torch.float64)


def rand_poly(B, K, m, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(B, K, m, generator=g)
    A = A / A.norm(dim=-1, keepdim=True)
    b = 0.05 + torch.rand(B, K, generator=g)
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[:, K - 2:] = torch.rand(B, 2, generator=g) > 0.5
    box = torch.cat([torch.eye(m), -torch.eye(m)], 0).expand(B, 2 * m, m)
    A = torch.cat([A, box], 1)
    b = torch.cat([b, torch.ones(B, 2 * m)], 1)
    mask = torch.cat([mask, torch.ones(B, 2 * m, dtype=torch.bool)], 1)
    return A, b, mask


def test_projection_matches_scipy_m3_m7():
    for m in (3, 7):
        A, b, mask = rand_poly(12, 8, m, 1 + m)
        y = 2.0 * torch.randn(12, m)
        u = project_nd(y, A, b, mask)
        assert ((A * u.unsqueeze(1)).sum(-1) <= b + 1e-9)[mask].all()
        for i in range(12):
            r = mask[i].numpy()
            Ai, bi = A[i].numpy()[r], b[i].numpy()[r]
            res = minimize(lambda x: ((x - y[i].numpy()) ** 2).sum(), np.zeros(m),
                           constraints=[{"type": "ineq", "fun": lambda x: bi - Ai @ x}], method="SLSQP",
                           options=dict(ftol=1e-14, maxiter=500))
            assert np.allclose(res.x, u[i].numpy(), atol=1e-4), (m, i, np.abs(res.x - u[i].numpy()).max())


def test_projection_gradient_rank_deficient_on_faces():
    A = torch.tensor([[[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]])
    b = torch.ones(1, 6)
    mask = torch.ones(1, 6, dtype=torch.bool)
    y = torch.tensor([[3.0, 3.0, 0.2]], requires_grad=True)   # corner of the cube in x,y; free in z
    u = project_nd(y, A, b, mask)
    J = torch.stack([torch.autograd.grad(u[0, j], y, retain_graph=True)[0][0] for j in range(3)])
    assert torch.allclose(J, torch.diag(torch.tensor([0.0, 0.0, 1.0])), atol=1e-8), J


def test_legendre_and_gauge_feasible_and_jacobian_spd_m7():
    A, b, mask = rand_poly(16, 8, 7, 5)
    nu = torch.empty(16, 7).uniform_(-1, 1).requires_grad_(True)
    u = legendre_map(nu, A, b, mask, 0.05)
    s = b - (A * u.unsqueeze(1)).sum(-1)
    assert (s[mask] > 0).all()
    J = torch.stack([torch.autograd.grad(u[:, j].sum(), nu, retain_graph=True)[0] for j in range(7)], 1)
    ev = torch.linalg.eigvalsh(0.5 * (J + J.transpose(1, 2)))
    assert (ev > 0).all()
    ug = gauge_nd(nu.detach(), A, b, mask)
    assert ((A * ug.unsqueeze(1)).sum(-1) <= b + 1e-9)[mask].all()
