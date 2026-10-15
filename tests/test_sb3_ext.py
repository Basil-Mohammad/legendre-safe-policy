import numpy as np, torch, pytest
from lsp2 import envs as E
from lsp2.sb3_ext import MapLayer
from lsp2.faults import FaultWrapper


@pytest.mark.parametrize("env", ["NavSI", "NavDI", "Unicycle", "PandaSafe"])
@pytest.mark.parametrize("kind", ["sp", "gauge", "legendre", "legendre_st"])
def test_map_outputs_feasible(env, kind):
    e = getattr(E, env)("A", 0)
    o, _ = e.reset(seed=0)
    rng = np.random.default_rng(0)
    L = MapLayer(e.spec, kind, 2.0)
    for _ in range(40):
        a = torch.as_tensor(rng.uniform(-1, 1, (1, e.spec.m)), dtype=torch.float32)
        w = L(a, torch.as_tensor(o)[None]).numpy()[0].astype(np.float64)
        assert np.isfinite(w).all()
        A, b, ok, u0 = e._rows64
        if o[e.spec.ofb_index] < 0.5:
            assert (A @ w - b)[ok].max() <= 1e-6
        o, r, te, tr, info = e.step(a.numpy()[0] * 0 + w)
        assert not info["collided"]
        if te or tr:
            o, _ = e.reset()


@pytest.mark.parametrize("fault", ["drop", "stale", "rowdrop"])
def test_fault_wrapper_runs(fault):
    e = FaultWrapper(E.Unicycle("A", 0), fault, seed=1)
    o, _ = e.reset(seed=0)
    for _ in range(60):
        o, r, te, tr, info = e.step(np.zeros(2))
        assert np.isfinite(o).all()
        if te or tr:
            o, _ = e.reset()
