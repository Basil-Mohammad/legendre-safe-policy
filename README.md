# Legendre Safe Policies (working title)

Safety as a **structural property of the policy class** instead of a reward penalty.
The policy outputs a preference vector `nu`; the executed action is the log-barrier (Legendre)
maximiser of `nu . u` over the state-dependent safe polytope `{u : A(x) u <= b(x)}` built from
discrete-time control barrier functions, actuation limits and run-time-specified keep-out zones.

Status: Phase-1 pilot finished. Pre-registered decision: **REVISE** (see `results/RESULTS.md`).

## Layout
- `lsp/layers.py`   polytope construction, Legendre map (custom autograd), exact 2-D projection, gauge map
- `lsp/env.py`      batched navigation environment with run-time constraints (conditions A/B/C)
- `lsp/agent.py`    TD3 with pluggable action map, evaluation, Jacobian diagnostics
- `scripts/`        `run_pilot.py`, `run_all_pilot.sh`, `aggregate.py`, `diag_gauge_pinch.py`
- `tests/`          unit tests (KKT, gradcheck, SPD Jacobian, projection vs scipy, no-collision invariance)
- `docs/`           `PREREGISTRATION.md`, `DEVIATIONS.md`
- `results/`        per-run JSON, `summary.md`, `RESULTS.md`, learning curves

## Reproduce (CPU, about 1.5 h on one core)
```bash
pip install -r requirements.txt
pytest -q
bash scripts/run_all_pilot.sh
python scripts/aggregate.py results
```
