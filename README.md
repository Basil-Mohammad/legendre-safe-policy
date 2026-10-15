# Legendre Safe Policies (working title)

Safety as a **structural property of the policy class** instead of a reward penalty.
The policy outputs a preference vector `nu`; the executed action is the log-barrier (Legendre)
maximiser of `nu . u` over the state-dependent safe polytope `{u : A(x) u <= b(x)}` built from
discrete-time control barrier functions, actuation limits and run-time-specified keep-out zones.

Status: Studies 1-5 finished (theory, planar benchmarks, pre-registered negative results). Study 6 (multi-environment, SAC/PPO, fault injection, ROS 2) in progress; see `docs/PREREGISTRATION_v6.md` locally and the paper.

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


## Study 6 (`lsp2/`)
- `lsp2/maps.py`  m-dimensional exact maps: robust Legendre (custom autograd), exact batched Euclidean projection (Lawson-Hanson dual active set), gauge map
- `lsp2/envs.py`  five environments with one row-observation interface: navsi, navdi, unicycle (MuJoCo), fetch (Gymnasium-Robotics), panda (MuJoCo Menagerie)
- `lsp2/sb3_ext.py`, `lsp2/train.py`  Stable-Baselines3 SAC (map inside the actor) and PPO (environment-side map)
- `lsp2/faults.py`  runtime faults (command blackout, watchdog, stale / dropped constraint rows, state latency)
- `lsp2/ros_nodes.py`, `scripts/ros_eval.py`  ROS 2 Jazzy (CycloneDDS) plant / policy / fault-bridge nodes
- `scripts/study6_driver.py`  tuning -> selection -> confirmatory runs for one environment; `scripts/aggregate_v6.py` pre-registered analyses
- `kaggle/run_shard.py`  CPU shards for Kaggle (see header)
```bash
pip install -r requirements.txt   # + stable-baselines3==2.9.0 gymnasium-robotics==1.4.2 mujoco==3.3.7
export MENAGERIE_PATH=/path/to/mujoco_menagerie   # commit 0059d43 (Panda)
pytest -q tests
python scripts/study6_driver.py --env navsi --workers 2 --out results_v6
python scripts/aggregate_v6.py results_v6
```
