#!/bin/bash
# Study 4: (1) baseline output-scale sweep on tuning seeds 100-102, (2) pre-registered selection,
# (3) confirmatory runs on seeds 20-31 (mu = 0.02 fixed a priori).  Two workers.  Resumable (skip-existing).
cd "$(dirname "$0")/.."
mkdir -p results_v4_tune results_v4
python3 scripts/make_jobs_v4.py tune | xargs -P 2 -I{} sh -c 'python3 scripts/run_study4.py {} --out results_v4_tune --steps 100000 --eval_eps 500 >> results_v4_tune/log.txt 2>&1'
python3 scripts/select_scale.py >> results_v4/log.txt 2>&1 || { echo SELECT_FAILED >> results_v4/log.txt; exit 1; }
python3 scripts/make_jobs_v4.py confirm | xargs -P 2 -I{} sh -c 'python3 scripts/run_study4.py {} --out results_v4 --steps 100000 --eval_eps 500 >> results_v4/log.txt 2>&1'
echo ALL_DONE >> results_v4/log.txt
