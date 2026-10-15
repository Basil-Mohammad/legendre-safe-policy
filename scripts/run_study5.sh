#!/bin/bash
# Study 5: (1) baseline output-scale sweep on tuning seeds 110-112, (2) pre-registered selection,
# (3) confirmatory runs on seeds 40-51 (mu = 0.05 as in Study 2).  Two workers.  Resumable (skip-existing).
cd "$(dirname "$0")/.."
mkdir -p results_v5_tune results_v5
python3 scripts/make_jobs_v5.py tune | xargs -P 2 -I{} sh -c 'python3 scripts/run_study5.py {} --out results_v5_tune --steps 100000 --eval_eps 500 >> results_v5_tune/log.txt 2>&1'
python3 scripts/select_scale_v5.py >> results_v5/log.txt 2>&1 || { echo SELECT_FAILED >> results_v5/log.txt; exit 1; }
python3 scripts/make_jobs_v5.py confirm | xargs -P 2 -I{} sh -c 'python3 scripts/run_study5.py {} --out results_v5 --steps 100000 --eval_eps 500 >> results_v5/log.txt 2>&1'
echo ALL_DONE >> results_v5/log.txt
