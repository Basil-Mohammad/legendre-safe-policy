#!/bin/bash
# Study 2 (confirmatory): seeds 3,4,5; two parallel workers. Resumable (finished runs are skipped).
cd "$(dirname "$0")/.."
mkdir -p results_v2
for seed in 3 4 5; do
  echo "--method legendre --seed $seed --mu 0.05"
  for m in sp se gauge sp_m09 sp_m07 gauge_m09; do echo "--method $m --seed $seed"; done
  for lam in 0.5 2.0 8.0; do echo "--method penalty --seed $seed --penalty $lam"; done
done | xargs -P 2 -I{} sh -c 'python3 scripts/run_pilot.py {} --out results_v2 --steps 100000 --eval_eps 500 >> results_v2/log.txt 2>&1'
echo ALL_DONE >> results_v2/log.txt
