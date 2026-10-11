#!/bin/bash
# Study 3 (confirmatory): double integrator, seeds 6,7,8; two parallel workers. Resumable.
cd "$(dirname "$0")/.."
mkdir -p results_v3
for seed in 6 7 8; do
  echo "--method legendre --seed $seed --mu 0.05"
  echo "--method legendre_st --seed $seed"
  for m in sp se gauge sp_m09 sp_m07; do echo "--method $m --seed $seed"; done
  for lam in 0.5 2.0 8.0; do echo "--method penalty --seed $seed --penalty $lam"; done
  echo "--method legendre --seed $seed --mu 0.02"
  echo "--method legendre --seed $seed --mu 0.2"
done | xargs -P 2 -I{} sh -c 'python3 scripts/run_study3.py {} --out results_v3 --steps 100000 --eval_eps 500 >> results_v3/log.txt 2>&1'
echo ALL_DONE >> results_v3/log.txt
