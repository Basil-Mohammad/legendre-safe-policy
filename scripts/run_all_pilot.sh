#!/bin/bash
# Sequential pilot queue (single CPU). Seeds outer loop so every method has seed 0 first.
cd "$(dirname "$0")/.."
mkdir -p results
for seed in 0 1 2; do
  for m in legendre sp gauge se penalty; do
    if [ -f results/${m}_s${seed}.json ] || [ -f results/${m}_mu0.05_s${seed}.json ]; then continue; fi
    python3 scripts/run_pilot.py --method $m --seed $seed --mu 0.05 --steps 100000 --eval_eps 500
  done
done
echo ALL_DONE
