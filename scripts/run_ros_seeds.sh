#!/bin/bash
# train (in-process venv) and then evaluate (ROS venv) PPO unicycle policies for extra seeds; usage: run_ros_seeds.sh 901 902
M=results_ros/models; O=results_ros/runs; mkdir -p $M $O
for seed in "$@"; do
  for cfg in "sp 4" "gauge 1" "legendre 20"; do set -- $cfg
    [ -f $M/unicycle_ppo_$1_s$2_seed$seed.json ] || PYTHONPATH=. /tmp/claude-0/v6/bin/python -m lsp2.train --env unicycle --algo ppo --method $1 --scale $2 --seed $seed --steps 200000 --eval_eps 30 --save_model --out $M > /dev/null 2>&1 &
  done; wait
done
