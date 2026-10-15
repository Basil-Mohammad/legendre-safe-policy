#!/bin/bash
# ROS 2 in-the-loop evaluation of the three PPO unicycle policies (run with the ROS python venv)
source /home/claude/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_LOCALHOST_ONLY=1
export PYTHONPATH="$PYTHONPATH:$(pwd)"
M=results_ros/models; O=results_ros/runs; mkdir -p $O
for cfg in "sp 4" "gauge 1" "legendre 20"; do
  set -- $cfg
  for b in none drop drop_wd; do
    f=$O/ppo_$1_$b.json
    [ -f $f ] && continue
    /home/claude/rosvenv/bin/python scripts/ros_eval.py --model $M/unicycle_ppo_$1_s$2_seed900.zip --algo ppo --method $1 --scale $2 --seed 900 --bridge $b --episodes 30 --out $f > $O/ppo_$1_$b.log 2>&1
  done
done
echo finished > $O/DONE
