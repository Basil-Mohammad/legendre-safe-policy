"""Print the job lines (arguments of scripts/run_study4.py) for Study 4.  Usage: make_jobs_v4.py tune|confirm"""
import json
import os
import sys

phase = sys.argv[1]
TUNE_SEEDS = (110, 111, 112)
CONF_SEEDS = tuple(range(40, 52))
SCALES = (0.5, 1.0, 2.0, 4.0)
if phase == "tune":
    for seed in TUNE_SEEDS:
        for sc in SCALES:
            for m in ("se", "sp", "gauge"):
                print(f"--method {m} --seed {seed} --scale {sc}")
else:
    sel = json.load(open("results_v5/selected_scales.json"))
    for seed in CONF_SEEDS:
        print(f"--method legendre --seed {seed} --mu 0.05")
        for m in ("se", "sp", "gauge"):
            print(f"--method {m} --seed {seed} --scale {sel[m]}")
        print(f"--method sp_m09 --seed {seed} --scale {sel['sp']}")
        for lam in (0.5, 2.0):
            print(f"--method penalty --seed {seed} --penalty {lam}")
