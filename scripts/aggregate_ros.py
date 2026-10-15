"""Aggregate ROS 2 runs: python scripts/aggregate_ros.py results_ros out.json"""
import glob, json, os, sys, re
import numpy as np
root = sys.argv[1]; out = sys.argv[2]
SC = dict(sp=4, gauge=1, legendre=20)
runs = {}
for f in glob.glob(os.path.join(root, "runs", "ppo_*.json")):
    d = json.load(open(f))
    runs[(d["method"], d["bridge"], d["seed"])] = d
res = {}
for m in SC:
    for b in ["none", "drop", "drop_wd"]:
        rs = [runs[(m, b, s)] for s in (900, 901, 902) if (m, b, s) in runs]
        if not rs: continue
        row = dict(n=len(rs), success=float(np.mean([r["success"] for r in rs])), collided=float(np.mean([r["collided"] for r in rs])),
                   viol=float(np.mean([r["viol"] for r in rs])), lat_median=float(np.mean([r["lat_ms"]["median"] for r in rs])),
                   lat_p99=float(np.max([r["lat_ms"]["p99"] for r in rs])), lat_max=float(np.max([r["lat_ms"]["max"] for r in rs])),
                   load=float(np.mean([r["load1"] for r in rs])))
        if b == "none":
            ip = [json.load(open(os.path.join(root, "models", f"unicycle_ppo_{m}_s{SC[m]}_seed{s}.json")))["eval"]["A"]["success"] for s in (900, 901, 902)]
            row["inproc_success"] = float(np.mean(ip))
            row["abs_diff_per_seed"] = [abs(runs[(m, b, s)]["success"] - ip[i]) for i, s in enumerate((900, 901, 902))]
            row["seed_success_ros"] = [runs[(m, b, s)]["success"] for s in (900, 901, 902)]
            row["seed_success_inproc"] = ip
        res[f"{m}/{b}"] = row
json.dump(res, open(out, "w"), indent=1)
for k, v in res.items():
    print(k, {a: (round(x, 3) if isinstance(x, float) else x) for a, x in v.items() if a not in ("seed_success_ros", "seed_success_inproc")})
    if "seed_success_ros" in v: print("   ros", v["seed_success_ros"], "inproc", v["seed_success_inproc"])
