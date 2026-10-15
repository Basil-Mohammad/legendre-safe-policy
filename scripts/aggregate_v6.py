"""Aggregate Study 6 (pre-registered analyses S1, S2, P1-P3, F1-F2).  python scripts/aggregate_v6.py results_v6 [out.json]"""
import glob, json, os, sys
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "results_v6"
outp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "summary_v6.json")
rng = np.random.default_rng(0)
B = 10_000
CONF = set(range(10))
SAFE = ["legendre", "sp", "gauge", "se"]
BASE = {"sac": ["sp", "gauge", "se"], "ppo": ["sp", "gauge"]}
FAULTS = ["none", "drop", "drop_wd", "stale", "rowdrop", "latency"]

runs = []
for f in glob.glob(os.path.join(root, "*", "*.json")):
    b = os.path.basename(f)
    if b.startswith("selected") or b.startswith("summary"):
        continue
    d = json.load(open(f))
    if "method" in d and "eval" in d:
        runs.append(d)
conf = [r for r in runs if r["seed"] in CONF]
tune = [r for r in runs if r["seed"] >= 100]


def cell(env, algo, method, seeds=CONF):
    """dict seed -> run, for confirmatory runs."""
    return {r["seed"]: r for r in conf if r["env"] == env and r["algo"] == algo and r["method"] == method}


def vec(d, cond="A", key="success"):
    ks = sorted(d)
    return ks, np.array([d[k]["eval"][cond][key] for k in ks])


def boot_diff(x, y, one_sided=True):
    """paired bootstrap over seed indices; x, y aligned arrays; returns mean diff, one-sided 95% lower bound, 2-sided 95% CI."""
    n = len(x)
    idx = rng.integers(0, n, (B, n))
    dm = (x[idx] - y[idx]).mean(1)
    return float(np.mean(x - y)), float(np.quantile(dm, 0.05)), (float(np.quantile(dm, 0.025)), float(np.quantile(dm, 0.975)))


def ci(x):
    n = len(x); idx = rng.integers(0, n, (B, n)); m = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


out = dict(n_runs=len(runs), n_conf=len(conf), table={}, S1={}, S2={}, P1={}, F={}, P2={}, P3={})
envs = sorted({r["env"] for r in conf}); algos = ["sac", "ppo"]
methods_all = ["legendre", "sp", "gauge", "se", "legendre_st", "pen2"]
for env in envs:
    for al in algos:
        for m in methods_all:
            c = cell(env, al, m)
            if not c:
                continue
            row = dict(n=len(c), scale=next(iter(c.values()))["scale"])
            for cond in "ABC":
                ks, s = vec(c, cond, "success"); _, co = vec(c, cond, "collided"); _, rs = vec(c, cond, "restorations")
                mu, lo, hi = ci(s)
                row[cond] = dict(success=mu, lo=lo, hi=hi, collided=float(co.mean()), restorations=float(rs.mean()), coll_total=int(sum(c[k]["eval"][cond]["coll_total"] for k in ks)))
            out["table"][f"{env}/{al}/{m}"] = row
# S1
for k, row in out["table"].items():
    m = k.split("/")[2]
    if m in SAFE + ["legendre_st"]:
        out["S1"][k] = {c: row[c]["coll_total"] for c in "ABC"}
out["S1_all_zero"] = all(v == 0 for r in out["S1"].values() for v in r.values())
# S2: tuning spread
for env in envs:
    for al in algos:
        for m in SAFE:
            sc = {}
            for r in tune:
                if r["env"] == env and r["algo"] == al and r["method"] == m:
                    sc.setdefault(r["scale"], []).append(r["eval"]["A"]["success"])
            if len(sc) >= 2:
                means = {s: float(np.mean(v)) for s, v in sc.items()}
                out["S2"][f"{env}/{al}/{m}"] = dict(means=means, spread=max(means.values()) - min(means.values()))
sp = [v["spread"] for k, v in out["S2"].items() if k.split("/")[2] != "legendre"]
out["S2_median_baseline_spread"] = float(np.median(sp)) if sp else None
# P1
for env in envs:
    for al in algos:
        L = cell(env, al, "legendre")
        bs = {m: cell(env, al, m) for m in BASE[al] if cell(env, al, m)}
        if not L or not bs:
            continue
        for cond in "ABC":
            _, lv = vec(L, cond)
            best = max(bs, key=lambda m: vec(bs[m], "A")[1].mean())          # reference chosen on condition A (pre-registered)
            best_c = max(bs, key=lambda m: vec(bs[m], cond)[1].mean())
            for tag_, ref in (("refA", best), ("refCond", best_c)):
                ks = sorted(set(L) & set(bs[ref]))
                x = np.array([L[k]["eval"][cond]["success"] for k in ks]); y = np.array([bs[ref][k]["eval"][cond]["success"] for k in ks])
                d, lo, ci2 = boot_diff(x, y)
                out["P1"][f"{env}/{al}/{cond}/{tag_}"] = dict(ref=ref, diff=d, lb95=lo, ci=ci2, superior=bool(d >= 0.03 and lo > 0),
                                                              noninferior=bool(lo > -0.05), inferior=bool(ci2[1] < 0), n=len(ks))
for cond in "ABC":
    ks = [k for k in out["P1"] if k.split("/")[2] == cond and k.endswith("refA")]
    out["P1_count_" + cond] = dict(cells=len(ks), superior=sum(out["P1"][k]["superior"] for k in ks),
                                   noninferior=sum(out["P1"][k]["noninferior"] for k in ks), inferior=sum(out["P1"][k]["inferior"] for k in ks))
# P2: SAC minus PPO contrast, pooled over envs (cond A, refA)
for cond in "ABC":
    cs = {}
    for env in envs:
        for al in algos:
            L = cell(env, al, "legendre"); bs = {m: cell(env, al, m) for m in BASE[al] if cell(env, al, m)}
            if not L or not bs:
                continue
            ref = max(bs, key=lambda m: vec(bs[m], "A")[1].mean())
            ks = sorted(set(L) & set(bs[ref]))
            cs[(env, al)] = np.array([L[k]["eval"][cond]["success"] - bs[ref][k]["eval"][cond]["success"] for k in ks])
    pooled = [e for e in envs if (e, "sac") in cs and (e, "ppo") in cs]
    if pooled:
        dm = []
        for _ in range(B):
            tot = 0.0
            for e in pooled:
                a_, p_ = cs[(e, "sac")], cs[(e, "ppo")]
                tot += a_[rng.integers(0, len(a_), len(a_))].mean() - p_[rng.integers(0, len(p_), len(p_))].mean()
            dm.append(tot / len(pooled))
        obs = np.mean([cs[(e, "sac")].mean() - cs[(e, "ppo")].mean() for e in pooled])
        out["P2"][cond] = dict(envs=pooled, mean=float(obs), lb95=float(np.quantile(dm, 0.05)), ci=(float(np.quantile(dm, 0.025)), float(np.quantile(dm, 0.975))))
# P3
for env in envs:
    L, S = cell(env, "sac", "legendre"), cell(env, "sac", "legendre_st")
    if L and S:
        for cond in "ABC":
            ks = sorted(set(L) & set(S))
            x = np.array([L[k]["eval"][cond]["success"] for k in ks]); y = np.array([S[k]["eval"][cond]["success"] for k in ks])
            d, lo, ci2 = boot_diff(x, y)
            out["P3"][f"{env}/{cond}"] = dict(diff=d, lb95=lo, ci=ci2)
# F: faults
for env in envs:
    for al in algos:
        for m in methods_all:
            c = cell(env, al, m)
            if not c or "faults" not in next(iter(c.values())) or not next(iter(c.values()))["faults"]:
                continue
            row = {}
            for f in FAULTS:
                ks = sorted(c)
                row[f] = dict(collided=float(np.mean([c[k]["faults"][f]["collided"] for k in ks])), success=float(np.mean([c[k]["faults"][f]["success"] for k in ks])),
                              viol=float(np.mean([c[k]["faults"][f]["viol"] for k in ks])))
            out["F"][f"{env}/{al}/{m}"] = row
# F1: Legendre vs sp, gauge on max violation, pooled over cells (seeds paired within cell)
for f in ["drop", "stale", "rowdrop"]:
    for ref in ["sp", "gauge"]:
        diffs = []
        for env in envs:
            for al in algos:
                L, R = cell(env, al, "legendre"), cell(env, al, ref)
                ks = sorted(set(L) & set(R))
                if ks and "faults" in L[ks[0]] and L[ks[0]]["faults"]:
                    diffs.append(np.array([L[k]["faults"][f]["viol"] - R[k]["faults"][f]["viol"] for k in ks]))
        if diffs:
            n = min(len(d) for d in diffs)
            dm = np.array([np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in diffs]) for _ in range(B)])
            out.setdefault("F1", {})[f"{f}/{ref}"] = dict(cells=len(diffs), mean=float(np.mean([d.mean() for d in diffs])), ub95=float(np.quantile(dm, 0.95)))
# F2: watchdog effect on collisions
for env in envs:
    for al in algos:
        for m in methods_all:
            k_ = f"{env}/{al}/{m}"
            if k_ in out["F"]:
                out.setdefault("F2", {})[k_] = dict(drop=out["F"][k_]["drop"]["collided"], drop_wd=out["F"][k_]["drop_wd"]["collided"])
json.dump(out, open(outp, "w"), indent=1)
print("runs", len(runs), "conf", len(conf), "tune", len(tune), "S1_all_zero", out["S1_all_zero"], "S2 median", out["S2_median_baseline_spread"])
for c in "ABC":
    print("P1", c, out.get("P1_count_" + c))
