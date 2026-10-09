"""Aggregate pilot results and evaluate the pre-registered hypotheses H1-H4."""
import glob
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

res_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(res_dir, "*.json")))]


def key(r):
    return f"legendre mu={r['mu']}" if r["method"] == "legendre" else r["method"]


groups = {}
for r in runs:
    groups.setdefault(key(r), []).append(r)
order = [k for k in ["penalty", "se", "sp", "gauge", "legendre mu=0.05", "legendre mu=0.02", "legendre mu=0.2"] if k in groups]


def ms(vals):
    v = np.array(vals, float)
    return float(np.nanmean(v)), float(np.nanstd(v))


lines = []
summary = {}
for cond, title in [("A", "A: in-distribution"), ("B", "B: more obstacles (OOD)"), ("C", "C: constraint shift (OOD)")]:
    lines.append(f"\n### Condition {title}\n")
    lines.append("| Method | seeds | Success % | Collision % | Time-to-goal | Deadlock % | Intervention |")
    lines.append("|---|---|---|---|---|---|---|")
    for k in order:
        rs = groups[k]
        f = [r["final"][cond] for r in rs]
        s, sd = ms([x["success"] * 100 for x in f]); c, cd = ms([x["collision_rate"] * 100 for x in f])
        t, td = ms([x["time_to_goal"] for x in f]); d, dd = ms([x["deadlock_rate"] * 100 for x in f])
        iv = ms([x["intervention"] for x in f])[0]
        summary[(k, cond)] = dict(success=s, ttg=t, coll=c, per_seed=[x["success"] * 100 for x in f])
        lines.append(f"| {k} | {len(rs)} | {s:.1f} ± {sd:.1f} | {c:.1f} ± {cd:.1f} | {t:.1f} ± {td:.1f} | "
                     f"{d:.1f} ± {dd:.1f} | {'–' if np.isnan(iv) else f'{iv:.3f}'} |")

lines.append("\n### Jacobian diagnostics (actor-layer d u_exec/d a, units of umax), pooled over A/B/C\n")
lines.append("| Method | frac sigma_min<1e-3 | frac <1e-6 | frac <1e-2 | near-obstacle frac <1e-3 | median log10 sigma_min |")
lines.append("|---|---|---|---|---|---|")
jac = {}
for k in order:
    J = [r["final"][c]["jacobian"] for r in groups[k] for c in "ABC"]
    g = lambda n: np.nanmean([j[n] for j in J])
    jac[k] = g("frac_rank_deficient")
    lines.append(f"| {k} | {g('frac_rank_deficient'):.3f} | {g('frac_smin_lt_1e6'):.3f} | {g('frac_smin_lt_1e2'):.3f} | "
                 f"{g('frac_rank_deficient_near'):.3f} | {g('median_log10_smin'):.2f} |")

lines.append("\n### Pre-registered hypotheses\n")
safe = [k for k in order if k in ("se", "sp", "gauge", "legendre mu=0.05")]
h1 = all(summary[(k, c)]["coll"] == 0 for k in safe for c in "ABC")
lines.append(f"- **H1 (zero collisions for se/sp/gauge/legendre, all conditions):** {'HOLDS' if h1 else 'FAILS'}")
if "legendre mu=0.05" in groups and "sp" in groups:
    h2 = jac["legendre mu=0.05"] == 0.0 and jac["sp"] > 0.0
    lines.append(f"- **H2 (sigma_min<1e-3: sp > 0, legendre = 0):** sp={jac['sp']:.3f}, legendre={jac['legendre mu=0.05']:.3f} -> "
                 f"{'HOLDS' if h2 else 'FAILS'}")
    wins, detail = 0, []
    for c in "ABC":
        L = summary[("legendre mu=0.05", c)]
        cands = [k for k in ("se", "sp", "gauge") if (k, c) in summary]
        bk = max(cands, key=lambda k: summary[(k, c)]["success"])
        best = summary[(bk, c)]
        d_succ = L["success"] - best["success"]
        ok = d_succ >= 3 or (abs(d_succ) <= 3 and (best["ttg"] - L["ttg"]) / best["ttg"] >= 0.10)
        wins += ok
        detail.append(f"{c}: dsucc={d_succ:+.1f} pts vs {bk}, TTG {L['ttg']:.1f} vs {best['ttg']:.1f} -> {'win' if ok else 'no'}")
    lines.append(f"- **H3 (>=2/3 conditions win):** {wins}/3 -> {'HOLDS' if wins >= 2 else 'FAILS'}  ({'; '.join(detail)})")
if "penalty" in groups:
    pc = [summary[("penalty", c)]["coll"] for c in "ABC"]
    lines.append(f"- **H4 (penalty collides in OOD):** collision % A/B/C = {pc[0]:.1f}/{pc[1]:.1f}/{pc[2]:.1f} -> "
                 f"{'HOLDS' if (pc[1] > 0 or pc[2] > 0) else 'FAILS'}")

open(os.path.join(res_dir, "summary.md"), "w").write("\n".join(lines))
print("\n".join(lines))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for k in order:
    cs = [r["curve"] for r in groups[k]]
    n = min(len(c) for c in cs)
    x = [cs[0][i]["steps"] for i in range(n)]
    for a_, m_ in zip(ax, ("success", "time_to_goal")):
        y = np.array([[c[i][m_] for i in range(n)] for c in cs])
        a_.plot(x, np.nanmean(y, 0), label=k)
        a_.fill_between(x, np.nanmean(y, 0) - np.nanstd(y, 0), np.nanmean(y, 0) + np.nanstd(y, 0), alpha=0.15)
ax[0].set_ylabel("success (cond. A)"); ax[1].set_ylabel("time-to-goal"); ax[0].legend(fontsize=7)
for a_ in ax: a_.set_xlabel("env steps")
plt.tight_layout(); plt.savefig(os.path.join(res_dir, "learning_curves.png"), dpi=150)
