"""Evaluate the Study-2 hypotheses exactly as written in docs/PREREGISTRATION_v4.md."""
import glob
import json
import os
import sys

import numpy as np

res_dir = sys.argv[1] if len(sys.argv) > 1 else "results_v5"
runs = [r for r in (json.load(open(f)) for f in sorted(glob.glob(os.path.join(res_dir, "*.json")))) if "method" in r]


def key(r):
    if r["method"] == "penalty":
        return f"penalty_l{r['penalty']}"
    return r["method"]


groups = {}
for r in runs:
    groups.setdefault(key(r), []).append(r)

ORDER = ["penalty_l0.5", "penalty_l2.0", "se", "sp", "gauge", "sp_m09", "legendre"]
order = [k for k in ORDER if k in groups]
SAFE = [k for k in order if not k.startswith("penalty")]
BASE = [k for k in ("se", "sp", "gauge", "sp_m09") if k in groups]


def vals(k, cond, field, scale=1.0):
    return [r["final"][cond][field] * scale for r in groups[k]]


def mean(k, cond, field, scale=1.0):
    return float(np.nanmean(vals(k, cond, field, scale)))


lines = []
for cond, title in (("A", "A: in-distribution"), ("B", "B: more obstacles (OOD)"), ("C", "C: constraint shift (OOD)")):
    lines += [f"\n### Condition {title}\n",
              "| Method | n seeds | Success % (mean ± sd) | per-seed | Collision % | Time-to-goal | Deadlock % |",
              "|---|---|---|---|---|---|---|"]
    for k in order:
        s = vals(k, cond, "success", 100)
        lines.append(
            f"| {k} | {len(s)} | {np.mean(s):.1f} ± {np.std(s):.1f} | {', '.join(f'{x:.1f}' for x in s)} | "
            f"{mean(k, cond, 'collision_rate', 100):.1f} | {mean(k, cond, 'time_to_goal'):.1f} | "
            f"{mean(k, cond, 'deadlock_rate', 100):.1f} |")

lines += ["\n### Jacobian diagnostic (actor-layer d u_exec / d a), pooled over conditions and seeds\n",
          "| Method | n states | frac sigma_min<1e-3 | median log10 sigma_min |", "|---|---|---|---|"]
jac = {}
for k in order:
    J = [r["final"][c]["jacobian"] for r in groups[k] for c in "ABC"]
    n = sum(j["n"] for j in J)
    f = sum(j["frac_rank_deficient"] * j["n"] for j in J) / n
    jac[k] = f
    lines.append(f"| {k} | {n} | {f:.5f} | {np.nanmean([j['median_log10_smin'] for j in J]):.2f} |")

lines.append("\n### Pre-registered hypotheses (v2)\n")
n_seeds = len(groups.get("legendre", []))
complete = all(len(groups[k]) == n_seeds for k in order) and n_seeds > 0
if not complete:
    lines.append(f"**INCOMPLETE: not all methods have the same number of seeds "
                 f"({ {k: len(groups[k]) for k in order} }). Hypotheses below are provisional.**\n")

h1 = all(mean(k, c, "collision_rate") == 0 and max(vals(k, c, "collision_rate")) == 0 for k in SAFE for c in "ABC")
lines.append(f"- **U1 (zero collisions, all safe-layer methods incl. controls):** {'HOLDS' if h1 else 'FAILS'}")

if "legendre" in groups and "sp" in groups:
    h2 = jac["legendre"] < 1e-3 and (jac["sp"] > 0 and jac["legendre"] <= jac["sp"] / 100)
    lines.append(f"- **U2 (legendre frac<0.1% and >=100x below sp):** legendre={jac['legendre']:.5f}, sp={jac['sp']:.5f} -> "
                 f"{'HOLDS' if h2 else 'FAILS'}")


def boot_lb(x, y, n=10000, seed=0):
    d = np.asarray(x) - np.asarray(y)
    rng = np.random.default_rng(seed)
    return float(np.percentile(d[rng.integers(0, len(d), (n, len(d)))].mean(1), 5))


def paired(k, ref, c):
    a = {r["seed"]: r["final"][c]["success"] * 100 for r in groups[k]}
    b = {r["seed"]: r["final"][c]["success"] * 100 for r in groups[ref]}
    ss = sorted(set(a) & set(b))
    return [a[s] for s in ss], [b[s] for s in ss]


def margin_test(name, rivals):
    wins, det = 0, []
    for c in "ABC":
        bk = max(rivals, key=lambda k: mean(k, c, "success"))
        x, y = paired("legendre", bk, c)
        d = float(np.mean(x) - np.mean(y))
        lb = boot_lb(x, y)
        wins += (d >= 3 and lb > 0)
        det.append(f"{c}: {d:+.1f} pts vs {bk}, 95% one-sided LB {lb:+.1f}, n={len(x)}")
    holds = wins >= 2
    lines.append(f"- **{name}:** {wins}/3 conditions with >= +3 pts AND bootstrap LB > 0 -> {'HOLDS' if holds else 'FAILS'}  ({'; '.join(det)})")
    return holds


h3 = margin_test("U3 (vs best tuned baseline of se/sp/gauge/sp_m09)", BASE) if "legendre" in groups and BASE else None

pens = [k for k in order if k.startswith("penalty")]
if pens:
    tuned = max(pens, key=lambda k: mean(k, "A", "success"))
    coll = [mean(tuned, c, "collision_rate", 100) for c in "ABC"]
    h4 = coll[1] > 0 or coll[2] > 0
    lines.append(f"- **U4 (tuned penalty = {tuned} collides in OOD):** collision % A/B/C = "
                 f"{coll[0]:.1f}/{coll[1]:.1f}/{coll[2]:.1f} -> {'HOLDS' if h4 else 'FAILS'}")

lines.append("\n### Decision (per preregistration)\n")
if not h1:
    lines.append("U1 FAILS or data incomplete -> STOP and diagnose.")
elif h3:
    lines.append("U1 and U3 hold: the relative-degree-one advantage survives scale-tuned baselines.")
else:
    lines.append("U3 FAILS: the relative-degree-one advantage does not survive scale-tuned baselines; report a negative result.")
if not complete:
    lines.append("\n(Provisional: not all runs finished.)")

open(os.path.join(res_dir, "summary_v5.md"), "w").write("\n".join(lines))
print("\n".join(lines))
