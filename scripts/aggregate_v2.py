"""Evaluate the Study-2 hypotheses exactly as written in docs/PREREGISTRATION_v2.md."""
import glob
import json
import os
import sys

import numpy as np

res_dir = sys.argv[1] if len(sys.argv) > 1 else "results_v2"
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(res_dir, "*.json")))]


def key(r):
    if r["method"] == "penalty":
        return f"penalty_l{r['penalty']}"
    return r["method"]


groups = {}
for r in runs:
    groups.setdefault(key(r), []).append(r)

ORDER = ["penalty_l0.5", "penalty_l2.0", "penalty_l8.0", "se", "sp", "gauge", "sp_m09", "sp_m07", "gauge_m09", "legendre"]
order = [k for k in ORDER if k in groups]
SAFE = [k for k in order if not k.startswith("penalty")]
BASE = [k for k in ("se", "sp", "gauge") if k in groups]
CTRL = [k for k in ("sp_m09", "sp_m07", "gauge_m09") if k in groups]


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
lines.append(f"- **H1 (zero collisions, all safe-layer methods incl. controls):** {'HOLDS' if h1 else 'FAILS'}")

if "legendre" in groups and "sp" in groups:
    h2 = jac["legendre"] < 1e-3 and (jac["sp"] > 0 and jac["legendre"] <= jac["sp"] / 100)
    lines.append(f"- **H2' (legendre frac<0.1% and >=100x below sp):** legendre={jac['legendre']:.5f}, sp={jac['sp']:.5f} -> "
                 f"{'HOLDS' if h2 else 'FAILS'}")


def margin_test(name, rivals):
    wins, det = 0, []
    for c in "ABC":
        bk = max(rivals, key=lambda k: mean(k, c, "success"))
        d = 100 * (mean("legendre", c, "success") - mean(bk, c, "success"))
        ok = d >= 3
        wins += ok
        det.append(f"{c}: {d:+.1f} pts vs {bk}")
    holds = wins >= 2
    lines.append(f"- **{name}:** {wins}/3 conditions with >= +3 pts -> {'HOLDS' if holds else 'FAILS'}  ({'; '.join(det)})")
    return holds


h3 = margin_test("H3' (vs best of se/sp/gauge)", BASE) if "legendre" in groups and BASE else None
h5 = margin_test("H5 (vs best margin control)", CTRL) if "legendre" in groups and CTRL else None

pens = [k for k in order if k.startswith("penalty")]
if pens:
    tuned = max(pens, key=lambda k: mean(k, "A", "success"))
    coll = [mean(tuned, c, "collision_rate", 100) for c in "ABC"]
    h4 = coll[1] > 0 or coll[2] > 0
    lines.append(f"- **H4' (tuned penalty = {tuned} collides in OOD):** collision % A/B/C = "
                 f"{coll[0]:.1f}/{coll[1]:.1f}/{coll[2]:.1f} -> {'HOLDS' if h4 else 'FAILS'}")

lines.append("\n### Decision (per preregistration)\n")
if h1 and h3 is not None and h5 is not None:
    if h5 and h3:
        lines.append("H5 and H3' hold with H1: the Legendre map itself is credited -> continue to Study 3.")
    elif h3 and not h5:
        lines.append("H3' holds but H5 FAILS: the advantage is explained by interior margin -> reframe the contribution "
                     "before any further compute.")
    else:
        lines.append("H3' fails: no advantage over unshrunk baselines on fresh seeds -> rethink the idea.")
else:
    lines.append("H1 FAILS or data incomplete -> STOP / wait for runs.")
if not complete:
    lines.append("\n(Provisional: not all runs finished.)")

open(os.path.join(res_dir, "summary_v2.md"), "w").write("\n".join(lines))
print("\n".join(lines))
