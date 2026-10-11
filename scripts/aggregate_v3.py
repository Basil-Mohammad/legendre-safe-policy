"""Evaluate the Study-3 hypotheses exactly as written in docs/PREREGISTRATION_v2.1.md (S1-S5)."""
import glob
import json
import os
import sys

import numpy as np

res_dir = sys.argv[1] if len(sys.argv) > 1 else "results_v3"
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(res_dir, "*.json")))]


def key(r):
    if r["method"] == "penalty":
        return f"penalty_l{r['penalty']}"
    if r["method"] == "legendre" and abs(r["mu"] - 0.05) > 1e-12:
        return f"legendre_mu{r['mu']}"
    return r["method"]


groups = {}
for r in runs:
    groups.setdefault(key(r), []).append(r)

ORDER = ["penalty_l0.5", "penalty_l2.0", "penalty_l8.0", "se", "sp", "gauge", "sp_m09", "sp_m07",
         "legendre_st", "legendre_mu0.02", "legendre_mu0.2", "legendre"]
order = [k for k in ORDER if k in groups]
SAFE = [k for k in order if not k.startswith("penalty")]
BASE = [k for k in ("se", "sp", "gauge", "sp_m09", "sp_m07") if k in groups]


def vals(k, cond, field, scale=1.0):
    return [r["final"][cond][field] * scale for r in groups[k]]


def mean(k, cond, field, scale=1.0):
    return float(np.nanmean(vals(k, cond, field, scale)))


lines = []
for cond, title in (("A", "A: in-distribution"), ("B", "B: more obstacles (OOD)"), ("C", "C: constraint shift (OOD)")):
    lines += [f"\n### Condition {title}\n",
              "| Method | n seeds | Success % (mean ± sd) | per-seed | Collision % | Fallback step rate % | "
              "Episodes using fallback % | Time-to-goal | Deadlock % |",
              "|---|---|---|---|---|---|---|---|---|"]
    for k in order:
        s = vals(k, cond, "success", 100)
        lines.append(
            f"| {k} | {len(s)} | {np.mean(s):.1f} ± {np.std(s):.1f} | {', '.join(f'{x:.1f}' for x in s)} | "
            f"{mean(k, cond, 'collision_rate', 100):.2f} | {mean(k, cond, 'fallback_step_rate', 100):.3f} | "
            f"{mean(k, cond, 'frac_episodes_using_fallback', 100):.1f} | {mean(k, cond, 'time_to_goal'):.1f} | "
            f"{mean(k, cond, 'deadlock_rate', 100):.1f} |")

lines += ["\n### Jacobian diagnostic (d a_exec / d nu), non-fallback states pooled over conditions and seeds\n",
          "| Method | n states | frac sigma_min<1e-3 | median log10 sigma_min |", "|---|---|---|---|"]
jac = {}
for k in order:
    J = [r["final"][c]["jacobian"] for r in groups[k] for c in "ABC"]
    n = sum(j["n"] for j in J)
    f = sum(j["frac_rank_deficient"] * j["n"] for j in J if j["n"]) / max(n, 1)
    jac[k] = f
    med = np.nanmean([j["median_log10_smin"] for j in J])
    lines.append(f"| {k} | {n} | {f:.5f} | {med:.2f} |")

n_seeds = len(groups.get("legendre", []))
complete = n_seeds > 0 and all(len(groups[k]) == n_seeds for k in order)
lines.append("\n### Pre-registered hypotheses (v2.1)\n")
if not complete:
    lines.append(f"**INCOMPLETE: seeds per method = { {k: len(groups[k]) for k in order} }. Provisional.**\n")

coll_bad = [(k, c, mean(k, c, "collision_rate", 100)) for k in SAFE for c in "ABC" if max(vals(k, c, "collision_rate")) > 0]
s1_collision = len(coll_bad) == 0
fb_A = mean("legendre", "A", "fallback_step_rate", 100) if "legendre" in groups else float("nan")
lines.append(f"- **S1 (zero collisions, all safe methods, with fallback):** {'HOLDS' if s1_collision else 'FAILS: ' + str(coll_bad)}")
lines.append(f"  - legendre fallback step rate in A: {fb_A:.3f} % (threshold < 1 %) -> {'met' if fb_A < 1 else 'NOT met'}")

if "legendre" in groups and "sp" in groups:
    s2 = jac["legendre"] < 1e-3 and jac["sp"] > 0 and jac["legendre"] <= jac["sp"] / 100
    lines.append(f"- **S2 (legendre < 0.1 % and >= 100x below sp):** legendre={jac['legendre']:.5f}, sp={jac['sp']:.5f} -> "
                 f"{'HOLDS' if s2 else 'FAILS'}")


def margin_test(name, rivals):
    wins, det = 0, []
    for c in "ABC":
        bk = max(rivals, key=lambda k: mean(k, c, "success"))
        d = 100 * (mean("legendre", c, "success") - mean(bk, c, "success"))
        wins += d >= 3
        det.append(f"{c}: {d:+.1f} pts vs {bk}")
    holds = wins >= 2
    lines.append(f"- **{name}:** {wins}/3 conditions with >= +3 pts -> {'HOLDS' if holds else 'FAILS'}  ({'; '.join(det)})")
    return holds


s3 = margin_test("S3 (vs best of se/sp/gauge/sp_m09/sp_m07)", BASE) if "legendre" in groups and BASE else None
pens = [k for k in order if k.startswith("penalty")]
if pens:
    tuned = max(pens, key=lambda k: mean(k, "A", "success"))
    coll = [mean(tuned, c, "collision_rate", 100) for c in "ABC"]
    lines.append(f"- **S4 (tuned penalty = {tuned} collides in OOD):** collision % A/B/C = "
                 f"{coll[0]:.1f}/{coll[1]:.1f}/{coll[2]:.1f} -> {'HOLDS' if (coll[1] > 0 or coll[2] > 0) else 'FAILS'}")
s5 = margin_test("S5 (mechanism: legendre vs legendre_st)", ["legendre_st"]) if "legendre" in groups and "legendre_st" in groups else None

lines.append("\n### Decision (per preregistration)\n")
if not s1_collision:
    lines.append("S1 FAILS (collision with fallback) -> STOP, find the cause before anything else.")
elif s3 is None:
    lines.append("Data incomplete.")
elif s3:
    lines.append("S1 and S3 hold -> the paper covers Study 1 (exploratory pilot), Study 2 and Study 3."
                 + ("" if s5 is None else (" S5 holds: gradient geometry contributes." if s5 else
                    " S5 fails: any advantage is attributed to the forward-map geometry, not the gradient geometry.")))
else:
    lines.append("S3 FAILS -> the advantage does not carry to relative degree 2; restrict the paper to the "
                 "single-integrator setting (stated in title/abstract) or rework the idea.")
if not complete:
    lines.append("\n(Provisional: not all runs finished.)")

open(os.path.join(res_dir, "summary_v3.md"), "w").write("\n".join(lines))
print("\n".join(lines))
