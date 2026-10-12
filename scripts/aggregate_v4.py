"""Evaluate the Study-4 hypotheses T1-T5 exactly as written in docs/PREREGISTRATION_v3.md."""
import glob
import json
import os
import sys

import numpy as np

res_dir = sys.argv[1] if len(sys.argv) > 1 else "results_v4"
runs = [r for r in (json.load(open(f)) for f in sorted(glob.glob(os.path.join(res_dir, "*.json")))) if "method" in r]  # skip selected_scales/tuning_table


def key(r):
    if r["method"] == "penalty":
        return f"penalty_l{r['penalty']}"
    return r["method"]


groups = {}
for r in runs:
    groups.setdefault(key(r), []).append(r)

ORDER = ["penalty_l0.5", "penalty_l2.0", "se", "sp", "gauge", "sp_m09", "legendre_st", "legendre"]
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
lines.append("\n### Pre-registered hypotheses (Study 4, PREREGISTRATION_v3)\n")
if not complete:
    lines.append(f"**INCOMPLETE: seeds per method = { {k: len(groups[k]) for k in order} }. Provisional.**\n")

coll_bad = [(k, c, mean(k, c, "collision_rate", 100)) for k in SAFE for c in "ABC" if max(vals(k, c, "collision_rate")) > 0]
s1_collision = len(coll_bad) == 0
fb_A = mean("legendre", "A", "fallback_step_rate", 100) if "legendre" in groups else float("nan")
lines.append(f"- **T1 (zero collisions, all safe methods, with fallback):** {'HOLDS' if s1_collision else 'FAILS: ' + str(coll_bad)}")
lines.append(f"  - legendre fallback step rate in A: {fb_A:.3f} % (threshold < 1 %) -> {'met' if fb_A < 1 else 'NOT met'}")

if "legendre" in groups and "sp" in groups:
    s2 = jac["legendre"] < 1e-3 and jac["sp"] > 0 and jac["legendre"] <= jac["sp"] / 100
    lines.append(f"- **T2 (legendre < 0.1 % and >= 100x below sp):** legendre={jac['legendre']:.5f}, sp={jac['sp']:.5f} -> "
                 f"{'HOLDS' if s2 else 'FAILS'}")


def boot_lb(a, b, n=10000, seed=0):
    """One-sided 95% lower bound of mean(a-b) (percentile bootstrap, seeds resampled jointly, in points)."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.percentile(m, 5))


def paired(k, ref, c):
    """Per-seed success of k and ref (points), aligned by seed."""
    a = {r["seed"]: r["final"][c]["success"] * 100 for r in groups[k]}
    b = {r["seed"]: r["final"][c]["success"] * 100 for r in groups[ref]}
    ss = sorted(set(a) & set(b))
    return [a[s] for s in ss], [b[s] for s in ss]


def margin_test(name, rivals):
    wins, det = 0, []
    for c in "ABC":
        bk = max(rivals, key=lambda k: mean(k, c, "success"))
        a, b = paired("legendre", bk, c)
        d = float(np.mean(a) - np.mean(b))
        lb = boot_lb(a, b)
        wins += (d >= 3 and lb > 0)
        det.append(f"{c}: {d:+.1f} pts vs {bk}, 95% one-sided LB {lb:+.1f}, n={len(a)}")
    holds = wins >= 2
    lines.append(f"- **{name}:** {wins}/3 conditions with >= +3 pts AND bootstrap LB > 0 -> {'HOLDS' if holds else 'FAILS'}  ({'; '.join(det)})")
    return holds


s3 = margin_test("T3 (vs best tuned baseline of se/sp/gauge/sp_m09)", BASE) if "legendre" in groups and BASE else None
pens = [k for k in order if k.startswith("penalty")]
if pens:
    tuned = max(pens, key=lambda k: mean(k, "A", "success"))
    coll = [mean(tuned, c, "collision_rate", 100) for c in "ABC"]
    lines.append(f"- **T4 (tuned penalty = {tuned} collides in OOD):** collision % A/B/C = "
                 f"{coll[0]:.1f}/{coll[1]:.1f}/{coll[2]:.1f} -> {'HOLDS' if (coll[1] > 0 or coll[2] > 0) else 'FAILS'}")
s5 = margin_test("T5 (mechanism: legendre vs legendre_st)", ["legendre_st"]) if "legendre" in groups and "legendre_st" in groups else None

lines.append("\n### Decision (per preregistration)\n")
if not s1_collision:
    lines.append("T1 FAILS (collision with fallback) -> STOP, find the cause before anything else.")
elif s3 is None:
    lines.append("Data incomplete.")
elif s3:
    lines.append("T1 and T3 hold -> the paper covers Study 1 (exploratory pilot), Study 2 and Study 3."
                 + ("" if s5 is None else (" T5 holds: gradient geometry contributes." if s5 else
                    " T5 fails: any advantage is attributed to the forward-map geometry, not the gradient geometry.")))
else:
    lines.append("T3 FAILS -> the advantage does not carry to relative degree 2; restrict the paper to the "
                 "single-integrator setting (stated in title/abstract) or rework the idea.")
if not complete:
    lines.append("\n(Provisional: not all runs finished.)")

open(os.path.join(res_dir, "summary_v4.md"), "w").write("\n".join(lines))
print("\n".join(lines))
