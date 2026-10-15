"""LaTeX tables and figures for Study 6 from summary_v6.json.   python scripts/make_tables_v6.py summary.json outdir"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = json.load(open(sys.argv[1])); out = sys.argv[2]
os.makedirs(out, exist_ok=True)
NAMES = dict(legendre="Legendre", sp="Projection (SP)", gauge="Gauge", se="Env.\\ projection (SE)", legendre_st="Legendre-ST", pen2="Penalty ($\\lambda{=}2$)")
ENVN = dict(navsi="navsi", navdi="navdi", unicycle="unicycle", fetch="fetch", panda="panda")
ORDER = ["legendre", "sp", "gauge", "se", "legendre_st", "pen2"]
pc = lambda x: f"{100*x:.1f}"

# ---- Table: main results per (env, algo)
L = []
for env in ["navsi", "navdi", "unicycle", "fetch", "panda"]:
    for al in ["sac", "ppo"]:
        rows = [(m, S["table"].get(f"{env}/{al}/{m}")) for m in ORDER]
        rows = [(m, r) for m, r in rows if r]
        if not rows:
            continue
        L.append(f"\\midrule\n\\multicolumn{{8}}{{l}}{{\\textit{{{ENVN[env]} / {al.upper()}}}}}\\\\")
        for m, r in rows:
            sc = "--" if m == "pen2" else f"{r['scale']:g}"
            cells = " & ".join(f"{pc(r[c]['success'])} \\,[{100*r[c]['lo']:.0f},{100*r[c]['hi']:.0f}]" for c in "ABC")
            coll = " / ".join(str(r[c]["coll_total"]) for c in "ABC")
            rest = f"{r['A']['restorations']:.1f}"
            b = "\\textbf{" if m == "legendre" else ""
            e = "}" if m == "legendre" else ""
            L.append(f"{b}{NAMES[m]}{e} & {sc} & {cells} & {coll} & {rest}\\\\")
open(os.path.join(out, "tab_main.tex"), "w").write(
    "\\begin{tabular}{llccccc}\n\\toprule\nMap & $s$ & A & B & C & Collisions (A/B/C, steps) & Restor.\\ / ep.\\\\\n" + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- Table: P1 contrasts
L = []
for k, v in sorted(S["P1"].items()):
    env, al, c, tag = k.split("/")
    if tag != "refA":
        continue
    flag = "S" if v["superior"] else ("I" if v["inferior"] else ("N" if v["noninferior"] else "--"))
    L.append(f"{ENVN[env]} & {al.upper()} & {c} & {v['ref'].upper()} & ${100*v['diff']:+.1f}$ & ${100*v['lb95']:+.1f}$ & [{100*v['ci'][0]:+.1f}, {100*v['ci'][1]:+.1f}] & {flag}\\\\")
open(os.path.join(out, "tab_p1.tex"), "w").write(
    "\\begin{tabular}{lllllccc}\n\\toprule\nEnv. & Alg. & Cond. & Ref. & $\\Delta$ (pts) & LB$_{95}$ & 95\\% CI & Verdict\\\\\n\\midrule\n" + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- Table: faults (collision rate % / violation x100), mean over cells for each map
FA = ["drop", "drop_wd", "stale", "rowdrop", "latency"]
agg = {}
for k, row in S["F"].items():
    env, al, m = k.split("/")
    for f in FA + ["none"]:
        agg.setdefault((m, f), []).append((row[f]["collided"], row[f]["viol"], row[f]["success"]))
L = []
for m in ORDER:
    if (m, "none") not in agg:
        continue
    cells = " & ".join(f"{100*np.mean([x[0] for x in agg[(m,f)]]):.1f}\\,/\\,{np.mean([x[1] for x in agg[(m,f)]]):.3f}" for f in ["none"] + FA)
    L.append(f"{NAMES[m]} & {cells}\\\\")
open(os.path.join(out, "tab_faults.tex"), "w").write(
    "\\begin{tabular}{lcccccc}\n\\toprule\nMap & none & drop & drop+WD & stale & rowdrop & latency\\\\\n\\midrule\n" + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- Figure: tuning curves
cells = sorted({(k.split("/")[0], k.split("/")[1]) for k in S["S2"]})
if cells:
    fig, axs = plt.subplots(1, len(cells), figsize=(2.6 * len(cells), 2.5), sharey=True, squeeze=False)
    col = dict(legendre="#1b6ca8", sp="#d95f02", gauge="#7570b3", se="#1b9e77")
    for ax, (env, al) in zip(axs[0], cells):
        for m in ["legendre", "sp", "gauge", "se"]:
            v = S["S2"].get(f"{env}/{al}/{m}")
            if not v:
                continue
            xs = sorted(map(float, v["means"]))
            ax.plot(range(len(xs)), [100 * v["means"][str(x) if str(x) in v["means"] else str(float(x))] for x in xs], "-o", ms=3, color=col[m], label=NAMES[m].replace("\\", "").replace("{", "").replace("}", "").replace("$", ""))
        ax.set_title(f"{env}/{al.upper()}", fontsize=8); ax.set_xticks(range(4)); ax.set_xticklabels(["1", "2", "3", "4"], fontsize=7)
        ax.set_xlabel("grid point", fontsize=7); ax.tick_params(labelsize=7)
    axs[0][0].set_ylabel("tuning success A (%)", fontsize=8)
    axs[0][-1].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_tuning.pdf")); plt.close(fig)

# ---- Figure: P1 forest (Legendre minus best baseline)
keys = [k for k in sorted(S["P1"]) if k.endswith("refA")]
if keys:
    fig, ax = plt.subplots(figsize=(5.2, 0.22 * len(keys) + 0.9))
    for i, k in enumerate(keys):
        v = S["P1"][k]
        c = "#1b6ca8" if v["superior"] else ("#b2182b" if v["inferior"] else "#777777")
        ax.plot([100 * v["ci"][0], 100 * v["ci"][1]], [i, i], color=c, lw=1.5)
        ax.plot(100 * v["diff"], i, "o", color=c, ms=3.5)
    ax.axvline(0, color="k", lw=0.6); ax.axvline(3, color="k", lw=0.4, ls=":"); ax.axvline(-5, color="k", lw=0.4, ls=":")
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([k.replace("/refA", "").replace("/", " ") for k in keys], fontsize=6)
    ax.set_xlabel("Legendre minus best baseline (success points, 95% CI)", fontsize=7); ax.tick_params(axis="x", labelsize=7)
    ax.invert_yaxis(); fig.tight_layout(); fig.savefig(os.path.join(out, "fig_p1.pdf")); plt.close(fig)
print("tables written to", out)
