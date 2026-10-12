"""Pre-registered selection of the baseline output scale (docs/PREREGISTRATION_v3.md, section 3).
Rule: for each of se/sp/gauge pick the scale with the highest mean condition-A success over the tuning seeds;
differences < 0.5 points count as ties, broken towards the scale closest to 1 (in log)."""
import glob
import json
import math
import os

runs = [json.load(open(f)) for f in glob.glob("results_v4_tune/*.json")]
sel, table = {}, {}
for m in ("se", "sp", "gauge"):
    rows = {}
    for r in runs:
        if r["method"] == m:
            rows.setdefault(r["scale"], []).append(100 * r["final"]["A"]["success"])
    assert rows and all(len(v) == 3 for v in rows.values()) and len(rows) == 4, (m, {k: len(v) for k, v in rows.items()})
    means = {k: sum(v) / len(v) for k, v in rows.items()}
    top = max(means.values())
    cands = [k for k, v in means.items() if top - v < 0.5]
    sel[m] = min(cands, key=lambda k: abs(math.log(k)))
    table[m] = {str(k): round(v, 2) for k, v in sorted(means.items())}
os.makedirs("results_v4", exist_ok=True)
json.dump(sel, open("results_v4/selected_scales.json", "w"), indent=1)
json.dump(table, open("results_v4/tuning_table.json", "w"), indent=1)
print("tuning means (cond A success %):", json.dumps(table, indent=1))
print("selected:", sel)
