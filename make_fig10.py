import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = json.load(open("final_analysis.json"))["10"]
NAMES = {"CMAES":"CMA-ES (IPOP)","LSHADE":"L-SHADE"}
order = sorted(R["ranks"], key=R["ranks"].get)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.3), dpi=150, gridspec_kw={"width_ratios":[1,1.15]})
cols = ["#8a3b12" if a=="MSBO" else "#8c96a3" for a in order]
bars = a1.barh([NAMES.get(a,a) for a in order][::-1], [R["ranks"][a] for a in order][::-1], color=cols[::-1])
for b, a in zip(bars, order[::-1]): a1.text(b.get_width()+0.08, b.get_y()+b.get_height()/2, f"{R['ranks'][a]:.2f}", va="center", fontsize=8)
a1.set_xlabel("Average Friedman rank (lower = better)"); a1.set_xlim(0, 8.8)
a1.set_title(f"(a) Ranks, D=10, {R['n_funcs']} functions, {R['runs']} runs", fontsize=10); a1.grid(axis="x", alpha=.3)
riv = ["GWO","GA","PSO","ABC","DE","LSHADE","CMAES"]
w = [R["wtl"][a]["win"] for a in riv]; t = [R["wtl"][a]["tie"] for a in riv]; l = [R["wtl"][a]["loss"] for a in riv]
y = np.arange(len(riv))[::-1]
a2.barh(y, w, color="#3b7d4f", label="MSBO wins"); a2.barh(y, t, left=w, color="#c9c9c9", label="tie")
a2.barh(y, l, left=np.array(w)+np.array(t), color="#b23a3a", label="MSBO loses")
for yi, wi, ti, li in zip(y, w, t, l):
    if wi: a2.text(wi/2, yi, str(wi), ha="center", va="center", color="white", fontsize=8)
    if ti: a2.text(wi+ti/2, yi, str(ti), ha="center", va="center", fontsize=8)
    if li: a2.text(wi+ti+li/2, yi, str(li), ha="center", va="center", color="white", fontsize=8)
a2.set_yticks(y); a2.set_yticklabels(["MSBO vs "+NAMES.get(a,a) for a in riv], fontsize=8.5)
a2.set_xlim(0, R["n_funcs"]); a2.set_xlabel("Number of the 29 functions")
a2.set_title("(b) Per-function outcome (Holm-adjusted, A12 ≥ 0.64 / ≤ 0.36)", fontsize=10)
a2.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
fig.tight_layout(); fig.savefig("fig10_full_d10.png", bbox_inches="tight"); print("saved")
