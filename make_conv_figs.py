import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
data = json.load(open("convergence_data.json"))
ALGOS = ["MSBO","GA","PSO","DE","ABC","GWO","CMAES","LSHADE"]
NAMES = {"CMAES":"CMA-ES (IPOP)"}
COL = {"MSBO":"#8a3b12","GA":"#4c72b0","PSO":"#55a868","DE":"#c44e52","ABC":"#8172b2","GWO":"#937860","CMAES":"#b8a000","LSHADE":"#2a9db5"}
for fk, title in (("9","F9 (simple multimodal)"),("16","F16 (hybrid)")):
    fig, ax = plt.subplots(figsize=(7.4,4.6), dpi=150)
    grid = np.linspace(0, 100000, 300)
    for a in ALGOS:
        ys = []
        for c in data[fk][a]:
            fes = np.array([p[0] for p in c]); err = np.array([max(p[1], 1e-9) for p in c])   # floor at the 1e-8 CEC zero-threshold region
            ys.append(np.interp(grid, fes, err))
        ax.plot(grid, np.median(np.array(ys), axis=0), label=NAMES.get(a,a), color=COL[a], lw=1.7)
    ax.axhline(1e-8, color="gray", ls=":", lw=1); ax.text(2000, 1.6e-8, "1e-8: counted as zero (CEC rule)", fontsize=7, color="gray")
    ax.set_yscale("log"); ax.set_xlabel("Function evaluations (FES)"); ax.set_ylabel("Median error (log scale), 5 runs")
    ax.set_title(f"Measured convergence, D=10, {title}"); ax.legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.92); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(f"conv_F{fk}.png"); print("saved", fk)
