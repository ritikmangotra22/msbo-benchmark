"""Small supplementary run to capture per-FES convergence histories (not saved by
run_pilot.py, which only kept final results). Real runs, same seeds convention."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
from problem import CEC2017Problem
from msbo import optimize as msbo_opt
import baselines as B

FUNCS = [9, 16]   # one simple-multimodal, one hybrid (representative, cheap to plot)
DIM = 10
MAX_FES = 10000 * DIM
N_REPEATS = 5
ALGOS = {"MSBO": (msbo_opt, {}), "GA": (B.ga, {}), "PSO": (B.pso, {}), "DE": (B.de, {}),
         "ABC": (B.abc, {}), "GWO": (B.gwo, {}), "CMAES": (B.cmaes, {}), "LSHADE": (B.lshade, {})}

out = {}
t0 = time.time()
for fn in FUNCS:
    out[fn] = {}
    for name, (fun, kw) in ALGOS.items():
        curves = []
        for r in range(N_REPEATS):
            seed = 5000 + 10 * fn + r
            problem = CEC2017Problem(fn, DIM)
            res = fun(problem, MAX_FES, seed=seed, log_every=200, **kw)
            curves.append(res["history"])
        out[fn][name] = curves
        print(f"F{fn} {name} done ({time.time()-t0:.0f}s)", flush=True)

with open("convergence_data.json", "w") as f:
    json.dump({str(k): v for k, v in out.items()}, f)
print("saved")
