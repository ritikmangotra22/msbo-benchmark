"""Convergence histories for IPOP-CMA-ES on F9 and F16 (D=10, 5 runs, same seeds as run_convergence.py),
merged into convergence_data.json. The earlier single-start curves are kept under 'CMAES_SS'."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from problem import CEC2017Problem
import baselines as B
data = json.load(open("convergence_data.json"))
for fn in (9, 16):
    k = str(fn)
    if "CMAES_SS" not in data[k]: data[k]["CMAES_SS"] = data[k]["CMAES"]
    curves = []
    for r in range(5):
        p = CEC2017Problem(fn, 10)
        res = B.cmaes_ipop(p, 100000, seed=5000 + 10*fn + r, log_every=200)
        curves.append(res["history"]); print(fn, r, res["best_error"], res["restarts"], flush=True)
    data[k]["CMAES"] = curves
json.dump(data, open("convergence_data.json", "w"))
print("saved")
