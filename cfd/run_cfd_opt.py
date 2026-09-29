"""
Small, real CFD calibration test: MSBO and DE optimizing the 5 k-epsilon
coefficients against the real OpenFOAM objective (cfd_problem.py), trying to
recover a disclosed synthetic target reattachment length. 3 seeds each,
population 10, budget 60 evaluations (~5 generations) -- modest by the paper's
own Section 16 standards, but genuine, measured CFD runs, not a surrogate.
"""
import sys, os, csv, time
sys.path.insert(0, '/home/claude/cfd')
sys.path.insert(0, '/home/claude/proj/src')
from cfd_problem import CFDCalibrationProblem
from msbo import optimize as msbo_optimize, MSBOParams
import baselines as B
import numpy as np

X_TARGET = 7.42315
BUDGET = 60
POP = 10
SEEDS = [0, 1, 2]

OUT = "/home/claude/cfd/results_cfd_opt.csv"
LOG = "/home/claude/cfd/cfd_opt_progress.log"

def log(msg):
    with open(LOG, "a") as f: f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

def main():
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=["algo", "seed", "best_J", "best_theta", "n_evals", "n_fail", "wall_s"])
    if new: w.writeheader(); fh.flush()
    done = set()
    if not new:
        import pandas as pd
        try:
            e = pd.read_csv(OUT)
            done = {(r.algo, int(r.seed)) for r in e.itertuples()}
        except Exception:
            pass

    for seed in SEEDS:
        for algo in ("MSBO", "DE"):
            if (algo, seed) in done:
                continue
            t0 = time.time()
            problem = CFDCalibrationProblem(x_target=X_TARGET)
            if algo == "MSBO":
                res = msbo_optimize(problem, BUDGET, params=MSBOParams(n0=POP, n_min=4), seed=seed)
            else:
                res = B.de(problem, BUDGET, pop_size=POP, seed=seed)
            dt = time.time() - t0
            w.writerow(dict(algo=algo, seed=seed, best_J=float(res["best_f"]),
                             best_theta=res["best_x"].tolist(), n_evals=problem.n_evals,
                             n_fail=problem.n_fail, wall_s=round(dt, 1)))
            fh.flush()
            log(f"{algo} seed={seed} done: best_J={res['best_f']:.4g} "
                f"n_evals={problem.n_evals} n_fail={problem.n_fail} wall={dt:.0f}s")
    fh.close()
    log("ALL DONE")

if __name__ == "__main__":
    main()
