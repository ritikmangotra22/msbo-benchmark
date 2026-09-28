"""Re-run the CMA-ES comparator with restarts (IPOP-CMA-ES) so it uses the full evaluation
budget like every other algorithm. Same seeds and same shared LHS initial point as
run_pilot.py, so the comparison remains paired. Resume-safe."""
import sys, os, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np, pandas as pd
from problem import CEC2017Problem
import baselines as B

OUT = os.path.join(os.path.dirname(__file__), "results_cmaes_ipop.csv")
LOG = os.path.join(os.path.dirname(__file__), "ipop_progress.log")
ALL_FUNCS = [f for f in range(1, 31) if f != 2]
PILOT_FUNCS = [1, 3, 4, 9, 13, 16, 21, 23, 27, 29]
PHASES = [(10, ALL_FUNCS, 51), (30, PILOT_FUNCS, 10)]   # D=10 full protocol first, then the D=30 pilot subset

def log(m):
    with open(LOG, "a") as fh: fh.write(f"{time.strftime('%H:%M:%S')} {m}\n")

def main():
    fields = ["algo", "func", "dim", "run", "seed", "final_error", "fes", "restarts", "wall_time_s"]
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline=""); w = csv.DictWriter(fh, fieldnames=fields)
    if new: w.writeheader(); fh.flush()
    done = set()
    if not new:
        try:
            e = pd.read_csv(OUT)
            done = {(int(r.func), int(r.dim), int(r.run)) for r in e.itertuples()}
        except Exception: pass
    t_start = time.time()
    for dim, funcs, n_runs in PHASES:
        max_fes = 10000 * dim
        for fn in funcs:
            for run in range(n_runs):
                if (fn, dim, run) in done: continue
                seed = 1000 * dim + 10 * fn + run            # identical to run_pilot.py
                rng0 = np.random.default_rng(seed)
                lb, ub = -100.0*np.ones(dim), 100.0*np.ones(dim)
                pool_n = max(18*dim, 50)
                pool = np.empty((pool_n, dim))
                for j in range(dim):                          # identical pool construction to run_pilot.py
                    perm = rng0.permutation(pool_n); u = rng0.uniform(0, 1, pool_n)
                    pool[:, j] = lb[j] + (ub[j]-lb[j])*(perm+1-u)/pool_n
                prob = CEC2017Problem(fn, dim)
                t0 = time.time()
                res = B.cmaes_ipop(prob, max_fes, seed=seed, x0_pop=pool[:1])
                w.writerow(dict(algo="CMAES_IPOP", func=fn, dim=dim, run=run, seed=seed,
                                final_error=float(res["best_error"]), fes=int(res["fes"]),
                                restarts=int(res["restarts"]), wall_time_s=round(time.time()-t0, 3)))
                fh.flush()
            log(f"D{dim} F{fn} done ({time.time()-t_start:.0f}s)")
    log(f"ALL DONE in {time.time()-t_start:.0f}s")

if __name__ == "__main__":
    main()
