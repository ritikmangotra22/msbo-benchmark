"""
Pilot experiment runner (NOT the full protocol of the paper's Section 11).
Reduced scale to fit a single-CPU-core sandbox in reasonable wall time:
  - 10 CEC2017 functions spanning all 4 categories (of 29 specified in the paper)
  - D in {10, 30} (paper also specifies D=50)
  - 15 independent runs per (algorithm, function, D) at D=10, 10 runs at D=30
    (paper specifies 51 runs)
  - Full CEC2017 budget MaxFES = 10000*D, as specified in the paper (Eq. Section 11)
All numbers are real, measured outputs of the code in src/. Nothing here is invented.
"""
import sys, os, time, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
from problem import CEC2017Problem
from msbo import optimize as msbo_opt, MSBOParams
import baselines as B

ALL_FUNCS = [f for f in range(1, 31) if f != 2]          # full CEC2017 suite (F2 excluded per organizer guidance)
SUBSET_FUNCS = [1, 3, 4, 9, 13, 16, 21, 23, 27, 29]      # 10-function subset spanning all four categories
# Usage:  python3 run_pilot.py full10     -> D=10, all 29 functions, 51 runs (the full protocol at D=10)
#         python3 run_pilot.py subset30   -> D=30, 10-function subset, 10 runs (the D=30 subset in the paper)
CONFIGS = {"full10": (ALL_FUNCS, [(10, 51)]), "subset30": (SUBSET_FUNCS, [(30, 10)])}
_cfg = sys.argv[1] if len(sys.argv) > 1 else "full10"
FUNCS, DIMS_RUNS = CONFIGS[_cfg]
ALGOS = {
    "MSBO": (msbo_opt, {}),
    "GA": (B.ga, {}),
    "PSO": (B.pso, {}),
    "DE": (B.de, {}),
    "ABC": (B.abc, {}),
    "GWO": (B.gwo, {}),
    "CMAES": (B.cmaes, {}),
    "LSHADE": (B.lshade, {}),
}
OUT = os.path.join(os.path.dirname(__file__), "results_pilot.csv")
LOG = os.path.join(os.path.dirname(__file__), "pilot_progress.log")

def log(msg):
    with open(LOG, "a") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)

def main():
    fieldnames = ["algo", "func", "dim", "run", "seed", "final_error", "fes", "wall_time_s"]
    write_header = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if write_header:
        writer.writeheader(); fh.flush()

    # resume support: skip combos already present
    done = set()
    if os.path.exists(OUT):
        import pandas as pd
        try:
            dfe = pd.read_csv(OUT)
            for _, row in dfe.iterrows():
                done.add((row["algo"], int(row["func"]), int(row["dim"]), int(row["run"])))
        except Exception:
            pass

    t_start = time.time()
    for dim, n_runs in DIMS_RUNS:
        max_fes = 10000 * dim
        for fn in FUNCS:
            for run in range(n_runs):
                seed = 1000 * dim + 10 * fn + run
                # shared LHS initial population across algorithms (common random numbers, Section 16)
                rng0 = np.random.default_rng(seed)
                lb, ub = -100.0 * np.ones(dim), 100.0 * np.ones(dim)
                # pop size varies per algorithm; generate a large shared pool and slice
                pool_n = max(18 * dim, 50)
                pool = np.empty((pool_n, dim))
                for j in range(dim):
                    perm = rng0.permutation(pool_n)
                    u = rng0.uniform(0, 1, pool_n)
                    pool[:, j] = lb[j] + (ub[j] - lb[j]) * (perm + 1 - u) / pool_n

                for algo_name, (fn_call, kw) in ALGOS.items():
                    key = (algo_name, fn, dim, run)
                    if key in done:
                        continue
                    problem = CEC2017Problem(fn, dim)
                    pop_needed = {
                        "MSBO": max(18 * dim, 4), "GA": 50, "PSO": 50, "DE": 50,
                        "ABC": 50, "GWO": 50, "CMAES": None, "LSHADE": max(18 * dim, 4),
                    }[algo_name]
                    x0 = pool[:pop_needed] if pop_needed else pool[:1]
                    t0 = time.time()
                    try:
                        res = fn_call(problem, max_fes, seed=seed, x0_pop=x0, **kw)
                        err = float(res["best_error"])
                        fes = int(res["fes"])
                    except Exception as e:
                        log(f"FAILED {algo_name} F{fn} D{dim} run{run}: {e!r}")
                        err, fes = float("nan"), 0
                    dt = time.time() - t0
                    writer.writerow({"algo": algo_name, "func": fn, "dim": dim, "run": run,
                                      "seed": seed, "final_error": err, "fes": fes,
                                      "wall_time_s": round(dt, 3)})
                    fh.flush()
                log(f"D{dim} F{fn} run{run}/{n_runs} done  (elapsed {time.time()-t_start:.0f}s)")
    fh.close()
    log(f"ALL DONE in {time.time()-t_start:.0f}s")

if __name__ == "__main__":
    main()
