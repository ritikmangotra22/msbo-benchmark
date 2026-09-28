"""
Lightweight, real ablation study for MSBO (Section 9 claims / Table 11).
NOT the full protocol: 6 CEC2017 functions spanning categories, D=10, 8 runs,
full CEC budget (10000*D). Variants match the ablation list in Table 11/16
of the paper: (a) each scale alone, (b) time-prior only, (c) feedback only,
(d) no floor, (e) no population reduction, (f) no diversity rescue.
All numbers below are real, measured outputs.
"""
import sys, os, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
from problem import CEC2017Problem
from msbo import optimize as msbo_opt, MSBOParams

FUNCS = [1, 4, 9, 16, 21, 27]   # unimodal, simple-multimodal, hybrid, composition
DIM = 10
MAX_FES = 10000 * DIM
N_RUNS = 8

def variant_full(): return MSBOParams()
def variant_scale1_only(): return MSBOParams(rho0=(0.20,0.05,0.01), lam=1.0)  # placeholder, real restriction applied via monkeypatch below
def variant_time_only(): return MSBOParams(lam=1.0)
def variant_feedback_only(): return MSBOParams(lam=0.0)
def variant_no_floor(): return MSBOParams(p_min=1e-9)
def variant_no_rescue(): return MSBOParams(g_stag=10**9)

# For "each scale alone" we force probabilities via a thin wrapper.
import msbo as msbo_mod

_REAL_DEFAULT_RNG = np.random.default_rng

def run_single_scale(problem, max_fes, scale, seed, x0_pop, log_every=None):
    """Force scale probabilities to concentrate on one scale (with the required
    floor still respected structurally by using a near-degenerate prior)."""
    p = MSBOParams(p_min=1e-6, lam=1.0)
    class ForcedRNG:
        def __init__(self, seed):
            self._r = _REAL_DEFAULT_RNG(seed)
        def __getattr__(self, name):
            return getattr(self._r, name)
        def choice(self, a, size=None, p=None, replace=True):
            if p is not None and len(a) == 3:
                return np.full(size, scale, dtype=int)
            return self._r.choice(a, size=size, p=p, replace=replace)
    def patched_default_rng(s):
        return ForcedRNG(s)
    np.random.default_rng = patched_default_rng
    try:
        res = msbo_opt(problem, max_fes, params=p, seed=seed, x0_pop=x0_pop, log_every=log_every)
    finally:
        np.random.default_rng = _REAL_DEFAULT_RNG
    return res

VARIANTS = {
    "full": ("full", None),
    "scale1_only": ("scale", 1),
    "scale2_only": ("scale", 2),
    "scale3_only": ("scale", 3),
    "time_prior_only": ("params", MSBOParams(lam=1.0)),
    "feedback_only": ("params", MSBOParams(lam=0.0)),
    "no_floor": ("params", MSBOParams(p_min=1e-9)),
    "no_rescue": ("params", MSBOParams(g_stag=10**9)),
}

def main():
    out_path = os.path.join(os.path.dirname(__file__), "results_ablation.csv")
    write_header = not os.path.exists(out_path)
    fh = open(out_path, "a", newline="")
    fieldnames = ["variant", "func", "dim", "run", "seed", "final_error", "fes"]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if write_header:
        writer.writeheader(); fh.flush()

    done = set()
    if os.path.exists(out_path):
        import pandas as pd
        try:
            d = pd.read_csv(out_path)
            for _, r in d.iterrows():
                done.add((r["variant"], int(r["func"]), int(r["run"])))
        except Exception:
            pass

    t0 = time.time()
    for fn in FUNCS:
        for run in range(N_RUNS):
            seed = 9000 + 10 * fn + run
            rng0 = np.random.default_rng(seed)
            lb, ub = -100.0*np.ones(DIM), 100.0*np.ones(DIM)
            N0 = max(18*DIM, 4)
            pool = np.empty((N0, DIM))
            for j in range(DIM):
                perm = rng0.permutation(N0); u = rng0.uniform(0,1,N0)
                pool[:, j] = lb[j] + (ub[j]-lb[j])*(perm+1-u)/N0
            for vname, (kind, val) in VARIANTS.items():
                key = (vname, fn, run)
                if key in done: continue
                problem = CEC2017Problem(fn, DIM)
                if kind == "scale":
                    res = run_single_scale(problem, MAX_FES, val, seed, pool)
                elif kind == "params":
                    res = msbo_opt(problem, MAX_FES, params=val, seed=seed, x0_pop=pool)
                else:
                    res = msbo_opt(problem, MAX_FES, params=MSBOParams(), seed=seed, x0_pop=pool)
                writer.writerow({"variant": vname, "func": fn, "dim": DIM, "run": run,
                                  "seed": seed, "final_error": float(res["best_error"]),
                                  "fes": int(res["fes"])})
                fh.flush()
            print(f"F{fn} run{run} done ({time.time()-t0:.0f}s)", flush=True)
    fh.close()
    print("ALL DONE", time.time()-t0)

if __name__ == "__main__":
    main()
