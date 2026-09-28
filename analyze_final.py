"""Final benchmark analysis, following Sections 11-12 of the paper:
 - CEC rule: final errors below 1e-8 are treated as exactly zero before ranking/testing
 - CMA-ES comparator = IPOP-CMA-ES (restarts, full budget); the single-start variant is reported
   only as a sensitivity check
 - Friedman + Iman-Davenport omnibus; post-hoc z-test on Friedman ranks with MSBO as control and
   Holm over the 7 rivals; per-function Wilcoxon + Holm(29) + Vargha-Delaney A12 win/tie/loss
 - Per-category average ranks"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import f as fdist
from stats_tools import friedman_iman_davenport, holm_correction, vargha_delaney_A12, wilcoxon_signed_rank

ALGOS = ["MSBO","GA","PSO","DE","ABC","GWO","CMAES","LSHADE"]
LABEL = {"CMAES":"CMA-ES (IPOP)"}

def load(dim):
    a = pd.read_csv("results_pilot.csv"); a = a[a.dim==dim]
    ss = a[a.algo=="CMAES"].copy(); ss["algo"] = "CMAES_SS"
    a = a[a.algo!="CMAES"]
    b = pd.read_csv("results_cmaes_ipop.csv"); b = b[b.dim==dim].copy(); b["algo"] = "CMAES"
    d = pd.concat([a, b[a.columns.intersection(b.columns)], ss], ignore_index=True)
    d["err0"] = np.where(d.final_error < 1e-8, 0.0, d.final_error)     # CEC zero rule
    return d

def analyze(dim, algos=ALGOS):
    d = load(dim)
    # only functions with all algorithms complete at the run count of the smallest cell
    need = set(algos)
    ok = []
    for f in sorted(int(x) for x in d.func.unique()):
        have = d[d.func==f].groupby("algo").run.nunique()
        if need.issubset(set(have.index)): ok.append(f)
    if not ok: raise ValueError('no function has all algorithms complete')
    d = d[d.func.isin(ok)]
    funcs = ok; n, k = len(funcs), len(algos)
    nruns = int(d[d.algo.isin(algos)].groupby(["algo","func"]).run.nunique().min())
    M = np.array([[np.median(d[(d.func==f)&(d.algo==a)].err0.values) for a in algos] for f in funcs])
    chi2, FF, ar = friedman_iman_davenport(M)
    pFF = 1 - fdist.cdf(FF, k-1, (k-1)*(n-1))
    ranks = dict(zip(algos, map(float, ar)))
    se = np.sqrt(k*(k+1)/(6.0*n)); rivals = [a for a in algos if a != "MSBO"]
    z = {a:(ranks[a]-ranks["MSBO"])/se for a in rivals}
    p2 = {a:float(2*(1-stats.norm.cdf(abs(z[a])))) for a in rivals}
    rej, padj = holm_correction([p2[a] for a in rivals])
    post = {a:dict(z=float(z[a]), p=p2[a], p_holm=float(pa), reject=bool(r),
                   better=("MSBO" if z[a]>0 else a)) for a,r,pa in zip(rivals,rej,padj)}
    wtl = {}
    for a in rivals:
        pv, a12s = [], []
        for f in funcs:
            m = d[(d.func==f)&(d.algo=="MSBO")].sort_values("run").err0.values
            r = d[(d.func==f)&(d.algo==a)].sort_values("run").err0.values
            nn = min(len(m), len(r))
            p,_ = wilcoxon_signed_rank(r[:nn], m[:nn]); pv.append(p); a12s.append(vargha_delaney_A12(r[:nn], m[:nn]))
        rj,_ = holm_correction(pv)
        w = sum(1 for x,y in zip(rj,a12s) if x and y>=0.64); l = sum(1 for x,y in zip(rj,a12s) if x and y<=0.36)
        wtl[a] = dict(win=w, tie=n-w-l, loss=l, mean_A12=float(np.mean(a12s)))
    cats = {"Unimodal":[1,3], "Simple multimodal":list(range(4,11)),
            "Hybrid":list(range(11,21)), "Composition":list(range(21,31))}
    catr = {}
    for name, fl in cats.items():
        fl = [f for f in fl if f in funcs]
        if len(fl) < 2: continue
        idx = [funcs.index(f) for f in fl]
        _,_,r_ = friedman_iman_davenport(M[idx]); catr[name] = dict(n=len(fl), ranks=dict(zip(algos, map(float, r_))))
    solved = {a:int(sum(1 for i in range(n) if M[i,algos.index(a)]==0)) for a in algos}
    table = {}
    for f in funcs:
        table[f] = {a:(float(np.median(d[(d.func==f)&(d.algo==a)].err0.values)),
                       float(np.std(d[(d.func==f)&(d.algo==a)].err0.values))) for a in algos}
    table = {int(f): v for f, v in table.items()}
    return dict(dim=dim, n_funcs=n, funcs=[int(f) for f in funcs], runs=nruns, chi2=float(chi2), FF=float(FF), pFF=float(pFF),
                df=[k-1,(k-1)*(n-1)], ranks=ranks, post=post, wtl=wtl, cat=catr, solved=solved, table=table)

if __name__ == "__main__":
    out = {}
    for dim in (10, 30):
        try:
            r = analyze(dim)
        except Exception as e:
            print(f"D={dim}: not analyzable yet ({e!r})"); continue
        out[str(dim)] = r
        print(f"\n=== D={dim}: {r['n_funcs']} functions, {r['runs']} runs ===")
        print(f"Friedman chi2={r['chi2']:.2f} F_F={r['FF']:.2f} df={r['df']} p={r['pFF']:.3g}")
        for a in sorted(r['ranks'], key=r['ranks'].get): print(f"  {a:7s} {r['ranks'][a]:.3f}   solved={r['solved'][a]}")
        for a,v in r['post'].items(): print(f"  post-hoc {a:7s} z={v['z']:+.2f} p_holm={v['p_holm']:.3g} reject={v['reject']} better={v['better']}")
        for a,v in r['wtl'].items(): print(f"  W/T/L vs {a:7s}: {v['win']}/{v['tie']}/{v['loss']}  A12={v['mean_A12']:.3f}")
        for c,v in r['cat'].items(): print(f"  {c:18s} n={v['n']:2d}", {a:round(x,2) for a,x in v['ranks'].items()})
        # sensitivity: substitute single-start CMA-ES
        alt = [("CMAES_SS" if a=="CMAES" else a) for a in ALGOS]
        try:
            r2 = analyze(dim, alt); print("  [sensitivity] single-start CMA-ES rank:", round(r2['ranks']['CMAES_SS'],2),
                  " MSBO rank:", round(r2['ranks']['MSBO'],2))
            out[str(dim)]["sensitivity_single_start"] = dict(cma_rank=r2['ranks']['CMAES_SS'], msbo_rank=r2['ranks']['MSBO'])
        except Exception as e: print("  sensitivity failed", e)
    json.dump(out, open("final_analysis.json","w"), indent=1, default=str)
