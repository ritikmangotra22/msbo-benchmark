# MSBO — code and data

Code and raw results behind the manuscript *"Development of a Multi-Scale Bio-inspired Optimization
Algorithm for Efficient Turbulence Modeling and CFD Optimization"* (Ritik and Sukanta Ghosh, Department
of Computer Applications, Lovely Professional University, Phagwara, India).

## What this repository does and does not contain

**Contains:** the MSBO algorithm; seven comparison algorithms; a wrapper for the CEC 2017 code; the benchmark
runners; the raw per-run results; the statistics and figure scripts; a small ablation study.

**Does not contain:** any CFD/OpenFOAM data (the CFD calibration in the paper has not been run);
D=50 results; a jSO implementation; tuned settings for any algorithm.

## Scope of the benchmark data (read this before using the numbers)

| Study | Scope |
|---|---|
| D=10 | Full protocol: all 29 functions (F2 excluded), 51 runs, 8 algorithms, budget 10,000·D |
| D=30 | Subset only: 10 functions (F1,F3,F4,F9,F13,F16,F21,F23,F27,F29), 10 runs |
| D=50 | Not run |
| Ablation | 6 functions, D=10, 8 runs, 8 MSBO variants |

Per the CEC protocol, final errors below 1e-8 are treated as zero before any ranking or test
(`analyze_final.py` does this). Nothing was tuned: every algorithm runs at literature-default settings.

## IMPORTANT: two CMA-ES variants are in the data

`results_pilot.csv` contains algorithm `CMAES`, a **single-start CMA-ES without restarts**. It stops when
it converges and used only ~8% of its evaluation budget at D=10 (~16% at D=30), so it is an under-powered
baseline. It is kept only to document that problem. The comparator used in the paper is
**IPOP-CMA-ES** (restarts with doubling population), stored in `results_cmaes_ipop.csv` as
`CMAES_IPOP`. `analyze_final.py` uses the IPOP results as "CMA-ES" and reports the single-start
substitution as a sensitivity check. Do not compare MSBO against the `CMAES` rows.

## Files

- `src/msbo.py` — MSBO (Eqs. 4-22 of the paper). `src/baselines.py` — GA, PSO, DE, ABC, GWO, L-SHADE (our
  implementation of the published description), single-start CMA-ES and IPOP-CMA-ES (both via `pycma`).
  `src/problem.py` — ctypes wrapper for the CEC 2017 reference code. `src/stats_tools.py` — Wilcoxon,
  Friedman/Iman-Davenport, Holm, Vargha-Delaney A12.
- `run_pilot.py` — main runner (`full10` or `subset30`); resumable; writes `results_pilot.csv`.
- `run_cmaes_ipop.py` — IPOP-CMA-ES runner (D=10 full, then the D=30 subset); writes `results_cmaes_ipop.csv`.
- `run_ablation.py` — ablation study; writes `results_ablation.csv`.
- `run_convergence.py`, `run_convergence_ipop.py` — 5-run convergence histories for F9 and F16
  (`convergence_data.json`). `make_conv_figs.py`, `make_fig10.py` — figures.
- `analyze_final.py` — final statistics; writes `final_analysis.json`. `analysis_output.txt` is its printed output.
- `cec17_wrapper.cpp` — our small wrapper that calls the CEC 2017 organizers' code. **The organizers' own files (`cec17_test_func.cpp` and the `input_data/` folder) are NOT included here; see "Getting the CEC 2017 code" below.**

## Getting the CEC 2017 code (needed only to re-run the experiments)

The organizers' code and shift/rotation data are not redistributed here. Download the C++ package
(`CEC17_fast_pow-C++.zip`) from the organizers' repository, https://github.com/P-N-Suganthan/CEC2017-BoundContrained
(Awad, Ali, Liang, Qu, Suganthan, 2016, Nanyang Technological University), unzip it, and copy **`cec17_test_func.cpp`**
and the **`input_data/`** folder into the top level of this repository (next to `cec17_wrapper.cpp`). Check the
organizers' terms of use.

## Reproducing

**Statistics and figures only (no CEC code needed).** The raw results are included, so this works straight away:
```bash
pip install numpy scipy pandas cma matplotlib
python3 analyze_final.py     # reproduces analysis_output.txt exactly
python3 make_fig10.py        # Figure 10 of the paper
```

**Re-running the experiments (needs the CEC code above).**
```bash
g++ -O3 -shared -fPIC -o libcec17.so cec17_wrapper.cpp   # builds the CEC 2017 library
python3 run_pilot.py full10        # ~5-6 h on one core (resumable: re-run to continue)
python3 run_pilot.py subset30      # ~1 h
python3 run_cmaes_ipop.py          # ~2-3 h (D=10) then ~40 min (D=30 subset)
python3 run_ablation.py            # ~6 min
python3 analyze_final.py
```
Seeds are `1000*D + 10*F + run` in the benchmark runners, and a common Latin-hypercube pool per seed gives every
population-based algorithm the same starting points.

## Third-party code

The CEC 2017 reference code and data (see above) are the organizers' work, not ours, and are deliberately left out of
this repository. `pycma` (used for CMA-ES) is installed via pip and is not redistributed.

## AI-assisted preparation

The manuscript and this code were prepared with substantial assistance from an AI system (Claude, Anthropic),
under the authors' direction. The authors are responsible for all claims. Please verify results by re-running the
code instead of relying on the manuscript's account of them.

## Status and license

Not yet archived: no permanent identifier exists yet, and no license has been chosen (choose one, e.g. MIT for code
and CC-BY for data, before publishing). Insert the archive link into the paper's Section 22 once it exists.

## CFD pilot (Section 18.7 of the paper)

`cfd/` contains the small real OpenFOAM pilot: `cfd_objective.py` (drives `simpleFoam` on the
built-in Pitz & Daily case), `cfd_problem.py` (the MSBO/DE-compatible problem wrapper, synthetic
target = X_r/H 7.423 from theta_true = (0.075, 1.38, 1.87, 0.85, 1.15)), `run_cfd_opt.py` (the
3-seed, budget-60 comparison), and its output `results_cfd_opt.csv`.

**This is NOT the Driver-Seegmiller case the paper specifies in Sections 13-17** — it is a
substitute case used because no mesh/data for that exact rig was available. It is a synthetic
parameter-recovery test, not a claim of experimental validation. Requires an OpenFOAM v1912
installation (`apt install openfoam` on Ubuntu) to re-run.

**Grid convergence** (Table 18a of the paper): `grid_study.py` builds coarse/production/fine
meshes from the same `blockMeshDict` and compares reattachment length; `vtu_reader.py` is a
minimal, dependency-free VTU parser written because `meshio` failed unpredictably on some
mesh sizes of OpenFOAM's `foamToVTK` output on this system.

**Citation for the real experimental case**: Pitz RW, Daily JW (1983) Combustion in a turbulent
mixing layer formed at a rearward-facing step. AIAA Journal 21(11):1565-1570.
https://doi.org/10.2514/3.8290 — OpenFOAM's own documentation confirms the tutorial's setup is
derived from this experiment. The specific numerical reattachment length reported in that paper
has not been extracted/verified here; the synthetic target in `cfd_problem.py` should be replaced
with it once available (see paper Table 21).
