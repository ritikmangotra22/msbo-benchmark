"""Grid convergence study: build 3 meshes at different refinement levels using the
real pitzDaily blockMeshDict (scaling cell counts), run standard k-epsilon coefficients
on each, and check whether the reattachment length has converged. This is the check
Section 14 of the paper specifies and which the pilot did not originally include."""
import subprocess, shutil, os, re
import numpy as np
import sys; sys.path.insert(0, '.')
from vtu_reader import read_cell_centers_and_U

BASE = "/home/claude/cfd/pitzDaily_base"
FOAM_ENV = {**os.environ, "WM_PROJECT_DIR": "/usr/share/openfoam", "FOAM_API": "1912"}

def build_and_run(scale_factor, tag):
    case_dir = f"/home/claude/cfd/grid_{tag}"
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)
    shutil.copytree(BASE, case_dir, ignore=shutil.ignore_patterns('VTK', 'log.*', 'postProcessing', '[1-9]*'))
    # scale the cell counts in blockMeshDict (find all hex(...) (nx ny nz) simpleGrading blocks)
    bmd_path = os.path.join(case_dir, "system", "blockMeshDict")
    s = open(bmd_path).read()
    def scale_counts(m):
        nums = [int(x) for x in m.group(1).split()]
        scaled = [max(1, round(n * scale_factor)) for n in nums]
        return f"({scaled[0]} {scaled[1]} {scaled[2]})"
    s2 = re.sub(r'\((\d+\s+\d+\s+\d+)\)\s*\n?simpleGrading', lambda m: scale_counts(m) + "\nsimpleGrading", s)
    # the actual pitzDaily blockMeshDict format: hex (...) (nx ny nz) simpleGrading (...)
    s2 = re.sub(r'hex \(([^)]*)\)\s*\n?\s*\((\d+)\s+(\d+)\s+(\d+)\)', 
                lambda m: f"hex ({m.group(1)}) ({max(1,round(int(m.group(2))*scale_factor))} {max(1,round(int(m.group(3))*scale_factor))} {m.group(4)})", s)
    open(bmd_path, "w").write(s2)
    r = subprocess.run(["blockMesh"], cwd=case_dir, env=FOAM_ENV, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return None, None, r.stdout[-2000:]
    ncells = None
    m = re.search(r'nCells:\s*(\d+)', r.stdout)
    if m: ncells = int(m.group(1))
    r2 = subprocess.run(["simpleFoam"], cwd=case_dir, env=FOAM_ENV, capture_output=True, text=True, timeout=120)
    converged = "SIMPLE solution converged" in r2.stdout
    # reattachment length
    times = [d for d in os.listdir(case_dir) if re.match(r'^\d+$', d)]
    t = str(max(int(x) for x in times)) if times else None
    if t is None:
        return ncells, None, "no time dir"
    subprocess.run(["foamToVTK", "-time", t], cwd=case_dir, env=FOAM_ENV, capture_output=True, timeout=60)
    vtk_dir = os.path.join(case_dir, "VTK")
    base = os.path.basename(case_dir)
    cand = [c for c in os.listdir(vtk_dir) if os.path.isdir(os.path.join(vtk_dir, c)) and c.startswith(base + "_")]
    vtu = os.path.join(vtk_dir, cand[0], "internal.vtu")
    centers, U = read_cell_centers_and_U(vtu)
    mask = (centers[:,0] > 0.0002) & (centers[:,0] < 0.199) & (centers[:,1] > -0.0254) & (centers[:,1] < -0.024)
    xs, ux = centers[mask,0], U[mask,0]
    order = np.argsort(xs); xs, ux = xs[order], ux[order]
    sc = np.where((ux[:-1] < 0) & (ux[1:] >= 0))[0]
    xr = xs[sc[-1]+1]/0.0254 if len(sc) else None
    shutil.rmtree(case_dir, ignore_errors=True)
    return ncells, xr, ("converged" if converged else "NOT converged")

results = {}
for factor, tag in [(0.6, "coarse"), (1.0, "medium"), (1.7, "fine")]:
    ncells, xr, status = build_and_run(factor, tag)
    results[tag] = (ncells, xr, status)
    print(f"{tag:8s} scale={factor}  cells={ncells}  X_r/H={xr}  {status}", flush=True)

print("\nGrid convergence index-style check:")
if all(results[t][1] is not None for t in ("coarse","medium","fine")):
    xc, xm, xf = results["coarse"][1], results["medium"][1], results["fine"][1]
    print(f"coarse->medium change: {abs(xm-xc)/xm*100:.2f}%")
    print(f"medium->fine change:   {abs(xf-xm)/xm*100:.2f}%")
