"""
Problem wrapper matching the interface MSBO/baselines expect (problem.evaluate(X),
problem.lb, problem.ub, problem.f_star), backed by the real OpenFOAM objective in
cfd_objective.py. Objective (paper's Eq. 37, reduced to its reattachment-length
term since no reference velocity/skin-friction profiles are available for this
case): J(theta) = ((X_r(theta) - X_r_target) / X_r_target)^2. Non-convergent or
non-reattaching runs return a fixed failure penalty, as Section 14 specifies.
"""
import numpy as np
import cfd_objective as cfo

STANDARD = np.array([0.09, 1.44, 1.92, 1.00, 1.30])
LB = 0.6 * STANDARD
UB = 1.5 * STANDARD
J_FAIL = 5.0  # normalized-objective failure penalty (must exceed any converged value seen)


class CFDCalibrationProblem:
    dim = 5
    f_star = 0.0

    def __init__(self, x_target):
        self.lb = LB.copy()
        self.ub = UB.copy()
        self.x_target = x_target
        self.n_evals = 0
        self.n_fail = 0
        self.log = []  # (theta, xr, converged, J)

    def evaluate(self, X):
        X = np.atleast_2d(X)
        out = np.empty(X.shape[0])
        for i, theta in enumerate(X):
            theta_c = np.clip(theta, self.lb, self.ub)
            tag = f"e{self.n_evals:06d}"
            try:
                xr, converged = cfo.evaluate(tuple(theta_c), tag=tag)
            except Exception:
                xr, converged = None, False
            self.n_evals += 1
            if xr is None or not converged:
                J = J_FAIL
                self.n_fail += 1
            else:
                J = ((xr - self.x_target) / self.x_target) ** 2
            out[i] = J
            self.log.append((theta_c.tolist(), xr, converged, float(J)))
        return out
