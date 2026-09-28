"""
Multi-Scale Bio-inspired Optimizer (MSBO).
Implements Sections 7-8 of the paper exactly (Eqs. 4-22), for box-constrained
problems only (V=0 throughout, so psi = f, matching Section 10's analysis scope).
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class MSBOParams:
    rho0: tuple = (0.20, 0.05, 0.01)      # Eq. 7
    rho_min: tuple = (0.02, 0.005, 1e-6)  # Eq. 7
    beta: float = 1.5                      # Levy index, Eq. 8-9
    p_e: float = 0.10                      # elite fraction, Eq. 5
    p_min: float = 0.05                    # probability floor, Eq. 19
    lam: float = 0.5                       # blend weight, Eq. 19
    alpha: float = 0.1                     # credit smoothing, Eq. 16
    w: float = 0.7                         # prior width, Eq. 18
    p_m: float = None                      # mask prob, Eq. 12 (default 3/D)
    theta_div: float = 1e-3                # Eq. 21 rescue trigger (normalized coords)
    g_stag: int = 20                       # rescue patience (generations)
    n0: int = None                         # initial pop size (default 18*D)
    n_min: int = 4                         # final pop size


def _levy(beta, size, rng):
    """Mantegna's algorithm, Eqs. (8)-(9)."""
    from math import gamma, sin, pi
    num = gamma(1 + beta) * sin(pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_a = (num / den) ** (1 / beta)
    a = rng.normal(0, sigma_a, size=size)
    b = rng.normal(0, 1, size=size)
    return a / (np.abs(b) ** (1 / beta))


def _lhs(n, dim, lb, ub, rng):
    """Latin hypercube sampling, Eq. (6)."""
    X = np.empty((n, dim))
    for j in range(dim):
        perm = rng.permutation(n)
        u = rng.uniform(0, 1, n)
        X[:, j] = lb[j] + (ub[j] - lb[j]) * (perm + 1 - u) / n
    return X


def _reflect_repair(V, lb, ub):
    """Eq. (13): reflect once, then clip (redraw-uniform is approximated by clip
    for any point still outside after one reflection, which is measure-zero in practice)."""
    below = V < lb
    above = V > ub
    V = np.where(below, 2 * lb - V, V)
    V = np.where(above, 2 * ub - V, V)
    return np.clip(V, lb, ub)


def optimize(problem, max_fes, params: MSBOParams = None, seed=0, x0_pop=None,
             log_every=None):
    """
    Minimize `problem.evaluate(X)-problem.f_star` is NOT what we minimize on;
    we minimize the RAW objective problem.evaluate, and report error via problem.f_star.
    (Bound-constrained only: V==0, psi=f, Sections 6-8.)

    Returns dict with best_x, best_f, history (list of (fes, best_error)).
    """
    rng = np.random.default_rng(seed)
    D = problem.dim
    lb, ub = problem.lb, problem.ub
    R = ub - lb
    p = params or MSBOParams()
    p_m = p.p_m if p.p_m is not None else min(1.0, 3.0 / D)
    N0 = p.n0 if p.n0 is not None else max(4, 18 * D)
    N_min = max(4, p.n_min)

    # --- Init (Eq. 6) ---
    X = x0_pop.copy() if x0_pop is not None else _lhs(N0, D, lb, ub, rng)
    f = problem.evaluate(X)
    fes = X.shape[0]
    credits = np.array([1 / 3, 1 / 3, 1 / 3])

    best_idx = np.argmin(f)
    best_x, best_f = X[best_idx].copy(), f[best_idx]
    history = [(fes, best_f - problem.f_star)]
    stall = 0
    div_max = 1e-12

    N_t = X.shape[0]
    while fes < max_fes:
        tau = min(1.0, fes / max_fes)

        # radii, Eq. 7
        rho = np.array(p.rho0) + (np.array(p.rho_min) - np.array(p.rho0)) * 0  # placeholder
        rho = np.array(p.rho_min) + (np.array(p.rho0) - np.array(p.rho_min)) * (1 - tau) ** 2

        # population size, Eq. 20
        N_next = int(round(N0 + (N_min - N0) * tau))
        N_next = max(N_min, min(N_next, N_t))

        # elites, Eq. 5
        order = np.argsort(f)
        n_elite = max(2, int(np.ceil(p.p_e * N_t)))
        elite_idx = order[:n_elite]
        E = X[elite_idx]

        # scale probabilities, Eqs. 17-19
        A = (credits + 1e-12)
        A = A / A.sum()
        mu = 1 + 2 * tau
        k = np.array([1, 2, 3])
        Pw = np.exp(-((k - mu) ** 2) / (2 * p.w ** 2))
        Pw = Pw / Pw.sum()
        blend = (1 - p.lam) * A + p.lam * Pw
        probs = p.p_min + (1 - 3 * p.p_min) * blend
        probs = probs / probs.sum()

        scales = rng.choice([1, 2, 3], size=N_t, p=probs)
        V = np.empty_like(X)

        idx_all = np.arange(N_t)
        for s in (1, 2, 3):
            sel = idx_all[scales == s]
            if sel.size == 0:
                continue
            if s == 1:
                L = _levy(p.beta, (sel.size, D), rng)
                V[sel] = X[sel] + rho[0] * R[None, :] * L
            elif s == 2:
                eq = E[rng.integers(0, E.shape[0], size=sel.size)]
                # a,b distinct from i and from each other
                Vab = np.empty((sel.size, D))
                for t_, i_ in enumerate(sel):
                    choices = idx_all[idx_all != i_]
                    a, b = rng.choice(choices, size=2, replace=False)
                    Vab[t_] = X[a] - X[b]
                r = rng.uniform(0, 1, size=(sel.size, D))
                Fi = rng.uniform(0.4, 0.9, size=(sel.size, 1))
                xi = rng.normal(0, 1, size=(sel.size, D))
                V[sel] = X[sel] + r * (eq - X[sel]) + Fi * Vab + rho[1] * R[None, :] * xi
            else:  # s == 3
                eq = E[rng.integers(0, E.shape[0], size=sel.size)]
                xi = rng.normal(0, 1, size=(sel.size, D))
                mask = (rng.uniform(0, 1, size=(sel.size, D)) < p_m)
                no_mask = ~mask.any(axis=1)
                if no_mask.any():
                    fix_cols = rng.integers(0, D, size=no_mask.sum())
                    mask[np.where(no_mask)[0], fix_cols] = True
                cand = eq + rho[2] * R[None, :] * xi
                V[sel] = np.where(mask, cand, X[sel])

        V = _reflect_repair(V, lb, ub)
        fv = problem.evaluate(V)
        fes += N_t

        # credit assignment (Eqs. 15-16)
        fmax, fmin = f.max(), f.min()
        denom = (fmax - fmin) + 1e-12
        improve = f - fv  # positive if trial is better (lower)
        accept = fv <= f
        new_credits = credits.copy()
        for s in (1, 2, 3):
            sel = idx_all[scales == s]
            if sel.size == 0:
                continue
            acc_sel = sel[accept[sel]]
            if acc_sel.size > 0:
                gains = improve[acc_sel] / denom
                q_hat = gains.mean()
            else:
                q_hat = 0.0
            new_credits[s - 1] = (1 - p.alpha) * credits[s - 1] + p.alpha * q_hat
        credits = new_credits

        # greedy selection (Eq. 14)
        X = np.where(accept[:, None], V, X)
        f = np.where(accept, fv, f)

        cur_best = f.min()
        if cur_best < best_f:
            best_f = cur_best
            best_x = X[np.argmin(f)].copy()
            stall = 0
        else:
            stall += 1

        # diversity (Eq. 21) & rescue
        Xn = (X - lb[None, :]) / R[None, :]
        med = np.median(Xn, axis=0)
        div = np.mean(np.abs(Xn - med[None, :]))
        div_max = max(div_max, div)

        if (div / div_max) < p.theta_div and stall >= p.g_stag:
            worst = np.argsort(f)[-max(1, int(np.ceil(0.2 * N_t))):]
            best_row = np.argmin(f)
            worst = worst[worst != best_row]
            Xr = _lhs(len(worst), D, lb, ub, rng)
            X[worst] = Xr
            fr = problem.evaluate(Xr)
            fes += len(worst)
            f[worst] = fr
            stall = 0

        # population reduction: drop worst until N_next remain
        if N_next < X.shape[0]:
            order = np.argsort(f)[:N_next]
            X, f = X[order], f[order]
        N_t = X.shape[0]

        if log_every and (len(history) == 0 or fes - history[-1][0] >= log_every):
            history.append((fes, best_f - problem.f_star))

    history.append((fes, best_f - problem.f_star))
    return {"best_x": best_x, "best_f": best_f, "best_error": best_f - problem.f_star,
            "history": history, "fes": fes}
