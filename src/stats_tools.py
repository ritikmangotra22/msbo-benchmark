"""Statistical toolkit matching Section 12 of the paper: Wilcoxon signed-rank,
Friedman + Iman-Davenport, Holm step-down correction, Vargha-Delaney A12."""
import numpy as np
from scipy import stats


def wilcoxon_signed_rank(a, b):
    """Paired Wilcoxon signed-rank test, Eq. (25). a,b: arrays of paired final errors."""
    a, b = np.asarray(a), np.asarray(b)
    if np.allclose(a, b):
        return 1.0, 0.0
    try:
        w, p = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        return 1.0, 0.0
    return p, w


def friedman_iman_davenport(rank_matrix):
    """rank_matrix: (n_problems, k_algorithms) of raw scores (lower=better).
    Returns (chi2_F, F_F, avg_ranks) per Eqs. (26)-(27)."""
    n, k = rank_matrix.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, rank_matrix)
    avg_ranks = ranks.mean(axis=0)
    chi2_F = (12 * n / (k * (k + 1))) * (np.sum(avg_ranks ** 2) - k * (k + 1) ** 2 / 4)
    if n - 1 == 0 or (n * (k - 1) - chi2_F) == 0:
        F_F = np.inf
    else:
        F_F = ((n - 1) * chi2_F) / (n * (k - 1) - chi2_F)
    return chi2_F, F_F, avg_ranks


def holm_correction(p_values, alpha=0.05):
    """Holm step-down procedure, Eq. (28). Returns boolean array of rejections
    (True = significant difference from the control after correction) and adjusted p."""
    p = np.asarray(p_values)
    m = len(p)
    order = np.argsort(p)
    reject = np.zeros(m, dtype=bool)
    p_adj = np.empty(m)
    max_adj = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p[idx]
        max_adj = max(max_adj, adj)
        p_adj[idx] = min(max_adj, 1.0)
    # sequential rejection: stop at first non-rejection
    stop = False
    for rank, idx in enumerate(order):
        if stop:
            reject[idx] = False
            continue
        thresh = alpha / (m - rank)
        if p[idx] <= thresh:
            reject[idx] = True
        else:
            reject[idx] = False
            stop = True
    return reject, p_adj


def vargha_delaney_A12(rival_errors, msbo_errors):
    """A12 effect size, Eq. (29). >0.5 favours MSBO (lower error)."""
    x = np.asarray(rival_errors)  # rival (m obs)
    y = np.asarray(msbo_errors)   # MSBO (n obs)
    m, n = len(x), len(y)
    pooled = np.concatenate([x, y])
    ranks = stats.rankdata(pooled)
    rank_x = ranks[:m]
    A12 = (rank_x.sum() - m * (m + 1) / 2) / (m * n)
    return A12


def effect_label(A12):
    d = abs(A12 - 0.5)
    if d >= 0.21:
        return "large"
    if d >= 0.14:
        return "medium"
    if d >= 0.06:
        return "small"
    return "negligible"
