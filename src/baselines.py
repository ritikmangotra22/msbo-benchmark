"""
Baseline optimizers for comparison against MSBO (Section 17 of the paper).
Each returns the same dict shape as msbo.optimize: best_x, best_f, best_error, history, fes.
All are bound-constrained, box repair by clipping unless noted, and use the
organizers' CEC2017 raw objective via `problem.evaluate`.

Implementations follow the cited original sources at the level of the core
update rule; they are standard textbook forms, not vendor-tuned code.
"""
import numpy as np


def _clip(X, lb, ub):
    return np.clip(X, lb, ub)


def _lhs(n, dim, lb, ub, rng):
    X = np.empty((n, dim))
    for j in range(dim):
        perm = rng.permutation(n)
        u = rng.uniform(0, 1, n)
        X[:, j] = lb[j] + (ub[j] - lb[j]) * (perm + 1 - u) / n
    return X


def _run_common_init(problem, pop_size, seed, x0_pop):
    rng = np.random.default_rng(seed)
    D = problem.dim
    lb, ub = problem.lb, problem.ub
    X = x0_pop.copy() if x0_pop is not None else _lhs(pop_size, D, lb, ub, rng)
    f = problem.evaluate(X)
    fes = X.shape[0]
    best_idx = np.argmin(f)
    return rng, D, lb, ub, X, f, fes, X[best_idx].copy(), f[best_idx]


def _log(history, fes, best_f, f_star, log_every):
    if not history or fes - history[-1][0] >= log_every:
        history.append((fes, best_f - f_star))


# ---------------------------------------------------------------- GA (real-coded)
def ga(problem, max_fes, pop_size=50, pc=0.9, eta_c=20, eta_m=20, pm=None,
       seed=0, x0_pop=None, log_every=None):
    """Real-coded GA: tournament-2 selection, SBX crossover, polynomial mutation, elitism-1. [24,10]"""
    rng, D, lb, ub, X, f, fes, bx, bf = _run_common_init(problem, pop_size, seed, x0_pop)
    pm = pm if pm is not None else 1.0 / D
    history = [(fes, bf - problem.f_star)]

    def tournament():
        i, j = rng.integers(0, pop_size, 2)
        return X[i] if f[i] < f[j] else X[j]

    def sbx(p1, p2):
        u = rng.uniform(0, 1, D)
        beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)),
                        (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
        c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        return c1, c2

    def poly_mut(c):
        u = rng.uniform(0, 1, D)
        delta = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1,
                         1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
        mask = rng.uniform(0, 1, D) < pm
        c2 = c.copy()
        c2[mask] = c[mask] + delta[mask] * (ub[mask] - lb[mask])
        return c2

    while fes < max_fes:
        newX = []
        while len(newX) < pop_size:
            p1, p2 = tournament(), tournament()
            if rng.uniform() < pc:
                c1, c2 = sbx(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            c1, c2 = poly_mut(c1), poly_mut(c2)
            newX.append(c1); newX.append(c2)
        newX = _clip(np.array(newX[:pop_size]), lb, ub)
        newf = problem.evaluate(newX)
        fes += pop_size
        # elitism: keep global best if lost
        worst = np.argmax(newf)
        if bf < newf.min():
            newX[worst], newf[worst] = bx, bf
        X, f = newX, newf
        i = np.argmin(f)
        if f[i] < bf:
            bf, bx = f[i], X[i].copy()
        if log_every: _log(history, fes, bf, problem.f_star, log_every)
    history.append((fes, bf - problem.f_star))
    return {"best_x": bx, "best_f": bf, "best_error": bf - problem.f_star, "history": history, "fes": fes}


# ---------------------------------------------------------------- PSO (constriction)
def pso(problem, max_fes, pop_size=50, chi=0.729, c1=1.49618, c2=1.49618,
        seed=0, x0_pop=None, log_every=None):
    """Constriction-coefficient PSO. [31,7]"""
    rng, D, lb, ub, X, f, fes, bx, bf = _run_common_init(problem, pop_size, seed, x0_pop)
    history = [(fes, bf - problem.f_star)]
    vmax = 0.2 * (ub - lb)
    V = rng.uniform(-vmax, vmax, size=(pop_size, D))
    pbest_x, pbest_f = X.copy(), f.copy()
    gbest_x, gbest_f = bx.copy(), bf

    while fes < max_fes:
        r1 = rng.uniform(0, 1, (pop_size, D))
        r2 = rng.uniform(0, 1, (pop_size, D))
        V = chi * (V + c1 * r1 * (pbest_x - X) + c2 * r2 * (gbest_x[None, :] - X))
        V = np.clip(V, -vmax, vmax)
        X = _clip(X + V, lb, ub)
        f = problem.evaluate(X)
        fes += pop_size
        improve = f < pbest_f
        pbest_x[improve], pbest_f[improve] = X[improve], f[improve]
        i = np.argmin(pbest_f)
        if pbest_f[i] < gbest_f:
            gbest_f, gbest_x = pbest_f[i], pbest_x[i].copy()
        bf, bx = gbest_f, gbest_x
        if log_every: _log(history, fes, bf, problem.f_star, log_every)
    history.append((fes, bf - problem.f_star))
    return {"best_x": bx, "best_f": bf, "best_error": bf - problem.f_star, "history": history, "fes": fes}


# ---------------------------------------------------------------- DE/rand/1/bin
def de(problem, max_fes, pop_size=50, F=0.5, CR=0.9,
       seed=0, x0_pop=None, log_every=None):
    """Classic DE/rand/1/bin. [55]"""
    rng, D, lb, ub, X, f, fes, bx, bf = _run_common_init(problem, pop_size, seed, x0_pop)
    history = [(fes, bf - problem.f_star)]
    idx_all = np.arange(pop_size)
    while fes < max_fes:
        V = np.empty_like(X)
        for i in range(pop_size):
            choices = idx_all[idx_all != i]
            r1, r2, r3 = rng.choice(choices, 3, replace=False)
            mutant = X[r1] + F * (X[r2] - X[r3])
            jrand = rng.integers(0, D)
            cross = rng.uniform(0, 1, D) < CR
            cross[jrand] = True
            V[i] = np.where(cross, mutant, X[i])
        V = _clip(V, lb, ub)
        fv = problem.evaluate(V)
        fes += pop_size
        better = fv <= f
        X[better], f[better] = V[better], fv[better]
        i = np.argmin(f)
        if f[i] < bf:
            bf, bx = f[i], X[i].copy()
        if log_every: _log(history, fes, bf, problem.f_star, log_every)
    history.append((fes, bf - problem.f_star))
    return {"best_x": bx, "best_f": bf, "best_error": bf - problem.f_star, "history": history, "fes": fes}


# ---------------------------------------------------------------- ABC
def abc(problem, max_fes, pop_size=50, limit=None, seed=0, x0_pop=None, log_every=None):
    """Artificial Bee Colony. [30]"""
    rng, D, lb, ub, X, f, fes, bx, bf = _run_common_init(problem, pop_size, seed, x0_pop)
    history = [(fes, bf - problem.f_star)]
    limit = limit if limit is not None else pop_size * D
    trials = np.zeros(pop_size)
    idx_all = np.arange(pop_size)

    def make_candidate(i):
        j = rng.integers(0, D)
        k = rng.choice(idx_all[idx_all != i])
        phi = rng.uniform(-1, 1)
        cand = X[i].copy()
        cand[j] = X[i, j] + phi * (X[i, j] - X[k, j])
        return _clip(cand[None, :], lb, ub)[0]

    while fes < max_fes:
        # employed bees
        for i in range(pop_size):
            cand = make_candidate(i)
            fc = problem.evaluate(cand[None, :])[0]
            fes += 1
            if fc < f[i]:
                X[i], f[i], trials[i] = cand, fc, 0
            else:
                trials[i] += 1
        # onlooker bees
        fit = np.where(f >= 0, 1 / (1 + f), 1 + np.abs(f))
        probs = fit / fit.sum()
        i = 0; t = 0
        while t < pop_size and fes < max_fes:
            if rng.uniform() < probs[i]:
                cand = make_candidate(i)
                fc = problem.evaluate(cand[None, :])[0]
                fes += 1
                if fc < f[i]:
                    X[i], f[i], trials[i] = cand, fc, 0
                else:
                    trials[i] += 1
                t += 1
            i = (i + 1) % pop_size
        # scout bees
        for i in range(pop_size):
            if trials[i] > limit:
                X[i] = _lhs(1, D, lb, ub, rng)[0]
                f[i] = problem.evaluate(X[i][None, :])[0]
                fes += 1
                trials[i] = 0
        j = np.argmin(f)
        if f[j] < bf:
            bf, bx = f[j], X[j].copy()
        if log_every: _log(history, fes, bf, problem.f_star, log_every)
    history.append((fes, bf - problem.f_star))
    return {"best_x": bx, "best_f": bf, "best_error": bf - problem.f_star, "history": history, "fes": fes}


# ---------------------------------------------------------------- GWO
def gwo(problem, max_fes, pop_size=50, seed=0, x0_pop=None, log_every=None):
    """Grey Wolf Optimizer. [42]"""
    rng, D, lb, ub, X, f, fes, bx, bf = _run_common_init(problem, pop_size, seed, x0_pop)
    history = [(fes, bf - problem.f_star)]
    max_iter = max(1, (max_fes - fes) // pop_size)
    it = 0
    while fes < max_fes:
        order = np.argsort(f)
        alpha, beta, delta = X[order[0]], X[order[1]], X[order[2]]
        a = 2 - 2 * it / max(1, max_iter)
        for k, leader in enumerate((alpha, beta, delta)):
            r1 = rng.uniform(0, 1, (pop_size, D)); r2 = rng.uniform(0, 1, (pop_size, D))
            Avec = 2 * a * r1 - a; Cvec = 2 * r2
            Dvec = np.abs(Cvec * leader[None, :] - X)
            Xk = leader[None, :] - Avec * Dvec
            if k == 0: Xsum = Xk
            else: Xsum = Xsum + Xk
        X = _clip(Xsum / 3.0, lb, ub)
        f = problem.evaluate(X)
        fes += pop_size; it += 1
        j = np.argmin(f)
        if f[j] < bf:
            bf, bx = f[j], X[j].copy()
        if log_every: _log(history, fes, bf, problem.f_star, log_every)
    history.append((fes, bf - problem.f_star))
    return {"best_x": bx, "best_f": bf, "best_error": bf - problem.f_star, "history": history, "fes": fes}


# ---------------------------------------------------------------- CMA-ES (via `cma` package, Hansen's reference implementation)
def cmaes(problem, max_fes, sigma0=None, pop_size=None, seed=0, x0_pop=None, log_every=None):
    """CMA-ES using the pycma reference implementation. [21]"""
    import cma
    D = problem.dim
    lb, ub = problem.lb, problem.ub
    rng = np.random.default_rng(seed)
    x0 = x0_pop[0] if x0_pop is not None else rng.uniform(lb, ub)
    sigma0 = sigma0 if sigma0 is not None else 0.3 * (ub - lb).mean()
    opts = {'bounds': [list(lb), list(ub)], 'seed': (seed % 2**31) + 1, 'verbose': -9,
            'maxfevals': max_fes}
    if pop_size: opts['popsize'] = pop_size
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    fes = 0
    bx, bf = x0.copy(), problem.evaluate(x0[None, :])[0]; fes += 1
    history = [(fes, bf - problem.f_star)]
    while not es.stop() and fes < max_fes:
        Xs = es.ask()
        Xa = _clip(np.array(Xs), lb, ub)
        fs = problem.evaluate(Xa)
        fes += len(Xs)
        es.tell(Xs, list(fs))
        j = np.argmin(fs)
        if fs[j] < bf:
            bf, bx = fs[j], Xa[j].copy()
        if log_every: _log(history, fes, bf, problem.f_star, log_every)
    history.append((fes, bf - problem.f_star))
    return {"best_x": bx, "best_f": bf, "best_error": bf - problem.f_star, "history": history, "fes": fes}


# ---------------------------------------------------------------- L-SHADE (success-history DE + linear pop reduction)
def lshade(problem, max_fes, pop_init=None, memory_size=6, arc_rate=2.6, p_best_rate=0.11,
           n_min=4, seed=0, x0_pop=None, log_every=None):
    """L-SHADE: DE/current-to-pbest/1 with an external archive, success-history
    parameter adaptation for F and CR, and linear population size reduction. [56]"""
    rng = np.random.default_rng(seed)
    D = problem.dim
    lb, ub = problem.lb, problem.ub
    N0 = pop_init if pop_init is not None else max(n_min, int(round(18 * D)))
    X = x0_pop.copy() if x0_pop is not None else _lhs(N0, D, lb, ub, rng)
    f = problem.evaluate(X)
    fes = X.shape[0]
    bi = np.argmin(f); bx, bf = X[bi].copy(), f[bi]
    history = [(fes, bf - problem.f_star)]

    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    mem_pos = 0
    archive = np.empty((0, D))
    N_t = N0
    N_INIT = N0

    while fes < max_fes:
        S_F, S_CR, S_w = [], [], []
        r_idx = rng.integers(0, memory_size, N_t)
        CRs = np.clip(rng.normal(M_CR[r_idx], 0.1), 0, 1)
        Fs = np.clip(_cauchy(M_F[r_idx], 0.1, rng), 1e-6, 1.0)
        order = np.argsort(f)
        p_num = max(2, int(round(p_best_rate * N_t)))
        pbest_pool = order[:p_num]
        V = np.empty_like(X)
        idx_all = np.arange(N_t)
        pool_all = np.vstack([X, archive]) if archive.size else X
        for i in range(N_t):
            pbest = X[rng.choice(pbest_pool)]
            r1 = rng.choice(idx_all[idx_all != i])
            r2 = rng.integers(0, pool_all.shape[0])
            mutant = X[i] + Fs[i] * (pbest - X[i]) + Fs[i] * (X[r1] - pool_all[r2])
            jrand = rng.integers(0, D)
            cross = rng.uniform(0, 1, D) < CRs[i]
            cross[jrand] = True
            V[i] = np.where(cross, mutant, X[i])
        V = _clip(V, lb, ub)
        fv = problem.evaluate(V)
        fes += N_t
        better = fv < f
        for i in np.where(better)[0]:
            S_F.append(Fs[i]); S_CR.append(CRs[i]); S_w.append(f[i] - fv[i])
        if archive.shape[0] < N_t:
            archive = np.vstack([archive, X[better]]) if archive.size else X[better].copy()
        elif better.any():
            n_repl = int(better.sum())
            if n_repl > 0 and archive.shape[0] > 0:
                repl_idx = rng.choice(archive.shape[0], min(n_repl, archive.shape[0]), replace=False)
                archive[repl_idx] = X[better][:len(repl_idx)]
        X[better], f[better] = V[better], fv[better]

        if S_F:
            w = np.array(S_w); w = w / w.sum()
            SF = np.array(S_F); SCR = np.array(S_CR)
            mean_F = (w * SF ** 2).sum() / (w * SF).sum()
            mean_CR = (w * SCR).sum()
            M_F[mem_pos] = mean_F
            M_CR[mem_pos] = mean_CR
            mem_pos = (mem_pos + 1) % memory_size

        j = np.argmin(f)
        if f[j] < bf:
            bf, bx = f[j], X[j].copy()

        # linear population size reduction, Eq. (20)-style
        N_next = int(round(N_INIT + (n_min - N_INIT) * (fes / max_fes)))
        N_next = max(n_min, N_next)
        if N_next < N_t:
            keep = np.argsort(f)[:N_next]
            X, f = X[keep], f[keep]
            N_t = N_next
            if archive.shape[0] > N_t:
                archive = archive[rng.choice(archive.shape[0], N_t, replace=False)]

        if log_every: _log(history, fes, bf, problem.f_star, log_every)
    history.append((fes, bf - problem.f_star))
    return {"best_x": bx, "best_f": bf, "best_error": bf - problem.f_star, "history": history, "fes": fes}


def _cauchy(loc, scale, rng):
    return loc + scale * np.tan(np.pi * (rng.uniform(0, 1, size=np.shape(loc)) - 0.5))


# ---------------------------------------------------------------- IPOP-CMA-ES (restarts with growing population)
def cmaes_ipop(problem, max_fes, sigma0=None, incpopsize=2.0, seed=0, x0_pop=None, log_every=None):
    """IPOP-CMA-ES: restart CMA-ES whenever its own stopping criteria fire, doubling the
    population size each restart, until the evaluation budget is used. Uses the pycma
    reference implementation for each individual run. Every objective call counts
    against `max_fes`. [21] (restart scheme: Auger & Hansen, 2005)."""
    import cma
    D = problem.dim
    lb, ub = problem.lb, problem.ub
    rng = np.random.default_rng(seed)
    sigma0 = sigma0 if sigma0 is not None else 0.3 * float((ub - lb).mean())
    base_lambda = 4 + int(3 * np.log(D))
    fes = 0
    bx, bf = None, np.inf
    history = []
    restart = 0
    while fes < max_fes:
        x0 = (x0_pop[0] if (restart == 0 and x0_pop is not None) else rng.uniform(lb, ub))
        lam = int(round(base_lambda * (incpopsize ** restart)))
        remaining = max_fes - fes
        opts = {'bounds': [list(lb), list(ub)], 'seed': int((seed * 1009 + restart) % 2**31) + 1,
                'verbose': -9, 'popsize': lam, 'maxfevals': remaining}
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        while not es.stop() and fes < max_fes:
            Xs = es.ask()
            Xa = _clip(np.array(Xs), lb, ub)
            fs = problem.evaluate(Xa)
            fes += len(Xs)
            es.tell(Xs, list(fs))
            j = int(np.argmin(fs))
            if fs[j] < bf:
                bf, bx = float(fs[j]), Xa[j].copy()
            if log_every and (not history or fes - history[-1][0] >= log_every):
                history.append((fes, bf - problem.f_star))
        restart += 1
        if lam > 200 * base_lambda:      # safety cap; budget will normally end first
            break
    history.append((fes, bf - problem.f_star))
    return {"best_x": bx, "best_f": bf, "best_error": bf - problem.f_star,
            "history": history, "fes": fes, "restarts": restart}
