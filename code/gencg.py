"""Generation-selection learning of disjunctive rules (k-GDI) with a MaxSAT oracle that
produces minimal generators of the negative class on demand.

Objective (as in DRL / MLIC):  f(R) = sum_{X in R} |X| + lam * (FN(R) + FP(R)),  |R| <= k.

Master  : given a pool C of itemsets, choose R subset of C, |R| <= k, minimizing f (MaxSAT).
Oracle  : given a residual problem (positives P_r, negatives N_r that the rest of the rule does
          not cover), find the itemset X maximizing gain(X) = lam*(|Cov(X,P_r)| - |Cov(X,N_r)|) - |X|
          (MaxSAT). Optimal oracle solutions are minimal generators of D- (Prop. in the paper);
          the freeness (witness) constraint can be added as a redundant constraint.
Loop    : R = {}; repeat: ask the oracle for an addition (if |R| < k) and for a replacement of each
          X_j in R; add every improving itemset to the pool; re-solve the master. Stop when no
          improving itemset exists (certificate: R is optimal among the pool and no single
          addition/replacement by an itemset improves it), or when time runs out.
"""
import os
import config
from config import RESULTS, PAPER, EGC
import os
import sys
import time

import numpy as np


from encode import WCNFWriter, _run_openwbo, _parse  # noqa: E402

TMP = config.TMP
os.makedirs(TMP, exist_ok=True)


def cover(B, X):
    c = np.ones(len(B), dtype=bool)
    for f, v in X:
        c &= B[:, f] == v
    return c


def objective(B, y, R, lam):
    pred = np.zeros(len(B), dtype=bool)
    for X in R:
        pred |= cover(B, X)
    return sum(len(X) for X in R) + lam * int((pred != (y == 1)).sum())


def groups(B, idx):
    """Distinct rows among B[idx] with multiplicities."""
    if len(idx) == 0:
        return np.zeros((0, B.shape[1]), dtype=B.dtype), np.zeros(0, dtype=int)
    u, cnt = np.unique(B[idx], axis=0, return_counts=True)
    return u, cnt


def solve_wcnf(path, timeout, exact_first=True):
    """Two-phase Open-WBO portfolio (linear search, then OLL); returns (status, model)."""
    t0 = time.time()
    rc, out = _run_openwbo(path, max(1, int(timeout / 2)), 1)
    st, model = _parse(out)
    if st == "OPTIMUM FOUND":
        return st, model
    remaining = timeout - (time.time() - t0)
    if remaining >= 1:
        rc, out = _run_openwbo(path, max(1, int(remaining)), 4)
        st2, model2 = _parse(out)
        if st2 == "OPTIMUM FOUND" or model is None:
            return st2, model2
        return (st2 or st), (model2 if model2 is not None else model)
    return st, model


# ----------------------------------------------------------------------------- oracle
class Oracle:
    def __init__(self, B, y, lam, freeness=False, implied_pairs=True, max_len=None, cap=600, seed=0):
        self.B, self.y, self.lam = B, y, lam
        self.cap, self.rng = cap, np.random.default_rng(seed)
        self.n = B.shape[1]
        self.items = [(f, v) for f in range(self.n) for v in (1, 0)]
        self.freeness, self.max_len = freeness, max_len
        Nall = B[y == 0]
        self.Ndist = np.unique(Nall, axis=0)
        self.pairs = []
        if implied_pairs:
            cov = {a: frozenset(np.where(Nall[:, a[0]] == a[1])[0]) for a in self.items}
            seen = set()
            for a in self.items:
                for b in self.items:
                    if a != b and a[0] != b[0] and cov[a] <= cov[b]:
                        key = frozenset((a, b))
                        if key not in seen:
                            seen.add(key)
                            self.pairs.append((a, b))

    def subsample(self, G, C):
        """If there are more than `cap` distinct transactions, keep a uniform sample of them and
        scale their multiplicities so that the sample represents the whole set. By Lemma 2, optimal
        answers on the sample are still minimal generators of D-; their gain is re-evaluated on the
        full residual by the caller."""
        if self.cap is None or len(G) <= self.cap:
            return G, C
        idx = self.rng.choice(len(G), self.cap, replace=False)
        scale = C.sum() / C[idx].sum()
        return G[idx], np.maximum(1, np.round(C[idx] * scale)).astype(int)

    def best(self, pos_idx, neg_idx, timeout, tag="o", block=()):
        """MaxSAT: argmax_X lam*(|Cov(X,P_r)|-|Cov(X,N_r)|) - |X| over non-empty itemsets
        satisfying the constraints. Returns (X, gain, proved_optimal, n_clauses)."""
        B, lam, n = self.B, self.lam, self.n
        path = f"{TMP}/{tag}_{os.getpid()}.wcnf"
        w = WCNFWriter(path)
        x = {a: w.new() for a in self.items}
        for f in range(n):
            w.hard([-x[(f, 1)], -x[(f, 0)]])
        w.hard(list(x.values()))  # non-empty
        for a, b in self.pairs:
            w.hard([-x[a], -x[b]])
        Pg, Pc = groups(B, pos_idx)
        Ng, Nc = groups(B, neg_idx)
        Pg, Pc = self.subsample(Pg, Pc)
        Ng, Nc = self.subsample(Ng, Nc)
        for T, c in zip(Pg, Pc):  # cp -> X covers T  (soft cp, weight lam*c)
            cp = w.new()
            for f in range(n):
                w.hard([-cp, -x[(f, 1 - int(T[f]))]])
            w.soft([cp], lam * int(c))
        for T, c in zip(Ng, Nc):  # X covers T -> cn  (soft -cn, weight lam*c)
            cn = w.new()
            w.hard([cn] + [x[(f, 1 - int(T[f]))] for f in range(n)])
            w.soft([-cn], lam * int(c))
        if self.freeness:  # every selected item has a witness among the negatives of D-
            zs = []
            for T in self.Ndist:
                z = w.new()
                absent = [x[(f, 1 - int(T[f]))] for f in range(n)]
                w.atmost1_if(absent, z)
                zs.append(z)
            for (f, v), xa in x.items():
                wit = [zs[t] for t, T in enumerate(self.Ndist) if T[f] != v]
                w.hard([-xa] + wit)
        if self.max_len:
            # sequential counter: at most max_len items
            lits = list(x.values())
            L = self.max_len
            s = [[w.new() for _ in range(L)] for _ in range(len(lits))]
            for i, l in enumerate(lits):
                w.hard([-l, s[i][0]])
                if i > 0:
                    for j in range(L):
                        w.hard([-s[i - 1][j], s[i][j]])
                    for j in range(1, L):
                        w.hard([-l, -s[i - 1][j - 1], s[i][j]])
                    w.hard([-l, -s[i - 1][L - 1]])
        for X in block:  # exclude already generated itemsets (exact blocking on x)
            w.hard([-x[a] if a in X else x[a] for a in self.items])
        for xa in x.values():
            w.soft([-xa], 1)
        w.close()
        st, model = solve_wcnf(path, timeout)
        ncl = w.n_clauses
        os.remove(path)
        if model is None:
            return None, None, False, ncl
        X = tuple(sorted(a for a, v in x.items() if model.get(v, False)))
        gain = lam * (int(cover(B[pos_idx], X).sum()) - int(cover(B[neg_idx], X).sum())) - len(X)
        return X, gain, st == "OPTIMUM FOUND", ncl


# ----------------------------------------------------------------------------- master
def master(B, y, pool, k, lam, timeout, tag="m"):
    """MaxSAT selection of at most k itemsets of the pool minimizing f."""
    if not pool:
        return [], True
    covs = np.stack([cover(B, X) for X in pool], axis=1)  # m x |pool|
    key = np.concatenate([covs.astype(np.uint8), y[:, None].astype(np.uint8)], axis=1)
    G, cnt = np.unique(key, axis=0, return_counts=True)
    path = f"{TMP}/{tag}_{os.getpid()}.wcnf"
    w = WCNFWriter(path)
    ys = [w.new() for _ in pool]
    for g, c in zip(G, cnt):
        cs = [ys[j] for j in range(len(pool)) if g[j]]
        label = g[-1]
        r = w.new()
        w.hard([-r] + cs)
        for v in cs:
            w.hard([r, -v])
        w.soft([r] if label == 1 else [-r], lam * int(c))
    # at most k selected (sequential counter)
    L = k
    if len(pool) > k:
        s = [[w.new() for _ in range(L)] for _ in range(len(ys))]
        for i, l in enumerate(ys):
            w.hard([-l, s[i][0]])
            if i > 0:
                for j in range(L):
                    w.hard([-s[i - 1][j], s[i][j]])
                for j in range(1, L):
                    w.hard([-l, -s[i - 1][j - 1], s[i][j]])
                w.hard([-l, -s[i - 1][L - 1]])
    for v, X in zip(ys, pool):
        w.soft([-v], max(1, len(X)))
    w.close()
    st, model = solve_wcnf(path, timeout)
    os.remove(path)
    if model is None:
        return None, False
    return [X for v, X in zip(ys, pool) if model.get(v, False)], st == "OPTIMUM FOUND"


# ----------------------------------------------------------------------------- main loop
def learn(B, y, k=3, lam=10, time_limit=60, freeness=False, oracle_cap=15, max_len=None, log=None, n_alt=1,
          init_rule=None, init_pool=None):
    t0 = time.time()
    orc = Oracle(B, y, lam, freeness=freeness, max_len=max_len)
    pool, R = list(init_pool or []), list(init_rule or [])
    allpos, allneg = np.where(y == 1)[0], np.where(y == 0)[0]
    it, certified, n_oracle, oracle_ncl = 0, False, 0, []
    trace = []
    while time.time() - t0 < time_limit - 1:
        it += 1
        found, all_proved = [], True
        queries = []
        covR = [cover(B, X) for X in R]
        if len(R) < k:
            c = np.zeros(len(B), dtype=bool)
            for cv in covR:
                c |= cv
            queries.append((None, 0, c))
        for j, X in enumerate(R):
            c = np.zeros(len(B), dtype=bool)
            for l, cv in enumerate(covR):
                if l != j:
                    c |= cv
            # current contribution of X_j on the residual of the others
            cj = covR[j]
            T = lam * (int((cj & ~c)[allpos].sum()) - int((cj & ~c)[allneg].sum())) - len(X)
            queries.append((j, T, c))
        for j, T, c in queries:
            rem = time_limit - (time.time() - t0)
            if rem < 1:
                all_proved = False
                break
            pr, nr = allpos[~c[allpos]], allneg[~c[allneg]]
            block = []
            for a_i in range(n_alt):
                rem = time_limit - (time.time() - t0)
                if rem < 1:
                    break
                X, gain, proved, ncl = orc.best(pr, nr, min(rem, oracle_cap), tag=f"o{j}", block=block)
                n_oracle += 1
                oracle_ncl.append(ncl)
                if a_i == 0:
                    all_proved &= proved
                if X is None or gain <= T:
                    break
                found.append((j, X, gain, T))
                block.append(X)
                if X not in pool:
                    pool.append(X)
        trace.append(dict(it=it, R=list(R), found=[(j, X, g, T) for j, X, g, T in found]))
        if not found:
            certified = all_proved
            break
        rem = time_limit - (time.time() - t0)
        if rem < 1:
            # apply the best single improvement without the master
            j, X, g, T = max(found, key=lambda t: t[2] - t[3])
            R = R + [X] if j is None else R[:j] + [X] + R[j + 1:]
            break
        newR, _ = master(B, y, pool, k, lam, min(rem, oracle_cap))
        if newR is None or objective(B, y, newR, lam) >= objective(B, y, R, lam):
            j, X, g, T = max(found, key=lambda t: t[2] - t[3])  # fallback: apply best swap
            newR = R + [X] if j is None else R[:j] + [X] + R[j + 1:]
        R = newR
        if log:
            log(f"it {it}: f={objective(B, y, R, lam)} |pool|={len(pool)} R={R}")
    return dict(rule=R, iterations=it, pool=len(pool), pool_items=list(pool), certified=certified,
                n_oracle=n_oracle, oracle_clauses=float(np.mean(oracle_ncl)) if oracle_ncl else 0,
                time=time.time() - t0, trace=trace)


# ----------------------------------------------------------------------------- LP column generation
def _price(orc, pos_groups, pos_w, neg_idx, timeout, block=(), scale=100, tag="p"):
    """Pricing oracle: argmin_X |X| + lam*|Cov(X,N)| - sum_{g covered} pos_w[g] over non-empty
    itemsets satisfying the oracle constraints (weighted MaxSAT, weights scaled to integers)."""
    B, lam, n = orc.B, orc.lam, orc.n
    path = f"{TMP}/{tag}_{os.getpid()}.wcnf"
    w = WCNFWriter(path)
    x = {a: w.new() for a in orc.items}
    for f in range(n):
        w.hard([-x[(f, 1)], -x[(f, 0)]])
    w.hard(list(x.values()))
    for a, b in orc.pairs:
        w.hard([-x[a], -x[b]])
    pg_s, pw_s = pos_groups, pos_w
    if orc.cap is not None and len(pos_groups) > orc.cap:
        idx = orc.rng.choice(len(pos_groups), orc.cap, replace=False)
        pg_s, pw_s = pos_groups[idx], pos_w[idx] * (pos_w.sum() / max(pos_w[idx].sum(), 1e-9))
    for T, wt in zip(pg_s, pw_s):
        iw = int(round(scale * wt))
        if iw <= 0:
            continue
        cp = w.new()
        for f in range(n):
            w.hard([-cp, -x[(f, 1 - int(T[f]))]])
        w.soft([cp], iw)
    Ng, Nc = orc.subsample(*groups(B, neg_idx))
    for T, c in zip(Ng, Nc):
        cn = w.new()
        w.hard([cn] + [x[(f, 1 - int(T[f]))] for f in range(n)])
        w.soft([-cn], scale * lam * int(c))
    if orc.freeness:
        zs = []
        for T in orc.Ndist:
            z = w.new()
            w.atmost1_if([x[(f, 1 - int(T[f]))] for f in range(n)], z)
            zs.append(z)
        for (f, v), xa in x.items():
            w.hard([-xa] + [zs[t] for t, T in enumerate(orc.Ndist) if T[f] != v])
    for X in block:
        w.hard([-x[a] if a in X else x[a] for a in orc.items])
    for xa in x.values():
        w.soft([-xa], scale)
    w.close()
    st, model = solve_wcnf(path, timeout)
    ncl = w.n_clauses
    os.remove(path)
    if model is None:
        return None, None, False, ncl
    X = tuple(sorted(a for a, v in x.items() if model.get(v, False)))
    cov_pos = np.array([all(T[f] == v for f, v in X) for T in pos_groups]) if len(pos_groups) else np.zeros(0, bool)
    val = len(X) + lam * int(cover(B[neg_idx], X).sum()) - float(pos_w[cov_pos].sum())
    return X, val, st == "OPTIMUM FOUND", ncl


def lp_master(Pg, Pc, pool_cov_pos, pool_cost, k, lam):
    """LP relaxation: min sum cost_X w_X + lam * sum_g c_g xi_g
       s.t. xi_g + sum_{X covers g} w_X >= 1, sum w_X <= k, 0<=w<=1, xi>=0.
       Returns (value, duals mu_g >= 0, kappa >= 0, w)."""
    from scipy.optimize import linprog
    G, C = len(Pg), len(pool_cost)
    c = np.concatenate([np.array(pool_cost, float), lam * Pc.astype(float)])
    A = np.zeros((G + 1, C + G))
    if C:
        A[:G, :C] = -np.asarray(pool_cov_pos, float).T  # pool_cov_pos is C x G
        A[G, :C] = 1
    A[:G, C:] = -np.eye(G)
    b = np.concatenate([-np.ones(G), [k]])
    bounds = [(0, 1)] * C + [(0, None)] * G
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    marg = -res.ineqlin.marginals
    return res.fun, np.maximum(marg[:G], 0), max(marg[G], 0.0), res.x[:C]


def learn_cg(B, y, k=3, lam=10, time_limit=60, freeness=False, oracle_cap=15, n_alt=3,
             cg_share=0.7, refine=True, log=None, greedy_share=0.3):
    """LP-guided generation of minimal generators, integer selection, then swap refinement."""
    t0 = time.time()
    orc = Oracle(B, y, lam, freeness=freeness)
    pos_idx, neg_idx = np.where(y == 1)[0], np.where(y == 0)[0]
    Pg, Pc = groups(B, pos_idx)
    pool, costs, covp = [], [], []
    greedy_res = None
    if greedy_share > 0:  # warm start: greedy generation (fast, good first columns)
        greedy_res = learn(B, y, k=k, lam=lam, time_limit=greedy_share * time_limit, freeness=freeness,
                           oracle_cap=oracle_cap)
        for X in greedy_res["pool_items"]:
            pool.append(X)
            costs.append(len(X) + lam * int(cover(B[neg_idx], X).sum()))
            covp.append(np.array([all(T[f] == v for f, v in X) for T in Pg]))
    cg_it, cg_converged, n_price, lp_vals = 0, False, 0, []
    while time.time() - t0 < cg_share * time_limit:
        cg_it += 1
        lpv, mu, kappa, _ = lp_master(Pg, Pc, np.array(covp) if covp else np.zeros((0, len(Pg))),
                                      costs, k, lam)
        lp_vals.append(lpv)
        added, proved_first = 0, False
        block = list(pool)
        for a_i in range(n_alt):
            rem = cg_share * time_limit - (time.time() - t0)
            if rem < 1:
                break
            X, val, proved, _ = _price(orc, Pg, mu, neg_idx, min(rem, oracle_cap), block=block)
            n_price += 1
            if a_i == 0:
                proved_first = proved
            if X is None or val + kappa >= -1e-6:
                break
            pool.append(X); block.append(X); added += 1
            costs.append(len(X) + lam * int(cover(B[neg_idx], X).sum()))
            covp.append(np.array([all(T[f] == v for f, v in X) for T in Pg]))
        if log:
            log(f"cg {cg_it}: LP={lpv:.2f} pool={len(pool)} added={added}")
        if added == 0:
            cg_converged = proved_first
            break
    rem = time_limit - (time.time() - t0)
    R = list(greedy_res["rule"]) if greedy_res else []
    if pool and rem >= 1:
        Rm, _ = master(B, y, pool, k, lam, min(rem, oracle_cap))
        if Rm is not None and objective(B, y, Rm, lam) <= objective(B, y, R, lam):
            R = Rm
    res_ref = None
    rem = time_limit - (time.time() - t0)
    if refine and rem >= 2:
        res_ref = learn(B, y, k=k, lam=lam, time_limit=rem, freeness=freeness, oracle_cap=oracle_cap,
                        init_rule=R, init_pool=pool)
        R = res_ref["rule"]
    return dict(rule=R, cg_iterations=cg_it, cg_converged=cg_converged, pool=len(pool),
                greedy_certified=bool(greedy_res and greedy_res["certified"]),
                lp_value=lp_vals[-1] if lp_vals else None, n_price=n_price,
                certified=bool(res_ref and res_ref["certified"]),
                ref_iterations=res_ref["iterations"] if res_ref else 0,
                pool_final=res_ref["pool"] if res_ref else len(pool), time=time.time() - t0)
