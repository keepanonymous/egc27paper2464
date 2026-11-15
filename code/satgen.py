"""SAT-based enumeration of minimal generators of the negative class under constraints.

Items: each binary feature f gives two items (f,1) "f=1" and (f,0) "f=0"; a transaction
contains exactly one of them per feature. An itemset X is a *minimal generator* in D- iff
every proper subset covers strictly more negative transactions, iff (it suffices to check
immediate subsets) every item a of X has a *witness*: a negative transaction that misses a
and contains all other items of X. The empty itemset is excluded.

Encoding (variables):
  x_a      item a is in X
  z_T      negative transaction T misses at most one item of X   (only the -> direction)
  cp_i     positive transaction i is covered by X (for the support constraint)
  cn_i     negative transaction i is covered by X (for the emergence constraint)
Constraints:
  exclusivity            not x_(f,1) or not x_(f,0)
  non-empty              OR_a x_a
  freeness (generator)   x_a -> OR_{T in D-, a not in T} z_T ;  z_T -> AMO{x_b : b not in T}
  [optional] Supp+ >= sigma, |X| <= L, Supp- <= theta  (cardinality constraints)
  [optional, redundant]  implied pairs: if Cov(a,D-) is a subset of Cov(b,D-) then not both
Enumeration: each model gives X = {a : x_a}; we block exactly this X and solve again.
"""
import itertools
import time

import numpy as np
from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from pysat.solvers import Solver


def items_of(n):
    return [(f, v) for f in range(n) for v in (1, 0)]


def dedup(B):
    return np.unique(B, axis=0) if len(B) else B


class GeneratorEnumerator:
    def __init__(self, B, y, min_pos=1, max_len=None, max_neg=None, implied_pairs=True,
                 solver="cd15"):
        self.B, self.y = B, y
        n = B.shape[1]
        self.items = items_of(n)
        P, N = B[y == 1], dedup(B[y == 0])  # duplicated negatives give identical witnesses
        self.P, self.N = P, B[y == 0]
        pool = IDPool()
        x = {a: pool.id(("x", a)) for a in self.items}
        cl = []
        for f in range(n):
            cl.append([-x[(f, 1)], -x[(f, 0)]])
        cl.append([x[a] for a in self.items])
        # freeness: witnesses among (deduplicated) negatives
        z = {}
        for t, T in enumerate(N):
            zt = pool.id(("z", t))
            z[t] = zt
            absent = [x[(f, 1 - int(T[f]))] for f in range(n)]  # items not in T
            amo = CardEnc.atmost(lits=absent, bound=1, vpool=pool, encoding=EncType.seqcounter)
            cl += [[-zt] + c for c in amo.clauses]
        for (f, v), xa in x.items():
            wit = [z[t] for t, T in enumerate(N) if T[f] != v]
            cl.append([-xa] + wit)  # empty wit -> item never selectable
        # redundant implied pairs on D-: Cov(a) subset of Cov(b) => not both
        if implied_pairs and len(N):
            cov = {a: frozenset(np.where(N[:, a[0]] == a[1])[0]) for a in self.items}
            seen = set()
            for a, b in itertools.permutations(self.items, 2):
                if a[0] != b[0] and cov[a] <= cov[b] and frozenset((a, b)) not in seen:
                    # X containing a and b: dropping b leaves Cov(X, D-) unchanged -> not free
                    seen.add(frozenset((a, b)))
                    cl.append([-x[a], -x[b]])
            self.n_implied = len(seen)
        # support in D+ (anti-monotone) and in D- (monotone)
        if min_pos and min_pos > 0:
            cps = []
            for i, T in enumerate(P):
                cp = pool.id(("cp", i))
                absent = [x[(f, 1 - int(T[f]))] for f in range(n)]
                cl += [[-cp, -a] for a in absent]  # cp -> covered
                cps.append(cp)
            card = CardEnc.atleast(lits=cps, bound=min_pos, vpool=pool, encoding=EncType.seqcounter)
            cl += card.clauses
        if max_neg is not None:
            cns = []
            for i, T in enumerate(self.N):
                cn = pool.id(("cn", i))
                absent = [x[(f, 1 - int(T[f]))] for f in range(n)]
                cl.append([cn] + absent)  # covered -> cn
                cns.append(cn)
            card = CardEnc.atmost(lits=cns, bound=max_neg, vpool=pool, encoding=EncType.seqcounter)
            cl += card.clauses
        if max_len is not None:
            card = CardEnc.atmost(lits=list(x.values()), bound=max_len, vpool=pool,
                                  encoding=EncType.seqcounter)
            cl += card.clauses
        self.x, self.clauses, self.pool = x, cl, pool
        self.solver_name = solver

    def enumerate(self, limit=None, time_limit=None):
        out = []
        t0 = time.time()
        with Solver(name=self.solver_name, bootstrap_with=self.clauses) as s:
            while s.solve():
                m = set(l for l in s.get_model() if l > 0)
                X = tuple(sorted(a for a, v in self.x.items() if v in m))
                out.append(X)
                s.add_clause([-self.x[a] if a in X else self.x[a] for a in self.items])
                if (limit and len(out) >= limit) or (time_limit and time.time() - t0 > time_limit):
                    return out, False
        return out, True


# ------------------------------------------------------------------ brute force (for checking)
def cover(B, X):
    c = np.ones(len(B), dtype=bool)
    for f, v in X:
        c &= B[:, f] == v
    return c


def is_minimal_generator(N, X):
    if not X:
        return False
    s = cover(N, X).sum()
    return all(cover(N, tuple(b for b in X if b != a)).sum() > s for a in X)


def brute_force(B, y, min_pos=1, max_len=None, max_neg=None):
    n = B.shape[1]
    P, N = B[y == 1], B[y == 0]
    L = max_len or n
    res = []
    feats = range(n)
    for r in range(1, L + 1):
        for fs in itertools.combinations(feats, r):
            for vals in itertools.product((1, 0), repeat=r):
                X = tuple(zip(fs, vals))
                if not is_minimal_generator(N, X):
                    continue
                if min_pos and cover(P, X).sum() < min_pos:
                    continue
                if max_neg is not None and cover(N, X).sum() > max_neg:
                    continue
                res.append(tuple(sorted(X)))
    return res
