"""Rule evaluation, post-processing (minimal generators, non-redundancy) and MLIC baseline."""
import numpy as np


def covers(B, pattern):
    """Boolean vector: rows of B containing every item (f, v) of the pattern."""
    c = np.ones(len(B), dtype=bool)
    for f, v in pattern:
        c &= B[:, f] == v
    return c


def predict(B, patterns):
    out = np.zeros(len(B), dtype=bool)
    for pat in patterns:
        out |= covers(B, pat)
    return out.astype(int)


def objective(B, y, patterns, lam):
    pred = predict(B, patterns)
    return sum(len(p) for p in patterns) + lam * int((pred != y).sum())


def postprocess(B, y, patterns, rng):
    """(i) make every pattern a minimal generator in D- by dropping items that do not change
    its negative cover; (ii) drop redundant patterns (covering no positive uniquely).
    Neither step can increase the objective of Eq. (1)."""
    neg = B[y == 0]
    pos = B[y == 1]
    pats = [list(p) for p in patterns]
    for p in pats:
        changed = True
        while changed and len(p) > 1:
            changed = False
            base = covers(neg, p).sum()
            for it in [p[i] for i in rng.permutation(len(p))]:
                cand = [a for a in p if a != it]
                if covers(neg, cand).sum() == base:
                    p.remove(it)
                    changed = True
                    break
    changed = True
    while changed:
        changed = False
        cov = [covers(pos, p) for p in pats]
        for j in range(len(pats)):
            others = np.zeros(len(pos), dtype=bool)
            for l in range(len(pats)):
                if l != j:
                    others |= cov[l]
            if not (cov[j] & ~others).any():
                pats.pop(j)
                changed = True
                break
    return pats


def rule_size(patterns):
    return sum(len(p) for p in patterns)


# ---------------------------------------------------------------- MLIC (CNF, Malioutov & Meel)
def build_mlic(w, B, y, k, lam):
    """CNF with k clauses over literals f / not f. Positive example: every clause satisfied
    (or noise). Negative: some clause falsified (or noise)."""
    m, n = B.shape
    b = [{(f, v): w.new() for f in range(n) for v in (1, 0)} for _ in range(k)]
    for j in range(k):
        for f in range(n):
            w.hard([-b[j][(f, 1)], -b[j][(f, 0)]])
    for i in range(m):
        x = B[i]
        eta = w.new()
        if y[i] == 1:
            for j in range(k):
                sat = [b[j][(f, int(x[f]))] for f in range(n)]
                w.hard([eta] + sat)
        else:
            zs = []
            for j in range(k):
                z = w.new()  # z -> clause j falsified by x
                for f in range(n):
                    w.hard([-z, -b[j][(f, int(x[f]))]])
                zs.append(z)
            w.hard([eta] + zs)
        w.soft([-eta], lam)
    for j in range(k):
        for v in b[j].values():
            w.soft([-v], 1)
    return b


def decode_mlic(b, model):
    return [[(f, v) for (f, v), var in bj.items() if model.get(var, False)] for bj in b]


def predict_cnf(B, clauses):
    out = np.ones(len(B), dtype=bool)
    for cl in clauses:
        sat = np.zeros(len(B), dtype=bool)
        for f, v in cl:
            sat |= B[:, f] == v
        out &= sat
    return out.astype(int)
