"""Exploratory test: enumerate minimal generators of D- (support+ >= 1, length <= L), remove exact
duplicates (same positive and negative covers, keep the shortest), then select at most k of them
with an exact MILP (HiGHS via scipy) for three objectives:
  f    : size + lam * (FN + FP)                      (objective of the paper, lam = 10)
  acc  : FN + FP, ties broken by size (lexicographic)  (accuracy, size bounded by k and L)
  cov  : number of covered positives, pure generators only (no negative covered), ties by size
Rules are evaluated on the outer test fold, as in the main experiments."""
import os
import config
from config import RESULTS, PAPER, EGC
import json, os, sys, time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix, vstack, hstack, identity, coo_matrix
from sklearn.model_selection import StratifiedKFold

from data import load_raw, Binarizer
from rules import predict
from encode import WCNFWriter
from gencg import solve_wcnf

LAM, K = 10, 3


def bits(col):
    v = 0
    for i in np.flatnonzero(col):
        v |= 1 << int(i)
    return v


def enumerate_gens(B, y, L, cap, time_limit):
    """Levelwise enumeration of minimal generators of D- with positive support >= 1."""
    n = B.shape[1]
    P, N = B[y == 1], B[y == 0]
    items = [(f, v) for f in range(n) for v in (1, 0)]
    pm = {a: bits(P[:, a[0]] == a[1]) for a in items}
    nm = {a: bits(N[:, a[0]] == a[1]) for a in items}
    allneg = (1 << len(N)) - 1
    t0 = time.time()
    level = {(a,): (pm[a], nm[a]) for a in items if nm[a] != allneg and pm[a] != 0}
    out = dict(level)
    complete = True
    for l in range(2, L + 1):
        new = {}
        pref = {}
        for X in sorted(level):
            pref.setdefault(X[:-1], []).append(X)
        for p, group in pref.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i][-1], group[j][-1]
                    if a[0] == b[0]:
                        continue
                    X = p + (a, b)
                    if any(X[:t] + X[t + 1:] not in level for t in range(l)):
                        continue
                    pos = level[group[i]][0] & pm[b]
                    if not pos:
                        continue
                    neg = level[group[i]][1] & nm[b]
                    if all(level[X[:t] + X[t + 1:]][1] != neg for t in range(l)):
                        new[X] = (pos, neg)
            if len(out) + len(new) > cap or time.time() - t0 > time_limit:
                complete = False
                break
        out.update(new)
        level = new
        if not complete or not level:
            break
    return out, complete, time.time() - t0


def dedup(gens):
    best = {}
    for X, (p, q) in gens.items():
        key = (p, q)
        if key not in best or len(X) < len(best[key]):
            best[key] = X
    return [(X, p, q) for (p, q), X in best.items()]


def to_matrix(bitsets, nrows):
    """List of int bitsets (one per column) -> sparse nrows x ncols boolean matrix."""
    rows, cols = [], []
    for j, v in enumerate(bitsets):
        i = 0
        while v:
            if v & 1:
                rows.append(i); cols.append(j)
            v >>= 1; i += 1
    return coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nrows, len(bitsets))).tocsr()


def solve_milp(cands, nP, nN, objective, time_limit):
    """Variables: s (|C|, binary), y (nP, [0,1]), z (nN, binary).  Returns selected indices."""
    if objective == "cov":
        cands = [c for c in cands if c[2] == 0]
    C = len(cands)
    if C == 0:
        return [], True, 0
    sizes = np.array([len(c[0]) for c in cands], float)
    Pm = to_matrix([c[1] for c in cands], nP)            # nP x C
    Nm = to_matrix([c[2] for c in cands], nN)            # nN x C
    if objective == "f":
        w_s, w_fn, w_fp = sizes, LAM, LAM
    else:  # lexicographic: one error outweighs any size (size <= K*L < 1/eps)
        eps = 1.0 / (sizes.sum() + 1) if objective == "cov" else 1.0 / (K * 4 + 1)
        w_s, w_fn, w_fp = eps * sizes, 1.0, 1.0
    use_z = objective != "cov"
    nz = nN if use_z else 0
    # minimize  w_s.s + w_fn * sum(1 - y) + w_fp * sum(z)  ->  constant dropped
    c = np.concatenate([w_s, -w_fn * np.ones(nP), w_fp * np.ones(nz)])
    cons = []
    # y_T - sum_{X covers T} s_X <= 0
    A1 = hstack([-Pm, identity(nP), csr_matrix((nP, nz))])
    cons.append(LinearConstraint(A1, -np.inf, 0))
    # sum s <= K
    A2 = hstack([csr_matrix(np.ones((1, C))), csr_matrix((1, nP)), csr_matrix((1, nz))])
    cons.append(LinearConstraint(A2, -np.inf, K))
    if use_z:  # sum_{X covers T} s_X - K z_T <= 0
        A3 = hstack([Nm, csr_matrix((nN, nP)), -K * identity(nN)])
        cons.append(LinearConstraint(A3, -np.inf, 0))
    integrality = np.concatenate([np.ones(C), np.zeros(nP), np.ones(nz)])
    bounds = Bounds(np.zeros(C + nP + nz), np.ones(C + nP + nz))
    res = milp(c, constraints=cons, integrality=integrality, bounds=bounds,
               options=dict(time_limit=time_limit, disp=False))
    if res.x is None:
        return None, False, C
    sel = [cands[j][0] for j in range(C) if res.x[j] > 0.5]
    return sel, res.status == 0, C


def filter_dom(cands):
    """Lossless dominance inside groups with identical negative cover: drop X if some Y has
    cov+(Y) superset of cov+(X) and |Y| <= |X| (Y is then at least as good in any rule)."""
    groups = {}
    for c in cands:
        groups.setdefault(c[2], []).append(c)
    out = []
    for q, g in groups.items():
        g.sort(key=lambda c: (-bin(c[1]).count("1"), len(c[0])))
        kept = []
        for c in g:
            if not any((d[1] | c[1]) == d[1] and len(d[0]) <= len(c[0]) for d in kept):
                kept.append(c)
        out += kept
    return out


def solve_maxsat(cands, nP, nN, objective, time_limit):
    """Same selection model as the GenSel master, over the whole candidate set (Open-WBO)."""
    if objective == "cov":
        cands = [c for c in cands if c[2] == 0]
    C = len(cands)
    if C == 0:
        return [], True, 0
    lam = LAM if objective == "f" else K * 4 + 1   # acc/cov: one error outweighs any size
    path = f"{config.TMP}/ex_{os.getpid()}.wcnf"
    w = WCNFWriter(path)
    s = [w.new() for _ in cands]
    def rows(idx_bits, n):
        lst = [[] for _ in range(n)]
        for j, v in enumerate(idx_bits):
            i = 0
            while v:
                if v & 1:
                    lst[i].append(j)
                v >>= 1; i += 1
        return lst
    for T, js in enumerate(rows([c[1] for c in cands], nP)):
        if js:  # positives covered by no candidate are a constant loss
            w.soft([s[j] for j in js], lam)
    if objective != "cov":
        for T, js in enumerate(rows([c[2] for c in cands], nN)):
            if js:
                z = w.new()
                for j in js:
                    w.hard([-s[j], z])
                w.soft([-z], lam)
    if C > K:
        cnt = [[w.new() for _ in range(K)] for _ in range(C)]
        for i, l in enumerate(s):
            w.hard([-l, cnt[i][0]])
            if i > 0:
                for j in range(K):
                    w.hard([-cnt[i - 1][j], cnt[i][j]])
                for j in range(1, K):
                    w.hard([-l, -cnt[i - 1][j - 1], cnt[i][j]])
                w.hard([-l, -cnt[i - 1][K - 1]])
    for v, c in zip(s, cands):
        w.soft([-v], len(c[0]))
    w.close()
    st, model = solve_wcnf(path, time_limit)
    os.remove(path)
    if model is None:
        return None, False, C
    return [c[0] for v, c in zip(s, cands) if model.get(v, False)], st == "OPTIMUM FOUND", C


def guarded(fn, args, hard_limit):
    """Run fn in a forked process, killed after hard_limit seconds (HiGHS may overrun its limit)."""
    import multiprocessing as mp
    q = mp.get_context("fork").Queue()
    p = mp.get_context("fork").Process(target=lambda: q.put(fn(*args)))
    p.start()
    p.join(hard_limit)
    if p.is_alive():
        p.kill(); p.join()
        return None, False, -1
    try:
        return q.get(timeout=5)
    except Exception:
        return None, False, -1


def objective_f(B, y, R):
    pred = predict(B, R)
    return sum(len(X) for X in R) + LAM * int((pred != y).sum())


def run(name, fold, L, out, cap=300_000, enum_limit=120, milp_limit=60):
    X, y = load_raw(name)
    tr, te = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))[fold]
    bz = Binarizer().fit(X.iloc[tr])
    Btr, Bte = bz.transform(X.iloc[tr]), bz.transform(X.iloc[te])
    ytr, yte = y[tr], y[te]
    nP, nN = int(ytr.sum()), int((1 - ytr).sum())
    gens, complete, t_enum = enumerate_gens(Btr, ytr, L, cap, enum_limit)
    cands = dedup(gens)
    n_dedup = len(cands)
    t1 = time.time()
    cands = filter_dom(cands)
    t_filter = time.time() - t1
    for obj in ["f", "acc", "cov"]:
        t0 = time.time()
        R, opt, C = guarded(solve_milp, (cands, nP, nN, obj, milp_limit), milp_limit + 30)
        rec = dict(dataset=name, fold=fold, L=L, objective=obj, n_gens=len(gens), n_dedup=n_dedup,
                   n_cands=len(cands), t_filter=t_filter,
                   n_cols=C, enum_complete=complete, t_enum=t_enum, t_milp=time.time() - t0, optimal=opt)
        if R is not None:
            rec.update(test_acc=float((predict(Bte, R) == yte).mean()),
                       train_acc=float((predict(Btr, R) == ytr).mean()),
                       size=sum(len(r) for r in R), n_rules=len(R), train_obj=objective_f(Btr, ytr, R))
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
