"""Count minimal generators of D- (with support >= 1 in D+) by length, levelwise (Apriori-style),
using the anti-monotonicity of freeness and support. Bitsets are Python integers."""
import os
import config
from config import RESULTS, PAPER, EGC
import json, sys, time
import numpy as np
from sklearn.model_selection import StratifiedKFold

from data import load_raw, Binarizer


def bits(col):
    v = 0
    for i in np.where(col)[0]:
        v |= 1 << int(i)
    return v


def count(B, y, Lmax=4, time_limit=300, max_level=2_000_000):
    n = B.shape[1]
    P, N = B[y == 1], B[y == 0]
    items = [(f, v) for f in range(n) for v in (1, 0)]
    pm = {a: bits(P[:, a[0]] == a[1]) for a in items}
    nm = {a: bits(N[:, a[0]] == a[1]) for a in items}
    allneg = (1 << len(N)) - 1
    t0 = time.time()
    # level 1: {a} free iff some negative misses a; support >= 1 in D+
    level = {}
    for a in items:
        if nm[a] != allneg and pm[a] != 0:
            level[(a,)] = (pm[a], nm[a])
    counts, complete = [len(level)], True
    for L in range(2, Lmax + 1):
        new = {}
        keys = sorted(level)
        # group by prefix for the join
        from collections import defaultdict
        pref = defaultdict(list)
        for X in keys:
            pref[X[:-1]].append(X)
        for p, group in pref.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i][-1], group[j][-1]
                    if a[0] == b[0]:
                        continue  # same feature (exclusivity)
                    X = p + (a, b)
                    # all (L-1)-subsets must be frequent generators (anti-monotone)
                    ok = True
                    for k in range(L):
                        if X[:k] + X[k + 1:] not in level:
                            ok = False
                            break
                    if not ok:
                        continue
                    pos = level[group[i]][0] & pm[b]
                    if pos == 0:
                        continue
                    neg = level[group[i]][1] & nm[b]
                    # freeness: removing any item must enlarge the negative cover
                    if all(level[X[:k] + X[k + 1:]][1] != neg for k in range(L)):
                        new[X] = (pos, neg)
                if time.time() - t0 > time_limit or len(new) > max_level:
                    complete = False
                    break
            if not complete:
                break
        counts.append(len(new))
        level = new
        if not complete or not level:
            break
    return counts, complete, time.time() - t0


if __name__ == "__main__":
    out = []
    for name in sys.argv[1].split(","):
        X, y = load_raw(name)
        tr, _ = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))[0]
        B = Binarizer().fit(X.iloc[tr]).transform(X.iloc[tr]); yy = y[tr]
        c, comp, t = count(B, yy, Lmax=int(sys.argv[2]) if len(sys.argv) > 2 else 4,
                           time_limit=int(sys.argv[3]) if len(sys.argv) > 3 else 300)
        rec = dict(dataset=name, n_bin=B.shape[1], counts=c, complete=comp, time=t)
        print(json.dumps(rec), flush=True)
        with open(os.path.join(RESULTS, "enum_counts.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
