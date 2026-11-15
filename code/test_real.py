"""SAT enumerator vs brute force on small real datasets (fold 0 of the training data)."""
import sys, time
import numpy as np
from sklearn.model_selection import StratifiedKFold

from satgen import GeneratorEnumerator, brute_force
from data import load_raw, Binarizer

name = sys.argv[1]
check_L = int(sys.argv[2]) if len(sys.argv) > 2 else 3
X, y = load_raw(name)
tr, _ = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))[0]
B = Binarizer().fit(X.iloc[tr]).transform(X.iloc[tr]); y = y[tr]
print(f"{name}: m={len(B)} (+{y.sum()}, -{(1-y).sum()}), n={B.shape[1]} features, {2*B.shape[1]} items")

# 1) correctness vs brute force for bounded length
for cfg in [dict(min_pos=1, max_len=2), dict(min_pos=1, max_len=check_L), dict(min_pos=1, max_len=check_L, max_neg=0)]:
    t = time.time(); g = GeneratorEnumerator(B, y, **cfg); sat, _ = g.enumerate(); ts = time.time() - t
    t = time.time(); bf = brute_force(B, y, **cfg); tb = time.time() - t
    ok = set(sat) == set(bf) and len(sat) == len(set(sat))
    print(f"  {str(cfg):45s} SAT={len(sat):6d} ({ts:5.1f}s)  brute={len(bf):6d} ({tb:6.1f}s)  {'OK' if ok else 'MISMATCH'}")

# 2) growth with the length bound (SAT only)
for L in [1, 2, 3, 4, 5]:
    t = time.time(); g = GeneratorEnumerator(B, y, min_pos=1, max_len=L)
    sat, done = g.enumerate(time_limit=120)
    by = np.bincount([len(X) for X in sat], minlength=L + 1)[1:]
    print(f"  L<={L}: {len(sat):7d} generators {'(complete)' if done else '(TIME LIMIT)'} {time.time()-t:6.1f}s  by length {list(by)}  implied pairs: {g.n_implied}")
