"""Check the SAT enumerator against brute force on the running example of DRL."""
import sys, time
import numpy as np

from satgen import GeneratorEnumerator, brute_force

F = "abcde"
D = [([1,1,1,1,0],1),([1,1,1,0,1],1),([1,1,0,0,1],1),([1,0,0,0,1],1),
     ([1,0,0,1,1],0),([1,1,0,0,0],0),([0,1,0,1,0],0),([0,1,0,1,1],0)]
B = np.array([x for x, _ in D], dtype=np.uint8); y = np.array([l for _, l in D])
name = lambda X: ' ∧ '.join(('' if v else '¬') + F[f] for f, v in X)

for cfg in [dict(min_pos=0), dict(min_pos=1), dict(min_pos=1, max_len=2), dict(min_pos=1, max_neg=0),
            dict(min_pos=2, max_neg=1)]:
    for ip in (False, True):
        t = time.time()
        g = GeneratorEnumerator(B, y, implied_pairs=ip, **cfg)
        sat, done = g.enumerate()
        bf = brute_force(B, y, **cfg)
        ok = set(sat) == set(bf) and len(sat) == len(set(sat))
        print(f"{str(cfg):38s} implied={ip!s:5s} SAT={len(sat):3d} brute={len(bf):3d} {'OK' if ok else 'MISMATCH'} {time.time()-t:.3f}s")
g = GeneratorEnumerator(B, y, min_pos=1, max_neg=0)
print("pure generators (JEP / LAD prime patterns):", [name(X) for X in g.enumerate()[0]])
