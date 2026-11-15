"""Random Minimal Support Sets (MinSS).

A set S of binary features is a support set of (P, N) iff no positive and no negative
example share the same projection on S, i.e. S is a transversal of the hypergraph whose
edges are p XOR n for (p, n) in P x N. A MinSS is an inclusion-minimal support set.

We draw random MinSS by elimination: start from all features (a support set once
conflicting examples are set aside) and try to drop features in a random order, keeping a
drop whenever the remaining set still separates P from N. Separation is checked with
incremental 64-bit row hashes, so each test costs O(m)."""
import numpy as np

_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def _hash_weights(n, rng):
    return rng.integers(1, 2**63 - 1, size=(n, 2), dtype=np.int64).astype(np.uint64)


def consistent_mask(B, y):
    """Examples whose full feature vector does not also appear in the opposite class."""
    keys = [r.tobytes() for r in B]
    pos = {k for k, l in zip(keys, y) if l == 1}
    neg = {k for k, l in zip(keys, y) if l == 0}
    clash = pos & neg
    return np.array([k not in clash for k in keys])


def _separates(hp, hn):
    # hp, hn: arrays of shape (*, 2) of uint64 hashes; separation iff no common row
    a = hp[:, 0] ^ (hp[:, 1] * np.uint64(1099511628211))
    b = hn[:, 0] ^ (hn[:, 1] * np.uint64(1099511628211))
    return len(np.intersect1d(a, b, assume_unique=False)) == 0


def random_minss(B, y, rng, cons=None):
    """Return a random inclusion-minimal support set (sorted list of feature indices)."""
    if cons is None:
        cons = consistent_mask(B, y)
    Bc, yc = B[cons], y[cons]
    P, N = Bc[yc == 1].astype(np.uint64), Bc[yc == 0].astype(np.uint64)
    n = B.shape[1]
    W = _hash_weights(n, rng)
    with np.errstate(over="ignore"):
        hp = P @ W  # wraps modulo 2^64
        hn = N @ W
        S = set(range(n))
        for f in rng.permutation(n):
            cand_p = hp - np.outer(P[:, f], W[f])
            cand_n = hn - np.outer(N[:, f], W[f])
            if _separates(cand_p, cand_n):
                hp, hn = cand_p, cand_n
                S.discard(int(f))
    return sorted(S)


def is_support_set(B, y, S, cons=None):
    if cons is None:
        cons = consistent_mask(B, y)
    Bc, yc = B[cons][:, S], y[cons]
    pos = {r.tobytes() for r in Bc[yc == 1]}
    neg = {r.tobytes() for r in Bc[yc == 0]}
    return not (pos & neg)
