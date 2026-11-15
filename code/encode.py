"""Weighted MaxSAT encodings for k-GDI classification rules (DNF of k itemsets).

Items: each binary feature f yields two items (f, 1) "f is true" and (f, 0) "f is false".
A transaction (example) x contains item (f, v) iff x[f] == v.

Three encodings:
  * drl   : faithful re-implementation of Hidouri et al. (ICTAI'24), constraints
            (2) exclusiveness, (3) cover [with the non-emptiness made explicit],
            (4) non-redundancy, (6) minimal generator in D-, (9) global cover.
            One cover variable per (transaction, pattern).
  * lite  : our simplified encoding on all features: exclusiveness, cover, global cover,
            pattern activation variables; no (4)/(6) (enforced by post-processing);
            cover variables shared by transactions with identical projection.
  * minss : same as lite but pattern j may only use the items of its MinSS S_j.
"""
import os
import config
from config import RESULTS, PAPER, EGC
import os
import subprocess
import tempfile
import time

import numpy as np

OPENWBO = config.OPENWBO
if not os.path.exists(OPENWBO):
    raise FileNotFoundError(f"Open-WBO not found at {OPENWBO}: run scripts/build_openwbo.sh "
                            "or set the OPENWBO environment variable")
UNIFORM_WEIGHTS = os.environ.get("UNIFORM_WEIGHTS", "0") == "1"


class WCNFWriter:
    """Streams clauses to disk to keep memory flat on large encodings."""

    def __init__(self, path, max_clauses=None):
        self.path = path
        self.body = open(path + ".body", "w")
        self.nv = 0
        self.n_hard = 0
        self.n_soft = 0
        self.soft_total = 0
        self.max_clauses = max_clauses

    def new(self):
        self.nv += 1
        return self.nv

    def hard(self, lits):
        self.body.write("H " + " ".join(map(str, lits)) + " 0\n")
        self.n_hard += 1
        if self.max_clauses and self.n_hard + self.n_soft > self.max_clauses:
            raise MemoryError("encoding too large")

    def soft(self, lits, w):
        self.body.write(f"{w} " + " ".join(map(str, lits)) + " 0\n")
        self.n_soft += 1
        self.soft_total += w

    def atmost1_if(self, lits, cond):
        """cond -> at most one of lits (sequential counter, Sinz 2005)."""
        n = len(lits)
        if n <= 1:
            return
        if n <= 4:
            for i in range(n):
                for j in range(i + 1, n):
                    self.hard([-cond, -lits[i], -lits[j]])
            return
        s = [self.new() for _ in range(n - 1)]
        self.hard([-cond, -lits[0], s[0]])
        for i in range(1, n - 1):
            self.hard([-cond, -lits[i], s[i]])
            self.hard([-cond, -s[i - 1], s[i]])
            self.hard([-cond, -lits[i], -s[i - 1]])
        self.hard([-cond, -lits[n - 1], -s[n - 2]])

    def close(self):
        self.body.close()
        top = self.soft_total + 1
        with open(self.path, "w") as out:
            out.write(f"p wcnf {self.nv} {self.n_hard + self.n_soft} {top}\n")
            with open(self.path + ".body") as b:
                for line in b:
                    if line[0] == "H":
                        out.write(f"{top}" + line[1:])
                    else:
                        out.write(line)
        os.remove(self.path + ".body")

    @property
    def n_clauses(self):
        return self.n_hard + self.n_soft


def _classes(B, feats):
    """Unique projections of rows of B on feats -> (representatives, class index per row)."""
    if len(feats) == 0:
        return np.zeros((1, 0), dtype=np.uint8), np.zeros(len(B), dtype=int)
    reps, inv = np.unique(B[:, feats], axis=0, return_inverse=True)
    return reps, inv.ravel()


def build(B, y, k, lam, mode, feat_sets=None, path=None, max_clauses=60_000_000):
    """Write the WCNF for `mode`; returns (writer, decode-info)."""
    m, n = B.shape
    w = WCNFWriter(path, max_clauses=max_clauses)
    if mode in ("drl", "lite"):
        feat_sets = [list(range(n))] * k
    p = []  # p[j][(f, v)] -> var
    for j in range(k):
        pj = {}
        for f in feat_sets[j]:
            pj[(f, 1)] = w.new()
            pj[(f, 0)] = w.new()
            w.hard([-pj[(f, 1)], -pj[(f, 0)]])  # (C1) exclusiveness
        p.append(pj)

    e = None
    if mode == "drl":
        _build_drl(w, B, y, k, lam, p)
    else:
        e = _build_grouped(w, B, y, k, lam, p, feat_sets)

    # sparsity: every selected item costs 1
    for j in range(k):
        for var in p[j].values():
            w.soft([-var], 1)
    w.close()
    return w, {"p": p, "e": e, "feat_sets": feat_sets, "mode": mode}


def _build_drl(w, B, y, k, lam, p):
    m = len(B)
    for j in range(k):  # (C2) patterns are non-empty
        w.hard(list(p[j].values()))
    q = [[w.new() for j in range(k)] for i in range(m)]
    for i in range(m):
        x = B[i]
        for j in range(k):
            # (C3) q_ij <-> no selected item is absent from T_i (pattern non-empty by C2)
            absent = [p[j][(f, 1 - int(x[f]))] for f, v in p[j] if v == 1]
            for a in absent:
                w.hard([-q[i][j], -a])
            w.hard([q[i][j]] + absent)
    # (C9) r_i <-> OR_j q_ij ; soft errors
    for i in range(m):
        r = w.new()
        w.hard([-r] + q[i])
        for j in range(k):
            w.hard([r, -q[i][j]])
        w.soft([r] if y[i] == 1 else [-r], lam)
    # (C4) non-redundancy: each pattern is the only one covering some positive
    pos = np.where(y == 1)[0]
    for j in range(k):
        us = []
        for i in pos:
            u = w.new()
            w.hard([-u, q[i][j]])
            for l in range(k):
                if l != j:
                    w.hard([-u, -q[i][l]])
            us.append(u)
        w.hard(us)
    # (C6) minimal generator in D-: for each selected item a of a pattern of size > 1 there
    # is a negative that is rejected by a only. z_ij -> exactly one selected item absent
    # from T_i (at-least-one is implied when used), w_j -> pattern has exactly one item.
    neg = np.where(y == 0)[0]
    for j in range(k):
        allp = list(p[j].values())
        wj = w.new()
        w.atmost1_if(allp, wj)
        z = {}
        for i in neg:
            x = B[i]
            zi = w.new()
            absent = [p[j][(f, 1 - int(x[f]))] for f, v in p[j] if v == 1]
            w.atmost1_if(absent, zi)
            z[i] = zi
        for (f, v), a in p[j].items():
            supp = [z[i] for i in neg if B[i, f] != v]
            w.hard([-a, wj] + supp)


def _build_grouped(w, B, y, k, lam, p, feat_sets):
    m = len(B)
    e = [w.new() for _ in range(k)]  # pattern activation
    cls_idx = []
    q = []
    for j in range(k):
        feats = feat_sets[j]
        reps, inv = _classes(B, feats)
        qj = []
        for c in range(len(reps)):
            qc = w.new()
            absent = [p[j][(f, 1 - int(reps[c][t]))] for t, f in enumerate(feats)]
            w.hard([-qc, e[j]])
            for a in absent:
                w.hard([-qc, -a])
            w.hard([qc, -e[j]] + absent)
            qj.append(qc)
        q.append(qj)
        cls_idx.append(inv)
    # global groups: same label and same class in every pattern
    key = np.stack(cls_idx + [y], axis=1)
    groups, ginv, gcount = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    for g, row in enumerate(groups):
        r = w.new()
        qs = [q[j][row[j]] for j in range(k)]
        w.hard([-r] + qs)
        for qq in qs:
            w.hard([r, -qq])
        if UNIFORM_WEIGHTS:  # one soft clause of weight lam per transaction
            for _ in range(int(gcount[g])):
                w.soft([r] if row[k] == 1 else [-r], lam)
        else:
            w.soft([r] if row[k] == 1 else [-r], lam * int(gcount[g]))
    return e


def _run_openwbo(path, timeout, alg):
    try:
        res = subprocess.run(["timeout", "-s", "TERM", str(timeout), OPENWBO, f"-algorithm={alg}", path],
                             capture_output=True, text=True, timeout=timeout + 60)
        return res.returncode, res.stdout
    except subprocess.TimeoutExpired as ex:
        return None, (ex.stdout.decode() if ex.stdout else "")


def _parse(out):
    status, model = None, None
    for line in out.splitlines():
        if line.startswith("s "):
            status = line[2:].strip()
        elif line.startswith("v "):
            vals = line[2:].split()
            if len(vals) == 1 and set(vals[0]) <= {"0", "1"}:
                model = {i + 1: vals[0][i] == "1" for i in range(len(vals[0]))}
            else:
                model = {abs(int(v)): int(v) > 0 for v in vals if v != "0"}
    return status, model


def solve(path, timeout):
    """Two-phase anytime portfolio with Open-WBO, identical for every encoding:
    (1) linear SAT-UNSAT search (-algorithm=1) for half of the budget; it is the best anytime
        strategy on small and medium instances (it crashes on some large instances, in which
        case we move on immediately);
    (2) if optimality was not proved, the core-guided OLL algorithm with stratification
        (-algorithm=4) for the remaining time, which behaves much better on large instances.
    On timeout (SIGTERM) Open-WBO prints its best model. All returned models are kept in
    solve.candidates; the caller keeps the one with the best objective."""
    t0 = time.time()
    cands = []
    rc, out = _run_openwbo(path, max(1, int(timeout / 2)), 1)
    st, model = _parse(out)
    if model is not None:
        cands.append((st, model, "linear"))
    if st != "OPTIMUM FOUND":
        remaining = int(timeout - (time.time() - t0))
        if remaining >= 1:
            rc, out = _run_openwbo(path, remaining, 4)
            st2, model2 = _parse(out)
            if model2 is not None:
                cands.append((st2, model2, "oll"))
    dt = time.time() - t0
    solve.candidates = cands
    opt = [c for c in cands if c[0] == "OPTIMUM FOUND"]
    best = opt[0] if opt else (cands[-1] if cands else (None, None, None))
    return best[0], best[1], None, dt


solve.candidates = []


def decode(info, model):
    """Return the list of active patterns (each a list of items (f, v))."""
    pats = []
    for j, pj in enumerate(info["p"]):
        if info["e"] is not None and not model.get(info["e"][j], False):
            continue  # inactive pattern
        pats.append([(f, v) for (f, v), var in pj.items() if model.get(var, False)])
    return pats
