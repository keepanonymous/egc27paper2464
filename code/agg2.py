"""Aggregate results of the second paper and compare with the first experiment (same protocol)."""
import os
import config
from config import RESULTS, PAPER, EGC
import json
from collections import defaultdict
import numpy as np

DS = ["iris", "heart", "ionosphere", "wdbc", "australian", "tictactoe", "spambase", "magic", "adult"]
DSN = {"iris": "Iris", "heart": "Heart", "ionosphere": "Ionosphere", "wdbc": "WDBC",
       "australian": "Australian", "tictactoe": "Tic-tac-toe", "spambase": "Spambase",
       "magic": "Magic", "adult": "Adult"}
LAM = 10
rows = []
for l in open(os.path.join(RESULTS, "results.jsonl")):
    r = json.loads(l)
    if r["method"] in ("RIPPER", "MLIC", "DRL", "DRL-lite", "MinSS-s", "LR", "RF"):
        rows.append(r)
for l in open(os.path.join(RESULTS, "results_gen.jsonl")):
    r = json.loads(l)
    r["method"] = {"Greedy": "GenSel-G", "CG": "GenSel", "CG-F": "GenSel-F"}[r["method"]]
    rows.append(r)

import sys

from data import load_raw
from sklearn.model_selection import StratifiedKFold
TRAIN_SIZE = {}
for _d in DS:
    _X, _y = load_raw(_d)
    for _f, (_tr, _te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(_X, _y)):
        TRAIN_SIZE[(_d, _f)] = len(_tr)
by = defaultdict(list)
for r in rows:
    by[(r["dataset"], r["method"])].append(r)
mtrain = {}
for r in rows:
    if "m" in r:
        mtrain[r["dataset"]] = r["m"]


def train_obj(r):
    """f on the training set (size + lam * training errors); None for non-rule models."""
    if r.get("error") is not None or "size" not in r or r.get("size") is None:
        return None
    if "train_obj" in r:
        return r["train_obj"]
    m_tr = TRAIN_SIZE[(r["dataset"], r["fold"])]
    err = round((1 - r["train_acc"]) * m_tr)
    return r["size"] + LAM * err


def mean(d, m, key):
    v = [r[key] for r in by.get((d, m), []) if r.get("error") is None and r.get(key) is not None]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


def mean_obj(d, m):
    v = [train_obj(r) for r in by.get((d, m), [])]
    v = [x for x in v if x is not None]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


def proved_opt(d, fold):
    """Proven optimum of Eq. (1) on this fold if DRL-lite (full feature space, at most k) proved it."""
    for r in by.get((d, "DRL-lite"), []):
        if r["fold"] == fold and r.get("status") == "OPTIMUM FOUND":
            return train_obj(r)
    return None


if __name__ == "__main__":
    M = ["RIPPER", "MLIC", "DRL", "DRL-lite", "MinSS-s", "GenSel-G", "GenSel", "GenSel-F"]
    print("TEST ACC")
    for d in DS:
        print(f"{d:11s}", " ".join(f"{m}={100*mean(d,m,'test_acc')[0]:.1f}" if mean(d, m, 'test_acc')[0] is not None else f"{m}=-" for m in M))
    print("TRAIN OBJ")
    for d in DS:
        print(f"{d:11s}", " ".join(f"{m}={mean_obj(d,m)[0]:.0f}/{mean_obj(d,m)[1]}" if mean_obj(d, m)[0] is not None else f"{m}=-" for m in M))
    print("GenSel extra")
    for d in DS:
        for m in ["GenSel-G", "GenSel", "GenSel-F"]:
            rs = by.get((d, m), [])
            if not rs:
                continue
            print(f"{d:11s} {m:9s} time={np.mean([r['time'] for r in rs]):5.1f} size={np.mean([r['size'] for r in rs]):5.1f} "
                  f"pool={np.mean([r.get('pool_final', r.get('pool', 0)) for r in rs]):5.1f} "
                  f"cert={sum(bool(r['certified']) for r in rs)} cgconv={sum(bool(r.get('cg_converged')) for r in rs)} "
                  f"optimum_reached={sum(1 for r in rs if proved_opt(d, r['fold']) is not None and r['train_obj'] <= proved_opt(d, r['fold']))}"
                  f"/{sum(1 for r in rs if proved_opt(d, r['fold']) is not None)}")
