"""Validation runs for selecting k: for one (dataset, outer fold, k), split the outer training fold
into an inner training part (80%) and a validation part (20%), stratified; fit the binarizer on the
inner training part; learn with GenSel, DRL and MLIC; record the validation accuracy."""
import os
import config
from config import RESULTS, PAPER, EGC
import json, os, sys, time
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from data import load_raw, Binarizer
from encode import WCNFWriter, build, solve, decode
from rules import predict, postprocess, rule_size, build_mlic, decode_mlic, predict_cnf, objective
from gencg import learn_cg

LAM = 10
TMP = config.TMP
os.makedirs(TMP, exist_ok=True)


def baseline(meth, Btr, ytr, Bva, yva, k, timeout, tag):
    """DRL or MLIC with the same solver portfolio as in the main experiments."""
    t0 = time.time()
    path = f"{TMP}/val_{tag}_{meth}.wcnf"
    try:
        if meth == "MLIC":
            w = WCNFWriter(path, max_clauses=20_000_000)
            b = build_mlic(w, Btr, ytr, k, LAM)
            w.close()
            info = None
        else:
            w, info = build(Btr, ytr, k, LAM, "drl", path=path)
    except MemoryError:
        for f in os.listdir(TMP):
            if f.startswith(f"val_{tag}_{meth}"):
                os.remove(os.path.join(TMP, f))
        return dict(error="encoding too large", time=time.time() - t0)
    status, model, _, _ = solve(path, max(1, int(timeout - (time.time() - t0))))
    os.remove(path)
    if model is None:
        return dict(error="no model", time=time.time() - t0)
    if status != "OPTIMUM FOUND" and len(solve.candidates) > 1:
        def _obj(mdl):
            if meth == "MLIC":
                cl = decode_mlic(b, mdl)
                return rule_size(cl) + LAM * int((predict_cnf(Btr, cl) != ytr).sum())
            return objective(Btr, ytr, decode(info, mdl), LAM)
        model = min(solve.candidates, key=lambda c: _obj(c[1]))[1]
    if meth == "MLIC":
        cl = decode_mlic(b, model)
        pva, size = predict_cnf(Bva, cl), rule_size(cl)
    else:
        pats = decode(info, model)
        pva, size = predict(Bva, pats), rule_size(pats)
    return dict(val_acc=float((pva == yva).mean()), size=size, status=status, time=time.time() - t0)


def gensel(Btr, ytr, Bva, yva, k, timeout, fold):
    t0 = time.time()
    res = learn_cg(Btr, ytr, k=k, lam=LAM, time_limit=timeout)
    R = postprocess(Btr, ytr, res["rule"], np.random.default_rng(fold))
    return dict(val_acc=float((predict(Bva, R) == yva).mean()), size=sum(len(r) for r in R),
                certified=res["certified"], time=time.time() - t0)


def run(name, fold, k, timeout, methods, out):
    X, y = load_raw(name)
    tr, _ = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))[fold]
    itr, iva = train_test_split(np.arange(len(tr)), test_size=0.2, stratify=y[tr], random_state=fold)
    Xtr = X.iloc[tr]
    bz = Binarizer().fit(Xtr.iloc[itr])
    Btr, Bva = bz.transform(Xtr.iloc[itr]), bz.transform(Xtr.iloc[iva])
    ytr, yva = y[tr][itr], y[tr][iva]
    for meth in methods:
        if meth == "GenSel":
            rec = gensel(Btr, ytr, Bva, yva, k, timeout, fold)
        else:
            rec = baseline(meth, Btr, ytr, Bva, yva, k, timeout, f"{name}_{fold}_{k}")
        rec = dict(dataset=name, fold=fold, k=k, method=meth, timeout=timeout, **rec)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5].split(","), sys.argv[6])
