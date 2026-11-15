"""Run one (dataset, fold) for every method and append JSON lines to results file."""
import os
import config
from config import RESULTS, PAPER, EGC
import json
import os
import sys
import time
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


from data import load_raw, Binarizer
from minss import random_minss, consistent_mask
from encode import WCNFWriter, build, solve, decode
from rules import predict, postprocess, rule_size, build_mlic, decode_mlic, predict_cnf

warnings.filterwarnings("ignore")
K, LAM = int(os.environ.get("K", "3")), 10
TMP = config.TMP
os.makedirs(TMP, exist_ok=True)


def run(name, fold, nfolds, timeout, methods, out, seed=0, sample=100):
    X, y = load_raw(name)
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
    tr, te = list(skf.split(X, y))[fold]
    bz = Binarizer().fit(X.iloc[tr])
    Btr, Bte = bz.transform(X.iloc[tr]), bz.transform(X.iloc[te])
    ytr, yte = y[tr], y[te]
    base = dict(dataset=name, fold=fold, m=len(X), n_orig=X.shape[1], n_bin=Btr.shape[1],
                timeout=timeout)

    def emit(rec):
        rec = {**base, **rec}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    for meth in methods:
        rng = np.random.default_rng(seed * 1000 + fold)
        t0 = time.time()
        rec = dict(method=meth, k=K)
        try:
            if meth in ("LR", "RF"):
                clf = LogisticRegression(max_iter=2000) if meth == "LR" else \
                    RandomForestClassifier(n_estimators=100, random_state=seed)
                clf.fit(Btr, ytr)
                rec.update(train_acc=float((clf.predict(Btr) == ytr).mean()),
                           test_acc=float((clf.predict(Bte) == yte).mean()), time=time.time() - t0)
            elif meth == "RIPPER":
                import pandas as pd
                import wittgenstein as lw
                clf = lw.RIPPER(random_state=seed)
                dtr = pd.DataFrame(Btr, columns=[f"f{i}" for i in range(Btr.shape[1])])
                dte = pd.DataFrame(Bte, columns=dtr.columns)
                dtr["cls"] = ytr
                clf.fit(dtr, class_feat="cls", pos_class=1)
                ptr = np.array(clf.predict(dtr.drop(columns="cls"))).astype(int)
                pte = np.array(clf.predict(dte)).astype(int)
                size = sum(len(r.conds) for r in clf.ruleset_.rules)
                rec.update(train_acc=float((ptr == ytr).mean()), test_acc=float((pte == yte).mean()),
                           time=time.time() - t0, size=size, n_rules=len(clf.ruleset_.rules))
            elif meth == "MinSS-R":
                # restarts: repeatedly draw k sampled MinSS, solve, post-process, and keep the
                # rule with the best training objective (Eq. 1) within the time budget
                from rules import objective
                path = f"{TMP}/{name}_{fold}_{meth}_k{K}.wcnf"
                best, n_iter, n_opt, sizes = None, 0, 0, []
                while time.time() - t0 < timeout - 2:
                    feat_sets = []
                    for _ in range(K):
                        idx = rng.choice(len(Btr), min(sample, len(Btr)), replace=False)
                        feat_sets.append(random_minss(Btr[idx], ytr[idx], rng))
                    sizes.append(np.mean([len(s) for s in feat_sets]))
                    w, info = build(Btr, ytr, K, LAM, "minss", feat_sets=feat_sets, path=path)
                    remaining = timeout - (time.time() - t0)
                    status, model, cost, dt = solve(path, max(1, int(min(remaining, timeout / 3))))
                    os.remove(path)
                    n_iter += 1
                    if model is None:
                        continue
                    n_opt += status == "OPTIMUM FOUND"
                    for _, mdl, _ in solve.candidates:
                        pats = postprocess(Btr, ytr, decode(info, mdl), rng)
                        obj = objective(Btr, ytr, pats, LAM)
                        if best is None or obj < best[0]:
                            best = (obj, pats, w.n_clauses)
                if best is None:
                    rec.update(error="no model", time=time.time() - t0)
                else:
                    obj, pats, ncl = best
                    ptr, pte = predict(Btr, pats), predict(Bte, pats)
                    rec.update(train_acc=float((ptr == ytr).mean()), test_acc=float((pte == yte).mean()),
                               size=rule_size(pats), n_rules=len(pats), time=time.time() - t0,
                               n_iter=n_iter, n_opt_iter=n_opt, minss_sizes=[float(np.mean(sizes))],
                               n_clauses=ncl, train_obj=obj,
                               rules=[[bz.names()[f] + ("" if v else " = 0") for f, v in p] for p in pats])
            else:
                path = f"{TMP}/{name}_{fold}_{meth}_k{K}.wcnf"
                if meth == "MLIC":
                    w = WCNFWriter(path, max_clauses=20_000_000)
                    b = build_mlic(w, Btr, ytr, K, LAM)
                    w.close()
                    info = None
                else:
                    feat_sets = None
                    if meth.startswith("MinSS"):
                        if meth == "MinSS":  # MinSS of the whole training set
                            cons = consistent_mask(Btr, ytr)
                            feat_sets = [random_minss(Btr, ytr, rng, cons) for _ in range(K)]
                        else:  # MinSS-s: MinSS of a random sample of `sample` training examples
                            feat_sets = []
                            for _ in range(K):
                                idx = rng.choice(len(Btr), min(sample, len(Btr)), replace=False)
                                feat_sets.append(random_minss(Btr[idx], ytr[idx], rng))
                        rec["minss_sizes"] = [len(s) for s in feat_sets]
                    mode = {"DRL": "drl", "DRL-lite": "lite"}.get(meth, "minss")
                    w, info = build(Btr, ytr, K, LAM, mode, feat_sets=feat_sets, path=path)
                t_build = time.time() - t0
                rec.update(n_vars=w.nv, n_clauses=w.n_clauses, t_build=t_build)
                status, model, cost, dt = solve(path, max(1, int(timeout - t_build)))
                os.remove(path)
                rec.update(status=status, t_solve=dt, time=time.time() - t0,
                           phases=[c[2] for c in solve.candidates])
                if model is not None and status != "OPTIMUM FOUND" and len(solve.candidates) > 1:
                    # keep the candidate model with the best training objective (Eq. 1)
                    from rules import objective
                    def _obj(mdl):
                        if meth == "MLIC":
                            cl = decode_mlic(b, mdl)
                            return rule_size(cl) + LAM * int((predict_cnf(Btr, cl) != ytr).sum())
                        return objective(Btr, ytr, decode(info, mdl), LAM)
                    scored = [(_obj(c[1]), i) for i, c in enumerate(solve.candidates)]
                    i_best = min(scored)[1]
                    model = solve.candidates[i_best][1]
                    rec["best_phase"] = solve.candidates[i_best][2]
                if model is None:
                    rec.update(error="no model")
                else:
                    if meth == "MLIC":
                        cl = decode_mlic(b, model)
                        ptr, pte = predict_cnf(Btr, cl), predict_cnf(Bte, cl)
                        size = rule_size(cl)
                        rec["n_rules"] = sum(1 for c in cl if c)
                    else:
                        pats = decode(info, model)
                        rec["size_raw"] = rule_size(pats)
                        if meth != "DRL":
                            pats = postprocess(Btr, ytr, pats, rng)
                        ptr, pte = predict(Btr, pats), predict(Bte, pats)
                        size = rule_size(pats)
                        rec["n_rules"] = len(pats)
                        rec["rules"] = [[bz.names()[f] + ("" if v else " = 0") for f, v in p] for p in pats]
                    rec.update(train_acc=float((ptr == ytr).mean()), test_acc=float((pte == yte).mean()),
                               size=size, time=time.time() - t0, optimal=status == "OPTIMUM FOUND")
        except MemoryError as ex:
            rec.update(error="encoding too large", time=time.time() - t0)
            for f in os.listdir(TMP):
                if f.startswith(f"{name}_{fold}_{meth}"):
                    os.remove(os.path.join(TMP, f))
        emit(rec)


if __name__ == "__main__":
    name, fold, nfolds, timeout, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    methods = sys.argv[6].split(",")
    run(name, fold, nfolds, timeout, methods, out)
