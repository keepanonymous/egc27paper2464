"""Run the generation-selection learner on one (dataset, fold), same protocol as exp/run.py."""
import json, os, sys, time
import numpy as np
from sklearn.model_selection import StratifiedKFold

from data import load_raw, Binarizer
from gencg import learn, learn_cg, objective, cover
from rules import postprocess, predict

K, LAM = int(os.environ.get("K", "3")), 10


def run(name, fold, timeout, methods, out, seed=0):
    X, y = load_raw(name)
    tr, te = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))[fold]
    bz = Binarizer().fit(X.iloc[tr])
    Btr, Bte = bz.transform(X.iloc[tr]), bz.transform(X.iloc[te])
    ytr, yte = y[tr], y[te]
    for meth in methods:
        fr = meth.endswith("-F")
        if meth.startswith("CG"):
            res = learn_cg(Btr, ytr, k=K, lam=LAM, time_limit=timeout, freeness=fr)
        else:  # greedy generation (oracle with lam on uncovered positives)
            res = learn(Btr, ytr, k=K, lam=LAM, time_limit=timeout, freeness=fr)
        R = res["rule"]
        f_raw = objective(Btr, ytr, R, LAM)
        R = postprocess(Btr, ytr, R, np.random.default_rng(fold))
        rec = dict(dataset=name, fold=fold, method=meth, k=K, m=len(X), n_bin=Btr.shape[1],
                   train_acc=float((predict(Btr, R) == ytr).mean()),
                   test_acc=float((predict(Bte, R) == yte).mean()),
                   size=sum(len(r) for r in R), n_rules=len(R), train_obj=objective(Btr, ytr, R, LAM),
                   train_obj_raw=f_raw, time=res["time"],
                   **{k2: v for k2, v in res.items() if k2 not in ("rule", "trace", "time")},
                   rules=[[bz.names()[f] + ("" if v else " = 0") for f, v in r] for r in R])
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec[k] for k in rec if k != "rules"}), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[5].split(","), sys.argv[4])
