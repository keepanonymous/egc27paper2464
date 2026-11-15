import os
import config
from config import RESULTS, PAPER, EGC
import glob, json, sys
import numpy as np

from agg2 import DS, DSN, mean, mean_obj
R = [json.loads(l) for p in glob.glob(os.path.join(RESULTS, "results_exact.jsonl*")) for l in open(p)]
cfg = sorted({(r["dataset"], r["L"]) for r in R}, key=lambda x: (DS.index(x[0]), x[1]))
print(f"{'dataset':11s} L  folds  #gen(dedup->filt)  | obj   f(opt)  acc%  | acc   f  acc% | cov  f  acc% || GenSel f acc% | DRL acc | MLIC acc")
for d, L in cfg:
    rr = [r for r in R if r["dataset"] == d and r["L"] == L]
    nf = len({r["fold"] for r in rr})
    row = f"{DSN.get(d,d):11s} {L}  {nf}  {np.mean([r['n_gens'] for r in rr]):8.0f}({np.mean([r['n_dedup'] for r in rr]):.0f}->{np.mean([r['n_cands'] for r in rr]):.0f}) |"
    for o in ("f", "acc", "cov"):
        x = [r for r in rr if r["objective"] == o]
        ok = [r for r in x if "train_obj" in r]
        nopt = sum(r["optimal"] for r in x)
        if len(ok) < len(x) or not ok:
            row += f" {o}: {len(ok)}/{len(x)} sol |"
            if not ok: continue
        row += f" {o} {np.mean([r['train_obj'] for r in ok]):7.0f} ({nopt}) {100*np.mean([r['test_acc'] for r in ok]):5.1f} |"
    g = mean_obj(d, "GenSel")[0]; ga = mean(d, "GenSel", "test_acc")[0]
    da = mean(d, "DRL", "test_acc")[0]; ma = mean(d, "MLIC", "test_acc")[0]
    row += f"| {g:7.0f} {100*ga:5.1f} | {'--' if da is None else f'{100*da:5.1f}'} | {100*ma:5.1f}"
    print(row)
