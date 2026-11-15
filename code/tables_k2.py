"""Sensitivity to k for the second paper: GenSel k in {1,2,3,5,8}; DRL and MLIC k in {3,8}."""
import os
import config
from config import RESULTS, PAPER, EGC
import json, sys
from collections import defaultdict
import numpy as np

from agg2 import DS, DSN, TRAIN_SIZE, LAM

rows = []
for l in open(os.path.join(RESULTS, "results_gen_k.jsonl")):
    r = json.loads(l); r["method"] = "GenSel"; rows.append(r)
for l in open(os.path.join(RESULTS, "results_gen.jsonl")):
    r = json.loads(l)
    if r["method"] == "CG":
        r["method"] = "GenSel"; r["k"] = 3; rows.append(r)
for l in open(os.path.join(RESULTS, "results.jsonl")):
    r = json.loads(l)
    if r["method"] in ("DRL", "MLIC"):
        r["k"] = 3; rows.append(r)
for l in open(os.path.join(RESULTS, "results_k.jsonl")):
    r = json.loads(l)
    if r["method"] in ("DRL", "MLIC") and r["k"] == 8:
        rows.append(r)
by = defaultdict(list)
for r in rows:
    by[(r["dataset"], r["method"], r["k"])].append(r)


def tobj(r):
    if r.get("error") is not None or r.get("size") is None:
        return None
    if "train_obj" in r:
        return r["train_obj"]
    return r["size"] + LAM * round((1 - r["train_acc"]) * TRAIN_SIZE[(r["dataset"], r["fold"])])


def mean(d, m, k, key):
    rs = [r for r in by.get((d, m, k), []) if r.get("error") is None and r.get(key) is not None]
    return (float(np.mean([r[key] for r in rs])), len(rs)) if rs else (None, 0)


def mobj(d, m, k):
    v = [tobj(r) for r in by.get((d, m, k), [])]
    v = [x for x in v if x is not None]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


KS = [1, 2, 3, 5, 8]
if __name__ == "__main__":
    print("GenSel test / obj / size / time / cert by k")
    for d in DS:
        cells = []
        for k in KS:
            a, n = mean(d, "GenSel", k, "test_acc")
            o, _ = mobj(d, "GenSel", k)
            s, _ = mean(d, "GenSel", k, "size")
            t, _ = mean(d, "GenSel", k, "time")
            nr, _ = mean(d, "GenSel", k, "n_rules")
            cert = sum(bool(r.get("certified")) for r in by.get((d, "GenSel", k), []))
            cells.append(f"k{k}: {100*a:5.1f} f={o:7.0f} sz={s:4.1f} nr={nr:3.1f} t={t:4.1f} c={cert}" if a is not None else f"k{k}: -")
        print(f"{d:11s} " + " | ".join(cells))
    print("DRL/MLIC k3 vs k8")
    for d in DS:
        out = []
        for m in ["DRL", "MLIC"]:
            for k in [3, 8]:
                a, n = mean(d, m, k, "test_acc"); o, _ = mobj(d, m, k)
                out.append(f"{m}{k}: " + (f"{100*a:5.1f}/{n} f={o:.0f}" if a is not None else "-"))
        print(f"{d:11s} " + " | ".join(out))

    # LaTeX
    L = [r"""\begin{table*}[t]
\centering
\caption{Sensitivity to the maximum number of itemsets $k$. Left: mean test accuracy (\%) of \ours{} for $k\in\{1,2,3,5,8\}$ (best per dataset in bold) and mean number of itemsets actually used for $k=8$. Right: mean training objective $f$ of \ours, DRL and MLIC for $k=3$ and $k=8$ (best in bold), and test accuracy of DRL and MLIC for $k=8$. ``--'': no rule returned in 60\,s (or encoding not generated); $^{\dagger}$: averaged over the folds where a rule was returned.}
\label{tab:k}
\setlength{\tabcolsep}{3pt}
\footnotesize
\begin{tabular}{l|rrrrr|r|rr|rr|rr|rr}
\toprule
& \multicolumn{5}{c|}{\ours: test accuracy} & \#itemsets & \multicolumn{2}{c|}{\ours: $f$} & \multicolumn{2}{c|}{DRL: $f$} & \multicolumn{2}{c|}{MLIC: $f$} & \multicolumn{2}{c}{acc.\ $k{=}8$}\\
Dataset & $k{=}1$ & $k{=}2$ & $k{=}3$ & $k{=}5$ & $k{=}8$ & $k{=}8$ & $k{=}3$ & $k{=}8$ & $k{=}3$ & $k{=}8$ & $k{=}3$ & $k{=}8$ & DRL & MLIC\\
\midrule"""]

    def fo(d, m, k):
        o, n = mobj(d, m, k)
        return None if o is None else (o, n)

    for d in DS:
        vals = {k: mean(d, "GenSel", k, "test_acc")[0] for k in KS}
        best = max(round(100 * v, 1) for v in vals.values() if v is not None)
        cells = []
        for k in KS:
            v = vals[k]
            if v is None:
                cells.append("--"); continue
            s = f"{100*v:.1f}"
            cells.append(r"\textbf{" + s + "}" if round(100 * v, 1) == best else s)
        nr8 = mean(d, "GenSel", 8, "n_rules")[0]
        cells.append("--" if nr8 is None else f"{nr8:.1f}")
        objs = {(m, k): fo(d, m, k) for m in ["GenSel", "DRL", "MLIC"] for k in [3, 8]}
        bo = min(v[0] for v in objs.values() if v is not None)
        for m in ["GenSel", "DRL", "MLIC"]:
            for k in [3, 8]:
                v = objs[(m, k)]
                if v is None:
                    cells.append("--"); continue
                s = f"{v[0]:.0f}" + (r"$^{\dagger}$" if v[1] < 5 else "")
                cells.append(r"\textbf{" + s + "}" if round(v[0]) == round(bo) else s)
        for m in ["DRL", "MLIC"]:
            a, n = mean(d, m, 8, "test_acc")
            cells.append("--" if a is None else f"{100*a:.1f}" + (r"$^{\dagger}$" if n < 5 else ""))
        L.append(f"{DSN[d]} & " + " & ".join(cells) + r"\\")
    L.append(r"""\bottomrule
\end{tabular}
\end{table*}""")
    open(os.path.join(PAPER, "table_k.tex"), "w").write("\n".join(L) + "\n")
