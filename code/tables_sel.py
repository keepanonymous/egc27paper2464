"""Validation-based selection of k for GenSel, DRL and MLIC.
For each (dataset, outer fold, method): k* = argmax over K of the validation accuracy (inner 80/20
split of the outer training fold, 30 s runs); ties broken towards the smallest k. The reported test
accuracy is that of the model trained on the whole outer training fold with k* (60 s runs)."""
import os
import config
from config import RESULTS, PAPER, EGC
import json, sys
from collections import defaultdict
import numpy as np

from agg2 import DS, DSN

KS = [1, 2, 3, 5, 8]
M = ["GenSel", "DRL", "MLIC"]

# ---------------- validation accuracies
val = defaultdict(dict)  # (d, fold, m) -> {k: acc or None}
for l in open(os.path.join(RESULTS, "results_val.jsonl")):
    r = json.loads(l)
    val[(r["dataset"], r["fold"], r["method"])][r["k"]] = r.get("val_acc")

# ---------------- full-training test results
full = {}  # (d, fold, m, k) -> record


def add(path, mapping, kdefault=None):
    for l in open(path):
        r = json.loads(l)
        m = mapping.get(r["method"])
        if m is None:
            continue
        k = r.get("k", kdefault)
        full[(r["dataset"], r["fold"], m, k)] = r


add(os.path.join(RESULTS, "results_gen.jsonl"), {"CG": "GenSel"}, 3)
add(os.path.join(RESULTS, "results_gen_k.jsonl"), {"CG": "GenSel"})
add(os.path.join(RESULTS, "results.jsonl"), {"DRL": "DRL", "MLIC": "MLIC"}, 3)
add(os.path.join(RESULTS, "results_k.jsonl"), {"DRL": "DRL", "MLIC": "MLIC"})
try:
    add(os.path.join(RESULTS, "results_k125.jsonl"), {"DRL": "DRL", "MLIC": "MLIC"})
except FileNotFoundError:
    pass


def ok(r):
    return r is not None and r.get("error") is None and r.get("test_acc") is not None


def select(d, fold, m):
    v = val.get((d, fold, m), {})
    cand = [(acc, -k) for k, acc in v.items() if acc is not None]
    if not cand:
        return None
    return -max(cand)[1]


def summary(d, m, mode):
    """mode = 'k3' or 'sel'. Returns mean test acc, #folds with a rule, mean k, mean size, #folds."""
    accs, ks, sizes, n_missing = [], [], [], 0
    for fold in range(5):
        k = 3 if mode == "k3" else select(d, fold, m)
        if k is None:
            n_missing += 1
            continue
        r = full.get((d, fold, m, k))
        if not ok(r):
            n_missing += 1
            continue
        accs.append(r["test_acc"]); ks.append(k); sizes.append(r.get("size", np.nan))
    if not accs:
        return None
    return dict(acc=float(np.mean(accs)), n=len(accs), k=float(np.mean(ks)), size=float(np.nanmean(sizes)))


if __name__ == "__main__":
    missing_full = [(d, f, m, k) for d in DS for f in range(5) for m in M for k in KS if (d, f, m, k) not in full]
    print("missing full-training runs:", len(missing_full), missing_full[:5])
    print("validation records:", sum(len(v) for v in val.values()))
    res = {(d, m, mode): summary(d, m, mode) for d in DS for m in M for mode in ("k3", "sel")}
    for d in DS:
        line = []
        for m in M:
            a, b = res[(d, m, "k3")], res[(d, m, "sel")]
            fa = f"{100*a['acc']:.1f}/{a['n']}" if a else "-"
            fb = f"{100*b['acc']:.1f}/{b['n']} k={b['k']:.1f} sz={b['size']:.1f}" if b else "-"
            line.append(f"{m}: k3 {fa} | sel {fb}")
        print(f"{d:11s} " + " || ".join(line))
    for m in M:
        for mode in ("k3", "sel"):
            v = [res[(d, m, mode)]["acc"] for d in DS if res[(d, m, mode)]]
            print(m, mode, f"mean {100*np.mean(v):.1f} over {len(v)}")

    # LaTeX table
    L = [r"""\begin{table}[t]
\centering
\caption{Mean test accuracy (\%) with $k=3$ and with $k\in\{1,2,3,5,8\}$ selected per fold on validation data (inner 80/20 split of the training fold, ties broken towards the smaller $k$), and mean selected $k$. ``--'': no rule returned; $^{\dagger}$: averaged over the folds where a rule was returned.}
\label{tab:sel}
\setlength{\tabcolsep}{2.6pt}
\footnotesize
\begin{tabular}{l|rrr|rrr|rrr}
\toprule
& \multicolumn{3}{c|}{\ours} & \multicolumn{3}{c|}{DRL} & \multicolumn{3}{c}{MLIC}\\
Dataset & $k{=}3$ & sel. & $\bar k$ & $k{=}3$ & sel. & $\bar k$ & $k{=}3$ & sel. & $\bar k$\\
\midrule"""]

    def cell(s):
        if s is None:
            return "--"
        return f"{100*s['acc']:.1f}" + (r"$^{\dagger}$" if s["n"] < 5 else "")

    for d in DS:
        best = max(round(100 * res[(d, m, "sel")]["acc"], 1) for m in M if res[(d, m, "sel")])
        cells = []
        for m in M:
            a, b = res[(d, m, "k3")], res[(d, m, "sel")]
            cb = cell(b)
            if b and round(100 * b["acc"], 1) == best:
                cb = r"\textbf{" + cb + "}"
            cells += [cell(a), cb, "--" if b is None else f"{b['k']:.1f}"]
        L.append(f"{DSN[d]} & " + " & ".join(cells) + r"\\")
    L.append(r"\midrule")
    means = []
    for m in M:
        for mode in ("k3", "sel"):
            v = [res[(d, m, mode)]["acc"] for d in DS if res[(d, m, mode)]]
            means.append(f"{100*np.mean(v):.1f}" + ("" if len(v) == len(DS) else "$^{*}$"))
        means.append("")
    L.append("Mean & " + " & ".join(means) + r"\\")
    L.append(r"""\bottomrule
\end{tabular}
\\[2pt]{\footnotesize $^{*}$ mean over the datasets where a rule was returned.}
\end{table}""")
    open(os.path.join(PAPER, "table_sel.tex"), "w").write("\n".join(L) + "\n")
