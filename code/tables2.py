"""LaTeX tables for the second paper."""
import os
import config
from config import RESULTS, PAPER, EGC
import json, sys
import numpy as np

from agg2 import DS, DSN, by, mean, mean_obj, proved_opt

out = []
# ---------------------------------------------------------------- enumeration sizes
enum = {}
try:
    for l in open(os.path.join(RESULTS, "enum_counts.jsonl")):
        r = json.loads(l)
        enum[r["dataset"]] = r
except FileNotFoundError:
    pass
out.append(r"""\begin{table}[t]
\centering
\caption{Number of minimal generators of $\Dm$ covering at least one positive transaction, by length $L$ (first training fold; levelwise enumeration stopped at 2 million generators per level, except for Heart, enumerated completely; ``$>$'': lower bound; ``--'': not reached, or out of memory for Adult at $L=4$), versus the mean final pool size of \ours.}
\label{tab:enum}
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{lr|rrrr|r}
\toprule
Dataset & $n_b$ & $L{=}1$ & $L{=}2$ & $L{=}3$ & $L{=}4$ & \ours{} pool\\
\midrule""")
for d in DS:
    if d not in enum:
        continue
    r = enum[d]
    c = r["counts"] + [None] * (4 - len(r["counts"]))
    cells = []
    for i, v in enumerate(c[:4]):
        if v is None:
            cells.append("--")
        else:
            last = (i == len(r["counts"]) - 1) and not r["complete"]
            cells.append((r"$>$" if last else "") + f"{v:,}".replace(",", "\\,"))
    rs = by.get((d, "GenSel"), [])
    pool = np.mean([x.get("pool_final", x.get("pool", 0)) for x in rs]) if rs else None
    out.append(f"{DSN[d]} & {r['n_bin']} & " + " & ".join(cells) + f" & {pool:.1f}" + r"\\")
out.append(r"""\bottomrule
\end{tabular}
\end{table}""")

# ---------------------------------------------------------------- test accuracy
M = ["LR", "RF", "RIPPER", "MLIC", "DRL", "GenSel-G", "GenSel", "GenSel-F"]
HEAD = ["LR", "RF", "RIPPER", "MLIC", "DRL", r"\ours-G", r"\ours", r"\ours-F"]
out.append(r"""\begin{table*}[t]
\centering
\caption{Mean test accuracy (\%) over 5-fold cross-validation ($k=3$, $\lambda=10$, 60\,s per fold). ``--'': no rule returned or not run (\ours-F on the three largest datasets); $^{\dagger}$: averaged over the folds where a rule was returned. Best rule-based result in bold.}
\label{tab:acc}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l|rr|rrr|rrr}
\toprule
Dataset & """ + " & ".join(HEAD) + r"""\\
\midrule""")
for d in DS:
    vals, cells = {}, []
    for m in M:
        a, n = mean(d, m, "test_acc")
        if a is None:
            cells.append("--"); continue
        s = f"{100*a:.1f}" + (r"$^{\dagger}$" if n < 5 else "")
        cells.append(s)
        if m not in ("LR", "RF"):
            vals[m] = round(100 * a, 1)
    best = max(vals.values())
    for i, m in enumerate(M):
        if m in vals and vals[m] == best:
            cells[i] = r"\textbf{" + cells[i] + "}"
    out.append(f"{DSN[d]} & " + " & ".join(cells) + r"\\")
avg = []
for m in M:
    v = [mean(d, m, "test_acc")[0] for d in DS]
    mark = "" if all(x is not None for x in v) else ("$^{\\ddagger}$" if m == "GenSel-F" else "$^{*}$")
    avg.append(f"{100*np.mean([x for x in v if x is not None]):.1f}" + mark)
out.append(r"\midrule")
out.append("Mean & " + " & ".join(avg) + r"\\")
out.append(r"""\bottomrule
\end{tabular}
\\[2pt]{\footnotesize $^{*}$ mean over the eight datasets where DRL returned a rule. $^{\ddagger}$ mean over the six datasets where \ours-F was run (on these, \ours: 86.0, \ours-G: 86.2).}
\end{table*}""")

# ---------------------------------------------------------------- objective, size, time, pool, certificates
OM = ["MLIC", "DRL", "GenSel-G", "GenSel", "GenSel-F"]
OH = ["MLIC", "DRL", "G", r"\ours", "F"]
out.append(r"""\begin{table*}[t]
\centering
\caption{Training objective $f$ (Eq.~\eqref{eq:obj}, mean over folds; lower is better; best in bold), rule size, training time (s), mean final pool size, and number of folds (out of 5) where \ours{} terminated with the certificate of Proposition~\ref{prop:term} (cert.) and where the LP-guided generation converged (LP). G and F stand for \ours-G and \ours-F.}
\label{tab:obj}
\setlength{\tabcolsep}{3pt}
\footnotesize
\begin{tabular}{l|rrrrr|rrrr|rrr|rr}
\toprule
& \multicolumn{5}{c|}{Training objective $f$} & \multicolumn{4}{c|}{Rule size} & \multicolumn{3}{c|}{Time (s)} & \multicolumn{2}{c}{\ours}\\
Dataset & """ + " & ".join(OH) + r""" & RIPPER & DRL & G & \ours & G & \ours & F & pool & cert./LP\\
\midrule""")
for d in DS:
    vals = {m: mean_obj(d, m)[0] for m in OM}
    best = min(v for v in vals.values() if v is not None)
    oc = []
    for m in OM:
        v = vals[m]
        if v is None:
            oc.append("--"); continue
        n = mean_obj(d, m)[1]
        s = f"{v:.0f}" + (r"$^{\dagger}$" if n < 5 else "")
        oc.append(r"\textbf{" + s + "}" if round(v) == round(best) else s)
    sz = []
    for m in ["RIPPER", "DRL", "GenSel-G", "GenSel"]:
        a, n = mean(d, m, "size")
        sz.append("--" if a is None else f"{a:.1f}" + (r"$^{\dagger}$" if n < 5 else ""))
    tm = []
    for m in ["GenSel-G", "GenSel", "GenSel-F"]:
        a, n = mean(d, m, "time")
        tm.append("--" if a is None else f"{a:.1f}")
    rs = by.get((d, "GenSel"), [])
    pool = np.mean([x.get("pool_final", x.get("pool", 0)) for x in rs])
    cert = sum(bool(x["certified"]) for x in rs)
    lp = sum(bool(x.get("cg_converged")) for x in rs)
    out.append(f"{DSN[d]} & " + " & ".join(oc + sz + tm) + f" & {pool:.1f} & {cert}/{lp}" + r"\\")
out.append(r"""\bottomrule
\end{tabular}
\end{table*}""")

open(os.path.join(PAPER, "tables.tex"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
