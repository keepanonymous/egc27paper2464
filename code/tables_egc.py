"""French tables for the EGC version (same data as the English paper)."""
import os
import config
from config import RESULTS, PAPER, EGC
import json, sys
import numpy as np

from agg2 import DS, DSN, by, mean, mean_obj
import tables_k2 as TK
import tables_sel as TS

DSF = dict(DSN, tictactoe="Tic-tac-toe")


def fr(x, d=1):
    return f"{x:.{d}f}".replace(".", "{,}")


def big(x):
    return f"{int(round(x)):,}".replace(",", "\\,")


out = []
# ------------------------------------------------------------ enumeration
enum = {json.loads(l)["dataset"]: json.loads(l) for l in open(os.path.join(RESULTS, "enum_counts.jsonl"))}
out.append(r"""\begin{table}[t]
\centering
\caption{Nombre de générateurs minimaux de $\Dm$ couvrant au moins une transaction positive, par longueur $L$ (premier pli d'apprentissage ; énumération par niveaux arrêtée à 2 millions de générateurs par niveau, sauf pour Heart, énuméré complètement ; « $>$ » : borne inférieure ; « -- » : non atteint, ou mémoire insuffisante pour Adult avec $L=4$), et taille moyenne du réservoir final de \ours.}
\label{tab:enum}
\small
\begin{tabular}{lr|rrrr|r}
\hline
Jeu & $n_b$ & $L{=}1$ & $L{=}2$ & $L{=}3$ & $L{=}4$ & Réservoir\\
\hline""")
for d in DS:
    r = enum[d]
    c = r["counts"] + [None] * (4 - len(r["counts"]))
    cells = []
    for i, v in enumerate(c[:4]):
        if v is None:
            cells.append("--")
        else:
            last = (i == len(r["counts"]) - 1) and not r["complete"]
            cells.append((r"$>$" if last else "") + big(v))
    pool = np.mean([x.get("pool_final", x.get("pool", 0)) for x in by[(d, "GenSel")]])
    out.append(f"{DSF[d]} & {r['n_bin']} & " + " & ".join(cells) + f" & {fr(pool)}" + r"\\")
out.append(r"""\hline
\end{tabular}
\end{table}""")

# ------------------------------------------------------------ accuracy k=3
M = ["LR", "RF", "RIPPER", "MLIC", "DRL", "GenSel-G", "GenSel", "GenSel-F"]
H = ["LR", "RF", "RIPPER", "MLIC", "DRL", r"\ours-G", r"\ours", r"\ours-F"]
out.append(r"""\begin{table}[t]
\centering
\caption{Précision moyenne en test (\%) en validation croisée à 5 plis ($k=3$, $\lambda=10$, 60\,s par pli). « -- » : aucune règle renvoyée, ou méthode non exécutée (\ours-F sur les trois plus grands jeux) ; $^{\dagger}$ : moyenne sur les plis où une règle a été renvoyée. En gras : meilleure méthode à base de règles.}
\label{tab:acc}
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{l|rr|rrr|rrr}
\hline
Jeu & """ + " & ".join(H) + r"""\\
\hline""")
for d in DS:
    vals, cells = {}, []
    for m in M:
        a, n = mean(d, m, "test_acc")
        if a is None:
            cells.append("--"); continue
        cells.append(fr(100 * a) + (r"$^{\dagger}$" if n < 5 else ""))
        if m not in ("LR", "RF"):
            vals[m] = round(100 * a, 1)
    b = max(vals.values())
    cells = [r"\textbf{" + c + "}" if (M[i] in vals and vals[M[i]] == b) else c for i, c in enumerate(cells)]
    out.append(f"{DSF[d]} & " + " & ".join(cells) + r"\\")
avg = []
for m in M:
    v = [mean(d, m, "test_acc")[0] for d in DS]
    vv = [x for x in v if x is not None]
    mark = "" if len(vv) == len(v) else ("$^{\\ddagger}$" if m == "GenSel-F" else "$^{*}$")
    avg.append(fr(100 * np.mean(vv)) + mark)
out.append(r"\hline")
out.append("Moyenne & " + " & ".join(avg) + r"\\")
out.append(r"""\hline
\end{tabular}\\[2pt]
{\footnotesize $^{*}$ moyenne sur les huit jeux où DRL renvoie une règle ; $^{\ddagger}$ moyenne sur les six jeux où \ours-F a été exécuté (sur ces jeux : \ours{} 86{,}0, \ours-G 86{,}2).}
\end{table}""")

# ------------------------------------------------------------ objective / size / time
OM = ["MLIC", "DRL", "GenSel-G", "GenSel", "GenSel-F"]
out.append(r"""\begin{table}[t]
\centering
\caption{Valeur moyenne de l'objectif $f$ (équation~\eqref{eq:obj}) sur les données d'apprentissage (plus petit est meilleur, meilleur en gras), taille des règles et temps (s) de \ours, taille moyenne du réservoir, et nombre de plis (sur 5) où \ours{} s'arrête avec le certificat de la proposition~\ref{prop:term} ($k=3$).}
\label{tab:obj}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l|rrrrr|rrrr}
\hline
& \multicolumn{5}{c|}{Objectif $f$} & \multicolumn{4}{c}{\ours}\\
Jeu & MLIC & DRL & \ours-G & \ours & \ours-F & taille & temps & réserv. & cert.\\
\hline""")
for d in DS:
    vals = {m: mean_obj(d, m) for m in OM}
    b = min(v[0] for v in vals.values() if v[0] is not None)
    cells = []
    for m in OM:
        v, n = vals[m]
        if v is None:
            cells.append("--"); continue
        s = big(v) + (r"$^{\dagger}$" if n < 5 else "")
        cells.append(r"\textbf{" + s + "}" if round(v) == round(b) else s)
    rs = by[(d, "GenSel")]
    cells += [fr(mean(d, "GenSel", "size")[0]), fr(mean(d, "GenSel", "time")[0]),
              fr(np.mean([x.get("pool_final", 0) for x in rs])), str(sum(bool(x["certified"]) for x in rs))]
    out.append(f"{DSF[d]} & " + " & ".join(cells) + r"\\")
out.append(r"""\hline
\end{tabular}
\end{table}""")

# ------------------------------------------------------------ sensitivity to k (condensed)
KS = [1, 2, 3, 5, 8]
out.append(r"""\begin{table}[t]
\centering
\caption{Sensibilité au nombre maximal d'itemsets $k$ : précision moyenne en test (\%) de \ours{} pour $k\in\{1,2,3,5,8\}$ (meilleure valeur en gras), nombre moyen d'itemsets réellement utilisés pour $k=8$, et objectif $f$ de \ours, DRL et MLIC pour $k=8$ (meilleur en gras). « -- » : aucune règle renvoyée en 60\,s ou encodage non généré.}
\label{tab:k}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l|rrrrr|r|rrr}
\hline
& \multicolumn{5}{c|}{\ours : précision} & & \multicolumn{3}{c}{Objectif $f$, $k=8$}\\
Jeu & $k{=}1$ & $k{=}2$ & $k{=}3$ & $k{=}5$ & $k{=}8$ & \#itemsets & \ours & DRL & MLIC\\
\hline""")
for d in DS:
    vals = {k: TK.mean(d, "GenSel", k, "test_acc")[0] for k in KS}
    b = max(round(100 * v, 1) for v in vals.values())
    cells = [(r"\textbf{" + fr(100 * vals[k]) + "}") if round(100 * vals[k], 1) == b else fr(100 * vals[k]) for k in KS]
    cells.append(fr(TK.mean(d, "GenSel", 8, "n_rules")[0]))
    objs = {m: TK.mobj(d, m, 8) for m in ["GenSel", "DRL", "MLIC"]}
    bo = min(v[0] for v in objs.values() if v[0] is not None)
    for m in ["GenSel", "DRL", "MLIC"]:
        v, n = objs[m]
        if v is None:
            cells.append("--"); continue
        s = big(v) + (r"$^{\dagger}$" if n < 5 else "")
        cells.append(r"\textbf{" + s + "}" if round(v) == round(bo) else s)
    out.append(f"{DSF[d]} & " + " & ".join(cells) + r"\\")
means = [fr(100 * np.mean([TK.mean(d, "GenSel", k, "test_acc")[0] for d in DS])) for k in KS]
out.append(r"\hline")
out.append("Moyenne & " + " & ".join(means) + r" & & & & \\")
out.append(r"""\hline
\end{tabular}
\end{table}""")

# ------------------------------------------------------------ validation-selected k
M3 = ["GenSel", "DRL", "MLIC"]
res = {(d, m, mode): TS.summary(d, m, mode) for d in DS for m in M3 for mode in ("k3", "sel")}
out.append(r"""\begin{table}[t]
\centering
\caption{Précision moyenne en test (\%) avec $k=3$ et avec $k\in\{1,2,3,5,8\}$ choisi pour chaque pli et chaque méthode sur des données de validation (découpage 80/20 du pli d'apprentissage ; en cas d'égalité, le plus petit $k$), et $k$ moyen choisi. Meilleure précision avec $k$ choisi en gras.}
\label{tab:sel}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l|rrr|rrr|rrr}
\hline
& \multicolumn{3}{c|}{\ours} & \multicolumn{3}{c|}{DRL} & \multicolumn{3}{c}{MLIC}\\
Jeu & $k{=}3$ & choisi & $\bar k$ & $k{=}3$ & choisi & $\bar k$ & $k{=}3$ & choisi & $\bar k$\\
\hline""")


def cell(s):
    return "--" if s is None else fr(100 * s["acc"]) + (r"$^{\dagger}$" if s["n"] < 5 else "")


for d in DS:
    b = max(round(100 * res[(d, m, "sel")]["acc"], 1) for m in M3 if res[(d, m, "sel")])
    cells = []
    for m in M3:
        a, s = res[(d, m, "k3")], res[(d, m, "sel")]
        cs = cell(s)
        if s and round(100 * s["acc"], 1) == b:
            cs = r"\textbf{" + cs + "}"
        cells += [cell(a), cs, "--" if s is None else fr(s["k"])]
    out.append(f"{DSF[d]} & " + " & ".join(cells) + r"\\")
means = []
for m in M3:
    for mode in ("k3", "sel"):
        v = [res[(d, m, mode)]["acc"] for d in DS if res[(d, m, mode)]]
        means.append(fr(100 * np.mean(v)) + ("" if len(v) == len(DS) else "$^{*}$"))
    means.append("")
out.append(r"\hline")
out.append("Moyenne & " + " & ".join(means) + r"\\")
out.append(r"""\hline
\end{tabular}
\end{table}""")

full = "\n".join(out)
parts = ["\\begin{table}" + p for p in full.split("\\begin{table}")[1:]]
assert len(parts) == 5
for name, t in zip(["enum", "acc", "obj", "k", "sel"], parts):
    open(os.path.join(EGC, f"tab_{name}.tex"), "w").write(t.strip() + "\n")
print("ok")
