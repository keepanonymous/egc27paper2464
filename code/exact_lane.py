import os
import config
from config import RESULTS, PAPER, EGC
import json, os, subprocess, sys
OUT = os.path.join(RESULTS, "results_exact.jsonl")
jobs = [(d, 4) for d in ["iris"]] + [(d, 3) for d in ["iris", "heart", "australian", "tictactoe"]] + \
       [(d, 2) for d in ["iris", "heart", "ionosphere", "wdbc", "australian", "tictactoe", "spambase", "magic", "adult"]]
jobs = [(d, L, f) for d, L in jobs for f in range(5)]
lane, nlanes = int(sys.argv[1]), int(sys.argv[2])
for i, (d, L, f) in enumerate(jobs):
    if i % nlanes != lane:
        continue
    done = set()
    import glob
    for p in glob.glob(OUT + ".lane*"):
        for l in open(p):
            r = json.loads(l); done.add((r["dataset"], r["L"], r["fold"], r["objective"]))
    if all((d, L, f, o) in done for o in ("f", "acc", "cov")):
        continue
    subprocess.run(["timeout", "600", "python3", "exact.py", d, str(f), str(L), OUT + f".lane{lane}"])
