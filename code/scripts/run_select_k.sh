#!/bin/bash
# Choice of k on validation data (Table VI):
#   "full" lines: DRL and MLIC on the outer training folds for k in {1,2,5} (60 s);
#   "val"  lines: GenSel, DRL and MLIC on the inner 80/20 split for k in {1,2,3,5,8} (30 s).
# Usage: scripts/run_select_k.sh [parallel_jobs=2]
cd "$(dirname "$0")/.."
P=${1:-2}; mkdir -p logs; export OUTDIR=${GENSEL_RESULTS:-results}; mkdir -p $OUTDIR
cat jobs/select_k.txt | xargs -P $P -L 1 bash -c '
if [ "$0" = "full" ]; then
  K=$3 python3 run.py $1 $2 5 60 $OUTDIR/results_k125.jsonl DRL,MLIC > logs/full_$1_$2_$3.log 2>&1
else
  python3 val_run.py $1 $2 $3 30 GenSel,DRL,MLIC $OUTDIR/results_val.jsonl > logs/val_$1_$2_$3.log 2>&1
fi'
