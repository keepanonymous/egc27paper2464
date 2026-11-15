#!/bin/bash
# GenSel-G, GenSel and GenSel-F with k=3 (Tables III and IV), then GenSel for k in {1,2,5,8} (Table V).
# Usage: scripts/run_gensel.sh [parallel_jobs=2]
cd "$(dirname "$0")/.."
P=${1:-2}; mkdir -p logs; export OUTDIR=${GENSEL_RESULTS:-results}; mkdir -p $OUTDIR
cat jobs/gensel.txt | xargs -P $P -L 1 bash -c 'python3 run_gen.py $0 $1 60 $OUTDIR/results_gen.jsonl $2 > logs/gen_$0_$1.log 2>&1'
cat jobs/gensel_k.txt | xargs -P $P -L 1 bash -c 'K=$2 python3 run_gen.py $0 $1 60 $OUTDIR/results_gen_k.jsonl CG > logs/gen_k_$0_$1_$2.log 2>&1'
