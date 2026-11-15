#!/bin/bash
# LR, RF, RIPPER, MLIC and DRL with k=3 (Tables III and IV), then DRL and MLIC with k=8 (Table V).
# Usage: scripts/run_baselines.sh [parallel_jobs=2]
cd "$(dirname "$0")/.."
P=${1:-2}; mkdir -p logs; export OUTDIR=${GENSEL_RESULTS:-results}; mkdir -p $OUTDIR
cat jobs/baselines.txt | xargs -P $P -L 1 bash -c 'python3 run.py $0 $1 5 60 $OUTDIR/results.jsonl LR,RF,RIPPER,MLIC,DRL > logs/base_$0_$1.log 2>&1'
cat jobs/baselines_k8.txt | xargs -P $P -L 1 bash -c 'K=$2 python3 run.py $0 $1 5 60 $OUTDIR/results_k.jsonl $3 > logs/base_k_$0_$1_$2.log 2>&1'
