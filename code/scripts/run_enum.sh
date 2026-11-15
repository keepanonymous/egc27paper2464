#!/bin/bash
# Enumeration experiments: counts of minimal generators (Table II) and exact selection over the
# enumerated generators (paragraph "Exact selection over enumerated generators").
cd "$(dirname "$0")/.."
mkdir -p logs
for d in iris heart ionosphere wdbc australian tictactoe spambase magic adult; do
  python3 enum_count.py $d > logs/enum_$d.log 2>&1
done
python3 exact_lane.py 0 2 > logs/exact0.log 2>&1 &
python3 exact_lane.py 1 2 > logs/exact1.log 2>&1
wait
python3 agg_exact.py
