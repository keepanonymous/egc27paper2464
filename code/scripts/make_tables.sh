#!/bin/bash
# Regenerates the LaTeX tables from results/*.jsonl (no solver needed):
#   ../paper/     English paper (tables.tex, table_k.tex, table_sel.tex)
#   ../paper_egc/ French EGC version (tab_*.tex)
cd "$(dirname "$0")/.."
if [ -d ../paper ]; then python3 tables2.py && python3 tables_k2.py && python3 tables_sel.py; fi
if [ -d ../paper_egc ]; then python3 tables_egc.py; fi
