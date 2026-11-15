# GenSel: learning disjunctive rules by generating minimal generators

Code, data and raw results for the paper GenSel (French version for EGC 2027). Everything needed to reproduce the experiments and the tables of
the paper is in this archive.

```
GenSel/
├── README.md            this file
├── LISEZMOI.txt         short French version of this file
├── paper_egc/           LaTeX sources of the French EGC 2027 version (tables are generated, see step 5)
└── code/
    ├── config.py        paths (data, results, solver), overridable by environment variables
    ├── data/            the 9 datasets (PMLB, .tsv.gz)
    ├── results/         raw results used in the paper (one JSON record per line)
    ├── jobs/            job lists (dataset, fold, ...) used by the run scripts
    ├── scripts/         build and run scripts
    └── *.py             source code (see "Source files" below)
```

## 1. Requirements

- Linux or macOS, Python ≥ 3.10, `git`, `make`, `g++`, `zlib` (Debian/Ubuntu:
  `sudo apt install build-essential zlib1g-dev git`).
- Python packages (versions used for the paper in `code/requirements.txt`):

```bash
cd GenSel/code
pip install -r requirements.txt
```

- The MaxSAT solver Open-WBO. The script below clones it from GitHub, checks out the version
  used in the paper and builds it into `code/bin/open-wbo_release`:

```bash
bash scripts/build_openwbo.sh
```

To use an Open-WBO binary installed elsewhere, set `OPENWBO=/path/to/open-wbo_release`.

## 2. Quick check (about 2 minutes)

All commands are run from `GenSel/code`. New results are written to a separate directory, so
that the results of the paper in `results/` are left untouched:

```bash
export GENSEL_RESULTS=$PWD/results_new
mkdir -p $GENSEL_RESULTS
python3 test_toy.py                                                # SAT encoding of minimal generators on the running example
python3 run_gen.py iris 0 10 $GENSEL_RESULTS/gen.jsonl Greedy,CG   # GenSel-G and GenSel, Iris, fold 0, 10 s
python3 run.py iris 0 5 10 $GENSEL_RESULTS/base.jsonl RIPPER,MLIC,DRL
```

On Iris fold 0, GenSel (`"method": "CG"`) should reach the training objective `train_obj = 8`,
which is the optimum.

## 3. Reproducing the experiments

Each script runs 2 jobs in parallel by default (the paper used a 2-core machine with 7 GB of
memory); pass another number as first argument to change this. Every run has a time limit of
60 s per fold and method (30 s on the validation splits), so the total wall-clock time is about
10 to 12 hours with 2 parallel jobs.

| Step | Command | Paper | Output in `$GENSEL_RESULTS` |
|---|---|---|---|
| Baselines (LR, RF, RIPPER, MLIC, DRL), k=3, and DRL/MLIC with k=8 | `bash scripts/run_baselines.sh` | Tables 2–4 | `results.jsonl`, `results_k.jsonl` |
| GenSel-G, GenSel, GenSel-F, k=3, and GenSel with k ∈ {1,2,5,8} | `bash scripts/run_gensel.sh` | Tables 2–4 | `results_gen.jsonl`, `results_gen_k.jsonl` |
| Choice of k on validation data | `bash scripts/run_select_k.sh` | Table 5 | `results_val.jsonl`, `results_k125.jsonl` |
| Counting minimal generators; exact selection over enumerated generators | `bash scripts/run_enum.sh` | Table 1 and paragraph « Sélection exacte sur les générateurs énumérés » | `enum_counts.jsonl`, `results_exact.jsonl.lane*` |

Without `GENSEL_RESULTS`, the scripts append to `code/results/`, which already contains the
results of the paper.

**Note on reproducibility.** Folds are fixed (stratified 5-fold, `random_state=0`) and the
binarization is fitted on the training folds only. The solvers are, however, stopped by time
limits, so on the larger datasets the results depend on the speed of the machine and can differ
slightly from the paper; on the six smaller datasets, GenSel usually terminates before the limit.

## 4. Running a single experiment

```bash
python3 run_gen.py  <dataset> <fold> <time_s> <out.jsonl> <methods>          # methods: Greedy, CG, CG-F
python3 run.py      <dataset> <fold> 5 <time_s> <out.jsonl> <methods>        # methods: LR, RF, RIPPER, MLIC, DRL
python3 val_run.py  <dataset> <fold> <k> <time_s> <methods> <out.jsonl>      # methods: GenSel, DRL, MLIC
python3 exact.py    <dataset> <fold> <L> <out.jsonl>                         # exact selection, length <= L
python3 enum_count.py <dataset> [Lmax=4] [time_s=300]
```

Datasets: `iris heart ionosphere wdbc australian tictactoe spambase magic adult`; folds 0 to 4.
The number of itemsets k is read from the environment variable `K` (default 3), e.g.
`K=5 python3 run_gen.py heart 0 60 out.jsonl CG`. In the code, `Greedy`, `CG` and `CG-F` are
GenSel-G, GenSel and GenSel-F of the paper.

Each line of an output file is one JSON record with, among others, `dataset`, `fold`, `method`,
`k`, `test_acc`, `train_obj` (objective (1) on the training data), `size`, `time`, `certified`
(certificate of Proposition 4) and the learned `rules`.

## 5. Regenerating the tables

```bash
bash scripts/make_tables.sh        # writes paper_egc/tab_*.tex
python3 agg_exact.py               # summary of the exact-selection experiment
cd ../paper_egc && pdflatex egc2027_gensel && bibtex egc2027_gensel && pdflatex egc2027_gensel && pdflatex egc2027_gensel
```

With the results shipped in `code/results/`, the generated tables are identical to those of
the paper. To build tables from new results, set `GENSEL_RESULTS` to their directory.

## Source files

| File | Content |
|---|---|
| `gencg.py` | GenSel: MaxSAT oracle, greedy generation, LP-guided generation (column generation), MaxSAT master, post-processing |
| `run_gen.py` | Runs GenSel-G / GenSel / GenSel-F on one fold |
| `satgen.py` | SAT encoding and enumeration of minimal generators (witness encoding); `test_toy.py`, `test_real.py` check it against brute force |
| `encode.py` | WCNF writer, DRL encoding, Open-WBO portfolio (linear search, then OLL) |
| `rules.py` | Rule evaluation, objective (1), post-processing, MLIC encoding |
| `run.py` | Baselines: LR, RF, RIPPER (wittgenstein), MLIC and DRL (re-implemented with the same solver). It also contains the MinSS methods of another paper, not used here |
| `val_run.py` | Validation runs used to choose k |
| `data.py` | Dataset loading and binarization (9 quantile thresholds, one-hot encoding) |
| `enum_count.py` | Levelwise enumeration of minimal generators (Table 1) |
| `exact.py`, `exact_lane.py`, `agg_exact.py` | Enumeration, lossless filtering and exact selection by integer programming (HiGHS) |
| `agg2.py`, `tables2.py`, `tables_k2.py`, `tables_sel.py`, `tables_egc.py` | Aggregation and LaTeX tables (English paper; `tables_egc.py` for the French EGC version) |
| `toy.py` | GenSel on the running example of the paper |

## Data

The datasets come from the UCI repository through PMLB (Olson et al., 2017) and are included in
`code/data/`. Positive class: versicolor (Iris), presence of disease (Heart), and the minority
class `>50K` (Adult); see `data.py` for the others.
