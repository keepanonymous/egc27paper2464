# GenSel: learning disjunctive rules by generating minimal generators

Here is the code, data and raw results for the GenSel approach. 

+++++++++++++++++++++++++++++++++
## 1. Requirements
+++++++++++++++++++++++++++++++++

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

+++++++++++++++++++++++++++++++++
## 2. Quick check (about 2 minutes)
+++++++++++++++++++++++++++++++++

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

+++++++++++++++++++++++++++++++++
## 3. Running a single experiment
+++++++++++++++++++++++++++++++++

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

+++++++++++++++++++++++++++++++++
## 4. Regenerating the tables
+++++++++++++++++++++++++++++++++

```bash
bash scripts/make_tables.sh        # writes paper_egc/tab_*.tex
python3 agg_exact.py               # summary of the exact-selection experiment
cd ../paper_egc && pdflatex egc2027_gensel && bibtex egc2027_gensel && pdflatex egc2027_gensel && pdflatex egc2027_gensel
```

With the results shipped in `code/results/`, the generated tables are identical to those of
the paper. To build tables from new results, set `GENSEL_RESULTS` to their directory.


## Data

The datasets come from the UCI repository through PMLB (Olson et al., 2017) and are included in
`code/data/`. Positive class: versicolor (Iris), presence of disease (Heart), and the minority
class `>50K` (Adult); see `data.py` for the others.
