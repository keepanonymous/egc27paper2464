"""Dataset loading and MLIC-style binarization."""
import os
import config
from config import RESULTS, PAPER, EGC
import numpy as np
import pandas as pd

DATA_DIR = config.DATA_DIR

# name -> (file, positive-class rule)
DATASETS = {
    "iris": ("iris", lambda y: (y == 1).astype(int)),          # versicolor vs rest (as in DRL paper)
    "heart": ("heart_disease_cleveland", lambda y: (y == 1).astype(int)),
    "ionosphere": ("ionosphere", lambda y: (y == 1).astype(int)),
    "wdbc": ("breast_cancer_wisconsin_diagnostic", lambda y: (y == 1).astype(int)),
    "australian": ("credit_approval_australia", lambda y: (y == 1).astype(int)),
    "tictactoe": ("tic_tac_toe", lambda y: (y == 1).astype(int)),
    "spambase": ("spambase", lambda y: (y == 1).astype(int)),
    "magic": ("magic", lambda y: (y == 1).astype(int)),
    "adult": ("adult", lambda y: (y == 0).astype(int)),        # minority class (>50K) as target
}


def load_raw(name):
    f, rule = DATASETS[name]
    d = pd.read_csv(f"{DATA_DIR}/{f}.tsv.gz", sep="\t")
    y = rule(d["target"].values)
    X = d.drop(columns=["target"])
    return X, y


class Binarizer:
    """Thresholds at quantiles for numeric columns (x <= t), one-hot for low-cardinality
    columns, as in MLIC. Each binary feature later yields two items (b and not b)."""

    def __init__(self, n_thresholds=9, max_categories=10):
        self.n_thresholds = n_thresholds
        self.max_categories = max_categories

    def fit(self, X):
        self.specs = []  # (col, kind, value)
        for c in X.columns:
            vals = X[c].values
            uniq = np.unique(vals)
            if len(uniq) <= 1:
                continue
            if len(uniq) == 2:
                self.specs.append((c, "eq", uniq[1]))
            elif len(uniq) <= self.max_categories:
                for v in uniq:
                    self.specs.append((c, "eq", v))
            else:
                qs = np.quantile(vals.astype(float), np.linspace(0, 1, self.n_thresholds + 2)[1:-1])
                for t in np.unique(qs):
                    self.specs.append((c, "le", t))
        return self

    def transform(self, X):
        cols = []
        for c, kind, v in self.specs:
            col = X[c].values
            cols.append((col == v) if kind == "eq" else (col.astype(float) <= v))
        B = np.stack(cols, axis=1).astype(np.uint8)
        return B

    def names(self):
        return [f"{c}={v}" if k == "eq" else f"{c}<={v:.4g}" for c, k, v in self.specs]
