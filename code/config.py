"""Paths used by all scripts (override with environment variables)."""
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("GENSEL_DATA", os.path.join(ROOT, "data"))
RESULTS = os.environ.get("GENSEL_RESULTS", os.path.join(ROOT, "results"))
TMP = os.environ.get("GENSEL_TMP", os.path.join(ROOT, "tmp"))
PAPER = os.environ.get("GENSEL_PAPER", os.path.join(ROOT, "..", "paper"))
EGC = os.environ.get("GENSEL_EGC", os.path.join(ROOT, "..", "paper_egc"))
OPENWBO = os.environ.get("OPENWBO", os.path.join(ROOT, "bin", "open-wbo_release"))
for _d in (RESULTS, TMP):
    os.makedirs(_d, exist_ok=True)
