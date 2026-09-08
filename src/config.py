"""Shared experiment constants."""
from __future__ import annotations
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"

SEEDS = list(range(10))
NOISE_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
NOISE_KINDS = ["symmetric", "asymmetric", "boundary", "far"]

PURITY_THRESHOLDS = [1.0, 0.9, 0.8]
PURITY_DEFAULT = 1.0
N_FOLDS = 5

# binary sets that load without network (openml/ucimlrepo are firewalled here)
DATASETS_BINARY = ["balance_scale", "breast_cancer", "SPECTF", "chess_krvkp"]
