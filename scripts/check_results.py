#!/usr/bin/env python
import argparse
from pathlib import Path

import pandas as pd


def pct(x):
    return 100.0 * float(x)


def close(actual, expected, tol=0.06):
    return abs(float(actual) - expected) <= tol


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()

    summary = pd.read_csv(args.analysis_dir / "same_config_summary.csv")
    diag = pd.read_csv(args.analysis_dir / "variance_diagnostics.csv")
    close_pairs = pd.read_csv(args.analysis_dir / "close_config_runlevel.csv")

    sasrec_20m = summary[(summary.model == "SASRec") & (summary.dataset == "Movielens-20m")].iloc[0]
    gru_20m = summary[(summary.model == "GRU4Rec") & (summary.dataset == "Movielens-20m")].iloc[0]
    beauty = summary[(summary.model == "SASRec") & (summary.dataset == "Beauty")].iloc[0]

    sasrec_diag = diag[(diag.model == "SASRec") & (diag.dataset == "Movielens-20m")].iloc[0]
    gru_diag = diag[(diag.model == "GRU4Rec") & (diag.dataset == "Movielens-20m")].iloc[0]

    require(close(pct(sasrec_20m.reject_005), 80.0), "SASRec ML-20M user-t rejection changed")
    require(close(pct(sasrec_20m.runlevel_reject_005), 4.3), "SASRec ML-20M run-level calibration changed")
    require(close(float(sasrec_diag.sqrt_VIF_pair), 9.23, tol=0.01), "SASRec ML-20M sqrt(VIF) changed")
    require(close(pct(gru_20m.reject_005), 44.8), "GRU4Rec ML-20M user-t rejection changed")
    require(close(float(gru_diag.sqrt_VIF_pair), 2.38, tol=0.01), "GRU4Rec ML-20M sqrt(VIF) changed")
    require(close(pct(beauty.wilcoxon_reject_005), 21.0), "Beauty Wilcoxon rejection changed")

    cross = close_pairs[
        (close_pairs.dataset == "Movielens-20m") & (close_pairs.comparison == "SASRec-GRU4Rec")
    ].iloc[0]
    require(close(float(cross.seed_snr), 3.43, tol=0.01), "SASRec-GRU4Rec SNR changed")

    print("included analysis files match the reported headline results")


if __name__ == "__main__":
    main()
