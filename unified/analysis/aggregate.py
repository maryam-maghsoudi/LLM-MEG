"""
aggregate.py — trial-level → fold-level aggregation.

For each (eval_scheme, fold_id, control):
  - Compute mean and std of scalar metrics across trials.
  - Compute mean recall@k curve across trials (per k).

Trials within a fold are NOT independent (shared subject/session/decoder),
so statistics always operate on fold-level aggregates, never raw trials.
"""

import numpy as np
import pandas as pd

SCALAR_METRICS = ["mrr", "word_acc", "bleu1", "wer", "recall_auc"]
GROUP_KEYS     = ["eval_scheme", "fold_id", "control"]


def aggregate_scalars(trials_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns fold_df: one row per (eval_scheme, fold_id, control).
    For each metric M, columns M_mean and M_std (within-fold std, ddof=1).
    """
    rows = []
    for keys, grp in trials_df.groupby(GROUP_KEYS, sort=True):
        row = dict(zip(GROUP_KEYS, keys))
        row["n_trials"] = len(grp)
        for m in SCALAR_METRICS:
            row[f"{m}_mean"] = float(grp[m].mean())
            row[f"{m}_std"]  = float(grp[m].std(ddof=1)) if len(grp) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_recall_curves(recall_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns fold_recall_df: one row per (eval_scheme, fold_id, control, k).
    Column recall_mean = mean recall@k across trials in that fold.
    """
    agg = (
        recall_df
        .groupby(GROUP_KEYS + ["k"], sort=True)["recall"]
        .mean()
        .reset_index()
        .rename(columns={"recall": "recall_mean"})
    )
    return agg


def get_fold_recall_matrix(
    fold_recall_df: pd.DataFrame,
    eval_scheme: str,
    control: str,
) -> tuple:
    """
    Return (matrix, ks) where matrix is (n_folds, min_V) and ks is the k values used.
    Rows are folds sorted by fold_id. Columns are k=1..min_V (truncated to shortest
    fold curve so all folds contribute equally).
    """
    sub = fold_recall_df[
        (fold_recall_df["eval_scheme"] == eval_scheme) &
        (fold_recall_df["control"]     == control)
    ]
    fold_ids = sorted(sub["fold_id"].unique())
    min_V    = sub.groupby("fold_id")["k"].max().min()
    ks       = list(range(1, int(min_V) + 1))

    matrix = np.array([
        sub[(sub["fold_id"] == fid) & (sub["k"] <= min_V)]
        .sort_values("k")["recall_mean"].values
        for fid in fold_ids
    ])  # (n_folds, min_V)
    return matrix, ks
