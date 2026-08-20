"""
stage1_validity.py — Q1: Is the model decoding real MEG signal?

For each eval scheme, compare real (none) vs shuffle_time and none vs zero,
using paired Wilcoxon signed-rank tests on fold-level metric means.

Returns raw (uncorrected) p-values. Holm-Bonferroni correction across all
30 tests (3 schemes × 5 metrics × 2 contrasts) is applied in run_all.py.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from .stats_utils import wilcoxon_test

EVAL_SCHEMES   = ["loso", "session_cv", "stimulus"]
SCALAR_METRICS = ["mrr", "word_acc", "bleu1", "wer", "recall_auc"]
CONTROLS       = ["shuffle_time", "zero"]


def run_stage1(fold_df: pd.DataFrame) -> List[Dict]:
    """
    Run all 30 paired Wilcoxon tests.

    Pairing is by fold_id within each eval_scheme: sub-01's none score is
    paired with sub-01's shuffle_time score, etc. Rows are sorted by fold_id
    before alignment.

    Returns list of test-record dicts. 'p_corrected' is None until filled
    by run_all.py.
    """
    records = []

    for scheme in EVAL_SCHEMES:
        scheme_df = fold_df[fold_df["eval_scheme"] == scheme]

        for metric in SCALAR_METRICS:
            none_df = (scheme_df[scheme_df["control"] == "none"]
                       .sort_values("fold_id"))
            none_vals = none_df[f"{metric}_mean"].values

            for ctrl in CONTROLS:
                ctrl_df  = (scheme_df[scheme_df["control"] == ctrl]
                            .sort_values("fold_id"))
                ctrl_vals = ctrl_df[f"{metric}_mean"].values

                if len(none_vals) == 0 or len(ctrl_vals) == 0:
                    continue
                if len(none_vals) != len(ctrl_vals):
                    print(f"  [warn] {scheme}/{metric}: fold count mismatch "
                          f"none={len(none_vals)} vs {ctrl}={len(ctrl_vals)} — skipping")
                    continue

                test = wilcoxon_test(none_vals, ctrl_vals)

                # Across-fold SEM (precision of the mean estimate)
                none_sem = (none_vals.std(ddof=1) / np.sqrt(len(none_vals))
                            if len(none_vals) > 1 else 0.0)
                ctrl_sem = (ctrl_vals.std(ddof=1) / np.sqrt(len(ctrl_vals))
                            if len(ctrl_vals) > 1 else 0.0)

                # Within-fold std: mean of per-fold stds (trial-level variability)
                none_std_w = float(none_df[f"{metric}_std"].mean())
                ctrl_std_w = float(ctrl_df[f"{metric}_std"].mean())

                records.append({
                    "eval_scheme":       scheme,
                    "metric":            metric,
                    "contrast":          f"none_vs_{ctrl}",
                    "none_mean":         float(none_vals.mean()),
                    "none_sem":          float(none_sem),
                    "none_std_within":   none_std_w,
                    "ctrl_name":         ctrl,
                    "ctrl_mean":         float(ctrl_vals.mean()),
                    "ctrl_sem":          float(ctrl_sem),
                    "ctrl_std_within":   ctrl_std_w,
                    "stat":              test["stat"],
                    "p_value":           test["p_value"],
                    "p_corrected":       None,   # set by run_all.py
                    "effect_size":       test["effect_size"],
                    "n":                 test["n"],
                })

    return records


def stage1_summary_df(records: List[Dict], scheme: str) -> pd.DataFrame:
    """
    Build a formatted summary DataFrame for one eval scheme.

    Columns:
      metric |
      none mean | none SEM | none std_within |
      shuffle_time mean | shuffle_time SEM | shuffle_time std_within |
        p(none vs shuffle, corrected) | ES(none vs shuffle) |
      zero mean | zero SEM | zero std_within |
        p(none vs zero, corrected) | ES(none vs zero)
    """
    sub = [r for r in records if r["eval_scheme"] == scheme]
    rows = []

    for metric in SCALAR_METRICS:
        metric_recs = [r for r in sub if r["metric"] == metric]
        if not metric_recs:
            continue

        row = {"metric": metric}

        # none values are the same across both contrasts for this metric
        ref = metric_recs[0]
        row.update({
            "none_mean":       ref["none_mean"],
            "none_sem":        ref["none_sem"],
            "none_std_within": ref["none_std_within"],
        })

        for ctrl in CONTROLS:
            rec = next((r for r in metric_recs if r["ctrl_name"] == ctrl), None)
            if rec is None:
                continue
            row[f"{ctrl}_mean"]       = rec["ctrl_mean"]
            row[f"{ctrl}_sem"]        = rec["ctrl_sem"]
            row[f"{ctrl}_std_within"] = rec["ctrl_std_within"]
            row[f"p_{ctrl}_corrected"]= rec["p_corrected"]
            row[f"es_{ctrl}"]         = rec["effect_size"]

        rows.append(row)

    return pd.DataFrame(rows)
