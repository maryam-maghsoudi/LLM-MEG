"""
stage2_generalization.py — Q2: How does performance compare across eval schemes?

Uses real (none) condition only. Fold counts differ across schemes (13, 5, 2),
and folds are NOT paired across schemes — DO NOT import Stage 1 pairing logic here.
Uses Kruskal-Wallis omnibus + pairwise Mann-Whitney U with Holm-Bonferroni.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from .stats_utils import kruskal_wallis, mann_whitney_u, holm_bonferroni

EVAL_SCHEMES   = ["loso", "session_cv", "stimulus"]
SCALAR_METRICS = ["mrr", "word_acc", "bleu1", "wer", "recall_auc"]
PAIRS          = [("loso", "session_cv"), ("loso", "stimulus"), ("session_cv", "stimulus")]


def run_stage2(fold_df: pd.DataFrame) -> List[Dict]:
    """
    For each metric:
      1. Kruskal-Wallis omnibus across the three eval schemes (real condition only).
      2. Pairwise Mann-Whitney U with Holm-Bonferroni correction within each metric.

    KW p-values are then Holm-corrected across all 5 metrics.
    Returns a flat list of per-metric × per-pair dicts.
    """
    real_df = fold_df[fold_df["control"] == "none"].copy()
    records = []

    kw_pvals_by_metric: Dict[str, float] = {}

    for metric in SCALAR_METRICS:
        groups = {
            s: real_df[real_df["eval_scheme"] == s][f"{metric}_mean"].values
            for s in EVAL_SCHEMES
        }

        kw = kruskal_wallis(*[groups[s] for s in EVAL_SCHEMES])
        kw_pvals_by_metric[metric] = kw["p_value"]

        # Per-metric pairwise MW with Holm correction within metric
        pw_raw = []
        for s1, s2 in PAIRS:
            mw = mann_whitney_u(groups[s1], groups[s2])
            pw_raw.append({
                "metric":      metric,
                "comparison":  f"{s1}_vs_{s2}",
                "kw_stat":     kw["stat"],
                "kw_p":        kw["p_value"],
                "kw_p_corrected": None,   # set below
                "mw_stat":     mw["stat"],
                "p_value":     mw["p_value"],
                "p_corrected": None,      # set below
                "effect_size": mw["effect_size"],
                "n_a":         mw["n_a"],
                "n_b":         mw["n_b"],
            })
            for scheme in EVAL_SCHEMES:
                g = groups[scheme]
                pw_raw[-1][f"{scheme}_mean"] = float(g.mean()) if len(g) else float("nan")
                pw_raw[-1][f"{scheme}_sem"]  = (float(g.std(ddof=1) / np.sqrt(len(g)))
                                                if len(g) > 1 else 0.0)
                pw_raw[-1][f"{scheme}_std_within"] = float(
                    real_df[real_df["eval_scheme"] == scheme][f"{metric}_std"].mean()
                )

        # Holm correct pairwise p-values within this metric (3 comparisons)
        corr = holm_bonferroni([r["p_value"] for r in pw_raw])
        for r, cp in zip(pw_raw, corr):
            r["p_corrected"] = float(cp)

        records.extend(pw_raw)

    # Holm correct KW p-values across all 5 metrics
    kw_corr = holm_bonferroni([kw_pvals_by_metric[m] for m in SCALAR_METRICS])
    kw_corr_map = dict(zip(SCALAR_METRICS, kw_corr))
    for r in records:
        r["kw_p_corrected"] = float(kw_corr_map.get(r["metric"], float("nan")))

    return records


def stage2_summary_df(records: List[Dict]) -> pd.DataFrame:
    """
    Summary table: one row per metric.
    Columns: metric, per-scheme mean/SEM/std_within, KW p (corrected),
             pairwise p's (corrected), effect sizes.
    """
    rows = []
    for metric in SCALAR_METRICS:
        metric_recs = [r for r in records if r["metric"] == metric]
        if not metric_recs:
            continue
        r0 = metric_recs[0]
        row = {
            "metric":         metric,
            "kw_stat":        r0["kw_stat"],
            "kw_p_corrected": r0["kw_p_corrected"],
        }
        for scheme in EVAL_SCHEMES:
            row[f"{scheme}_mean"]       = r0.get(f"{scheme}_mean", float("nan"))
            row[f"{scheme}_sem"]        = r0.get(f"{scheme}_sem",  float("nan"))
            row[f"{scheme}_std_within"] = r0.get(f"{scheme}_std_within", float("nan"))
        for rec in metric_recs:
            cmp = rec["comparison"]
            row[f"p_{cmp}_corrected"] = rec["p_corrected"]
            row[f"es_{cmp}"]          = rec["effect_size"]
        rows.append(row)
    return pd.DataFrame(rows)
