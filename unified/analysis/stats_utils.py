"""
stats_utils.py — statistical helper functions.
"""

from typing import Callable, List, Sequence, Tuple

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
#  Wilcoxon signed-rank test (paired)
# ---------------------------------------------------------------------------

def wilcoxon_test(a: Sequence[float], b: Sequence[float]) -> dict:
    """
    Two-sided paired Wilcoxon signed-rank test.
    Warns when n < 6 (underpowered; stimulus has only 2 folds).
    Returns stat, p_value, effect_size (rank-biserial r), n.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n    = len(a)
    if n < 6:
        print(f"  [warn] Wilcoxon n={n} — underpowered (min recommended n=6). "
              f"Interpret p-value with caution.")
    diffs = a - b
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return {"stat": 0.0, "p_value": 1.0, "effect_size": 0.0, "n": n}
    stat, p = stats.wilcoxon(a, b, alternative="two-sided")
    return {
        "stat":        float(stat),
        "p_value":     float(p),
        "effect_size": float(_rank_biserial(a, b)),
        "n":           n,
    }


def _rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation as effect size for Wilcoxon."""
    diffs  = np.asarray(a, float) - np.asarray(b, float)
    diffs  = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0
    ranks   = stats.rankdata(np.abs(diffs))
    w_plus  = float(ranks[diffs > 0].sum())
    w_minus = float(ranks[diffs < 0].sum())
    total   = w_plus + w_minus
    return (w_plus - w_minus) / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
#  Mann-Whitney U test (unpaired)
# ---------------------------------------------------------------------------

def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> dict:
    """Two-sided Mann-Whitney U test. Effect size = rank-biserial r."""
    a, b   = np.asarray(a, float), np.asarray(b, float)
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    n_a, n_b = len(a), len(b)
    r = 1.0 - (2.0 * stat) / (n_a * n_b)
    return {
        "stat":        float(stat),
        "p_value":     float(p),
        "effect_size": float(r),
        "n_a":         n_a,
        "n_b":         n_b,
    }


# ---------------------------------------------------------------------------
#  Kruskal-Wallis (omnibus, unpaired)
# ---------------------------------------------------------------------------

def kruskal_wallis(*groups: Sequence[float]) -> dict:
    """Kruskal-Wallis H-test across ≥2 independent groups."""
    stat, p = stats.kruskal(*groups)
    return {"stat": float(stat), "p_value": float(p), "n_groups": len(groups)}


# ---------------------------------------------------------------------------
#  Holm-Bonferroni correction
# ---------------------------------------------------------------------------

def holm_bonferroni(p_values: List[float]) -> np.ndarray:
    """Return Holm-Bonferroni corrected p-values (same length as input)."""
    if not p_values:
        return np.array([])
    _, corrected, _, _ = multipletests(p_values, method="holm")
    return corrected


# ---------------------------------------------------------------------------
#  Bootstrap CI for recall curves (resampling folds)
# ---------------------------------------------------------------------------

def bootstrap_recall_ci(
    fold_recall_matrix: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap CI for mean recall@k curve by resampling rows (folds).
    fold_recall_matrix : (n_folds, V)
    Returns (lo, hi) each of shape (V,).
    """
    rng  = np.random.default_rng(seed)
    n    = fold_recall_matrix.shape[0]
    boot = np.array([
        fold_recall_matrix[rng.integers(0, n, size=n)].mean(axis=0)
        for _ in range(n_boot)
    ])
    alpha = 1.0 - ci
    lo = np.percentile(boot, 100 * alpha / 2,       axis=0)
    hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


# ---------------------------------------------------------------------------
#  Bootstrap CI for a scalar (resampling folds)
# ---------------------------------------------------------------------------

def bootstrap_scalar_ci(
    values: np.ndarray,
    func: Callable = np.mean,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """Bootstrap CI for a scalar statistic by resampling the input array."""
    rng  = np.random.default_rng(seed)
    n    = len(values)
    boot = [func(values[rng.integers(0, n, size=n)]) for _ in range(n_boot)]
    alpha = 1.0 - ci
    return (float(np.percentile(boot, 100 * alpha / 2)),
            float(np.percentile(boot, 100 * (1 - alpha / 2))))
