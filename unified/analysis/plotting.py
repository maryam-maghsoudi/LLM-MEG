"""
plotting.py — all figures for the twostage analysis pipeline.

Figure 1 (×3, one per eval scheme): validity — real vs controls.
  Subplot 0: recall@k curves with bootstrap CI.
  Subplots 1-4: box+strip plots for mrr, word_acc, bleu1, wer.
               Dots = fold-level means with within-fold std error bars.
               Lines connect each fold across conditions (paired).

Figure 2 (×3, one per eval scheme): training curves — Stage 1 and Stage 2.
  All folds overlaid (light) + mean curve (bold), train and val loss.

Figure 3: generalization — scalar metrics across eval schemes (real only).
  One panel per metric; x-axis = eval scheme.

Figure 4: generalization — recall@k curves, one per eval scheme (real only).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .aggregate import get_fold_recall_matrix
from .stats_utils import bootstrap_recall_ci

# ---------------------------------------------------------------------------
#  Style constants
# ---------------------------------------------------------------------------

CTRL_COLORS = {
    "none":         "#1f77b4",   # blue
    "shuffle_time": "#ff7f0e",   # orange
    "zero":         "#d62728",   # red
}
CTRL_LABELS = {
    "none":         "Real",
    "shuffle_time": "Shuffle time",
    "zero":         "Zero MEG",
}
SCHEME_COLORS = {
    "loso":       "#2ca02c",   # green
    "session_cv": "#9467bd",   # purple
    "stimulus":   "#8c564b",   # brown
}
SCHEME_LABELS = {
    "loso":       "LOSO",
    "session_cv": "Session CV",
    "stimulus":   "Stimulus",
}
METRIC_LABELS = {
    "mrr":        "MRR",
    "word_acc":   "Word Accuracy",
    "bleu1":      "BLEU-1",
    "wer":        "WER",
    "recall_auc": "Recall AUC",
}

CONTROLS       = ["none", "shuffle_time", "zero"]
EVAL_SCHEMES   = ["loso", "session_cv", "stimulus"]
SCALAR_METRICS = ["mrr", "word_acc", "bleu1", "wer"]


# ---------------------------------------------------------------------------
#  Helper: recall@k subplot
# ---------------------------------------------------------------------------

def _plot_recall_ax(
    ax,
    fold_recall_df: pd.DataFrame,
    eval_scheme: str,
    controls: List[str] = CONTROLS,
    n_boot: int = 1000,
):
    for ctrl in controls:
        try:
            matrix, ks = get_fold_recall_matrix(fold_recall_df, eval_scheme, ctrl)
        except (ValueError, KeyError):
            continue
        if matrix.shape[0] == 0:
            continue

        mean_curve = matrix.mean(axis=0)
        lo, hi     = bootstrap_recall_ci(matrix, n_boot=n_boot)
        color      = CTRL_COLORS[ctrl]
        label      = CTRL_LABELS[ctrl]

        ax.plot(ks, mean_curve, color=color, lw=1.8, label=label)
        ax.fill_between(ks, lo, hi, color=color, alpha=0.15)

    ax.set_xlabel("k", fontsize=9)
    ax.set_ylabel("Recall@k", fontsize=9)
    ax.set_title("Recall@k curve", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
#  Helper: scalar metric subplot (box + strip + error bars + paired lines)
# ---------------------------------------------------------------------------

def _plot_metric_ax(
    ax,
    fold_df: pd.DataFrame,
    eval_scheme: str,
    metric: str,
    controls: List[str] = CONTROLS,
):
    scheme_df = fold_df[fold_df["eval_scheme"] == eval_scheme]
    x_pos     = {c: i for i, c in enumerate(controls)}
    rng       = np.random.default_rng(42)

    # Collect per-control data sorted by fold_id
    fold_data = {}
    for ctrl in controls:
        sub = scheme_df[scheme_df["control"] == ctrl].sort_values("fold_id")
        fold_data[ctrl] = sub

    # Connecting lines between paired folds
    sets = [set(fold_data[c]["fold_id"]) for c in controls if c in fold_data]
    common_folds = set.intersection(*sets) if sets else set()
    for fid in sorted(common_folds):
        xs, ys = [], []
        for ctrl in controls:
            row = fold_data[ctrl][fold_data[ctrl]["fold_id"] == fid]
            if len(row):
                xs.append(x_pos[ctrl])
                ys.append(row[f"{metric}_mean"].values[0])
        if len(xs) > 1:
            ax.plot(xs, ys, color="gray", alpha=0.2, lw=0.7, zorder=1)

    # Box plot (distribution of fold-level means)
    box_data = [fold_data[c][f"{metric}_mean"].values
                for c in controls if c in fold_data]
    bp = ax.boxplot(
        box_data,
        positions=list(range(len(controls))),
        widths=0.35,
        patch_artist=True,
        zorder=2,
        flierprops=dict(marker="", linestyle="none"),
        medianprops=dict(color="black", lw=1.5),
        whiskerprops=dict(lw=1.0),
        capprops=dict(lw=1.0),
    )
    for patch, ctrl in zip(bp["boxes"], controls):
        patch.set_facecolor(CTRL_COLORS[ctrl])
        patch.set_alpha(0.25)

    # Scatter dots + within-fold std error bars
    for ctrl in controls:
        if ctrl not in fold_data:
            continue
        x    = x_pos[ctrl]
        sub  = fold_data[ctrl]
        means = sub[f"{metric}_mean"].values
        stds  = sub[f"{metric}_std"].values
        jitter = rng.uniform(-0.1, 0.1, len(means))
        ax.errorbar(
            np.full(len(means), x) + jitter,
            means,
            yerr=stds,
            fmt="o",
            color=CTRL_COLORS[ctrl],
            ms=4, lw=0, elinewidth=0.9, capsize=2, alpha=0.85, zorder=3,
        )

    ax.set_xticks(range(len(controls)))
    ax.set_xticklabels([CTRL_LABELS[c] for c in controls], fontsize=8)
    ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
#  Figure 1: validity (per eval scheme)
# ---------------------------------------------------------------------------

def plot_fig1_validity(
    fold_df: pd.DataFrame,
    fold_recall_df: pd.DataFrame,
    eval_scheme: str,
    out_dir: Path,
    n_boot: int = 1000,
):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    fig.suptitle(
        f"Validity — {SCHEME_LABELS.get(eval_scheme, eval_scheme)} "
        f"(real vs controls, twostage)",
        fontsize=12, fontweight="bold",
    )

    _plot_recall_ax(axes[0], fold_recall_df, eval_scheme, n_boot=n_boot)
    for ax, metric in zip(axes[1:], SCALAR_METRICS):
        _plot_metric_ax(ax, fold_df, eval_scheme, metric)

    # Legend for control colours
    handles = [mpatches.Patch(color=CTRL_COLORS[c], label=CTRL_LABELS[c])
               for c in CONTROLS]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ("pdf", "png"):
        path = out_dir / f"fig1_validity_{eval_scheme}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Figure 3: generalization — scalar metrics across eval schemes
# ---------------------------------------------------------------------------

def plot_fig3_generalization_scalars(
    fold_df: pd.DataFrame,
    out_dir: Path,
):
    all_metrics = SCALAR_METRICS + ["recall_auc"]
    n_metrics   = len(all_metrics)
    fig, axes   = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 4.5))
    fig.suptitle(
        "Generalization — real condition across eval schemes (twostage)",
        fontsize=12, fontweight="bold",
    )

    real_df = fold_df[fold_df["control"] == "none"]
    rng     = np.random.default_rng(42)

    for ax, metric in zip(axes, all_metrics):
        x_pos = {s: i for i, s in enumerate(EVAL_SCHEMES)}

        # Collect per-scheme data
        scheme_data = {}
        for scheme in EVAL_SCHEMES:
            sub = real_df[real_df["eval_scheme"] == scheme].sort_values("fold_id")
            scheme_data[scheme] = sub

        # Box plot
        box_data = [scheme_data[s][f"{metric}_mean"].values for s in EVAL_SCHEMES]
        bp = ax.boxplot(
            box_data,
            positions=list(range(len(EVAL_SCHEMES))),
            widths=0.4,
            patch_artist=True,
            zorder=2,
            flierprops=dict(marker="", linestyle="none"),
            medianprops=dict(color="black", lw=1.5),
        )
        for patch, scheme in zip(bp["boxes"], EVAL_SCHEMES):
            patch.set_facecolor(SCHEME_COLORS[scheme])
            patch.set_alpha(0.25)

        # Scatter + within-fold std error bars
        for scheme in EVAL_SCHEMES:
            x    = x_pos[scheme]
            sub  = scheme_data[scheme]
            means = sub[f"{metric}_mean"].values
            stds  = sub[f"{metric}_std"].values
            jitter = rng.uniform(-0.12, 0.12, len(means))
            ax.errorbar(
                np.full(len(means), x) + jitter,
                means,
                yerr=stds,
                fmt="o",
                color=SCHEME_COLORS[scheme],
                ms=4, lw=0, elinewidth=0.9, capsize=2, alpha=0.85, zorder=3,
            )

        ax.set_xticks(range(len(EVAL_SCHEMES)))
        ax.set_xticklabels([SCHEME_LABELS[s] for s in EVAL_SCHEMES],
                           fontsize=8, rotation=15, ha="right")
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"fig3_generalization_scalars.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Figure 4: generalization — recall@k curves across eval schemes
# ---------------------------------------------------------------------------

def plot_fig4_generalization_recall(
    fold_recall_df: pd.DataFrame,
    out_dir: Path,
    n_boot: int = 1000,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        "Recall@k — real condition across eval schemes (twostage)",
        fontsize=12, fontweight="bold",
    )

    for scheme in EVAL_SCHEMES:
        try:
            matrix, ks = get_fold_recall_matrix(fold_recall_df, scheme, "none")
        except (ValueError, KeyError):
            continue
        if matrix.shape[0] == 0:
            continue

        mean_curve = matrix.mean(axis=0)
        lo, hi     = bootstrap_recall_ci(matrix, n_boot=n_boot)
        color      = SCHEME_COLORS[scheme]
        label      = SCHEME_LABELS[scheme]

        ax.plot(ks, mean_curve, color=color, lw=2.0, label=label)
        ax.fill_between(ks, lo, hi, color=color, alpha=0.15)

    ax.set_xlabel("k", fontsize=11)
    ax.set_ylabel("Recall@k", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"fig4_generalization_recall.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Figure 2: training curves (per eval scheme, real condition only)
# ---------------------------------------------------------------------------

_OUT_ROOT = Path(__file__).parent.parent / "out" / "twostage" / "HuggingFaceTB_SmolLM2-360M"


def _load_histories(eval_scheme: str, out_root: Path = _OUT_ROOT) -> Dict[str, Dict]:
    """
    Return {fold_id: {"stage1": {"train_loss": [...], "val_loss": [...]},
                       "stage2": {...}}}
    for the real (none) condition of the given eval scheme.
    """
    from .load_data import _parse_dir
    histories = {}
    for h_path in sorted(out_root.glob("*/history.json")):
        dir_name = h_path.parent.name
        try:
            scheme, fold_id, control = _parse_dir(dir_name)
        except ValueError:
            continue
        if scheme != eval_scheme or control != "none":
            continue
        histories[fold_id] = json.loads(h_path.read_text())
    return histories


def plot_fig2_training_curves(
    eval_scheme: str,
    out_dir: Path,
    out_root: Path = _OUT_ROOT,
):
    histories = _load_histories(eval_scheme, out_root)
    if not histories:
        print(f"  [skip] No history.json found for {eval_scheme}")
        return

    stages = ["stage1", "stage2"]
    stage_labels = {"stage1": "Stage 1 (InfoNCE)", "stage2": "Stage 2 (KL)"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        f"Training curves — {SCHEME_LABELS.get(eval_scheme, eval_scheme)} "
        f"(real condition, twostage)",
        fontsize=12, fontweight="bold",
    )

    for ax, stage in zip(axes, stages):
        train_curves, val_curves = [], []

        for fold_id, hist in histories.items():
            if stage not in hist:
                continue
            train_curves.append(hist[stage]["train_loss"])
            val_curves.append(hist[stage]["val_loss"])

        if not train_curves:
            continue

        # Pad to common length (early stopping → variable lengths)
        max_len = max(len(c) for c in train_curves)
        def _pad(curves):
            return np.array([
                c + [c[-1]] * (max_len - len(c)) for c in curves
            ])

        train_arr = _pad(train_curves)   # (n_folds, max_epochs)
        val_arr   = _pad(val_curves)
        epochs    = np.arange(1, max_len + 1)

        # Individual folds (light)
        for row in train_arr:
            ax.plot(epochs, row, color="#1f77b4", alpha=0.15, lw=0.8)
        for row in val_arr:
            ax.plot(epochs, row, color="#ff7f0e", alpha=0.15, lw=0.8)

        # Mean curves (bold)
        ax.plot(epochs, train_arr.mean(axis=0), color="#1f77b4",
                lw=2.0, label="Train (mean)")
        ax.plot(epochs, val_arr.mean(axis=0),   color="#ff7f0e",
                lw=2.0, label="Val (mean)")

        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax.set_title(stage_labels.get(stage, stage), fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"fig2_training_curves_{eval_scheme}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved {path}")
    plt.close(fig)
