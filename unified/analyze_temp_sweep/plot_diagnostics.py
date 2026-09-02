#!/usr/bin/env python3
"""
plot_diagnostics.py — Visualize embedding diagnostics across the temperature sweep.

Reads sweep_diagnostics.json produced by collect_diagnostics.py and generates:

  figures/diagnostics_summary.png
      6-panel figure, one panel per metric.
      Each panel: x = temperature, 3 bands:
        - seen_single (blue)   : one training subject per checkpoint
        - seen_avg   (green)   : mean over all 12 training subjects
        - unseen     (red)     : the LOSO heldout subject
      Bold line = mean across 13 LOSO folds; shading = ±1 std.
      Thin coloured lines = individual LOSO folds.

  figures/diagnostics_by_subject_{condition}.png  (3 files)
      Same layout, but shows every subject as a labelled thin line,
      separately for each condition (unseen / seen_single / seen_avg).

Run from llm_decoder/:
    python -m unified.analyze_temp_sweep.plot_diagnostics
    python -m unified.analyze_temp_sweep.plot_diagnostics --diag_path unified/analyze_temp_sweep/sweep_diagnostics.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

_HERE = Path(__file__).resolve().parent

# ─── Metric display metadata ──────────────────────────────────────────────────

METRICS = [
    ("meg_effective_rank",    "MEG effective rank",      "↑ good (less collapse)"),
    ("text_effective_rank",   "Text effective rank",     "informational"),
    ("meg_pairwise_cos_mean", "MEG pairwise cos (mean)", "↓ good (less collapse)"),
    ("intra_trial_cos_mean",  "Intra-trial cos (mean)",  "↓ good (less collapse)"),
    ("nn_exact_match",        "NN exact match (R@1)",    "↑ good"),
    ("nn_purity_top5",        "NN top-5 purity",         "↓ good (less freq-bias)"),
]

CONDITIONS = [
    ("seen_single", "Seen (single)",  "#2196F3"),   # blue
    ("seen_avg",    "Seen (avg×12)",  "#4CAF50"),   # green
    ("unseen",      "Unseen (LOSO)",  "#F44336"),   # red
]


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot temperature sweep diagnostics")
    p.add_argument("--diag_path",
                   default=str(_HERE / "sweep_diagnostics.json"),
                   help="Path to sweep_diagnostics.json")
    p.add_argument("--out_dir",
                   default=str(_HERE / "figures"),
                   help="Output directory for PNG figures")
    return p.parse_args()


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(path: Path):
    d        = json.loads(path.read_text())
    temps    = sorted(set(e["temperature"] for e in d["entries"]))
    subjects = d["subjects"]
    entries  = d["entries"]
    return temps, subjects, entries


def pivot(entries, temps, subjects, condition, metric) -> np.ndarray:
    """
    Return array (n_subjects, n_temps) of metric values for a given condition.
    NaN where data is missing.
    """
    temp_idx = {t: i for i, t in enumerate(temps)}
    sub_idx  = {s: i for i, s in enumerate(subjects)}
    arr = np.full((len(subjects), len(temps)), np.nan)

    for e in entries:
        ti = temp_idx.get(e["temperature"])
        si = sub_idx.get(e["heldout"])
        if ti is None or si is None:
            continue
        cond_data = e.get(condition)
        if cond_data is None:
            continue
        val = cond_data.get(metric, np.nan)
        if val is not None:
            arr[si, ti] = val
    return arr


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def _plot_band_and_lines(
    ax, temps, arr, color, label,
    alpha_band=0.15, alpha_line=0.25, lw_mean=2.5,
):
    """
    arr: (n_subjects, n_temps)
    Plots mean±std band + individual subject thin lines.
    """
    n_subs = arr.shape[0]
    sub_colors = cm.tab20(np.linspace(0, 1, n_subs))

    # Individual subject lines
    for i in range(n_subs):
        row = arr[i]
        mask = ~np.isnan(row)
        if mask.sum() > 1:
            ax.plot(np.array(temps)[mask], row[mask],
                    color=color, lw=0.8, alpha=alpha_line)

    # Mean ± std band
    mean = np.nanmean(arr, axis=0)
    std  = np.nanstd(arr,  axis=0)
    valid = ~np.isnan(mean)
    ax.fill_between(np.array(temps)[valid],
                    (mean - std)[valid], (mean + std)[valid],
                    color=color, alpha=alpha_band)
    ax.plot(np.array(temps)[valid], mean[valid],
            color=color, lw=lw_mean, label=label, zorder=4)


def _plot_subject_lines(
    ax, temps, arr, subjects, condition_label, color,
):
    """
    arr: (n_subjects, n_temps)
    Plots individually labelled lines per subject.
    """
    n_subs = arr.shape[0]
    colors = cm.tab20(np.linspace(0, 1, n_subs))
    for i, sub in enumerate(subjects):
        row = arr[i]
        mask = ~np.isnan(row)
        if mask.sum() < 2:
            continue
        ax.plot(np.array(temps)[mask], row[mask],
                color=colors[i], lw=1.5, label=sub, marker=".", markersize=5)
    ax.set_title(condition_label, fontsize=11)


# ─── Figure 1: summary (all conditions, mean±std) ────────────────────────────

def plot_summary(temps, subjects, entries, out_path: Path):
    n_metrics = len(METRICS)
    n_cols    = 3
    n_rows    = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

    for idx, (metric, title, note) in enumerate(METRICS):
        ax = axes[idx // n_cols][idx % n_cols]

        for cond_key, cond_label, color in CONDITIONS:
            arr = pivot(entries, temps, subjects, cond_key, metric)
            _plot_band_and_lines(ax, temps, arr, color, cond_label)

        ax.set_title(f"{title}\n({note})", fontsize=10)
        ax.set_xlabel("Temperature")
        ax.set_xticks(temps)
        ax.set_xticklabels([str(t) for t in temps], rotation=45, fontsize=8)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=9, loc="best", framealpha=0.8)

    # Hide unused axes
    for idx in range(n_metrics, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(
        "Embedding diagnostics vs InfoNCE temperature\n"
        "Bold line = mean ± std across 13 LOSO folds; thin = individual folds",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ─── Figure 2: per-subject lines, one figure per condition ───────────────────

def plot_by_subject(temps, subjects, entries, condition_key, condition_label,
                    color, out_path: Path):
    n_metrics = len(METRICS)
    n_cols    = 3
    n_rows    = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

    for idx, (metric, title, note) in enumerate(METRICS):
        ax = axes[idx // n_cols][idx % n_cols]
        arr = pivot(entries, temps, subjects, condition_key, metric)

        # Mean band in the background
        mean  = np.nanmean(arr, axis=0)
        std   = np.nanstd(arr,  axis=0)
        valid = ~np.isnan(mean)
        ax.fill_between(np.array(temps)[valid],
                        (mean - std)[valid], (mean + std)[valid],
                        color=color, alpha=0.12)
        ax.plot(np.array(temps)[valid], mean[valid],
                color=color, lw=2.5, zorder=4, label="mean")

        # Individual subject lines
        _plot_subject_lines(ax, temps, arr, subjects, "", color)

        ax.set_title(f"{title}\n({note})", fontsize=10)
        ax.set_xlabel("Temperature")
        ax.set_xticks(temps)
        ax.set_xticklabels([str(t) for t in temps], rotation=45, fontsize=8)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=6, ncol=2, loc="best", framealpha=0.7)

    for idx in range(n_metrics, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(
        f"Diagnostics vs Temperature — {condition_label}\n"
        "Each line = one LOSO heldout subject; bold = mean",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ─── Figure 3: seen_single vs seen_avg comparison (key metrics) ──────────────

def plot_single_vs_avg(temps, subjects, entries, out_path: Path):
    """Side-by-side: seen_single (solid) vs seen_avg (dashed) per metric."""
    key_metrics = [
        ("meg_effective_rank",    "MEG effective rank",  "↑ good"),
        ("meg_pairwise_cos_mean", "MEG pairwise cos",    "↓ good"),
        ("nn_exact_match",        "NN exact match (R@1)","↑ good"),
        ("nn_purity_top5",        "NN top-5 purity",     "↓ good"),
    ]
    fig, axes = plt.subplots(1, len(key_metrics),
                             figsize=(5 * len(key_metrics), 4.5), squeeze=False)

    for idx, (metric, title, note) in enumerate(key_metrics):
        ax = axes[0][idx]
        for cond_key, ls, lbl in [("seen_single", "-", "single"),
                                   ("seen_avg",    "--", "avg×12")]:
            arr  = pivot(entries, temps, subjects, cond_key, metric)
            mean = np.nanmean(arr, axis=0)
            std  = np.nanstd(arr,  axis=0)
            valid = ~np.isnan(mean)
            c = "#2196F3" if "single" in cond_key else "#4CAF50"
            ax.fill_between(np.array(temps)[valid],
                            (mean - std)[valid], (mean + std)[valid],
                            color=c, alpha=0.15)
            ax.plot(np.array(temps)[valid], mean[valid],
                    color=c, lw=2.5, ls=ls, label=f"seen {lbl}", zorder=4)

        ax.set_title(f"{title}\n({note})", fontsize=10)
        ax.set_xlabel("Temperature")
        ax.set_xticks(temps)
        ax.set_xticklabels([str(t) for t in temps], rotation=45, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Seen single vs seen avg×12 — mean ± std across LOSO folds",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    diag_p  = Path(args.diag_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not diag_p.exists():
        raise FileNotFoundError(f"sweep_diagnostics.json not found at {diag_p}\n"
                                f"Run collect_diagnostics.py first.")

    print(f"Loading {diag_p} ...")
    temps, subjects, entries = load_data(diag_p)
    print(f"  {len(temps)} temperatures: {temps}")
    print(f"  {len(subjects)} subjects")
    print(f"  {len(entries)} entries\n")

    print("Generating figures ...")
    plot_summary(temps, subjects, entries,
                 out_dir / "diagnostics_summary.png")

    for cond_key, cond_label, color in CONDITIONS:
        safe = cond_key.replace(" ", "_")
        plot_by_subject(temps, subjects, entries, cond_key, cond_label, color,
                        out_dir / f"diagnostics_by_subject_{safe}.png")

    plot_single_vs_avg(temps, subjects, entries,
                       out_dir / "diagnostics_single_vs_avg.png")

    print(f"\nDone. Figures → {out_dir}/")


if __name__ == "__main__":
    main()
