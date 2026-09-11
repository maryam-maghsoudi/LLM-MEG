"""
plot_beam_grid_imagined.py

Visualise the beam-search fusion hyperparameter grid sweep for IMAGINED MEG
(imagined → predicted-listened → decode via the Stage 1 contrastive decoder).

Analogue of ../plot_beam_grid.py, adapted for the imagined filename convention:
    {subj}_beamfusion_B{B}_top{k}_{llm_tag}_{norm}_{mapping_key}.json

Two figures are produced:

  1. Grid of alpha-sweep curves  (N_B rows × N_k cols)
     Each cell shows word_acc (or bleu1) vs alpha for every available subject
     (thin coloured lines) plus the mean ± SEM (thick black line + shaded band).
     Cells with fewer than all 13 subjects are annotated with the subject count.

  2. Heatmap of best alpha
     Rows = beam_width, cols = top_k.
     Left panel: alpha that maximises mean metric.
     Right panel: mean metric value at that alpha (%).
     Incomplete cells (< min_subjects) are greyed out.

Usage (run from inside contrastive_multimodal/):
    python imagined_eval/plot_beam_grid_imagined.py
    python imagined_eval/plot_beam_grid_imagined.py --metric bleu1
    python imagined_eval/plot_beam_grid_imagined.py --mapping_key CNN1D_full
    python imagined_eval/plot_beam_grid_imagined.py --norm row_zscore --out_dir some/dir
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Allow running from contrastive_multimodal/ as well as from imagined_eval/
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))

SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]
K_VALUES = [1, 3, 5, 7, 10, 15, 20, 25, 30]
B_VALUES = [3, 5, 7, 9]


def _json_path(fusion_dir: Path, subj: str, B: int, k: int,
               llm_tag: str, norm: str, mapping_key: str) -> Path:
    return fusion_dir / f"{subj}_beamfusion_B{B}_top{k}_{llm_tag}_{norm}_{mapping_key}.json"


def load_curves(
    fusion_dir: Path,
    B: int,
    k: int,
    llm_tag: str,
    norm: str,
    mapping_key: str,
    metric: str,
) -> Tuple[Optional[np.ndarray], np.ndarray, List[str]]:
    """
    Load metric curves for all available subjects at a given (B, k).

    Returns
    -------
    alphas        : (n_alphas,) or None if no files found
    curves        : (n_found_subjects, n_alphas)
    found_subjects: list of subject IDs that had a file
    """
    curves = []
    found = []
    ref_alphas = None
    for subj in SUBJECTS:
        path = _json_path(fusion_dir, subj, B, k, llm_tag, norm, mapping_key)
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        items = sorted(data["results"].items(), key=lambda kv: float(kv[0]))
        alphas = np.array([float(a) for a, _ in items])
        vals   = np.array([v[metric] for _, v in items])
        if ref_alphas is None:
            ref_alphas = alphas
        curves.append(vals)
        found.append(subj)
    if not curves:
        return None, np.empty((0,)), []
    return ref_alphas, np.stack(curves, axis=0), found


def plot_grid(
    fusion_dir: Path,
    llm_tag: str,
    norm: str,
    mapping_key: str,
    metric: str,
    out_path: Path,
    min_subjects: int = 1,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_B = len(B_VALUES)
    n_k = len(K_VALUES)
    metric_label = {"bleu1": "BLEU-1", "word_acc": "Word accuracy"}.get(metric, metric)

    fig, axes = plt.subplots(
        n_B, n_k,
        figsize=(2.8 * n_k, 2.6 * n_B),
        sharey=False, sharex=True,
    )

    cmap = plt.get_cmap("tab20")

    for row, B in enumerate(B_VALUES):
        for col, k in enumerate(K_VALUES):
            ax = axes[row][col]
            alphas, curves, found = load_curves(
                fusion_dir, B, k, llm_tag, norm, mapping_key, metric
            )

            if alphas is None or len(found) < min_subjects:
                ax.set_facecolor("#e8e8e8")
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="gray")
                ax.set_title(f"B={B}  k={k}", fontsize=7, pad=2)
                ax.tick_params(labelsize=6)
                continue

            pct = curves * 100.0

            mean = pct.mean(axis=0)
            sem  = pct.std(axis=0, ddof=1) / np.sqrt(len(found)) if len(found) > 1 else np.zeros_like(mean)
            best_idx = int(np.argmax(mean))

            for i, subj in enumerate(found):
                ax.plot(alphas, pct[i], color=cmap(SUBJECTS.index(subj) % 20),
                        alpha=0.35, linewidth=0.7)

            ax.plot(alphas, mean, color="black", linewidth=1.8, zorder=5)
            ax.fill_between(alphas, mean - sem, mean + sem,
                            color="black", alpha=0.18, zorder=4)
            ax.scatter([alphas[best_idx]], [mean[best_idx]],
                       color="crimson", s=25, zorder=6)

            ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_xlim(-0.03, 1.03)
            ax.tick_params(labelsize=6)

            title = f"B={B}  k={k}"
            if len(found) < len(SUBJECTS):
                title += f"  (n={len(found)})"
            ax.set_title(title, fontsize=7, pad=2)

    for row, B in enumerate(B_VALUES):
        axes[row][0].set_ylabel(f"{metric_label} (%)", fontsize=7)
    for col in range(n_k):
        axes[-1][col].set_xlabel("α", fontsize=7)

    fig.suptitle(
        f"Imagined MEG beam-search fusion grid — {metric_label} vs α\n"
        f"mapping: {mapping_key}  LLM: {llm_tag}  norm: {norm}  "
        f"(thin=subjects, thick=mean±SEM, ●=best α)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved grid figure → {out_path}")


def plot_heatmap(
    fusion_dir: Path,
    llm_tag: str,
    norm: str,
    mapping_key: str,
    metric: str,
    out_path: Path,
    min_subjects: int = 1,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_alpha = np.full((len(B_VALUES), len(K_VALUES)), np.nan)
    best_val   = np.full((len(B_VALUES), len(K_VALUES)), np.nan)
    n_subj_mat = np.zeros((len(B_VALUES), len(K_VALUES)), dtype=int)

    for row, B in enumerate(B_VALUES):
        for col, k in enumerate(K_VALUES):
            alphas, curves, found = load_curves(
                fusion_dir, B, k, llm_tag, norm, mapping_key, metric
            )
            if alphas is None or len(found) < min_subjects:
                continue
            mean = curves.mean(axis=0)
            best_idx = int(np.argmax(mean))
            best_alpha[row, col] = alphas[best_idx]
            best_val[row, col]   = mean[best_idx] * 100.0
            n_subj_mat[row, col] = len(found)

    metric_label = {"bleu1": "BLEU-1", "word_acc": "Word accuracy"}.get(metric, metric)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    def _draw(ax, data, title, cmap, vmin, vmax, fmt, cbar_label, mask):
        masked = np.where(mask, data, np.nan)
        im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(K_VALUES)))
        ax.set_xticklabels(K_VALUES)
        ax.set_yticks(range(len(B_VALUES)))
        ax.set_yticklabels(B_VALUES)
        ax.set_xlabel("top_k", fontsize=11)
        ax.set_ylabel("beam_width B", fontsize=11)
        ax.set_title(title, fontsize=11)
        for r in range(len(B_VALUES)):
            for c in range(len(K_VALUES)):
                if mask[r, c]:
                    n = n_subj_mat[r, c]
                    suffix = f"\n(n={n})" if n < len(SUBJECTS) else ""
                    ax.text(c, r, fmt.format(data[r, c]) + suffix,
                            ha="center", va="center", fontsize=7, color="black")
                else:
                    ax.text(c, r, "—", ha="center", va="center",
                            fontsize=9, color="gray")
        plt.colorbar(im, ax=ax, label=cbar_label)

    valid = ~np.isnan(best_alpha)

    _draw(axes[0], best_alpha,
          f"Best α (argmax mean {metric_label})",
          "RdYlGn_r", 0.0, 1.0, "{:.2f}", "Best α", valid)

    vmin = float(np.nanmin(best_val)) if valid.any() else 0.0
    vmax = float(np.nanmax(best_val)) if valid.any() else 1.0
    _draw(axes[1], best_val,
          f"Mean {metric_label} at best α (%)",
          "YlGn", vmin, vmax, "{:.1f}%", f"{metric_label} (%)", valid)

    fig.suptitle(
        f"Imagined MEG beam-search grid — best α and {metric_label} | "
        f"mapping: {mapping_key}  LLM: {llm_tag}  norm: {norm}",
        fontsize=11,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap figure → {out_path}")


def main(args):
    fusion_dir = Path(args.fusion_results_dir)
    out_dir    = Path(args.out_dir)
    llm_tag    = args.llm_name.replace("/", "_")

    stem = f"imagined_grid_{args.mapping_key}_{llm_tag}_{args.norm}_{args.metric}"
    plot_grid(
        fusion_dir, llm_tag, args.norm, args.mapping_key, args.metric,
        out_dir / f"{stem}_curves.png",
        min_subjects=args.min_subjects,
    )
    plot_heatmap(
        fusion_dir, llm_tag, args.norm, args.mapping_key, args.metric,
        out_dir / f"{stem}_heatmap.png",
        min_subjects=args.min_subjects,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Plot imagined MEG beam-search grid sweep: alpha-curve grid + best-alpha heatmap."
    )
    p.add_argument("--fusion_results_dir", type=str,
                   default="imagined_eval/beam_fusion_results",
                   help="Directory containing the imagined beam fusion JSON files.")
    p.add_argument("--mapping_key", type=str, default="RNN_full",
                   help="img→lis mapping architecture key (e.g. RNN_full, CNN1D_full).")
    p.add_argument("--metric", type=str, default="word_acc",
                   choices=["word_acc", "bleu1"])
    p.add_argument("--llm_name", type=str, default="gpt2")
    p.add_argument("--norm", type=str, default="row_zscore",
                   choices=["row_zscore", "logsoftmax"])
    p.add_argument("--out_dir", type=str,
                   default="imagined_eval/beam_fusion_results/figures",
                   help="Output directory for the two PNGs.")
    p.add_argument("--min_subjects", type=int, default=1,
                   help="Minimum subjects required to plot a cell (others shown as 'no data').")
    main(p.parse_args())
