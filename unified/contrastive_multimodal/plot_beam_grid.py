"""
plot_beam_grid.py

Visualise the beam-search fusion hyperparameter grid sweep
(top_k × beam_width × subject × alpha).

Two figures are produced:

  1. Grid of alpha-sweep curves  (N_B rows × N_k cols)
     Each cell shows word_acc vs alpha for every subject (thin coloured lines)
     plus the mean ± SEM (thick black line + shaded band). No legend.

  2. Heatmap of best alpha
     Rows = beam_width, cols = top_k.  Cell value = alpha that maximises the
     mean word_acc curve for that (B, k) pair.

Usage (run from inside contrastive_multimodal/):
    python plot_beam_grid.py
    python plot_beam_grid.py --metric bleu1
    python plot_beam_grid.py --norm logsoftmax --out_dir some/other/dir
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]
K_VALUES = [1, 3, 5, 7, 10, 15, 20, 25, 30]
B_VALUES = [3, 5, 7, 9]


def load_curves(
    fusion_dir: Path,
    B: int,
    k: int,
    llm_tag: str,
    norm: str,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load metric curves for all subjects at a given (B, k).

    Returns
    -------
    alphas  : (n_alphas,)
    curves  : (n_subjects, n_alphas)  — values in [0, 1]
    """
    curves = []
    ref_alphas = None
    for subj in SUBJECTS:
        path = fusion_dir / f"{subj}_beamfusion_B{B}_top{k}_{llm_tag}_{norm}.json"
        with open(path) as f:
            data = json.load(f)
        items = sorted(data["results"].items(), key=lambda kv: float(kv[0]))
        alphas = np.array([float(a) for a, _ in items])
        vals   = np.array([v[metric] for _, v in items])
        if ref_alphas is None:
            ref_alphas = alphas
        curves.append(vals)
    return ref_alphas, np.stack(curves, axis=0)   # (S, A)


def plot_grid(fusion_dir: Path, llm_tag: str, norm: str, metric: str,
              out_path: Path):
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

    # Collect global y range for shared y-axis
    all_means = []

    # First pass — load everything and draw
    for row, B in enumerate(B_VALUES):
        for col, k in enumerate(K_VALUES):
            ax = axes[row][col]
            alphas, curves = load_curves(fusion_dir, B, k, llm_tag, norm, metric)
            pct = curves * 100.0        # → %

            mean = pct.mean(axis=0)
            sem  = pct.std(axis=0, ddof=1) / np.sqrt(len(SUBJECTS))
            best_idx = int(np.argmax(mean))
            all_means.append(mean)

            for i in range(len(SUBJECTS)):
                ax.plot(alphas, pct[i], color=cmap(i % 20),
                        alpha=0.35, linewidth=0.7)

            ax.plot(alphas, mean, color="black", linewidth=1.8, zorder=5)
            ax.fill_between(alphas, mean - sem, mean + sem,
                            color="black", alpha=0.18, zorder=4)
            ax.scatter([alphas[best_idx]], [mean[best_idx]],
                       color="crimson", s=25, zorder=6)

            ax.axvline(0.0, color="gray", linestyle="--",
                       linewidth=0.5, alpha=0.5)
            ax.axvline(1.0, color="gray", linestyle="--",
                       linewidth=0.5, alpha=0.5)
            ax.set_xlim(-0.03, 1.03)
            ax.tick_params(labelsize=6)

            # Cell annotation
            ax.set_title(f"B={B}  k={k}", fontsize=7, pad=2)

    # Shared axis labels — row and col headers
    for col, k in enumerate(K_VALUES):
        axes[0][col].set_xlabel("")      # top row gets title already
    for row, B in enumerate(B_VALUES):
        axes[row][0].set_ylabel(f"{metric_label} (%)", fontsize=7)

    # Bottom row x-labels
    for col in range(n_k):
        axes[-1][col].set_xlabel("α", fontsize=7)

    fig.suptitle(
        f"Beam-search fusion grid — {metric_label} vs α\n"
        f"LLM: {llm_tag}  norm: {norm}  "
        f"(thin=subjects, thick=mean±SEM, ●=best α)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved grid figure → {out_path}")


def plot_heatmap(fusion_dir: Path, llm_tag: str, norm: str, metric: str,
                 out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_alpha = np.zeros((len(B_VALUES), len(K_VALUES)))
    best_val   = np.zeros((len(B_VALUES), len(K_VALUES)))

    for row, B in enumerate(B_VALUES):
        for col, k in enumerate(K_VALUES):
            alphas, curves = load_curves(fusion_dir, B, k, llm_tag, norm, metric)
            mean = curves.mean(axis=0)
            best_idx = int(np.argmax(mean))
            best_alpha[row, col] = alphas[best_idx]
            best_val[row, col]   = mean[best_idx] * 100.0

    metric_label = {"bleu1": "BLEU-1", "word_acc": "Word accuracy"}.get(metric, metric)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # --- left: best alpha heatmap ---
    ax = axes[0]
    im = ax.imshow(best_alpha, aspect="auto", cmap="RdYlGn_r",
                   vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels(K_VALUES)
    ax.set_yticks(range(len(B_VALUES)))
    ax.set_yticklabels(B_VALUES)
    ax.set_xlabel("top_k", fontsize=11)
    ax.set_ylabel("beam_width B", fontsize=11)
    ax.set_title(f"Best α (argmax mean {metric_label})", fontsize=11)
    for row in range(len(B_VALUES)):
        for col in range(len(K_VALUES)):
            ax.text(col, row, f"{best_alpha[row, col]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black")
    plt.colorbar(im, ax=ax, label="Best α")

    # --- right: best value heatmap ---
    ax = axes[1]
    vmin = best_val.min()
    vmax = best_val.max()
    im2 = ax.imshow(best_val, aspect="auto", cmap="YlGn",
                    vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels(K_VALUES)
    ax.set_yticks(range(len(B_VALUES)))
    ax.set_yticklabels(B_VALUES)
    ax.set_xlabel("top_k", fontsize=11)
    ax.set_ylabel("beam_width B", fontsize=11)
    ax.set_title(f"Mean {metric_label} at best α (%)", fontsize=11)
    for row in range(len(B_VALUES)):
        for col in range(len(K_VALUES)):
            ax.text(col, row, f"{best_val[row, col]:.1f}%",
                    ha="center", va="center", fontsize=8,
                    color="black")
    plt.colorbar(im2, ax=ax, label=f"{metric_label} (%)")

    fig.suptitle(
        f"Beam-search grid — best α and {metric_label} | "
        f"LLM: {llm_tag}  norm: {norm}",
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

    stem = f"grid_{llm_tag}_{args.norm}_{args.metric}"
    plot_grid(
        fusion_dir, llm_tag, args.norm, args.metric,
        out_dir / f"{stem}_curves.png",
    )
    plot_heatmap(
        fusion_dir, llm_tag, args.norm, args.metric,
        out_dir / f"{stem}_heatmap.png",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Plot beam-search grid sweep: alpha-curve grid + best-alpha heatmap."
    )
    p.add_argument("--fusion_results_dir", type=str, default="fusion_results",
                   help="Directory containing the beam fusion JSON files.")
    p.add_argument("--metric", type=str, default="word_acc",
                   choices=["word_acc", "bleu1"])
    p.add_argument("--llm_name", type=str, default="gpt2")
    p.add_argument("--norm", type=str, default="row_zscore",
                   choices=["row_zscore", "logsoftmax"])
    p.add_argument("--out_dir", type=str, default="fusion_results/figures",
                   help="Output directory for the two PNGs.")
    main(p.parse_args())
