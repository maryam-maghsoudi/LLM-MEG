"""
plot_beam_k_overlay.py

For each beam width B, overlay the subject-averaged alpha-sweep curves for every
top_k on a single axis. Larger k is drawn darker (grayscale) AND thicker, so the
effect of increasing top_k is readable at a glance.

One figure per B (4 total), for the listened OR imagined grid (via --mapping_key).
Missing (B, k) cells are skipped; each curve is the mean over whatever subjects
are available, so partial grids still plot.

Reuses SUBJECTS / K_VALUES / B_VALUES / load_curves from plot_beam_grid.py so the
two scripts read the grid identically.

Usage (run from inside contrastive_multimodal_v2/):
    # listened
    python plot_beam_k_overlay.py --fusion_results_dir fusion_grid
    # imagined
    python plot_beam_k_overlay.py \
        --fusion_results_dir imagined_eval/fusion_grid_imagined --mapping_key RNN_full
    # bleu1 instead of word_acc
    python plot_beam_k_overlay.py --fusion_results_dir fusion_grid --metric bleu1
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from plot_beam_grid import B_VALUES, K_VALUES, SUBJECTS, load_curves


def _k_style(rank: int, n_k: int):
    """Grayscale shade + linewidth for the k at position `rank` (0=smallest k)."""
    frac = rank / (n_k - 1) if n_k > 1 else 1.0
    shade = 0.78 * (1.0 - frac)       # small k -> light grey (0.78), large k -> black (0.0)
    lw = 0.9 + 2.2 * frac             # small k -> thin (0.9), large k -> thick (3.1)
    return str(shade), lw


def plot_one_B(fusion_dir: Path, B: int, llm_tag: str, norm: str, metric: str,
               mapping_key: Optional[str], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    metric_label = {"bleu1": "BLEU-1", "word_acc": "Word accuracy"}.get(metric, metric)
    n_k = len(K_VALUES)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    legend_handles = []
    any_data = False

    for rank, k in enumerate(K_VALUES):
        alphas, curves, present_idx = load_curves(
            fusion_dir, B, k, llm_tag, norm, metric, mapping_key,
        )
        shade, lw = _k_style(rank, n_k)
        if alphas is None:
            # still add a legend entry so the k -> shade key is complete
            legend_handles.append(
                Line2D([0], [0], color=shade, lw=lw, label=f"k={k} (n=0)"))
            continue

        any_data = True
        mean = curves.mean(axis=0) * 100.0
        n_present = curves.shape[0]
        ax.plot(alphas, mean, color=shade, linewidth=lw, zorder=2 + rank)
        # mark the best-alpha point in the same shade
        best_idx = int(np.argmax(mean))
        ax.scatter([alphas[best_idx]], [mean[best_idx]], color=shade,
                   s=18, zorder=20, edgecolors="crimson", linewidths=0.6)
        label = f"k={k}" if n_present == len(SUBJECTS) else f"k={k} (n={n_present})"
        legend_handles.append(Line2D([0], [0], color=shade, lw=lw, label=label))

    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel(r"$\alpha$  (0 = MEG only, 1 = LLM only)", fontsize=11)
    ax.set_ylabel(f"{metric_label} (%)  — mean over subjects", fontsize=11)

    cond = f"  |  mapping: {mapping_key}" if mapping_key else ""
    ax.set_title(
        f"Beam-search fusion, B={B} — {metric_label} vs "
        r"$\alpha$" + " across top_k\n"
        f"LLM: {llm_tag}  norm: {norm}{cond}  (darker + thicker = larger k, "
        "dot = best "r"$\alpha$)",
        fontsize=10,
    )
    ax.legend(handles=legend_handles, title="top_k", fontsize=8,
              title_fontsize=9, loc="best", frameon=True, ncol=2)

    if not any_data:
        ax.text(0.5, 0.5, f"no data for B={B}", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved B={B} overlay -> {out_path}")


def main(args):
    fusion_dir = Path(args.fusion_results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else fusion_dir / "figures"
    llm_tag = args.llm_name.replace("/", "_")
    mapping_key = args.mapping_key or None

    for B in B_VALUES:
        stem = f"beam_koverlay_B{B}_{llm_tag}_{args.norm}_{args.metric}"
        if mapping_key:
            stem += f"_{mapping_key}"
        plot_one_B(fusion_dir, B, llm_tag, args.norm, args.metric, mapping_key,
                   out_dir / f"{stem}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Per-B overlay of subject-averaged alpha curves across top_k "
                    "(grayscale + linewidth encode k). Listened or imagined."
    )
    p.add_argument("--fusion_results_dir", type=str, default="fusion_grid",
                   help="Directory with the beam fusion JSON files "
                        "(fusion_grid, or imagined_eval/fusion_grid_imagined).")
    p.add_argument("--mapping_key", type=str, default=None,
                   help="Imagined mapping key (e.g. RNN_full). Omit for listened.")
    p.add_argument("--metric", type=str, default="word_acc",
                   choices=["word_acc", "bleu1"])
    p.add_argument("--llm_name", type=str, default="gpt2")
    p.add_argument("--norm", type=str, default="row_zscore",
                   choices=["row_zscore", "logsoftmax"])
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output dir for the per-B PNGs "
                        "(default: <fusion_results_dir>/figures).")
    main(p.parse_args())
