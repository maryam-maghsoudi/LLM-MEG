"""
plot_beam_fusion_summary_imagined.py

Aggregate beam-search fusion results for imagined MEG (imagined → predicted-
listened → decode) and plot BLEU-1 and word_acc vs alpha, one thin line per
subject plus a thick mean ± STD/SEM band.

JSON files are written by fusion_beamsearch_imagined.py and live under
    imagined_eval/beam_fusion_results/
with names like:
    sub-01_beamfusion_B5_top5_gpt2_row_zscore_RNN_full.json

Each JSON stores:
    data["results"][str(alpha)] = {"bleu1": .., "word_acc": .., "n_valid": .., "n_trials": ..}

Usage (run from inside contrastive_multimodal/):
    python imagined_eval/plot_beam_fusion_summary_imagined.py
    python imagined_eval/plot_beam_fusion_summary_imagined.py --mapping_key CNN1D_full
    python imagined_eval/plot_beam_fusion_summary_imagined.py --band std
    python imagined_eval/plot_beam_fusion_summary_imagined.py \\
        --beam_width 5 --top_k 5 --llm_name gpt2 --normalization row_zscore
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent


def _glob_pattern(beam_width: int, top_k: int, llm_tag: str, normalization: str,
                  mapping_key: str) -> str:
    return (f"sub-*_beamfusion_B{beam_width}_top{top_k}_"
            f"{llm_tag}_{normalization}_{mapping_key}.json")


def load_subject_curves(
    results_dir: str,
    beam_width: int,
    top_k: int,
    llm_tag: str,
    normalization: str,
    mapping_key: str,
    metric: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Glob all matching imagined-fusion JSONs and extract one metric curve per subject.

    Returns
    -------
    alphas         : np.ndarray (n_alphas,)  — shared alpha grid (sorted)
    subject_curves : dict subject -> np.ndarray (n_alphas,)  values in [0, 1]
    """
    pattern = os.path.join(
        results_dir,
        _glob_pattern(beam_width, top_k, llm_tag, normalization, mapping_key),
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")

    subject_curves: Dict[str, np.ndarray] = {}
    ref_alphas: Optional[List[float]] = None

    for path in paths:
        m = re.match(r"(sub-\d+)_", os.path.basename(path))
        subject = m.group(1) if m else os.path.basename(path)

        with open(path) as f:
            data = json.load(f)

        items = sorted(data["results"].items(), key=lambda kv: float(kv[0]))
        alphas = [float(a) for a, _ in items]
        values = [vals[metric] for _, vals in items]

        if ref_alphas is None:
            ref_alphas = alphas
        elif alphas != ref_alphas:
            print(f"  [warn] {subject}: alpha grid differs — skipping "
                  f"({len(alphas)} alphas vs expected {len(ref_alphas)}).")
            continue

        subject_curves[subject] = np.array(values, dtype=float)

    return np.array(ref_alphas, dtype=float), subject_curves


def plot_metric(
    alphas: np.ndarray,
    subject_curves: Dict[str, np.ndarray],
    metric: str,
    band: str,
    out_path: str,
    mapping_key: str,
    llm_name: str,
    normalization: str,
    beam_width: int,
    top_k: int,
    normalize_per_subject: bool = False,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subjects = sorted(subject_curves.keys())

    if normalize_per_subject:
        rows = []
        for s in subjects:
            c = subject_curves[s].astype(float)
            span = c.max() - c.min()
            rows.append((c - c.min()) / span if span > 0 else np.zeros_like(c))
        curves = np.stack(rows, axis=0)          # (n_subj, n_alpha), unitless [0,1]
    else:
        curves = np.stack([subject_curves[s] for s in subjects], axis=0) * 100.0  # %

    mean_curve = curves.mean(axis=0)
    std_curve = (curves.std(axis=0, ddof=1)
                 if curves.shape[0] > 1 else np.zeros_like(mean_curve))
    band_curve = std_curve / np.sqrt(curves.shape[0]) if band == "sem" else std_curve
    band_label = "SEM" if band == "sem" else "STD"

    best_idx = int(np.argmax(mean_curve))

    fig, ax = plt.subplots(figsize=(9, 6))

    cmap = plt.get_cmap("tab20")
    for i, s in enumerate(subjects):
        ax.plot(alphas, curves[i], color=cmap(i % 20), alpha=0.45,
                linewidth=1.0, label=s)

    ax.plot(alphas, mean_curve, color="black", linewidth=2.8,
            label=f"Mean (n={len(subjects)})", zorder=6)
    ax.fill_between(alphas, mean_curve - band_curve, mean_curve + band_curve,
                    color="black", alpha=0.15, zorder=5, label=f"±{band_label}")

    unit = "" if normalize_per_subject else "%"
    best_val = (f"{mean_curve[best_idx]:.2f}" if normalize_per_subject
                else f"{mean_curve[best_idx]:.1f}%")
    ax.scatter([alphas[best_idx]], [mean_curve[best_idx]], color="crimson",
               zorder=7, s=70,
               label=f"Best α={alphas[best_idx]:.2f} → {best_val}")

    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ymin, ymax = ax.get_ylim()
    ytxt = ymin + 0.03 * (ymax - ymin)
    ax.text(0.01, ytxt, "MEG only\n(imagined)", fontsize=8, color="gray")
    ax.text(0.88, ytxt, "LLM only", fontsize=8, color="gray")

    metric_label = {"bleu1": "BLEU-1", "word_acc": "Word accuracy"}.get(metric, metric)
    norm_tag = "  [per-subject min-max normalized]" if normalize_per_subject else ""
    ylabel = (f"{metric_label} (normalized 0–1)" if normalize_per_subject
              else f"{metric_label} (%)")
    ax.set_xlabel("Alpha (LLM weight)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(
        f"Imagined MEG beam-search fusion — {metric_label} vs alpha{norm_tag}\n"
        f"B={beam_width}  top_k={top_k}  mapping: {mapping_key}  "
        f"LLM: {llm_name}  norm: {normalization}  (band: ±{band_label})",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".",
                exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}  "
          f"(mean best α={alphas[best_idx]:.2f}, {mean_curve[best_idx]:.2f}{unit})")


def main(args):
    llm_tag = args.llm_name.replace("/", "_")
    results_dir = args.results_dir or str(
        _THIS_DIR / "beam_fusion_results"
    )
    prefix = (f"imagined_beam_summary_B{args.beam_width}_top{args.top_k}_"
              f"{llm_tag}_{args.mapping_key}")

    for metric in ("bleu1", "word_acc"):
        alphas, subject_curves = load_subject_curves(
            results_dir, args.beam_width, args.top_k,
            llm_tag, args.normalization, args.mapping_key, metric,
        )
        print(f"[{metric}] loaded {len(subject_curves)} subjects, "
              f"{len(alphas)} alpha values")

        out_path = os.path.join(
            args.out_dir,
            f"{prefix}_{args.normalization}_{metric}_{args.band}.png",
        )
        plot_metric(
            alphas, subject_curves, metric, args.band, out_path,
            args.mapping_key, args.llm_name, args.normalization,
            args.beam_width, args.top_k,
        )

        # Also produce a per-subject normalized word_acc figure so shape is
        # comparable across subjects sitting at different absolute levels.
        if metric == "word_acc":
            norm_out = os.path.join(
                args.out_dir,
                f"{prefix}_{args.normalization}_{metric}_normalized_{args.band}.png",
            )
            plot_metric(
                alphas, subject_curves, metric, args.band, norm_out,
                args.mapping_key, args.llm_name, args.normalization,
                args.beam_width, args.top_k, normalize_per_subject=True,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Aggregate + plot imagined-MEG beam-search fusion BLEU-1 "
                    "and word_acc vs alpha across all heldout subjects."
    )
    p.add_argument("--mapping_key", type=str, default="RNN_full",
                   help="img→lis mapping model key, e.g. RNN_full, CNN1D_full.")
    p.add_argument("--results_dir", type=str, default=None,
                   help="Directory containing the fusion JSON files. "
                        "Default: imagined_eval/beam_fusion_results/ relative to "
                        "contrastive_multimodal/.")
    p.add_argument("--beam_width", type=int, default=5)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--llm_name", type=str, default="gpt2")
    p.add_argument("--normalization", type=str, default="row_zscore",
                   choices=["logsoftmax", "row_zscore"])
    p.add_argument("--band", type=str, default="sem", choices=["sem", "std"],
                   help="Shaded error band: SEM (default) or STD.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory for PNGs. "
                        "Default: same as results_dir.")
    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = args.results_dir or str(_THIS_DIR / "beam_fusion_results")

    main(args)
