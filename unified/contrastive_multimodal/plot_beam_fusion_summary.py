"""
plot_beam_fusion_summary.py

Aggregate fusion results across all heldout subjects and plot BLEU-1 vs alpha
and word_acc vs alpha as two separate figures. Works for both fusion modes:

  --mode beam           beam-search fusion (fusion_beamsearch.py), files like
                        fusion_results/sub-03_beamfusion_B5_top5_gpt2_row_zscore.json
  --mode teacher_forced teacher-forced fusion (fusion_teacher_forced.py), files like
                        fusion_results/sub-03_fusion_gpt2_row_zscore.json

Each JSON stores  data["results"][str(alpha)] = {"bleu1": .., "word_acc": ..}.

Each figure shows one thin line per subject plus a thick mean line with a
shaded error band (SEM by default, --band std for standard deviation).

Usage (run from inside contrastive_multimodal/):
    python plot_beam_fusion_summary.py                       # beam (default)
    python plot_beam_fusion_summary.py --mode teacher_forced
    python plot_beam_fusion_summary.py --band std
    python plot_beam_fusion_summary.py --mode beam --beam_width 5 --top_k 5 \
        --llm_name gpt2 --normalization row_zscore
"""

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np


def _glob_pattern(mode, beam_width, top_k, llm_tag, normalization):
    """Filename glob for the requested fusion mode."""
    if mode == "beam":
        return f"sub-*_beamfusion_B{beam_width}_top{top_k}_{llm_tag}_{normalization}.json"
    elif mode == "teacher_forced":
        return f"sub-*_fusion_{llm_tag}_{normalization}.json"
    raise ValueError(f"unknown mode: {mode!r}")


def load_subject_curves(
    fusion_results_dir: str,
    mode: str,
    beam_width: int,
    top_k: int,
    llm_tag: str,
    normalization: str,
    metric: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Glob all matching fusion JSONs and extract one metric curve per subject.

    Returns
    -------
    alphas         : np.ndarray (n_alphas,)  — shared alpha grid (sorted)
    subject_curves : dict subject -> np.ndarray (n_alphas,)  values in [0, 1]
    """
    pattern = os.path.join(
        fusion_results_dir,
        _glob_pattern(mode, beam_width, top_k, llm_tag, normalization),
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")

    subject_curves: Dict[str, np.ndarray] = {}
    ref_alphas: List[float] = None

    for path in paths:
        m = re.match(r"(sub-\d+)_", os.path.basename(path))
        subject = m.group(1) if m else os.path.basename(path)

        with open(path) as f:
            data = json.load(f)

        # Sort alphas numerically (JSON keys are strings)
        items = sorted(data["results"].items(), key=lambda kv: float(kv[0]))
        alphas = [float(a) for a, _ in items]
        values = [vals[metric] for _, vals in items]

        if ref_alphas is None:
            ref_alphas = alphas
        elif alphas != ref_alphas:
            print(f"  [warn] {subject}: alpha grid differs from the first file — "
                  f"skipping (has {len(alphas)} alphas, expected {len(ref_alphas)}).")
            continue

        subject_curves[subject] = np.array(values, dtype=float)

    return np.array(ref_alphas, dtype=float), subject_curves


def plot_metric(
    alphas: np.ndarray,
    subject_curves: Dict[str, np.ndarray],
    metric: str,
    band: str,
    out_path: str,
    mode: str,
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
        # Min-max normalize each subject's curve to [0, 1] so that the SHAPE
        # (where each subject's optimum alpha sits) is comparable across
        # subjects that sit at different absolute levels. Flat curves (max==min)
        # map to all-zeros.
        rows = []
        for s in subjects:
            c = subject_curves[s].astype(float)
            span = c.max() - c.min()
            rows.append((c - c.min()) / span if span > 0 else np.zeros_like(c))
        curves = np.stack(rows, axis=0)          # (n_subj, n_alpha), unitless [0,1]
    else:
        curves = np.stack([subject_curves[s] for s in subjects], axis=0) * 100.0  # (n_subj, n_alpha), %

    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros_like(mean_curve)
    if band == "sem":
        band_curve = std_curve / np.sqrt(curves.shape[0])
        band_label = "SEM"
    else:
        band_curve = std_curve
        band_label = "STD"

    best_idx = int(np.argmax(mean_curve))

    fig, ax = plt.subplots(figsize=(9, 6))

    # Per-subject thin lines
    cmap = plt.get_cmap("tab20")
    for i, s in enumerate(subjects):
        ax.plot(alphas, curves[i], color=cmap(i % 20), alpha=0.45,
                linewidth=1.0, label=s)

    # Mean + shaded band
    ax.plot(alphas, mean_curve, color="black", linewidth=2.8,
            label=f"Mean (n={len(subjects)})", zorder=6)
    ax.fill_between(alphas, mean_curve - band_curve, mean_curve + band_curve,
                    color="black", alpha=0.15, zorder=5,
                    label=f"±{band_label}")

    unit = "" if normalize_per_subject else "%"
    best_val = f"{mean_curve[best_idx]:.2f}" if normalize_per_subject else f"{mean_curve[best_idx]:.1f}%"

    # Best-alpha marker on the mean
    ax.scatter([alphas[best_idx]], [mean_curve[best_idx]], color="crimson",
               zorder=7, s=70,
               label=f"Best α={alphas[best_idx]:.2f} → {best_val}")

    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ymin, ymax = ax.get_ylim()
    ytxt = ymin + 0.03 * (ymax - ymin)
    ax.text(0.01, ytxt, "MEG only", fontsize=8, color="gray")
    ax.text(0.90, ytxt, "LLM only", fontsize=8, color="gray")

    metric_label = {"bleu1": "BLEU-1", "word_acc": "Word accuracy"}.get(metric, metric)
    mode_label = "Beam-search fusion" if mode == "beam" else "Teacher-forced fusion"
    cfg = (f"B={beam_width}  top_k={top_k}  " if mode == "beam" else "")
    norm_tag = "  [per-subject min-max normalized]" if normalize_per_subject else ""
    ylabel = (f"{metric_label} (normalized 0–1)" if normalize_per_subject
              else f"{metric_label} (%)")
    ax.set_xlabel("Alpha (LLM weight)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(
        f"{mode_label} — {metric_label} vs alpha{norm_tag}\n"
        f"{cfg}LLM: {llm_name}  norm: {normalization}  "
        f"(band: ±{band_label})",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}  (mean best α={alphas[best_idx]:.2f}, {mean_curve[best_idx]:.2f}{unit})")


def main(args):
    llm_tag = args.llm_name.replace("/", "_")

    if args.mode == "beam":
        prefix = f"beam_summary_B{args.beam_width}_top{args.top_k}_{llm_tag}"
    else:
        prefix = f"tf_summary_{llm_tag}"

    for metric in ("bleu1", "word_acc"):
        alphas, subject_curves = load_subject_curves(
            args.fusion_results_dir, args.mode, args.beam_width, args.top_k,
            llm_tag, args.normalization, metric,
        )
        print(f"[{metric}] loaded {len(subject_curves)} subjects, "
              f"{len(alphas)} alpha values")

        out_path = os.path.join(
            args.out_dir,
            f"{prefix}_{args.normalization}_{metric}_{args.band}.png",
        )
        plot_metric(
            alphas, subject_curves, metric, args.band, out_path,
            args.mode, args.llm_name, args.normalization, args.beam_width, args.top_k,
        )

        # Beam mode: an additional per-subject-normalized word_acc figure so
        # curves at different absolute levels can be compared by shape.
        if args.mode == "beam" and metric == "word_acc":
            norm_out = os.path.join(
                args.out_dir,
                f"{prefix}_{args.normalization}_{metric}_normalized_{args.band}.png",
            )
            plot_metric(
                alphas, subject_curves, metric, args.band, norm_out,
                args.mode, args.llm_name, args.normalization, args.beam_width, args.top_k,
                normalize_per_subject=True,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Aggregate + plot fusion BLEU-1 and word_acc vs alpha "
                    "across all heldout subjects (beam or teacher-forced)."
    )
    p.add_argument("--mode", type=str, default="beam",
                   choices=["beam", "teacher_forced"],
                   help="beam = fusion_beamsearch.py outputs; "
                        "teacher_forced = fusion_teacher_forced.py outputs.")
    p.add_argument("--fusion_results_dir", type=str, default="fusion_results",
                   help="Directory containing the fusion JSON files.")
    p.add_argument("--beam_width", type=int, default=5)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--llm_name", type=str, default="gpt2")
    p.add_argument("--normalization", type=str, default="row_zscore",
                   choices=["logsoftmax", "row_zscore"])
    p.add_argument("--band", type=str, default="sem", choices=["sem", "std"],
                   help="Shaded error band around the mean: SEM (default) or STD.")
    p.add_argument("--out_dir", type=str, default="fusion_results/figures",
                   help="Output directory for the two summary PNGs.")
    main(p.parse_args())
