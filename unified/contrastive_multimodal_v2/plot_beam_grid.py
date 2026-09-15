"""
plot_beam_grid.py

Visualise the beam-search fusion hyperparameter grid sweep
(top_k x beam_width x subject x alpha) for contrastive_multimodal_v2.

Handles BOTH the listened and imagined grids via --mapping_key:
  - listened : {subj}_beamfusion_B{B}_top{k}_{llm_tag}_{norm}.json
  - imagined : {subj}_beamfusion_B{B}_top{k}_{llm_tag}_{norm}_{mapping_key}.json

The v2 grids are usually still running, so any (B, k, subject) cell whose JSON
is missing is silently skipped: the mean/SEM in a cell is computed only over the
subjects that are present, and a fully-empty cell is left blank. Each cell is
annotated with how many subjects contributed (n=<present>/<total>).

Subject colours are fixed globally (sub-01 is always the same colour, whether or
not its neighbours are present), so curves are comparable across cells.

Two figures are produced:

  1. Grid of alpha-sweep curves  (N_B rows x N_k cols)
     Each cell: metric vs alpha for every present subject (thin coloured lines)
     plus mean +/- SEM (thick black line + shaded band) and the best-alpha dot.

  2. Heatmap of best alpha + best value
     Rows = beam_width, cols = top_k. Missing cells shown in grey.

Usage (run from inside contrastive_multimodal_v2/):
    # listened grid
    python plot_beam_grid.py --fusion_results_dir fusion_grid
    python plot_beam_grid.py --fusion_results_dir fusion_grid --metric bleu1

    # imagined grid
    python plot_beam_grid.py \
        --fusion_results_dir imagined_eval/fusion_grid_imagined \
        --mapping_key RNN_full
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]
K_VALUES = [1, 3, 5, 7, 10, 15, 20, 25, 30]
B_VALUES = [3, 5, 7, 9]


def _subject_colors():
    """Fixed colour per subject (index-stable across cells and runs)."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    return {subj: cmap(i % 20) for i, subj in enumerate(SUBJECTS)}


def _cell_filename(subj: str, B: int, k: int, llm_tag: str, norm: str,
                   mapping_key: Optional[str]) -> str:
    base = f"{subj}_beamfusion_B{B}_top{k}_{llm_tag}_{norm}"
    if mapping_key:
        base += f"_{mapping_key}"
    return base + ".json"


def load_curves(
    fusion_dir: Path,
    B: int,
    k: int,
    llm_tag: str,
    norm: str,
    metric: str,
    mapping_key: Optional[str],
) -> Tuple[Optional[np.ndarray], np.ndarray, List[int]]:
    """
    Load metric curves for all AVAILABLE subjects at a given (B, k).

    Missing / unreadable files are skipped rather than raising, so partially
    completed grids still plot.

    Returns
    -------
    alphas       : (n_alphas,) or None if no subject present
    curves       : (n_present, n_alphas)  — values in [0, 1]
    present_idx  : list of global SUBJECTS indices that contributed (same
                   order as rows of `curves`), used to colour consistently.
    """
    curves: List[np.ndarray] = []
    present_idx: List[int] = []
    ref_alphas: Optional[np.ndarray] = None

    for i, subj in enumerate(SUBJECTS):
        path = fusion_dir / _cell_filename(subj, B, k, llm_tag, norm, mapping_key)
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            items = sorted(data["results"].items(), key=lambda kv: float(kv[0]))
            alphas = np.array([float(a) for a, _ in items])
            vals = np.array([v[metric] for _, v in items], dtype=float)
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            # incomplete write or malformed file — skip this subject
            continue

        if ref_alphas is None:
            ref_alphas = alphas
        elif alphas.shape != ref_alphas.shape:
            # different alpha grid (shouldn't happen for finished cells) — skip
            continue

        curves.append(vals)
        present_idx.append(i)

    if not curves:
        return None, np.empty((0, 0)), []
    return ref_alphas, np.stack(curves, axis=0), present_idx


def plot_grid(fusion_dir: Path, llm_tag: str, norm: str, metric: str,
              mapping_key: Optional[str], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    subj_colors = _subject_colors()
    n_B = len(B_VALUES)
    n_k = len(K_VALUES)
    metric_label = {"bleu1": "BLEU-1", "word_acc": "Word accuracy"}.get(metric, metric)

    fig, axes = plt.subplots(
        n_B, n_k,
        figsize=(2.8 * n_k, 2.6 * n_B),
        sharey=False, sharex=True, squeeze=False,
    )

    for row, B in enumerate(B_VALUES):
        for col, k in enumerate(K_VALUES):
            ax = axes[row][col]
            alphas, curves, present_idx = load_curves(
                fusion_dir, B, k, llm_tag, norm, metric, mapping_key,
            )

            if alphas is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
                ax.set_title(f"B={B}  k={k}  (n=0/{len(SUBJECTS)})",
                             fontsize=7, pad=2)
                ax.set_xlim(-0.03, 1.03)
                ax.tick_params(labelsize=6)
                continue

            pct = curves * 100.0        # -> %
            n_present = pct.shape[0]

            # per-subject thin lines, coloured by their GLOBAL identity
            for row_i, gidx in enumerate(present_idx):
                ax.plot(alphas, pct[row_i], color=subj_colors[SUBJECTS[gidx]],
                        alpha=0.35, linewidth=0.7)

            mean = pct.mean(axis=0)
            ax.plot(alphas, mean, color="black", linewidth=1.8, zorder=5)
            if n_present >= 2:
                sem = pct.std(axis=0, ddof=1) / np.sqrt(n_present)
                ax.fill_between(alphas, mean - sem, mean + sem,
                                color="black", alpha=0.18, zorder=4)

            best_idx = int(np.argmax(mean))
            ax.scatter([alphas[best_idx]], [mean[best_idx]],
                       color="crimson", s=25, zorder=6)

            ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_xlim(-0.03, 1.03)
            ax.tick_params(labelsize=6)
            ax.set_title(f"B={B}  k={k}  (n={n_present}/{len(SUBJECTS)})",
                         fontsize=7, pad=2)

    for row in range(n_B):
        axes[row][0].set_ylabel(f"{metric_label} (%)", fontsize=7)
    for col in range(n_k):
        axes[-1][col].set_xlabel(r"$\alpha$", fontsize=7)

    # shared subject-colour legend (colours are consistent across every cell)
    handles = [Line2D([0], [0], color=subj_colors[s], lw=2, label=s)
               for s in SUBJECTS]
    fig.legend(handles=handles, loc="lower center", ncol=len(SUBJECTS),
               fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.015))

    cond = f"mapping: {mapping_key}  " if mapping_key else ""
    fig.suptitle(
        f"Beam-search fusion grid — {metric_label} vs "
        r"$\alpha$" + "\n"
        f"LLM: {llm_tag}  norm: {norm}  {cond}"
        f"(thin=subjects, thick=mean+/-SEM, dot=best alpha)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved grid figure -> {out_path}")


def plot_heatmap(fusion_dir: Path, llm_tag: str, norm: str, metric: str,
                 mapping_key: Optional[str], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_alpha = np.full((len(B_VALUES), len(K_VALUES)), np.nan)
    best_val = np.full((len(B_VALUES), len(K_VALUES)), np.nan)
    n_present = np.zeros((len(B_VALUES), len(K_VALUES)), dtype=int)

    for row, B in enumerate(B_VALUES):
        for col, k in enumerate(K_VALUES):
            alphas, curves, present_idx = load_curves(
                fusion_dir, B, k, llm_tag, norm, metric, mapping_key,
            )
            n_present[row, col] = len(present_idx)
            if alphas is None:
                continue
            mean = curves.mean(axis=0)
            best_idx = int(np.argmax(mean))
            best_alpha[row, col] = alphas[best_idx]
            best_val[row, col] = mean[best_idx] * 100.0

    metric_label = {"bleu1": "BLEU-1", "word_acc": "Word accuracy"}.get(metric, metric)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    def _style(ax):
        ax.set_xticks(range(len(K_VALUES)))
        ax.set_xticklabels(K_VALUES)
        ax.set_yticks(range(len(B_VALUES)))
        ax.set_yticklabels(B_VALUES)
        ax.set_xlabel("top_k", fontsize=11)
        ax.set_ylabel("beam_width B", fontsize=11)

    # --- left: best alpha heatmap ---
    ax = axes[0]
    cmap_a = plt.get_cmap("RdYlGn_r").copy()
    cmap_a.set_bad(color="lightgray")
    im = ax.imshow(np.ma.masked_invalid(best_alpha), aspect="auto",
                   cmap=cmap_a, vmin=0.0, vmax=1.0)
    _style(ax)
    ax.set_title(f"Best alpha (argmax mean {metric_label})", fontsize=11)
    for row in range(len(B_VALUES)):
        for col in range(len(K_VALUES)):
            txt = "—" if np.isnan(best_alpha[row, col]) else f"{best_alpha[row, col]:.2f}"
            ax.text(col, row, txt, ha="center", va="center", fontsize=8, color="black")
    plt.colorbar(im, ax=ax, label="Best alpha")

    # --- right: best value heatmap ---
    ax = axes[1]
    cmap_v = plt.get_cmap("YlGn").copy()
    cmap_v.set_bad(color="lightgray")
    if np.isnan(best_val).all():
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.nanmin(best_val), np.nanmax(best_val)
    im2 = ax.imshow(np.ma.masked_invalid(best_val), aspect="auto",
                    cmap=cmap_v, vmin=vmin, vmax=vmax)
    _style(ax)
    ax.set_title(f"Mean {metric_label} at best alpha (%)", fontsize=11)
    for row in range(len(B_VALUES)):
        for col in range(len(K_VALUES)):
            if np.isnan(best_val[row, col]):
                txt = f"—\n(n={n_present[row, col]})"
            else:
                txt = f"{best_val[row, col]:.1f}%\n(n={n_present[row, col]})"
            ax.text(col, row, txt, ha="center", va="center", fontsize=7, color="black")
    plt.colorbar(im2, ax=ax, label=f"{metric_label} (%)")

    cond = f"  mapping: {mapping_key}" if mapping_key else ""
    fig.suptitle(
        f"Beam-search grid — best alpha and {metric_label} | "
        f"LLM: {llm_tag}  norm: {norm}{cond}  (grey = not yet run)",
        fontsize=11,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap figure -> {out_path}")


def collect_results(fusion_dir: Path, llm_tag: str, norm: str,
                    mapping_key: Optional[str]):
    """
    Read every available (subject, B, k) cell once and build:

    cells      : per-subject records with the full alpha sweep plus that
                 subject's best alpha / best word_acc.
    aggregate  : per-(B, k) summary using the mean-over-subjects curve
                 (best_alpha_of_mean is what the heatmap's left panel shows).

    Missing / unreadable cells are skipped, so partial grids still dump.
    """
    cells = []
    aggregate = []
    for B in B_VALUES:
        for k in K_VALUES:
            wa_curves = []
            present = []
            ref_alphas: Optional[list] = None
            for subj in SUBJECTS:
                path = fusion_dir / _cell_filename(subj, B, k, llm_tag, norm,
                                                   mapping_key)
                if not path.exists():
                    continue
                try:
                    with open(path) as f:
                        data = json.load(f)
                    items = sorted(data["results"].items(),
                                   key=lambda kv: float(kv[0]))
                    alphas = [float(a) for a, _ in items]
                    word_acc = [float(v["word_acc"]) for _, v in items]
                    bleu1 = [float(v["bleu1"]) for _, v in items]
                except (json.JSONDecodeError, KeyError, ValueError, OSError):
                    continue

                if ref_alphas is None:
                    ref_alphas = alphas
                elif len(alphas) != len(ref_alphas):
                    continue

                bi = int(np.argmax(word_acc))
                cells.append({
                    "subject": subj, "B": B, "k": k,
                    "alphas": alphas,
                    "word_acc": word_acc,
                    "bleu1": bleu1,
                    "best_alpha": alphas[bi],
                    "best_word_acc": word_acc[bi],
                })
                wa_curves.append(word_acc)
                present.append(subj)

            if wa_curves:
                mean = np.mean(np.array(wa_curves), axis=0)
                bi = int(np.argmax(mean))
                aggregate.append({
                    "B": B, "k": k,
                    "n_present": len(present),
                    "subjects_present": present,
                    "best_alpha_of_mean": ref_alphas[bi],
                    "best_mean_word_acc": float(mean[bi]),
                })
            else:
                aggregate.append({
                    "B": B, "k": k, "n_present": 0,
                    "subjects_present": [],
                    "best_alpha_of_mean": None,
                    "best_mean_word_acc": None,
                })
    return cells, aggregate


def dump_results_json(fusion_dir: Path, llm_tag: str, norm: str,
                      mapping_key: Optional[str], out_path: Path):
    cells, aggregate = collect_results(fusion_dir, llm_tag, norm, mapping_key)
    payload = {
        "condition": "imagined" if mapping_key else "listened",
        "mapping_key": mapping_key,
        "llm": llm_tag,
        "norm": norm,
        "subjects": SUBJECTS,
        "B_values": B_VALUES,
        "k_values": K_VALUES,
        "cells": cells,
        "aggregate": aggregate,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    n_cells_total = len(B_VALUES) * len(K_VALUES)
    n_complete = sum(a["n_present"] == len(SUBJECTS) for a in aggregate)
    print(f"Saved results JSON -> {out_path}  "
          f"({len(cells)} subject-cells; {n_complete}/{n_cells_total} (B,k) complete)")


def main(args):
    fusion_dir = Path(args.fusion_results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else fusion_dir / "figures"
    llm_tag = args.llm_name.replace("/", "_")
    mapping_key = args.mapping_key or None

    stem = f"grid_{llm_tag}_{args.norm}_{args.metric}"
    if mapping_key:
        stem += f"_{mapping_key}"

    plot_grid(fusion_dir, llm_tag, args.norm, args.metric, mapping_key,
              out_dir / f"{stem}_curves.png")
    plot_heatmap(fusion_dir, llm_tag, args.norm, args.metric, mapping_key,
                 out_dir / f"{stem}_heatmap.png")

    # machine-readable dump: per-subject best alpha / best word_acc + aggregate.
    # (metric-independent — always records both word_acc and bleu1 sweeps.)
    results_stem = f"grid_results_{llm_tag}_{args.norm}"
    if mapping_key:
        results_stem += f"_{mapping_key}"
    dump_results_json(fusion_dir, llm_tag, args.norm, mapping_key,
                      out_dir / f"{results_stem}.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Plot beam-search grid sweep (listened or imagined): "
                    "alpha-curve grid + best-alpha heatmap. Missing cells skipped."
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
                   help="Output dir for the two PNGs "
                        "(default: <fusion_results_dir>/figures).")
    main(p.parse_args())
