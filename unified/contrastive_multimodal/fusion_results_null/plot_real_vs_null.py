"""
plot_real_vs_null.py — compare real MEG-guided beam fusion vs. the
random-candidate null baseline, alpha sweep, for a given (B, k).

Reads:
  real : ../fusion_results/sub-*_beamfusion_B{B}_top{k}_{llm}_{norm}.json
  null : ./sub-*_beamfusion_null_B{B}_top{k}_{llm}_{norm}_seed{seed}.json

Plots, per metric (bleu1 / word_acc):
  - each REAL subject   : thin solid line
  - REAL mean           : thick solid line
  - each NULL subject   : thin dashed line
  - NULL mean           : thick dashed line

Usage (run from anywhere; paths are resolved relative to this file):
    python fusion_results_null/plot_real_vs_null.py --B 5 --top_k 5
    python fusion_results_null/plot_real_vs_null.py --B 5 --top_k 5 --metric bleu1
    python fusion_results_null/plot_real_vs_null.py --B 5 --top_k 5 \\
        --norm row_zscore --llm gpt2 --seed 0
"""

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent          # .../fusion_results_null


def _load_group(files, metric):
    """Return {subject: (alphas_sorted, values)} for one metric."""
    curves = {}
    for f in sorted(files):
        d = json.load(open(f))
        subj = d["heldout_subject"]
        r = d["results"]
        alphas = sorted(float(a) for a in r.keys())
        vals = [r[str(a) if str(a) in r else f"{a:g}"][metric] * 100 for a in alphas]
        curves[subj] = (np.array(alphas), np.array(vals))
    return curves


def _mean_curve(curves):
    """Mean over subjects on the alpha grid common to all of them."""
    if not curves:
        return None, None
    common = None
    for alphas, _ in curves.values():
        s = set(np.round(alphas, 4))
        common = s if common is None else (common & s)
    common = np.array(sorted(common))
    stacked = []
    for alphas, vals in curves.values():
        idx = {round(a, 4): v for a, v in zip(alphas, vals)}
        stacked.append([idx[round(a, 4)] for a in common])
    return common, np.array(stacked).mean(axis=0)


def plot_metric(real_curves, null_curves, metric, out_path, B, k, llm, norm):
    fig, ax = plt.subplots(figsize=(9, 6))

    REAL_C, NULL_C = "#1f77b4", "#d62728"   # blue / red

    # per-subject thin lines (label only once each for the legend)
    for i, (alphas, vals) in enumerate(real_curves.values()):
        ax.plot(alphas, vals, color=REAL_C, alpha=0.25, linewidth=1.0,
                label="Real (per subject)" if i == 0 else None)
    for i, (alphas, vals) in enumerate(null_curves.values()):
        ax.plot(alphas, vals, color=NULL_C, alpha=0.25, linewidth=1.0,
                linestyle="--", label="Null (per subject)" if i == 0 else None)

    # means (thick)
    ra, rm = _mean_curve(real_curves)
    na, nm = _mean_curve(null_curves)
    if rm is not None:
        bi = int(np.argmax(rm))
        ax.plot(ra, rm, color=REAL_C, linewidth=2.8,
                label=f"Real mean (n={len(real_curves)})")
        ax.scatter([ra[bi]], [rm[bi]], color=REAL_C, s=55, zorder=5,
                   label=f"Real best α={ra[bi]:.2f} → {rm[bi]:.1f}%")
    if nm is not None:
        ax.plot(na, nm, color=NULL_C, linewidth=2.8, linestyle="--",
                label=f"Null mean (n={len(null_curves)})")

    ax.axvline(0.0, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.axvline(1.0, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.text(0.005, ax.get_ylim()[0], "MEG only", fontsize=8, color="gray", va="bottom")
    ax.text(0.90, ax.get_ylim()[0], "LLM only", fontsize=8, color="gray", va="bottom")
    ax.set_xlabel("Alpha (LLM weight)")
    ax.set_ylabel(f"{metric} (%)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(f"Real vs. random-candidate null — {metric}\n"
                 f"B={B}  top_k={k}  LLM={llm}  norm={norm}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, required=True, help="beam width")
    p.add_argument("--top_k", type=int, required=True, help="MEG top-k / null candidate count")
    p.add_argument("--norm", type=str, default="row_zscore",
                   choices=["row_zscore", "logsoftmax"])
    p.add_argument("--llm", type=str, default="gpt2")
    p.add_argument("--seed", type=int, default=0, help="null seed in the filename")
    p.add_argument("--metric", type=str, default="both",
                   choices=["bleu1", "word_acc", "both"])
    p.add_argument("--real_dir", type=str, default=str(_HERE.parent / "fusion_results"))
    p.add_argument("--null_dir", type=str, default=str(_HERE))
    p.add_argument("--out_dir", type=str, default=str(_HERE / "figures"))
    args = p.parse_args()

    real_glob = os.path.join(
        args.real_dir, f"sub-*_beamfusion_B{args.B}_top{args.top_k}_{args.llm}_{args.norm}.json")
    null_glob = os.path.join(
        args.null_dir,
        f"sub-*_beamfusion_null_B{args.B}_top{args.top_k}_{args.llm}_{args.norm}_seed{args.seed}.json")
    real_files = glob.glob(real_glob)
    null_files = glob.glob(null_glob)
    print(f"Real files ({len(real_files)}): {real_glob}")
    print(f"Null files ({len(null_files)}): {null_glob}")
    if not real_files:
        raise SystemExit(f"No real files matched:\n  {real_glob}")
    if not null_files:
        raise SystemExit(f"No null files matched:\n  {null_glob}")

    metrics = ["bleu1", "word_acc"] if args.metric == "both" else [args.metric]
    for metric in metrics:
        real_curves = _load_group(real_files, metric)
        null_curves = _load_group(null_files, metric)
        out_path = os.path.join(
            args.out_dir,
            f"real_vs_null_B{args.B}_top{args.top_k}_{args.llm}_{args.norm}_{metric}.png")
        plot_metric(real_curves, null_curves, metric, out_path,
                    args.B, args.top_k, args.llm, args.norm)


if __name__ == "__main__":
    main()
