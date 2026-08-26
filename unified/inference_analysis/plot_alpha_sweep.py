"""
plot_alpha_sweep.py — Alpha sweep curves for LLM+MEG fusion results.

Loads per-fold fusion JSONs from:
    unified/out/{method}/{model_tag}/loso_{heldout}/fusion/fusion_{llm_tag}_{norm}.json

Produces 5 figures (one per metric): R@1, MRR, Word Accuracy, BLEU-1, WER.
Each figure: two side-by-side subplots (logsoftmax | row_zscore), shared y-axis.
Within each subplot: one thin line per LOSO subject + one thick mean line.

Usage (from llm_decoder/):
    python -m unified.inference_analysis.plot_alpha_sweep
    python -m unified.inference_analysis.plot_alpha_sweep \\
        --method inference \\
        --out_root unified/out \\
        --out_dir unified/inference_analysis/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

_HERE    = Path(__file__).parent
_OUT_ROOT = _HERE.parent / "out"
_OUT_DIR  = _HERE / "figures"

METRICS = [
    ("mean_R@1",      "R@1",          False),   # (json_key, label, lower_is_better)
    ("mean_MRR",      "MRR",          False),
    ("mean_accuracy", "Word Accuracy", False),
    ("mean_BLEU1",    "BLEU-1",       False),
    ("mean_WER",      "WER",          True),
]

NORMS = ["logsoftmax", "row_zscore"]
NORM_LABELS = {
    "logsoftmax": "log-softmax",
    "row_zscore": "row z-score",
}


def _model_tag(method: str, llm_tag: str, bert_tag: str) -> str:
    if method == "inference":
        return bert_tag.replace("/", "_").replace("-", "_")
    return llm_tag.replace("/", "_")


def load_fold_results(out_root: Path, method: str, model_tag: str,
                      fusion_llm_tag: str, norm: str,
                      control: str = "none") -> dict:
    """
    Glob for all per-fold fusion JSONs matching the given configuration.

    Returns {subject_or_fold_key: per_alpha_dict}
    where per_alpha_dict maps alpha_str → scalar_summary.
    """
    ctrl_suffix = f"_ctrl_{control}" if control != "none" else ""
    pattern = f"loso_*{ctrl_suffix}"
    base_dir = out_root / method / model_tag

    fold_data = {}
    for fold_dir in sorted(base_dir.glob(pattern)):
        fusion_file = fold_dir / "fusion" / f"fusion_{fusion_llm_tag}_{norm}.json"
        if not fusion_file.exists():
            continue
        data = json.loads(fusion_file.read_text())
        # Derive subject key from directory name, e.g. "loso_sub-01" → "sub-01"
        dir_name = fold_dir.name
        fold_key = dir_name[len("loso_"):]
        if ctrl_suffix:
            fold_key = fold_key[:fold_key.index(ctrl_suffix)]
        fold_data[fold_key] = data["per_alpha"]

    return fold_data


def extract_curves(fold_data: dict, alphas: np.ndarray, metric_key: str):
    """Returns (per_subject_curves, mean_curve) where each curve is np.array(n_alphas)."""
    curves = {}
    for subj, per_alpha in sorted(fold_data.items()):
        vals = np.array([per_alpha[str(a)][metric_key] for a in alphas])
        curves[subj] = vals
    mean_curve = np.stack(list(curves.values())).mean(axis=0)
    return curves, mean_curve


def plot_metric(metric_key: str, metric_label: str, lower_is_better: bool,
                alphas: np.ndarray, results_by_norm: dict, method: str,
                out_dir: Path):
    subjects = sorted(results_by_norm[NORMS[0]].keys())
    colors   = cm.tab20(np.linspace(0, 1, max(len(subjects), 1)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(f"{method.capitalize()} — {metric_label} vs alpha  (all LOSO subjects)",
                 fontsize=13, fontweight="bold")

    for ax, norm in zip(axes, NORMS):
        fold_data = results_by_norm[norm]
        curves, mean_curve = extract_curves(fold_data, alphas, metric_key)

        for i, (subj, vals) in enumerate(sorted(curves.items())):
            ax.plot(alphas, vals, color=colors[i], alpha=0.55,
                    linewidth=1.0, label=subj)

        ax.plot(alphas, mean_curve, color="black", linewidth=2.5,
                linestyle="--", label="mean", zorder=5)

        ax.set_title(NORM_LABELS[norm], fontsize=11)
        ax.set_xlabel("alpha  (0 = pure MEG, 1 = pure LLM)", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel(metric_label, fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

        best_idx = int(mean_curve.argmin() if lower_is_better else mean_curve.argmax())
        best_a   = alphas[best_idx]
        best_v   = mean_curve[best_idx]
        ax.axvline(best_a, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.annotate(f"α={best_a:.2f}\n{metric_label}={best_v:.3f}",
                    xy=(best_a, best_v),
                    xytext=(best_a + 0.04, best_v),
                    fontsize=7.5, color="black",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    # Legend on right subplot only
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, fontsize=7, loc="upper left",
                   ncol=2, framealpha=0.7)

    fig.tight_layout()

    safe_label = metric_label.lower().replace(" ", "_").replace("-", "")
    for ext in ("pdf", "png"):
        out_path = out_dir / f"alpha_sweep_{method}_{safe_label}.{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  saved: {out_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method",        default="inference",
                   choices=["inference", "twostage", "interleaved"])
    p.add_argument("--out_root",      default=str(_OUT_ROOT))
    p.add_argument("--out_dir",       default=str(_OUT_DIR))
    p.add_argument("--fusion_llm",    default="HuggingFaceTB/SmolLM2-360M",
                   help="Fusion LLM (used to find the JSON filename)")
    p.add_argument("--bert_name",     default="bert-base-uncased")
    p.add_argument("--llm_name",      default="HuggingFaceTB/SmolLM2-360M")
    p.add_argument("--control",       default="none")
    args = p.parse_args()

    out_root       = Path(args.out_root)
    out_dir        = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fusion_llm_tag = args.fusion_llm.replace("/", "_")
    model_tag      = _model_tag(args.method, args.llm_name, args.bert_name)

    print(f"Searching: {out_root / args.method / model_tag}/loso_*/fusion/fusion_{fusion_llm_tag}_*.json")

    results_by_norm = {}
    alphas_ref = None
    for norm in NORMS:
        fold_data = load_fold_results(
            out_root, args.method, model_tag, fusion_llm_tag, norm, args.control
        )
        if not fold_data:
            raise FileNotFoundError(
                f"No fusion results found for method={args.method} norm={norm}. "
                f"Run fuse_eval.py first."
            )
        results_by_norm[norm] = fold_data

        # Infer alphas from first fold's keys (consistent across folds)
        first_fold = next(iter(fold_data.values()))
        alphas_found = sorted(float(k) for k in first_fold.keys())
        if alphas_ref is None:
            alphas_ref = alphas_found
        print(f"  {norm}: {len(fold_data)} folds, {len(alphas_found)} alpha values")

    alphas = np.array(alphas_ref)
    subjects = sorted(results_by_norm[NORMS[0]].keys())
    print(f"\nSubjects ({len(subjects)}): {subjects}")
    print(f"\nGenerating figures → {out_dir}")

    for metric_key, metric_label, lower_is_better in METRICS:
        print(f"  {metric_label} ...")
        plot_metric(metric_key, metric_label, lower_is_better,
                    alphas, results_by_norm, args.method, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
