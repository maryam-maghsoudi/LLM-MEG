"""
plot_alpha_sweep.py — Alpha sweep curves for LLM+MEG fusion results.

Loads per-fold fusion JSONs from:
    unified/out/{method}/{model_tag}/{scheme}_{fold}/{fusion_subdir}/fusion_{llm_tag}_{norm}.json

fusion_subdir is "fusion" (test set, default) or "fusion_on_val" (validation set).

Produces 5 figures (one per metric) for each eval scheme (loso, session_cv, stimulus):
R@1, MRR, Word Accuracy, BLEU-1, WER.
Each figure: two side-by-side subplots (logsoftmax | row_zscore), shared y-axis.
Within each subplot: one thin line per fold + one thick mean line.

Figures are saved to:
    {out_dir}/{eval_scheme}/alpha_sweep_{method}_{metric}.{pdf,png}          (test)
    {out_dir}/{eval_scheme}_val/alpha_sweep_{method}_{metric}.{pdf,png}      (val)

Usage (from llm_decoder/):
    python -m unified.method1_analysis.fusion.plot_alpha_sweep
    python -m unified.method1_analysis.fusion.plot_alpha_sweep \\
        --method inference \\
        --out_root unified/out \\
        --out_dir unified/method1_analysis/fusion/figures
    python -m unified.method1_analysis.fusion.plot_alpha_sweep \\
        --fusion_subdir fusion_on_val \\
        --schemes loso session_cv stimulus
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

_HERE     = Path(__file__).parent
_OUT_ROOT = _HERE.parent.parent / "out"
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

SCHEME_CONFIG = {
    "loso":       {"prefix": "loso_",            "strip": "loso_",            "label": "LOSO subjects"},
    "session_cv": {"prefix": "session_cv_fold",  "strip": "session_cv_fold",  "label": "session CV folds"},
    "stimulus":   {"prefix": "stimulus_lines",   "strip": "stimulus_lines",   "label": "stimulus lines"},
}


def _model_tag(method: str, llm_tag: str, bert_tag: str) -> str:
    if method == "inference":
        return bert_tag.replace("/", "_").replace("-", "_")
    return llm_tag.replace("/", "_")


def load_fold_results(out_root: Path, method: str, model_tag: str,
                      eval_scheme: str, fusion_llm_tag: str, norm: str,
                      control: str = "none",
                      fusion_subdir: str = "fusion") -> dict:
    """
    Glob for all per-fold fusion JSONs for the given eval scheme.

    Returns {fold_key: per_alpha_dict}
    where per_alpha_dict maps alpha_str → scalar_summary.
    fold_key is e.g. "sub-01" for loso or "0" for session_cv.
    """
    cfg        = SCHEME_CONFIG[eval_scheme]
    ctrl_suffix = f"_ctrl_{control}" if control != "none" else ""
    pattern    = f"{cfg['prefix']}*{ctrl_suffix}"
    base_dir   = out_root / method / model_tag

    fold_data = {}
    for fold_dir in sorted(base_dir.glob(pattern)):
        fusion_file = fold_dir / fusion_subdir / f"fusion_{fusion_llm_tag}_{norm}.json"
        if not fusion_file.exists():
            continue
        data = json.loads(fusion_file.read_text())
        dir_name = fold_dir.name
        fold_key = dir_name[len(cfg["strip"]):]
        if ctrl_suffix and fold_key.endswith(ctrl_suffix):
            fold_key = fold_key[: -len(ctrl_suffix)]
        fold_data[fold_key] = data["per_alpha"]

    return fold_data


def extract_curves(fold_data: dict, alphas: np.ndarray, metric_key: str):
    """Returns (per_fold_curves, mean_curve) where each curve is np.array(n_alphas)."""
    curves = {}
    for key, per_alpha in sorted(fold_data.items()):
        vals = np.array([per_alpha[str(a)][metric_key] for a in alphas])
        curves[key] = vals
    mean_curve = np.stack(list(curves.values())).mean(axis=0)
    return curves, mean_curve


def plot_metric(metric_key: str, metric_label: str, lower_is_better: bool,
                alphas: np.ndarray, results_by_norm: dict,
                method: str, eval_scheme: str, out_dir: Path,
                scheme_dir_suffix: str = ""):
    fold_keys = sorted(results_by_norm[NORMS[0]].keys())
    colors    = cm.tab20(np.linspace(0, 1, max(len(fold_keys), 1)))
    scheme_label = SCHEME_CONFIG[eval_scheme]["label"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(
        f"{method.capitalize()} — {metric_label} vs alpha  ({scheme_label})",
        fontsize=13, fontweight="bold",
    )

    for ax, norm in zip(axes, NORMS):
        fold_data = results_by_norm[norm]
        curves, mean_curve = extract_curves(fold_data, alphas, metric_key)

        for i, (key, vals) in enumerate(sorted(curves.items())):
            ax.plot(alphas, vals, color=colors[i], alpha=0.55,
                    linewidth=1.0, label=key)

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

    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, fontsize=7, loc="upper left",
                   ncol=2, framealpha=0.7)

    fig.tight_layout()

    safe_label = metric_label.lower().replace(" ", "_").replace("-", "")
    scheme_dir = out_dir / f"{eval_scheme}{scheme_dir_suffix}"
    scheme_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = scheme_dir / f"alpha_sweep_{method}_{safe_label}.{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  saved: {out_path}")
    plt.close(fig)


def _load_scheme(out_root, method, model_tag, eval_scheme,
                 fusion_llm_tag, control, fusion_subdir="fusion"):
    """Load both norms for one eval scheme; return (results_by_norm, alphas) or None."""
    results_by_norm = {}
    alphas_ref = None
    for norm in NORMS:
        fold_data = load_fold_results(
            out_root, method, model_tag, eval_scheme,
            fusion_llm_tag, norm, control, fusion_subdir,
        )
        if not fold_data:
            print(f"  [skip] no results for scheme={eval_scheme} norm={norm} subdir={fusion_subdir}")
            return None, None
        results_by_norm[norm] = fold_data
        first_fold = next(iter(fold_data.values()))
        alphas_found = sorted(float(k) for k in first_fold.keys())
        if alphas_ref is None:
            alphas_ref = alphas_found
        print(f"  {norm}: {len(fold_data)} folds, {len(alphas_found)} alpha values")
    return results_by_norm, np.array(alphas_ref)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method",     default="inference",
                   choices=["inference", "twostage", "interleaved"])
    p.add_argument("--out_root",   default=str(_OUT_ROOT))
    p.add_argument("--out_dir",    default=str(_OUT_DIR))
    p.add_argument("--fusion_llm", default="HuggingFaceTB/SmolLM2-360M")
    p.add_argument("--bert_name",  default="bert-base-uncased")
    p.add_argument("--llm_name",   default="HuggingFaceTB/SmolLM2-360M")
    p.add_argument("--control",    default="none")
    p.add_argument("--schemes",    nargs="+", default=["loso", "session_cv"],
                   choices=["loso", "session_cv", "stimulus"],
                   help="Eval schemes to plot (default: loso session_cv)")
    p.add_argument("--fusion_subdir", default="fusion",
                   help="Subdirectory inside each checkpoint dir containing fusion JSONs "
                        "(default: 'fusion'; use 'fusion_on_val' for validation-set results)")
    args = p.parse_args()

    out_root       = Path(args.out_root)
    out_dir        = Path(args.out_dir)
    fusion_llm_tag = args.fusion_llm.replace("/", "_")
    model_tag      = _model_tag(args.method, args.llm_name, args.bert_name)
    # Append "_val" to output scheme dirs when plotting validation-set results
    scheme_dir_suffix = "_val" if args.fusion_subdir != "fusion" else ""

    for eval_scheme in args.schemes:
        print(f"\n=== {eval_scheme} (subdir={args.fusion_subdir}) ===")
        print(f"Searching: {out_root / args.method / model_tag}/{eval_scheme}_*/{args.fusion_subdir}/")

        results_by_norm, alphas = _load_scheme(
            out_root, args.method, model_tag, eval_scheme,
            fusion_llm_tag, args.control, args.fusion_subdir,
        )
        if results_by_norm is None:
            continue

        fold_keys = sorted(results_by_norm[NORMS[0]].keys())
        print(f"Folds ({len(fold_keys)}): {fold_keys}")
        out_scheme_dir = out_dir / f"{eval_scheme}{scheme_dir_suffix}"
        print(f"Generating figures → {out_scheme_dir}")

        for metric_key, metric_label, lower_is_better in METRICS:
            print(f"  {metric_label} ...")
            plot_metric(metric_key, metric_label, lower_is_better,
                        alphas, results_by_norm, args.method, eval_scheme, out_dir,
                        scheme_dir_suffix)

    print("\nDone.")


if __name__ == "__main__":
    main()
