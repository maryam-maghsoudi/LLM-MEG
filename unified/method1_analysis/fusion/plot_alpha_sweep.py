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

--diff_test mode: compute BLEU(diff_alpha) − BLEU(1.0) per fold across train/val/test
splits, run one-sided Wilcoxon signed-rank test (H1: median > 0), and plot distributions.
Useful for testing whether the MEG signal contributes beyond pure LLM at a given alpha.

Figure output layout:
    figures/smolLM/{eval_scheme}/          SmolLM2, test set (default)
    figures/smolLM/{eval_scheme}_val/      SmolLM2, val set
    figures/gpt2/{eval_scheme}/            GPT-2, open (per-trial) vocab
    figures/gpt2/{eval_scheme}_closed76/   GPT-2, closed 76-word vocab

Usage (from llm_decoder/):
    # SmolLM2 (default --out_dir figures/smolLM)
    python -m unified.method1_analysis.fusion.plot_alpha_sweep
    python -m unified.method1_analysis.fusion.plot_alpha_sweep \\
        --fusion_subdir fusion_on_val --schemes loso session_cv

    # GPT-2, open vocab
    python -m unified.method1_analysis.fusion.plot_alpha_sweep \\
        --fusion_llm gpt2 --out_dir unified/method1_analysis/fusion/figures/gpt2

    # GPT-2, closed 76-word vocab
    python -m unified.method1_analysis.fusion.plot_alpha_sweep \\
        --fusion_llm gpt2 --vocab_suffix _closed76 \\
        --out_dir unified/method1_analysis/fusion/figures/gpt2

    # Significance test (BLEU diff analysis)
    python -m unified.method1_analysis.fusion.plot_alpha_sweep --diff_test
    python -m unified.method1_analysis.fusion.plot_alpha_sweep --diff_test --diff_alpha 0.5
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy import stats

_HERE     = Path(__file__).parent
_OUT_ROOT = _HERE.parent.parent / "out"
_OUT_DIR  = _HERE / "figures" / "smolLM"   # default; use --out_dir figures/gpt2 for GPT-2 runs

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

# Maps human-readable split name → fusion subdir
SPLIT_CONFIG = {
    "train": "fusion_on_train",
    "val":   "fusion_on_val",
    "test":  "fusion",
}


def _model_tag(method: str, llm_tag: str, bert_tag: str) -> str:
    if method == "inference":
        return bert_tag.replace("/", "_").replace("-", "_")
    return llm_tag.replace("/", "_")


def load_fold_results(out_root: Path, method: str, model_tag: str,
                      eval_scheme: str, fusion_llm_tag: str, norm: str,
                      control: str = "none",
                      fusion_subdir: str = "fusion",
                      vocab_suffix: str = "") -> dict:
    """
    Glob for all per-fold fusion JSONs for the given eval scheme.

    Returns {fold_key: per_alpha_dict}
    where per_alpha_dict maps alpha_str → scalar_summary.
    fold_key is e.g. "sub-01" for loso or "0" for session_cv.
    vocab_suffix is appended before .json, e.g. "_closed76".
    """
    cfg        = SCHEME_CONFIG[eval_scheme]
    ctrl_suffix = f"_ctrl_{control}" if control != "none" else ""
    pattern    = f"{cfg['prefix']}*{ctrl_suffix}"
    base_dir   = out_root / method / model_tag

    fold_data = {}
    for fold_dir in sorted(base_dir.glob(pattern)):
        fusion_file = fold_dir / fusion_subdir / f"fusion_{fusion_llm_tag}_{norm}{vocab_suffix}.json"
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
    all_fold_keys = sorted(set(k for fd in results_by_norm.values() for k in fd.keys()))
    colors        = cm.tab20(np.linspace(0, 1, max(len(all_fold_keys), 1))
                             )
    color_idx     = {k: i for i, k in enumerate(all_fold_keys)}
    scheme_label = SCHEME_CONFIG[eval_scheme]["label"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(
        f"{method.capitalize()} — {metric_label} vs alpha  ({scheme_label})",
        fontsize=13, fontweight="bold",
    )

    for ax, norm in zip(axes, NORMS):
        fold_data = results_by_norm[norm]
        curves, mean_curve = extract_curves(fold_data, alphas, metric_key)

        for key, vals in sorted(curves.items()):
            ax.plot(alphas, vals, color=colors[color_idx[key]], alpha=0.55,
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
                 fusion_llm_tag, control, fusion_subdir="fusion", vocab_suffix=""):
    """Load both norms for one eval scheme; return (results_by_norm, alphas) or None."""
    results_by_norm = {}
    alphas_ref = None
    for norm in NORMS:
        fold_data = load_fold_results(
            out_root, method, model_tag, eval_scheme,
            fusion_llm_tag, norm, control, fusion_subdir, vocab_suffix,
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


def run_bleu_diff_analysis(out_root: Path, method: str, model_tag: str,
                            fusion_llm_tag: str, control: str,
                            eval_schemes: list, diff_alpha: float,
                            out_dir: Path):
    """
    Compute BLEU(diff_alpha) − BLEU(1.0) per fold across train/val/test splits.

    For each (eval_scheme, split, norm), runs a one-sided Wilcoxon signed-rank test
    (H1: median > 0; falls back to one-sample t-test when n < 6).
    Positive difference means MEG contributes beyond the pure-LLM baseline.

    Produces one figure per norm (logsoftmax / row_zscore):
        rows = eval_schemes, cols = splits (train | val | test)
    Saved to: {out_dir}/bleu_diff_{method}_alpha{diff_alpha}_{norm}.{pdf,png}
    """
    splits = ["train", "val", "test"]
    rng = np.random.default_rng(42)

    for norm in NORMS:
        n_rows = len(eval_schemes)
        n_cols = len(splits)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4.5 * n_cols, 4.0 * n_rows),
                                 squeeze=False)
        fig.suptitle(
            f"{method.capitalize()} — BLEU({diff_alpha:.2f}) − BLEU(1.0) per fold"
            f"  [{NORM_LABELS[norm]}]\n"
            "Positive = MEG helps; negative = MEG hurts",
            fontsize=12, fontweight="bold",
        )

        print(f"\n=== BLEU diff analysis | norm={norm} | alpha={diff_alpha} ===")

        for row, eval_scheme in enumerate(eval_schemes):
            scheme_label = SCHEME_CONFIG[eval_scheme]["label"]
            for col, split in enumerate(splits):
                fusion_subdir = SPLIT_CONFIG[split]
                ax = axes[row][col]

                fold_data = load_fold_results(
                    out_root, method, model_tag, eval_scheme,
                    fusion_llm_tag, norm, control, fusion_subdir,
                )

                if not fold_data:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes, fontsize=10, color="gray")
                    ax.set_title(f"{scheme_label}\n{split}", fontsize=9)
                    continue

                # Compute per-fold BLEU difference
                fold_keys = sorted(fold_data.keys())
                avail_alphas = sorted(float(k) for k in next(iter(fold_data.values())).keys())
                closest_alpha = min(avail_alphas, key=lambda x: abs(x - diff_alpha))

                diffs = []
                for fk in fold_keys:
                    per_alpha = fold_data[fk]
                    b_meg = per_alpha[str(closest_alpha)]["mean_BLEU1"]
                    b_llm = per_alpha["1.0"]["mean_BLEU1"]
                    diffs.append(b_meg - b_llm)

                diffs = np.array(diffs)
                n = len(diffs)
                mean_diff = float(diffs.mean())

                # One-sided significance test: H1: median/mean > 0
                if n >= 6:
                    try:
                        _, p_val = stats.wilcoxon(diffs, alternative="greater")
                        test_name = "Wilcoxon"
                    except ValueError:
                        p_val = float("nan")
                        test_name = "Wilcoxon"
                else:
                    _, p_val = stats.ttest_1samp(diffs, 0.0, alternative="greater")
                    test_name = "t-test"

                sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else
                      ("*" if p_val < 0.05 else ("†" if p_val < 0.10 else "ns")))

                # Jittered strip plot
                x_jitter = rng.uniform(-0.18, 0.18, size=n)
                dot_colors = ["#2166ac" if d > 0 else "#d6604d" for d in diffs]
                ax.scatter(x_jitter, diffs, s=70, zorder=3,
                           color=dot_colors, edgecolors="white", linewidths=0.6)

                # Annotate fold keys next to dots
                for xi, yi, fk in zip(x_jitter, diffs, fold_keys):
                    short = fk.replace("sub-", "s") if fk.startswith("sub-") else fk
                    ax.text(xi + 0.02, yi, short, fontsize=5.5, va="center",
                            color="gray", clip_on=True)

                # Reference lines
                ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
                ax.axhline(mean_diff, color="navy", linewidth=1.8, linestyle="-",
                           alpha=0.85, zorder=4)

                # p-value box
                p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                ax.text(0.97, 0.97,
                        f"{test_name}\n{p_str}  {sig}\nmean={mean_diff:+.4f}",
                        ha="right", va="top", transform=ax.transAxes,
                        fontsize=7.5,
                        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.75))

                ax.set_xlim(-0.7, 0.7)
                ax.set_xticks([])
                ax.set_ylabel("ΔBLEU-1" if col == 0 else "", fontsize=9)
                ax.grid(True, axis="y", alpha=0.3)
                title_line = f"{scheme_label} | {split}  (n={n})"
                if abs(closest_alpha - diff_alpha) > 1e-4:
                    title_line += f"\n[α≈{closest_alpha}]"
                ax.set_title(title_line, fontsize=8.5)

                print(f"  {eval_scheme:10s} | {split:5s} | {norm}: "
                      f"n={n:2d}  mean={mean_diff:+.5f}  "
                      f"{test_name} p={p_val:.4f}  {sig}")

        fig.tight_layout()

        safe_alpha = f"{diff_alpha:.2f}".replace(".", "p")
        fname = f"bleu_diff_{method}_alpha{safe_alpha}_{norm}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            out_path = out_dir / f"{fname}.{ext}"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"  saved: {out_path}")
        plt.close(fig)


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
    p.add_argument("--vocab_suffix", default="",
                   help="Suffix appended to the fusion JSON filename before the extension "
                        "(e.g. '_closed76' loads fusion_gpt2_row_zscore_closed76.json). "
                        "Default: '' (open/per-trial vocab)")
    # BLEU-diff analysis flags
    p.add_argument("--diff_test",  action="store_true",
                   help="Run BLEU(diff_alpha)−BLEU(1.0) significance analysis across "
                        "train/val/test splits instead of the per-metric sweep plots")
    p.add_argument("--diff_alpha", type=float, default=0.75,
                   help="Alpha to compare against 1.0 in diff analysis (default: 0.75)")
    args = p.parse_args()

    out_root       = Path(args.out_root)
    out_dir        = Path(args.out_dir)
    fusion_llm_tag = args.fusion_llm.replace("/", "_")
    model_tag      = _model_tag(args.method, args.llm_name, args.bert_name)

    vocab_suffix = args.vocab_suffix  # e.g. "" or "_closed76"

    if args.diff_test:
        run_bleu_diff_analysis(
            out_root=out_root,
            method=args.method,
            model_tag=model_tag,
            fusion_llm_tag=fusion_llm_tag,
            control=args.control,
            eval_schemes=args.schemes,
            diff_alpha=args.diff_alpha,
            out_dir=out_dir,
        )
        print("\nDone.")
        return

    # Derive output suffix from subdir name: "fusion" → "", "fusion_on_val" → "_val"
    if args.fusion_subdir == "fusion":
        scheme_dir_suffix = ""
    elif args.fusion_subdir.startswith("fusion_on_"):
        scheme_dir_suffix = "_" + args.fusion_subdir[len("fusion_on_"):]
    else:
        scheme_dir_suffix = "_" + args.fusion_subdir
    # Append vocab suffix so closed76 figures go to a separate folder
    scheme_dir_suffix += vocab_suffix

    for eval_scheme in args.schemes:
        print(f"\n=== {eval_scheme} (subdir={args.fusion_subdir}, vocab={vocab_suffix or 'per_trial'}) ===")
        print(f"Searching: {out_root / args.method / model_tag}/{eval_scheme}_*/{args.fusion_subdir}/")

        results_by_norm, alphas = _load_scheme(
            out_root, args.method, model_tag, eval_scheme,
            fusion_llm_tag, args.control, args.fusion_subdir, vocab_suffix,
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
