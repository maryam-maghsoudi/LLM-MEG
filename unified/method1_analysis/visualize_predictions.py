"""
visualize_predictions.py — Show predicted vs. ground-truth word sequences
for the inference method (Method 1), LOSO evaluation.

For each heldout subject: 2 trials from poem1, 2 trials from poem2.
Words are colour-coded: green = correct, red = wrong, grey = invalid MEG window.

Usage (from llm_decoder/ parent):
    python -m unified.method1_analysis.visualize_predictions
    python -m unified.method1_analysis.visualize_predictions --sessions 0 5
    python -m unified.method1_analysis.visualize_predictions --no_figs   # text only
"""

import argparse
import sys
from pathlib import Path

import torch

_HERE     = Path(__file__).parent
CKPT_ROOT = _HERE.parent / "out" / "inference" / "bert_base_uncased"
OUT_DIR   = _HERE / "results" / "predictions"

sys.path.insert(0, str(_HERE.parent.parent))   # llm_decoder/ on path

SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]
POEMS = ["poem1", "poem2"]


# ---------------------------------------------------------------------------
#  Text summary (always produced)
# ---------------------------------------------------------------------------

def _word_line(words, pred, valid, width=12):
    """Format a single row of aligned words."""
    return "  ".join(f"{w:<{width}}" for w, p, v in zip(words, pred, valid))


def print_trial(subject, poem, session, words, pred_top1, valid, metrics):
    n = len(words)
    correct = sum(w == p for w, p, v in zip(words, pred_top1, valid) if v)
    n_valid = sum(valid)

    print(f"\n  Subject={subject}  Poem={poem}  Session={session}  "
          f"({correct}/{n_valid} correct, R@1={metrics['R@1']:.3f}, "
          f"MRR={metrics['MRR']:.3f}, WER={metrics['WER']:.3f})")
    print(f"  {'':4}  {'GROUND TRUTH':<15}  {'PREDICTION':<15}  {'OK'}")
    print(f"  {'-'*50}")
    for i, (w, p, v) in enumerate(zip(words, pred_top1, valid)):
        ok = "✓" if (v and w == p) else ("✗" if v else "—")
        tag = "" if v else " [no MEG]"
        print(f"  {i:3d}  {w:<15}  {p:<15}  {ok}{tag}")


# ---------------------------------------------------------------------------
#  Figure (one page per subject)
# ---------------------------------------------------------------------------

def make_figure(subject, trials_data, out_path):
    """
    trials_data: list of dicts with keys
        poem, session, words, pred_top1, valid, metrics
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [warn] matplotlib not available — skipping figures")
        return

    n_trials = len(trials_data)
    max_words = max(len(t["words"]) for t in trials_data)

    # Layout: one column per trial, rows = word positions
    fig_w = 3.2 * n_trials
    fig_h = max(6, 0.28 * max_words + 2.5)
    fig, axes = plt.subplots(1, n_trials, figsize=(fig_w, fig_h))
    if n_trials == 1:
        axes = [axes]

    fig.suptitle(f"Inference Method — {subject}", fontsize=13, fontweight="bold", y=0.98)

    for ax, td in zip(axes, trials_data):
        words     = td["words"]
        pred      = td["pred_top1"]
        valid     = td["valid"]
        metrics   = td["metrics"]
        poem      = td["poem"]
        session   = td["session"]
        N         = len(words)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, N - 0.5)
        ax.invert_yaxis()
        ax.axis("off")

        title = (f"{poem}  sess={session}\n"
                 f"R@1={metrics['R@1']:.3f}  MRR={metrics['MRR']:.3f}\n"
                 f"WER={metrics['WER']:.3f}  BLEU-1={metrics['BLEU1']:.3f}")
        ax.set_title(title, fontsize=8, loc="left", pad=4)

        # Column headers
        ax.text(0.05, -0.45, "Truth",      fontsize=7.5, fontweight="bold", va="center")
        ax.text(0.55, -0.45, "Prediction", fontsize=7.5, fontweight="bold", va="center")

        for i, (w, p, v) in enumerate(zip(words, pred, valid)):
            correct = v and (w == p)
            wrong   = v and (w != p)

            # Background strip
            if correct:
                bg = "#d4edda"   # light green
            elif wrong:
                bg = "#f8d7da"   # light red
            else:
                bg = "#f0f0f0"   # light grey (no MEG)

            rect = mpatches.FancyBboxPatch(
                (0.0, i - 0.45), 1.0, 0.88,
                boxstyle="round,pad=0.02",
                linewidth=0, facecolor=bg,
            )
            ax.add_patch(rect)

            # Word index
            ax.text(0.01, i, str(i), fontsize=6, color="#888888", va="center")

            # Ground truth
            ax.text(0.08, i, w, fontsize=7.5, va="center",
                    color="#1a472a" if correct else ("#7b2d30" if wrong else "#555555"),
                    fontweight="bold" if correct else "normal")

            # Prediction
            ax.text(0.58, i, p, fontsize=7.5, va="center",
                    color="#1a472a" if correct else ("#7b2d30" if wrong else "#555555"))

            # Tick mark
            if v:
                marker = "✓" if correct else "✗"
                col    = "#2e7d32" if correct else "#c62828"
                ax.text(0.50, i, marker, fontsize=7, va="center",
                        ha="center", color=col, fontweight="bold")
            else:
                ax.text(0.50, i, "—", fontsize=7, va="center",
                        ha="center", color="#aaaaaa")

    # Legend
    legend_patches = [
        mpatches.Patch(color="#d4edda", label="Correct"),
        mpatches.Patch(color="#f8d7da", label="Wrong"),
        mpatches.Patch(color="#f0f0f0", label="No MEG window"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Figure → {out_path}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", type=int, nargs=2, default=[0, 5],
                   help="Two session indices to use per poem (default: 0 5)")
    p.add_argument("--device",  default="cpu")
    p.add_argument("--no_figs", action="store_true")
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Subset of subjects (default: all 13)")
    return p.parse_args()


def _run_predict(subject, poem, session, ckpt_dir, device):
    from unified.predict import predict
    from unified.evaluate import eval_option_a, eval_option_b
    # Patch load_bert_hiddens with a cached version so BERT is only loaded once
    # per process (same model+layer args always produce identical output).
    import unified.methods.models as _models
    import functools
    if not isinstance(_models.load_bert_hiddens, functools._lru_cache_wrapper):
        _models.load_bert_hiddens = functools.lru_cache(maxsize=4)(_models.load_bert_hiddens)

    result = predict(
        subject   = subject,
        session   = session,
        condition = "lis",
        poem      = poem,
        method    = "inference",
        ckpt_dir  = str(ckpt_dir),
        device    = device,
    )
    a = eval_option_a(result["scores"], result["vocab"],
                      result["words"], result["valid"])
    b = eval_option_b(result["pred_top1"], result["words"], result["valid"])
    metrics = {
        "R@1":  a["recall_at_k"][0],
        "MRR":  a["mrr"],
        "WER":  b["wer"],
        "BLEU1": b["bleu1"],
    }
    return result["words"], result["pred_top1"], result["valid"], metrics


def main():
    args     = parse_args()
    subjects = args.subjects or SUBJECTS
    sessions = args.sessions
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Open a combined text log
    log_path = OUT_DIR / "predictions_text.txt"
    log_lines = []

    for subject in subjects:
        ckpt_dir = CKPT_ROOT / f"loso_{subject}"
        if not ckpt_dir.exists():
            print(f"[skip] no checkpoint for {subject}")
            continue

        print(f"\n{'='*60}")
        print(f"  {subject}")
        print(f"{'='*60}")

        trials_data = []
        for poem in POEMS:
            for session in sessions:
                print(f"  predicting {poem} session={session} ...", end=" ", flush=True)
                try:
                    words, pred_top1, valid, metrics = _run_predict(
                        subject, poem, session, ckpt_dir, args.device
                    )
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue
                print(f"R@1={metrics['R@1']:.3f}  MRR={metrics['MRR']:.3f}")

                trials_data.append(dict(
                    poem=poem, session=session,
                    words=words, pred_top1=pred_top1, valid=valid,
                    metrics=metrics,
                ))

                # Print to console + log
                block = []
                block.append(
                    f"\nSubject={subject}  Poem={poem}  Session={session}  "
                    f"R@1={metrics['R@1']:.3f}  MRR={metrics['MRR']:.3f}  "
                    f"WER={metrics['WER']:.3f}  BLEU-1={metrics['BLEU1']:.3f}"
                )
                block.append(f"  {'#':>3}  {'Ground Truth':<18}  {'Prediction':<18}  OK")
                block.append(f"  {'-'*55}")
                for i, (w, p, v) in enumerate(zip(words, pred_top1, valid)):
                    ok = "✓" if (v and w == p) else ("✗" if v else "—")
                    block.append(f"  {i:3d}  {w:<18}  {p:<18}  {ok}")
                block_str = "\n".join(block)
                print(block_str)
                log_lines.append(block_str)

        if not args.no_figs and trials_data:
            fig_path = OUT_DIR / f"{subject}_predictions.png"
            make_figure(subject, trials_data, fig_path)

    log_path.write_text("\n".join(log_lines))
    print(f"\nText log → {log_path}")
    print(f"Figures  → {OUT_DIR}")


if __name__ == "__main__":
    main()
