"""
visualize_fusion_predictions.py — Word-level fusion predictions across alpha values.

For a selected (subject, poem, session), runs predict() to get MEG scores and
compute_llm_scores() for LLM teacher-forced scores, then shows the top-1 prediction
at each alpha value side by side in a single figure.

Figure layout (one per trial):
    Rows    = word positions
    Columns = [# | Ground Truth | α=0.0 (MEG) | α=0.25 | … | α=1.0 (LLM)]
    Colors  = green  (correct top-1)
              red    (wrong top-1)
              grey   (no valid MEG window)

Metric header per alpha column: R@1 / BLEU-1 for that alpha.

Usage (from llm_decoder/):
    python -m unified.method1_analysis.fusion.visualize_fusion_predictions
    python -m unified.method1_analysis.fusion.visualize_fusion_predictions \\
        --subjects sub-01 sub-03 \\
        --poem poem1 --session 0 \\
        --alphas 0.0 0.25 0.5 0.75 1.0 \\
        --norm row_zscore \\
        --fusion_llm HuggingFaceTB/SmolLM2-360M
"""

import argparse
import functools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch

_HERE     = Path(__file__).parent
CKPT_ROOT = _HERE.parent.parent / "out" / "inference" / "bert_base_uncased"
OUT_DIR   = _HERE / "figures" / "predictions"

sys.path.insert(0, str(_HERE.parent.parent.parent))  # llm_decoder/ on path

SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]
DEFAULT_ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.0]


# ---------------------------------------------------------------------------
#  Colour palette
# ---------------------------------------------------------------------------

BG_CORRECT = "#c8e6c9"   # light green
BG_WRONG   = "#ffcdd2"   # light red
BG_NOWIN   = "#f0f0f0"   # light grey
TC_CORRECT = "#1b5e20"   # dark green text
TC_WRONG   = "#b71c1c"   # dark red text
TC_NOWIN   = "#888888"   # grey text
TC_TRUTH   = "#333333"   # ground truth column text


# ---------------------------------------------------------------------------
#  Per-trial figure
# ---------------------------------------------------------------------------

def make_figure(subject, poem, session, words, valid, vocab,
                meg_scores, llm_scores, alphas, normalization, out_path):
    """
    One figure for a single (subject, poem, session) trial.
    Columns: ground-truth | one column per alpha.
    """
    from unified.methods.fusion import fuse_scores
    from unified.evaluate import eval_option_a, eval_option_b

    N        = len(words)
    n_alphas = len(alphas)

    # Compute fused scores and per-alpha metrics
    alpha_data = {}   # alpha → {top1: List[str], r1: float, bleu: float}
    for alpha in alphas:
        fused     = fuse_scores(meg_scores, llm_scores, alpha, normalization)
        pred_top1 = [vocab[fused[i].argmax().item()] for i in range(N)]
        a = eval_option_a(fused, vocab, words, valid)
        b = eval_option_b(pred_top1, words, valid)
        alpha_data[alpha] = {
            "top1": pred_top1,
            "r1":   a["recall_at_k"][0] if a["recall_at_k"] else 0.0,
            "bleu": b["bleu1"],
        }

    # Figure dimensions
    # Columns: 1 (index) + 1 (ground truth) + n_alphas (predictions)
    n_cols   = 2 + n_alphas
    col_w    = 1.8
    row_h    = 0.28
    header_h = 1.5          # extra space at top for column labels + metrics
    fig_w    = col_w * n_cols
    fig_h    = header_h + row_h * N + 0.8   # 0.8 for legend
    fig_w    = max(fig_w, 8)
    fig_h    = max(fig_h, 6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(-header_h / row_h, N)
    ax.invert_yaxis()
    ax.axis("off")

    fig.suptitle(
        f"Fusion Predictions — {subject}  |  {poem}  sess={session}  "
        f"[{normalization}]",
        fontsize=11, fontweight="bold", y=0.99,
    )

    # ---- Column headers ----
    header_y = -header_h / row_h + 0.3   # just above row 0

    ax.text(0.1,        header_y + 1.0, "#",           fontsize=7, fontweight="bold",
            va="center", ha="left", color="#444444")
    ax.text(0.35,       header_y + 1.0, "Ground Truth", fontsize=7, fontweight="bold",
            va="center", ha="left", color="#444444")

    for ci, alpha in enumerate(alphas):
        cx   = 2 + ci
        label = f"α={alpha:.2f}"
        if alpha == 0.0:
            label += "\n(pure MEG)"
        elif alpha == 1.0:
            label += "\n(pure LLM)"
        r1   = alpha_data[alpha]["r1"]
        bleu = alpha_data[alpha]["bleu"]
        ax.text(cx + 0.5, header_y + 0.8, label,
                fontsize=7, fontweight="bold", va="center", ha="center",
                color="#222222")
        ax.text(cx + 0.5, header_y + 1.6,
                f"R@1={r1:.3f}\nBLEU={bleu:.3f}",
                fontsize=6.5, va="center", ha="center", color="#555555")

    # Separator line between header and rows
    ax.axhline(0, color="#cccccc", linewidth=0.8)

    # ---- Word rows ----
    for i, (w, v) in enumerate(zip(words, valid)):
        # Background for ground-truth column
        rect = mpatches.FancyBboxPatch(
            (0.0, i - 0.44), 2.0, 0.86,
            boxstyle="round,pad=0.02",
            linewidth=0, facecolor="#f7f7f7",
        )
        ax.add_patch(rect)

        # Word index
        ax.text(0.12, i, str(i), fontsize=6, color="#aaaaaa", va="center", ha="center")

        # Ground truth word
        ax.text(0.35, i, w, fontsize=7.5, va="center", ha="left",
                color=TC_TRUTH, fontweight="bold")

        # Alpha columns
        for ci, alpha in enumerate(alphas):
            cx   = 2 + ci
            pred = alpha_data[alpha]["top1"][i]

            if not v:
                bg = BG_NOWIN
                tc = TC_NOWIN
                txt = "—"
            elif pred == w:
                bg = BG_CORRECT
                tc = TC_CORRECT
                txt = pred
            else:
                bg = BG_WRONG
                tc = TC_WRONG
                txt = pred

            rect = mpatches.FancyBboxPatch(
                (cx + 0.04, i - 0.44), 0.90, 0.86,
                boxstyle="round,pad=0.02",
                linewidth=0, facecolor=bg,
            )
            ax.add_patch(rect)

            ax.text(cx + 0.49, i, txt,
                    fontsize=7, va="center", ha="center",
                    color=tc,
                    fontweight="bold" if (v and pred == w) else "normal")

            # Tiny tick marker
            if v:
                marker = "✓" if pred == w else "✗"
                mc = "#2e7d32" if pred == w else "#c62828"
                ax.text(cx + 0.93, i, marker, fontsize=5.5, va="center",
                        ha="right", color=mc, fontweight="bold")

        # Alternating row tint
        if i % 2 == 1:
            stripe = mpatches.FancyBboxPatch(
                (0.0, i - 0.44), n_cols, 0.86,
                boxstyle="square,pad=0",
                linewidth=0, facecolor="#00000008", zorder=0,
            )
            ax.add_patch(stripe)

    # ---- Vertical separator after ground-truth column ----
    ax.axvline(2.0, color="#cccccc", linewidth=0.8, ymin=0, ymax=1)

    # ---- Legend ----
    legend_patches = [
        mpatches.Patch(color=BG_CORRECT, label="Correct top-1"),
        mpatches.Patch(color=BG_WRONG,   label="Wrong top-1"),
        mpatches.Patch(color=BG_NOWIN,   label="No MEG window"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects",    nargs="+", default=None,
                   help="Subjects to visualize (default: first 4)")
    p.add_argument("--poems",       nargs="+", default=["poem1", "poem2"])
    p.add_argument("--sessions",    type=int, nargs="+", default=[0],
                   help="Session indices to visualize (default: 0)")
    p.add_argument("--condition",   default="lis")
    p.add_argument("--alphas",      type=float, nargs="+", default=DEFAULT_ALPHAS,
                   help="Alpha values to visualize (default: 0.0 0.25 0.5 0.75 1.0)")
    p.add_argument("--norm",        default="row_zscore",
                   choices=["logsoftmax", "row_zscore"],
                   help="Fusion normalization (default: row_zscore)")
    p.add_argument("--fusion_llm",  default="HuggingFaceTB/SmolLM2-360M",
                   help="LLM for teacher-forced scoring")
    p.add_argument("--device",      default="cpu")
    p.add_argument("--out_dir",     default=str(OUT_DIR))
    return p.parse_args()


def main():
    args     = parse_args()
    subjects = args.subjects or SUBJECTS[:4]
    device   = torch.device(args.device)
    out_dir  = Path(args.out_dir)
    alphas   = sorted(args.alphas)

    # Lazy imports (keeps startup fast when not on GPU node)
    from unified.predict import predict
    from unified.methods.fusion import load_fusion_llm, compute_llm_scores
    import unified.methods.models as _models

    # Cache BERT across predict() calls
    if not isinstance(_models.load_bert_hiddens, functools._lru_cache_wrapper):
        _models.load_bert_hiddens = functools.lru_cache(maxsize=4)(_models.load_bert_hiddens)

    # Load the fusion LLM once
    tokenizer, llm_model = load_fusion_llm(args.fusion_llm, device)

    llm_score_cache = {}   # poem → Tensor(N, |V|)
    vocab_cache     = {}   # poem → List[str]

    for subject in subjects:
        ckpt_dir = CKPT_ROOT / f"loso_{subject}"
        if not ckpt_dir.exists():
            print(f"[skip] no checkpoint for {subject}")
            continue

        print(f"\n{'='*60}")
        print(f"  {subject}")
        print(f"{'='*60}")

        for poem in args.poems:
            for session in args.sessions:
                print(f"  predict  {poem}  sess={session} ...", end=" ", flush=True)

                try:
                    result = predict(
                        subject   = subject,
                        session   = session,
                        condition = args.condition,
                        poem      = poem,
                        method    = "inference",
                        ckpt_dir  = str(ckpt_dir),
                        device    = args.device,
                    )
                except Exception as exc:
                    print(f"SKIP ({exc})")
                    continue

                words      = result["words"]
                vocab      = result["vocab"]
                meg_scores = result["scores"]   # Tensor(N, |V|)
                valid      = result["valid"]

                # LLM scores — cached per poem (text is the same across subjects)
                if poem not in llm_score_cache:
                    print("(computing LLM scores) ...", end=" ", flush=True)
                    llm_score_cache[poem] = compute_llm_scores(
                        words, vocab, tokenizer, llm_model, device
                    )
                    vocab_cache[poem] = vocab

                if vocab != vocab_cache[poem]:
                    print(f"SKIP (vocab mismatch for {poem})")
                    continue

                llm_scores = llm_score_cache[poem]
                print("done")

                # Print text summary to console
                from unified.methods.fusion import fuse_scores
                from unified.evaluate import eval_option_a, eval_option_b
                print(f"\n  {'#':>3}  {'Truth':<16}  " +
                      "  ".join(f"α={a:.2f}" for a in alphas))
                print(f"  {'-'*60}")
                pred_by_alpha = {}
                for alpha in alphas:
                    fused = fuse_scores(meg_scores, llm_scores, alpha, args.norm)
                    pred_by_alpha[alpha] = [vocab[fused[i].argmax().item()]
                                            for i in range(len(words))]
                for i, (w, v) in enumerate(zip(words, valid)):
                    row = f"  {i:3d}  {w:<16}  "
                    for alpha in alphas:
                        p = pred_by_alpha[alpha][i] if v else "—"
                        mark = "✓" if (v and p == w) else ("✗" if v else "—")
                        row += f"{p:<10}{mark}  "
                    print(row)

                # Compute and print per-alpha metrics
                print()
                print(f"  {'alpha':>8}  {'R@1':>7}  {'BLEU-1':>8}")
                for alpha in alphas:
                    fused = fuse_scores(meg_scores, llm_scores, alpha, args.norm)
                    pred  = pred_by_alpha[alpha]
                    a = eval_option_a(fused, vocab, words, valid)
                    b = eval_option_b(pred, words, valid)
                    print(f"  {alpha:8.2f}  {a['recall_at_k'][0]:7.3f}  {b['bleu1']:8.3f}")

                # Figure
                fname = f"{subject}_{poem}_sess{session}_{args.norm}.png"
                make_figure(
                    subject       = subject,
                    poem          = poem,
                    session       = session,
                    words         = words,
                    valid         = valid,
                    vocab         = vocab,
                    meg_scores    = meg_scores,
                    llm_scores    = llm_scores,
                    alphas        = alphas,
                    normalization = args.norm,
                    out_path      = out_dir / fname,
                )

    print(f"\nFigures → {out_dir}")


if __name__ == "__main__":
    main()
