"""
visualize_predictions_topk.py — Like visualize_predictions.py but with
graded colour coding based on where the correct word ranks.

Colour scheme per word position:
  Dark green   — correct word is rank 1  (top-1 hit)
  Light green  — correct word is rank 2–5 (top-5 hit, not top-1)
  Red          — correct word not in top 5
  Grey         — no valid MEG window

Each row also shows the top-5 predicted words so you can see what the
model actually considered.

Usage (from llm_decoder/ parent):
    python -m unified.inference_analysis.visualize_predictions_topk
    python -m unified.inference_analysis.visualize_predictions_topk --topk 10
    python -m unified.inference_analysis.visualize_predictions_topk --sessions 0 5
    python -m unified.inference_analysis.visualize_predictions_topk --no_figs
"""

import argparse
import functools
import sys
from pathlib import Path

import torch

_HERE     = Path(__file__).parent
CKPT_ROOT = _HERE.parent / "out" / "inference" / "bert_base_uncased"
OUT_DIR   = _HERE / "results" / "predictions_topk"

sys.path.insert(0, str(_HERE.parent.parent))

SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]
POEMS = ["poem1", "poem2"]


# ---------------------------------------------------------------------------
#  Rank computation
# ---------------------------------------------------------------------------

def get_rank(scores: torch.Tensor, vocab: list, word: str) -> int:
    """1-indexed rank of `word` in vocab by descending score. Returns len(vocab)+1 if absent."""
    if word not in vocab:
        return len(vocab) + 1
    idx = vocab.index(word)
    rank = int((scores > scores[idx]).sum().item()) + 1
    return rank


def get_topk_words(scores: torch.Tensor, vocab: list, k: int) -> list:
    """Top-k vocab words by descending score."""
    topk_idx = scores.topk(min(k, len(vocab))).indices.tolist()
    return [vocab[i] for i in topk_idx]


# ---------------------------------------------------------------------------
#  Text summary
# ---------------------------------------------------------------------------

def print_trial(subject, poem, session, words, pred_top1, valid, vocab_per_pos,
                scores_per_pos, topk, metrics):
    print(f"\n  Subject={subject}  Poem={poem}  Session={session}  "
          f"R@1={metrics['R@1']:.3f}  MRR={metrics['MRR']:.3f}  "
          f"WER={metrics['WER']:.3f}  BLEU-1={metrics['BLEU1']:.3f}")
    print(f"  {'#':>3}  {'Truth':<14}  {'Rank':>4}  {'Top-1':<14}  Top-{topk} predictions")
    print(f"  {'-'*75}")
    for i, (w, p, v) in enumerate(zip(words, pred_top1, valid)):
        if not v:
            print(f"  {i:3d}  {w:<14}  {'—':>4}  {'—':<14}  [no MEG window]")
            continue
        rank  = get_rank(scores_per_pos[i], vocab_per_pos, w)
        top5  = get_topk_words(scores_per_pos[i], vocab_per_pos, topk)
        top5_str = "  ".join(
            f"[{ww}]" if ww == w else ww for ww in top5
        )
        hit1 = "✓" if rank == 1 else f"@{rank}" if rank <= topk else "✗"
        print(f"  {i:3d}  {w:<14}  {hit1:>4}  {p:<14}  {top5_str}")


# ---------------------------------------------------------------------------
#  Figure
# ---------------------------------------------------------------------------

def make_figure(subject, trials_data, out_path, topk):
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

    # Each trial gets two columns: truth+rank | top-k predictions
    n_cols   = n_trials * 2
    fig_w    = 2.5 * n_cols
    fig_h    = max(6, 0.26 * max_words + 3.0)
    fig, all_axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h))
    if n_cols == 1:
        all_axes = [all_axes]

    fig.suptitle(f"Inference Method — {subject}  (top-{topk} shown)",
                 fontsize=13, fontweight="bold", y=0.99)

    # Colour palette
    C_TOP1   = "#2e7d32"   # dark green text
    C_TOPK   = "#558b2f"   # medium green text
    C_MISS   = "#c62828"   # red text
    C_NOWIN  = "#999999"   # grey text
    BG_TOP1  = "#c8e6c9"   # dark green bg
    BG_TOPK  = "#f1f8e9"   # light green bg
    BG_MISS  = "#ffcdd2"   # light red bg
    BG_NOWIN = "#f5f5f5"   # light grey bg

    axes_pairs = [(all_axes[2*i], all_axes[2*i+1]) for i in range(n_trials)]

    for (ax_left, ax_right), td in zip(axes_pairs, trials_data):
        words    = td["words"]
        pred     = td["pred_top1"]
        valid    = td["valid"]
        scores_t = td["scores"]       # (N, |V|) tensor
        vocab    = td["vocab"]
        metrics  = td["metrics"]
        poem     = td["poem"]
        session  = td["session"]
        N        = len(words)

        for ax in (ax_left, ax_right):
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, N - 0.5)
            ax.invert_yaxis()
            ax.axis("off")

        title = (f"{poem}  sess={session}\n"
                 f"R@1={metrics['R@1']:.3f}  MRR={metrics['MRR']:.3f}\n"
                 f"top-{topk} hit: {metrics[f'R@{topk}']:.3f}")
        ax_left.set_title(title, fontsize=8, loc="left", pad=4)

        # Column headers
        ax_left.text(0.02, -0.45,  "#",       fontsize=7, fontweight="bold", va="center")
        ax_left.text(0.12, -0.45,  "Truth",   fontsize=7, fontweight="bold", va="center")
        ax_left.text(0.72, -0.45,  "Rank",    fontsize=7, fontweight="bold", va="center")
        ax_right.text(0.02, -0.45, f"Top-{topk} predictions", fontsize=7,
                      fontweight="bold", va="center")

        for i, (w, p, v) in enumerate(zip(words, pred, valid)):
            if not v:
                bg = BG_NOWIN
                tc = C_NOWIN
                rank = None
            else:
                rank = get_rank(scores_t[i], vocab, w)
                if rank == 1:
                    bg, tc = BG_TOP1, C_TOP1
                elif rank <= topk:
                    bg, tc = BG_TOPK, C_TOPK
                else:
                    bg, tc = BG_MISS, C_MISS

            # Background strip (spans both sub-axes via figure coords, but
            # we just draw in each axis separately)
            for ax in (ax_left, ax_right):
                rect = mpatches.FancyBboxPatch(
                    (0.0, i - 0.44), 1.0, 0.86,
                    boxstyle="round,pad=0.02",
                    linewidth=0, facecolor=bg,
                )
                ax.add_patch(rect)

            # Left axis: index + ground truth + rank badge
            ax_left.text(0.02, i, str(i), fontsize=6, color="#888888", va="center")
            ax_left.text(0.12, i, w, fontsize=7.5, va="center", color=tc,
                         fontweight="bold" if rank == 1 else "normal")
            if v:
                badge = f"#{rank}" if rank <= topk else f"#{rank}"
                badge_col = C_TOP1 if rank == 1 else (C_TOPK if rank <= topk else C_MISS)
                ax_left.text(0.75, i, badge, fontsize=6.5, va="center",
                             ha="center", color=badge_col, fontweight="bold")
            else:
                ax_left.text(0.75, i, "—", fontsize=6.5, va="center",
                             ha="center", color=C_NOWIN)

            # Right axis: top-k words, bold+underline if matches truth
            if v:
                top_words = get_topk_words(scores_t[i], vocab, topk)
                x_step = 0.95 / topk
                for j, tw in enumerate(top_words):
                    is_correct = (tw == w)
                    ax_right.text(
                        0.02 + j * x_step, i,
                        tw,
                        fontsize=6.5 if topk <= 5 else 5.5,
                        va="center",
                        color=C_TOP1 if (is_correct and j == 0) else
                              (C_TOPK if is_correct else "#444444"),
                        fontweight="bold" if is_correct else "normal",
                        style="italic" if is_correct else "normal",
                    )
            else:
                ax_right.text(0.02, i, "[no MEG window]", fontsize=6.5,
                              va="center", color=C_NOWIN, style="italic")

    # Legend
    legend_patches = [
        mpatches.Patch(color=BG_TOP1,  label="Top-1 correct"),
        mpatches.Patch(color=BG_TOPK,  label=f"Top-{topk} correct (not top-1)"),
        mpatches.Patch(color=BG_MISS,  label=f"Not in top-{topk}"),
        mpatches.Patch(color=BG_NOWIN, label="No MEG window"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Figure → {out_path}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--topk",     type=int, default=5)
    p.add_argument("--sessions", type=int, nargs=2, default=[0, 5])
    p.add_argument("--device",   default="cpu")
    p.add_argument("--no_figs",  action="store_true")
    p.add_argument("--subjects", nargs="+", default=None)
    return p.parse_args()


def _run_predict(subject, poem, session, ckpt_dir, device):
    from unified.predict import predict
    from unified.evaluate import eval_option_a, eval_option_b
    import unified.methods.models as _models
    # Cache BERT across calls (same args → same output)
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
    return result   # caller computes metrics with chosen topk


def main():
    args     = parse_args()
    topk     = args.topk
    subjects = args.subjects or SUBJECTS
    sessions = args.sessions
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
                    result = _run_predict(subject, poem, session, ckpt_dir, args.device)
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

                from unified.evaluate import eval_option_a, eval_option_b
                a = eval_option_a(result["scores"], result["vocab"],
                                  result["words"], result["valid"])
                b = eval_option_b(result["pred_top1"], result["words"], result["valid"])

                metrics = {
                    "R@1":        a["recall_at_k"][0],
                    f"R@{topk}":  a["recall_at_k"][topk - 1] if topk <= len(a["recall_at_k"]) else 1.0,
                    "MRR":        a["mrr"],
                    "WER":        b["wer"],
                    "BLEU1":      b["bleu1"],
                }
                print(f"R@1={metrics['R@1']:.3f}  R@{topk}={metrics[f'R@{topk}']:.3f}  MRR={metrics['MRR']:.3f}")

                td = dict(
                    poem=poem, session=session,
                    words=result["words"],
                    pred_top1=result["pred_top1"],
                    valid=result["valid"],
                    scores=result["scores"],
                    vocab=result["vocab"],
                    metrics=metrics,
                )
                trials_data.append(td)

                # Text log
                block = []
                block.append(
                    f"\nSubject={subject}  Poem={poem}  Session={session}  "
                    f"R@1={metrics['R@1']:.3f}  R@{topk}={metrics[f'R@{topk}']:.3f}  "
                    f"MRR={metrics['MRR']:.3f}  WER={metrics['WER']:.3f}"
                )
                block.append(f"  {'#':>3}  {'Truth':<14}  {'Rank':>5}  Top-{topk} predictions")
                block.append(f"  {'-'*70}")
                for i, (w, p, v) in enumerate(zip(
                        result["words"], result["pred_top1"], result["valid"])):
                    if not v:
                        block.append(f"  {i:3d}  {w:<14}  {'—':>5}  [no MEG window]")
                        continue
                    rank     = get_rank(result["scores"][i], result["vocab"], w)
                    top_words= get_topk_words(result["scores"][i], result["vocab"], topk)
                    top_str  = "  ".join(f"[{ww}]" if ww == w else ww for ww in top_words)
                    hit      = f"#{rank}"
                    block.append(f"  {i:3d}  {w:<14}  {hit:>5}  {top_str}")
                block_str = "\n".join(block)
                print(block_str)
                log_lines.append(block_str)

        if not args.no_figs and trials_data:
            fig_path = OUT_DIR / f"{subject}_predictions_top{topk}.png"
            make_figure(subject, trials_data, fig_path, topk)

    log_path = OUT_DIR / f"predictions_top{topk}_text.txt"
    log_path.write_text("\n".join(log_lines))
    print(f"\nText log → {log_path}")
    print(f"Figures  → {OUT_DIR}")


if __name__ == "__main__":
    main()
