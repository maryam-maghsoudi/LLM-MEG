"""
visualize_fusion_predictions.py

Word-level prediction table for teacher-forced fusion across multiple alpha values.

Columns (left to right):
    Truth | MEG only (α=0) | Best α* | α=0.10 | α=0.20 | α=0.50 | α=0.70 | α=1.00

Cell colors:
    green  = prediction matches ground truth
    pink   = prediction is wrong
    grey   = invalid MEG window (no encoder output)

Usage (run from inside contrastive_multimodal/):
    python visualize_fusion_predictions.py --heldout_subject sub-01
    python visualize_fusion_predictions.py --heldout_subject sub-01 \\
        --sessions 0 5 --normalization row_zscore
    python visualize_fusion_predictions.py --heldout_subject sub-01 \\
        --best_alpha 0.35          # override; skip loading JSON
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from fusion_teacher_forced import (
    load_stage1_checkpoint,
    build_candidate_bank,
    run_encoder_on_trial,
    meg_scores_to_type_level,
    load_fusion_llm,
    compute_llm_scores,
    fuse_scores,
    _load_onsets,
    ALPHA_GRID,
)
from new_dataset import MEGContinuousTrialDataset, collate_continuous_trials, MEG_BASE
from splits import make_loso_splits


# ---------------------------------------------------------------------------
#  Column spec
# ---------------------------------------------------------------------------

FIXED_ALPHAS = [0.10, 0.20, 0.50, 0.70, 1.00]
# Full ordered column list: MEG-only, best-alpha, then fixed alphas.
# "best" placeholder is replaced with the actual best alpha at runtime.


# ---------------------------------------------------------------------------
#  Figure drawing
# ---------------------------------------------------------------------------

def _col_label(alpha: float, best_alpha: float) -> str:
    star = "*" if abs(alpha - best_alpha) < 1e-6 else ""
    if alpha == 0.0:
        return f"MEG\nonly{star}"
    if alpha == 1.0:
        return f"LLM\nonly{star}"
    return f"α={alpha:.2f}{star}"


def draw_panel(ax, words, valid, preds_by_alpha, col_alphas, best_alpha,
               bleu1_by_alpha, poem, session, normalization):
    """
    Draw one trial panel on ax.

    preds_by_alpha : {alpha: List[str]}  — top-1 prediction per position
    col_alphas     : ordered list of alpha values (MEG-only first, then best, then fixed)
    bleu1_by_alpha : {alpha: float}  — per-alpha BLEU-1 for header display
    """
    import matplotlib.patches as mpatches

    TRUTH_W = 2.6         # width of truth column (data units)
    PRED_W  = 1.85        # width of each prediction column
    ROW_H   = 1.0         # height of each word row
    HEADER_H = 3.5        # rows reserved for title + column headers

    n_rows   = len(words)
    n_pcols  = len(col_alphas)
    total_w  = TRUTH_W + n_pcols * PRED_W

    ax.set_xlim(0, total_w)
    ax.set_ylim(-HEADER_H, n_rows)
    ax.invert_yaxis()
    ax.axis("off")

    # ── Panel title ──────────────────────────────────────────────────────────
    title_lines = [
        f"{poem}  sess={session}",
        f"norm={normalization}",
    ]
    ax.text(0.0, -HEADER_H + 0.3, "\n".join(title_lines),
            fontsize=7, va="top", ha="left", fontweight="bold",
            color="#222222")

    # ── Column headers ────────────────────────────────────────────────────────
    ax.text(TRUTH_W / 2, -1.6, "Truth",
            fontsize=7, va="center", ha="center", fontweight="bold")

    for ci, alpha in enumerate(col_alphas):
        cx = TRUTH_W + ci * PRED_W + PRED_W / 2
        label = _col_label(alpha, best_alpha)
        bleu  = bleu1_by_alpha.get(alpha, float("nan"))
        header_text = f"{label}\nBLEU={bleu*100:.1f}%"
        ax.text(cx, -1.6, header_text,
                fontsize=5.5, va="center", ha="center",
                color="#000080" if abs(alpha - best_alpha) < 1e-6 else "#333333",
                fontweight="bold" if abs(alpha - best_alpha) < 1e-6 else "normal")

    # ── Separator line below headers ─────────────────────────────────────────
    ax.plot([0, total_w], [-0.5, -0.5], color="#aaaaaa", linewidth=0.6)

    # ── Word rows ─────────────────────────────────────────────────────────────
    for i, (word, v) in enumerate(zip(words, valid)):
        y_top = i - 0.46
        y_ctr = i

        # Truth cell (always white background)
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.0, y_top), TRUTH_W - 0.05, 0.88,
            boxstyle="round,pad=0.01", linewidth=0, facecolor="#f9f9f9",
        ))
        ax.text(0.18, y_ctr, str(i), fontsize=5, color="#aaaaaa", va="center")
        ax.text(0.42, y_ctr, word, fontsize=6.5, va="center",
                fontweight="bold", color="#222222")

        # Prediction cells
        for ci, alpha in enumerate(col_alphas):
            x0  = TRUTH_W + ci * PRED_W
            pred = preds_by_alpha.get(alpha, [""] * len(words))[i]

            if not v:
                bg    = "#eeeeee"   # grey — no MEG window
                fg    = "#999999"
            elif pred == word:
                bg    = "#d4edda"   # green — correct
                fg    = "#1a472a"
            else:
                bg    = "#f8d7da"   # pink — wrong
                fg    = "#7b2d30"

            ax.add_patch(mpatches.FancyBboxPatch(
                (x0 + 0.04, y_top), PRED_W - 0.10, 0.88,
                boxstyle="round,pad=0.01", linewidth=0, facecolor=bg,
            ))
            ax.text(x0 + PRED_W / 2, y_ctr, pred,
                    fontsize=6, va="center", ha="center", color=fg)

    # ── Bottom separator ──────────────────────────────────────────────────────
    ax.plot([0, total_w], [n_rows - 0.5, n_rows - 0.5],
            color="#aaaaaa", linewidth=0.6)


def make_figure(subject, panels, col_alphas, best_alpha, normalization, out_path):
    """
    panels : list of dicts — one per trial (poem+session):
        words, valid, preds_by_alpha, bleu1_by_alpha, poem, session
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n_panels  = len(panels)
    max_words = max(len(p["words"]) for p in panels)
    PRED_W    = 1.85
    TRUTH_W   = 2.6
    n_pcols   = len(col_alphas)
    HEADER_H  = 3.5

    # Each panel is one subplot; lay them in a row
    panel_w_in = (TRUTH_W + n_pcols * PRED_W) * 0.22   # inches per data unit
    panel_h_in = max_words * 0.19 + 2.0
    fig_w = panel_w_in * n_panels + 0.5
    fig_h = panel_h_in + 1.2

    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(f"Teacher-forced fusion — {subject}", fontsize=12,
                 fontweight="bold", y=0.995)

    for ax, panel in zip(axes, panels):
        draw_panel(
            ax,
            words          = panel["words"],
            valid          = panel["valid"],
            preds_by_alpha = panel["preds_by_alpha"],
            col_alphas     = col_alphas,
            best_alpha     = best_alpha,
            bleu1_by_alpha = panel["bleu1_by_alpha"],
            poem           = panel["poem"],
            session        = panel["session"],
            normalization  = normalization,
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color="#d4edda", label="Correct"),
        mpatches.Patch(color="#f8d7da", label="Wrong"),
        mpatches.Patch(color="#eeeeee", label="No MEG window"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def find_best_alpha(fusion_json_path: str, metric: str = "bleu1") -> float:
    """Return the alpha with the highest aggregate metric in the fusion JSON."""
    with open(fusion_json_path) as f:
        data = json.load(f)
    best_a, best_v = 0.0, -1.0
    for a_str, vals in data["results"].items():
        v = vals.get(metric, 0.0)
        if v > best_v:
            best_v = v
            best_a = float(a_str)
    print(f"Best alpha from {Path(fusion_json_path).name}: α={best_a:.2f}  {metric}={best_v*100:.2f}%")
    return best_a


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  subject={args.heldout_subject}  norm={args.normalization}")

    # ── Best alpha ───────────────────────────────────────────────────────────
    if args.best_alpha is not None:
        best_alpha = args.best_alpha
        print(f"Using user-specified best alpha: {best_alpha}")
    else:
        llm_tag       = args.llm_name.replace("/", "_")
        fusion_json   = os.path.join(
            args.fusion_results_dir,
            f"{args.heldout_subject}_fusion_{llm_tag}_{args.normalization}.json",
        )
        if not os.path.exists(fusion_json):
            print(f"[warn] Fusion JSON not found: {fusion_json}")
            print("       Using α=0.50 as fallback. Run fusion_teacher_forced.py first, "
                  "or pass --best_alpha explicitly.")
            best_alpha = 0.50
        else:
            best_alpha = find_best_alpha(fusion_json, metric="bleu1")

    # ── Column order: MEG-only, best, then fixed alphas (skip if duplicate) ──
    col_alphas = [0.0]
    if abs(best_alpha - 0.0) > 1e-6:
        col_alphas.append(best_alpha)
    for a in FIXED_ALPHAS:
        if all(abs(a - ca) > 1e-6 for ca in col_alphas):
            col_alphas.append(a)
    print(f"Alpha columns: {col_alphas}")

    # ── Models ───────────────────────────────────────────────────────────────
    encoder, word_head, pooling_module, pooling_mode, _ = load_stage1_checkpoint(
        args.stage1_checkpoint_path, device
    )
    teacher_cache = torch.load(args.teacher_cache_path, weights_only=False)
    bank_vectors, _, bank_word_types, *_ = build_candidate_bank(teacher_cache)
    bank_vectors = bank_vectors.to(device)

    tokenizer, llm_model = load_fusion_llm(args.llm_name, device)

    # ── Pre-compute LLM scores once per poem ─────────────────────────────────
    llm_cache:   Dict[str, torch.Tensor] = {}
    vocab_cache: Dict[str, List[str]]    = {}
    for poem in ("poem1", "poem2"):
        onsets     = _load_onsets(poem)
        word_texts = [e["word"].strip().lower() for e in onsets]
        vocab      = sorted(set(word_texts))
        llm_cache[poem]   = compute_llm_scores(word_texts, vocab, tokenizer, llm_model, device)
        vocab_cache[poem] = vocab

    # ── Pre-compute per-alpha BLEU-1 from fusion JSON (for column headers) ───
    llm_tag     = args.llm_name.replace("/", "_")
    fusion_json = os.path.join(
        args.fusion_results_dir,
        f"{args.heldout_subject}_fusion_{llm_tag}_{args.normalization}.json",
    )
    json_bleu: Dict[float, float] = {}
    if os.path.exists(fusion_json):
        with open(fusion_json) as f:
            fd = json.load(f)
        for a_str, vals in fd["results"].items():
            json_bleu[float(a_str)] = vals.get("bleu1", 0.0)

    # ── Select test trials for the requested sessions ─────────────────────────
    splits = make_loso_splits(args.heldout_subject)
    subj   = args.heldout_subject
    target_trials = [
        (subj, poem, sess)
        for poem in ["poem1", "poem2"]
        for sess in args.sessions
    ]

    # Only keep trials present in the test split
    test_trial_set = {(s, po, se) for s, po, se in splits["test"]["trials"]}
    target_trials  = [t for t in target_trials if t in test_trial_set]
    if not target_trials:
        print(f"[error] No matching test trials found for sessions {args.sessions}")
        return

    ds = MEGContinuousTrialDataset(
        target_trials,
        word_filter=splits["test"]["word_filter"],
        meg_base=args.meg_base,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=collate_continuous_trials)
    print(f"Running fusion on {len(ds)} selected trials ...")

    # ── Collect per-trial predictions ─────────────────────────────────────────
    panels = []
    for batch in loader:
        z_word, valid_mask, word_texts, poem = run_encoder_on_trial(
            batch, encoder, word_head, pooling_module, pooling_mode, device
        )
        session = int(batch["session"][0]) if "session" in batch else "?"

        vocab      = vocab_cache[poem]
        llm_scores = llm_cache[poem]
        meg_scores = meg_scores_to_type_level(
            z_word.to(device), bank_vectors, bank_word_types, vocab
        )

        preds_by_alpha: Dict[float, List[str]] = {}
        for alpha in col_alphas:
            fused = fuse_scores(meg_scores, llm_scores, alpha, args.normalization)
            preds_by_alpha[alpha] = [
                vocab[int(fused[i].argmax().item())]
                for i in range(fused.shape[0])
            ]

        # Per-alpha BLEU-1 for this trial's column headers (from JSON if available)
        bleu1_by_alpha = {a: json_bleu.get(a, float("nan")) for a in col_alphas}

        panels.append({
            "words":          word_texts,
            "valid":          valid_mask.tolist(),
            "preds_by_alpha": preds_by_alpha,
            "bleu1_by_alpha": bleu1_by_alpha,
            "poem":           poem,
            "session":        session,
        })
        print(f"  {poem} sess={session}: {sum(valid_mask.tolist())} valid positions")

    # ── Figure ────────────────────────────────────────────────────────────────
    if panels:
        llm_tag  = args.llm_name.replace("/", "_")
        sess_str = "_".join(str(s) for s in args.sessions)
        out_path = os.path.join(
            args.out_dir,
            f"{args.heldout_subject}_fusion_predictions_{llm_tag}_{args.normalization}_sess{sess_str}.png",
        )
        make_figure(
            subject       = args.heldout_subject,
            panels        = panels,
            col_alphas    = col_alphas,
            best_alpha    = best_alpha,
            normalization = args.normalization,
            out_path      = out_path,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Word-level prediction table for teacher-forced fusion across alpha values."
    )
    p.add_argument("--heldout_subject", type=str, default="sub-01")
    p.add_argument("--sessions", type=int, nargs="+", default=[0, 5],
                   help="Sessions to show (one panel per poem×session). Default: 0 5")
    p.add_argument("--normalization", type=str, default="logsoftmax",
                   choices=["logsoftmax", "row_zscore"])
    p.add_argument("--llm_name", type=str, default="gpt2")
    p.add_argument("--best_alpha", type=float, default=None,
                   help="Override best alpha (skips loading fusion JSON).")
    p.add_argument("--stage1_checkpoint_path", type=str, default=None,
                   help="Defaults to checkpoints/joint_annealed_exact/stage1_best_{subject}_*.pt")
    p.add_argument("--teacher_cache_path", type=str, default="teacher_cache.pt")
    p.add_argument("--meg_base", type=str, default=None)
    p.add_argument("--fusion_results_dir", type=str, default="fusion_results",
                   help="Directory containing *_fusion_*.json files (for best-alpha and BLEU-1 headers).")
    p.add_argument("--out_dir", type=str, default="fusion_results/figures",
                   help="Output directory for PNG figures.")
    args = p.parse_args()

    if args.stage1_checkpoint_path is None:
        args.stage1_checkpoint_path = (
            f"checkpoints/joint_annealed_exact/"
            f"stage1_best_{args.heldout_subject}_joint_annealed_exact.pt"
        )

    main(args)
