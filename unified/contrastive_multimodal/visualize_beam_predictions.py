"""
visualize_beam_predictions.py

Word-level prediction table for beam-search fusion across multiple alpha values.
Reads predicted sequences from the beam fusion JSON produced by fusion_beamsearch.py.

Columns: Truth | MEG-only (α=0) | Best-α* | α=0.10 | α=0.20 | α=0.50 | α=0.70 | α=1.00
Colors: green=correct, pink=wrong, grey=no MEG window.

Usage (run from inside contrastive_multimodal/):
    python visualize_beam_predictions.py --heldout_subject sub-01
    python visualize_beam_predictions.py --heldout_subject sub-01 \\
        --sessions 0 5 --normalization row_zscore --beam_width 5 --top_k 5
    python visualize_beam_predictions.py --heldout_subject sub-01 \\
        --alphas 0.0 0.25 0.5 0.75 1.0 --best_alpha 0.35
"""

import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))
from visualize_fusion_predictions import draw_panel, make_figure, find_best_alpha

FIXED_ALPHAS = [0.10, 0.20, 0.50, 0.70, 1.00]


def main(args):
    # ── Locate beam JSON ─────────────────────────────────────────────────────
    if args.beam_json_path:
        json_path = args.beam_json_path
    else:
        llm_tag   = args.llm_name.replace("/", "_")
        json_path = os.path.join(
            args.fusion_results_dir,
            f"{args.heldout_subject}_beamfusion_B{args.beam_width}"
            f"_top{args.top_k}_{llm_tag}_{args.normalization}.json",
        )
    if not os.path.exists(json_path):
        print(f"[error] Beam JSON not found: {json_path}")
        print("        Run fusion_beamsearch.py first.")
        return

    with open(json_path) as f:
        data = json.load(f)

    if "trials" not in data:
        print("[error] JSON has no 'trials' key.")
        print("        Re-run fusion_beamsearch.py to regenerate with per-trial predictions.")
        return

    # ── Best alpha ───────────────────────────────────────────────────────────
    best_alpha = (args.best_alpha if args.best_alpha is not None
                  else find_best_alpha(json_path, metric="bleu1"))

    # ── Column order ─────────────────────────────────────────────────────────
    fixed = args.alphas if args.alphas else FIXED_ALPHAS
    col_alphas = [0.0]
    if abs(best_alpha - 0.0) > 1e-6:
        col_alphas.append(best_alpha)
    for a in fixed:
        if all(abs(a - ca) > 1e-6 for ca in col_alphas):
            col_alphas.append(a)
    print(f"Alpha columns: {col_alphas}")

    # ── Per-alpha aggregate BLEU-1 for column headers ────────────────────────
    json_bleu = {
        float(a_str): vals.get("bleu1", 0.0)
        for a_str, vals in data["results"].items()
    }

    # ── Index trials by (poem, session) ──────────────────────────────────────
    # trials_index[(poem, session)][alpha] = pred_sequence
    trials_index: Dict[tuple, Dict[float, List[str]]] = {}
    word_texts_index: Dict[tuple, List[str]]           = {}
    valid_mask_index: Dict[tuple, List[bool]]          = {}

    for a_str, trial_list in data["trials"].items():
        alpha = float(a_str)
        for rec in trial_list:
            key = (rec["poem"], rec["session"])
            if key not in trials_index:
                trials_index[key]    = {}
                word_texts_index[key] = rec["word_texts"]
                valid_mask_index[key] = rec["valid_mask"]
            trials_index[key][alpha] = rec["pred_sequence"]

    # ── Build panels for requested sessions ──────────────────────────────────
    panels = []
    for poem in ("poem1", "poem2"):
        for sess in args.sessions:
            key = (poem, sess)
            if key not in trials_index:
                print(f"  [skip] {poem} sess={sess} not found in JSON")
                continue
            preds_by_alpha = {
                alpha: trials_index[key].get(alpha, [""] * len(word_texts_index[key]))
                for alpha in col_alphas
            }
            bleu1_by_alpha = {a: json_bleu.get(a, float("nan")) for a in col_alphas}
            n_valid = sum(valid_mask_index[key])
            print(f"  {poem} sess={sess}: {n_valid} valid positions")
            panels.append({
                "words":          word_texts_index[key],
                "valid":          valid_mask_index[key],
                "preds_by_alpha": preds_by_alpha,
                "bleu1_by_alpha": bleu1_by_alpha,
                "poem":           poem,
                "session":        sess,
            })

    if not panels:
        print("[error] No panels found for requested sessions.")
        return

    # ── Save figure ──────────────────────────────────────────────────────────
    llm_tag  = data["llm_name"].replace("/", "_")
    norm     = data["normalization"]
    B        = data["beam_width"]
    topk     = data["top_k"]
    sess_str = "_".join(str(s) for s in args.sessions)
    out_path = os.path.join(
        args.out_dir,
        f"{args.heldout_subject}_beam_predictions_B{B}_top{topk}"
        f"_{llm_tag}_{norm}_sess{sess_str}.png",
    )
    make_figure(
        subject       = args.heldout_subject,
        panels        = panels,
        col_alphas    = col_alphas,
        best_alpha    = best_alpha,
        normalization = norm,
        out_path      = out_path,
        title         = f"Beam-search fusion (B={B}, top_k={topk}) — {args.heldout_subject}",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Word-level prediction table for beam-search fusion."
    )
    p.add_argument("--heldout_subject", type=str, default="sub-01")
    p.add_argument("--sessions", type=int, nargs="+", default=[0, 5],
                   help="Sessions to show (one panel per poem×session). Default: 0 5")
    p.add_argument("--normalization", type=str, default="logsoftmax",
                   choices=["logsoftmax", "row_zscore"])
    p.add_argument("--llm_name", type=str, default="gpt2")
    p.add_argument("--beam_width", type=int, default=5,
                   help="Must match the beam_width used in fusion_beamsearch.py.")
    p.add_argument("--top_k", type=int, default=5,
                   help="Must match the top_k used in fusion_beamsearch.py.")
    p.add_argument("--best_alpha", type=float, default=None,
                   help="Override best alpha (skips reading from JSON results).")
    p.add_argument("--alphas", type=float, nargs="+", default=None,
                   help="Override fixed alpha columns. Default: 0.10 0.20 0.50 0.70 1.00")
    p.add_argument("--beam_json_path", type=str, default=None,
                   help="Explicit path to beam fusion JSON. Auto-detected from other flags if not set.")
    p.add_argument("--fusion_results_dir", type=str, default="fusion_results",
                   help="Directory containing beam fusion JSONs.")
    p.add_argument("--out_dir", type=str, default="fusion_results/figures",
                   help="Output directory for PNG figures.")
    args = p.parse_args()
    main(args)
