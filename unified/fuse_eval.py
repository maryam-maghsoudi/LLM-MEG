"""
fuse_eval.py — Alpha-sweep fusion of MEG scores with LLM next-word scores.

For each test trial the script:
  1. Runs predict() to get MEG-based scores over the eval vocabulary.
  2. Computes teacher-forced LLM next-word scores over the same vocabulary
     (cached per poem — word texts do not vary across subjects/sessions).
  3. Fuses the two with log-linear mixing:
         fused = (1-alpha)*normalize(meg) + alpha*normalize(llm)
  4. Evaluates all metrics (R@1, MRR, BLEU-1, WER, word accuracy) for each alpha.
  5. Saves per-fold results alongside each checkpoint.

Output
------
Per-fold JSON written to:
    {ckpt_dir}/fusion/fusion_{fusion_llm_tag}_{normalization}.json

Structure:
    {
      "meta": {
          "method": ..., "eval_scheme": ..., "heldout": ...,
          "fusion_llm": ..., "fusion_normalization": ..., "alphas": [...]
      },
      "alphas": [0.0, 0.05, ..., 1.0],
      "scale_diagnostics": {...},
      "per_alpha": {
          "0.0":  {"mean_R@1": ..., "mean_MRR": ..., "mean_accuracy": ...,
                   "mean_BLEU1": ..., "mean_WER": ..., "n_trials": ...},
          "0.5":  {...},
          ...
      }
    }

Usage examples
--------------
# LOSO, all 13 subjects, default alpha grid
python -m unified.fuse_eval \\
    --method inference \\
    --eval_scheme loso \\
    --fusion_llm_name HuggingFaceTB/SmolLM2-360M \\
    --device cuda

# LOSO, single subject
python -m unified.fuse_eval \\
    --method twostage \\
    --eval_scheme loso \\
    --heldout sub-01 \\
    --llm_name HuggingFaceTB/SmolLM2-360M \\
    --fusion_llm_name HuggingFaceTB/SmolLM2-360M \\
    --fusion_normalization row_zscore \\
    --device cuda
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))   # llm_decoder/ on path

from unified.data.splits import (
    make_loso_splits, make_session_cv_splits, make_stimulus_splits, SUBJECTS,
)
from unified.predict import predict
from unified.methods.fusion import (
    load_fusion_llm, compute_llm_scores, sweep_alphas, sweep_alphas_beam,
)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Alpha-sweep LLM+MEG fusion evaluation")

    p.add_argument("--method",      required=True,
                   choices=["inference", "twostage", "interleaved"])
    p.add_argument("--eval_scheme", required=True,
                   choices=["loso", "session_cv", "stimulus"])

    # Split selectors (same as train.py)
    p.add_argument("--heldout",  default=None,
                   help="Single held-out subject (loso only); omit to run all 13")
    p.add_argument("--fold",     type=int, default=None,
                   help="Fold index 0–4 (session_cv only); omit to run all 5 folds")
    p.add_argument("--n_lines",  type=int, default=2, choices=[2, 4])

    # Control (must match trained model)
    p.add_argument("--control",  default="none",
                   choices=["none", "zero", "shuffle_time"])

    # Model tags — used to reconstruct ckpt_dir paths
    p.add_argument("--llm_name",  default="HuggingFaceTB/SmolLM2-360M",
                   help="LLM used during training (Methods 2 and 3)")
    p.add_argument("--bert_name", default="bert-base-uncased",
                   help="BERT used during training (Method 1)")

    # Fusion LLM (can differ from training LLM)
    p.add_argument("--fusion_llm_name", default=None,
                   help="LLM for computing teacher-forced fusion scores "
                        "(default: same as --llm_name, or gpt2 for Method 1)")

    # Alpha grid
    p.add_argument("--alphas", type=float, nargs="+", default=None,
                   help="Alpha values to sweep (default: 0.0, 0.05, ..., 1.0)")

    # Fusion normalization
    p.add_argument("--fusion_normalization", default="logsoftmax",
                   choices=["logsoftmax", "row_zscore"],
                   help="Per-row normalization before mixing MEG and LLM scores.")

    # Paths
    p.add_argument("--out_root", default=str(_HERE / "out"),
                   help="Root dir containing trained model checkpoints")

    p.add_argument("--device",    default=None)
    p.add_argument("--condition", default="lis",
                   help="MEG condition suffix (default: lis)")

    # Fixed evaluation vocabulary
    p.add_argument("--closed_vocab_path", default=None,
                   help="Path to a vocab_info.json with a 'restricted_words' key. "
                        "When set, all trials are evaluated over this fixed vocabulary "
                        "regardless of which words appear in the poem. MEG scores for "
                        "out-of-trial words are set to the row minimum (ranked last). "
                        "Output filename gets a _closed{N} suffix. "
                        "Example: llm_twostage/cache/gpt2/vocab_info.json")

    # Beam-search fusion (MEG-guided LLM context)
    p.add_argument("--beam_width", type=int, default=0,
                   help="Beam width for MEG-guided beam-search fusion. "
                        "0 (default) = disabled; use teacher-forced LLM scores instead. "
                        "Output written to fusion_beam{B}_top{k}_*.json.")
    p.add_argument("--top_k", type=int, default=5,
                   help="MEG top-k candidates to consider at each beam step "
                        "(ignored when beam_width=0)")
    p.add_argument("--no_repeat_ngram", type=int, default=0,
                   help="Block repeated n-grams of this length during beam search. "
                        "0 (default) = disabled. 2 = no consecutive word repeats "
                        "(e.g. 'flash flash'). 3 = no repeated trigrams. "
                        "Ignored when beam_width=0.")

    return p.parse_args()


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _resolve_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.tensor([1.0]).cuda()
        return torch.device("cuda")
    except RuntimeError as e:
        print(f"[warn] CUDA unusable ({e}); falling back to CPU.")
        return torch.device("cpu")


def _model_tag(method: str, llm_name: str, bert_name: str) -> str:
    if method == "inference":
        return bert_name.replace("/", "_").replace("-", "_")
    return llm_name.replace("/", "_")


def _ckpt_dir(
    out_root: Path, method: str, model_tag: str,
    eval_scheme: str, heldout: Optional[str],
    fold: Optional[int], n_lines: int, control: str,
) -> Path:
    ctrl = f"_ctrl_{control}" if control != "none" else ""
    if eval_scheme == "loso":
        split_tag = f"loso_{heldout}"
    elif eval_scheme == "session_cv":
        split_tag = f"session_cv_fold{fold}"
    else:
        split_tag = f"stimulus_lines{n_lines}"
    return out_root / method / model_tag / f"{split_tag}{ctrl}"


def _iter_folds(args) -> List[Dict]:
    """
    Return a list of fold descriptors, each with keys:
        heldout, fold, splits, ckpt_dir
    """
    out_root  = Path(args.out_root)
    model_tag = _model_tag(args.method, args.llm_name, args.bert_name)
    folds = []

    if args.eval_scheme == "loso":
        subjects = [args.heldout] if args.heldout else SUBJECTS
        for subj in subjects:
            splits  = make_loso_splits(subj)
            ckpt    = _ckpt_dir(out_root, args.method, model_tag,
                                "loso", subj, None, args.n_lines, args.control)
            folds.append({"heldout": subj, "fold": None, "splits": splits, "ckpt_dir": ckpt})

    elif args.eval_scheme == "session_cv":
        fold_ids = [args.fold] if args.fold is not None else list(range(5))
        for k in fold_ids:
            splits  = make_session_cv_splits(k)
            ckpt    = _ckpt_dir(out_root, args.method, model_tag,
                                "session_cv", None, k, args.n_lines, args.control)
            folds.append({"heldout": None, "fold": k, "splits": splits, "ckpt_dir": ckpt})

    else:  # stimulus
        splits = make_stimulus_splits(args.n_lines)
        ckpt   = _ckpt_dir(out_root, args.method, model_tag,
                           "stimulus", None, None, args.n_lines, args.control)
        folds.append({"heldout": None, "fold": None, "splits": splits, "ckpt_dir": ckpt})

    return folds


def _remap_to_closed_vocab(
    meg_scores:   torch.Tensor,   # (N, |V_trial|)
    trial_vocab:  List[str],
    closed_vocab: List[str],
) -> torch.Tensor:
    """
    Remap per-trial MEG scores to a fixed closed vocabulary.
    Words in closed_vocab absent from the trial get the row-minimum score
    (ranked below all in-trial words, no information).
    """
    N        = meg_scores.shape[0]
    trial_idx = {w: i for i, w in enumerate(trial_vocab)}
    row_min  = meg_scores.min(dim=-1, keepdim=True).values   # (N, 1)
    remapped = row_min.expand(N, len(closed_vocab)).clone()
    for j, w in enumerate(closed_vocab):
        if w in trial_idx:
            remapped[:, j] = meg_scores[:, trial_idx[w]]
    return remapped


def _scalar_summary(trial_metrics: List[Dict]) -> Dict:
    """Average key scalars over a list of per-trial metric dicts (teacher-forced path)."""
    r1s  = [m["option_a"]["recall_at_k"][0] for m in trial_metrics]
    mrrs = [m["option_a"]["mrr"]             for m in trial_metrics]
    accs = [m["option_b"]["word_accuracy"]   for m in trial_metrics]
    bleu = [m["option_b"]["bleu1"]           for m in trial_metrics]
    wers = [m["option_b"]["wer"]             for m in trial_metrics]
    return {
        "mean_R@1":      float(np.mean(r1s)),
        "mean_MRR":      float(np.mean(mrrs)),
        "mean_accuracy": float(np.mean(accs)),
        "mean_BLEU1":    float(np.mean(bleu)),
        "mean_WER":      float(np.mean(wers)),
        "n_trials":      len(trial_metrics),
    }


def _scalar_summary_beam(trial_metrics: List[Dict]) -> Dict:
    """Average option_b scalars over a list of beam-search trial results."""
    accs = [m["option_b"]["word_accuracy"] for m in trial_metrics]
    bleu = [m["option_b"]["bleu1"]         for m in trial_metrics]
    wers = [m["option_b"]["wer"]           for m in trial_metrics]
    return {
        "mean_accuracy": float(np.mean(accs)),
        "mean_BLEU1":    float(np.mean(bleu)),
        "mean_WER":      float(np.mean(wers)),
        "n_trials":      len(trial_metrics),
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = _resolve_device(args.device)

    beam_mode = args.beam_width > 0

    # Alpha grid
    if args.alphas is not None:
        alphas = args.alphas
    else:
        coarse = list(np.round(np.linspace(0, 0.90, 19), 3))
        fine   = list(np.round(np.arange(0.91, 1.001, 0.01), 3))
        alphas = sorted(set(coarse + fine))

    # Fusion LLM
    fusion_llm = args.fusion_llm_name
    if fusion_llm is None:
        fusion_llm = args.llm_name if args.method != "inference" else "gpt2"

    tokenizer, llm_model = load_fusion_llm(fusion_llm, device)
    fusion_llm_tag = fusion_llm.replace("/", "_")
    norm_tag = args.fusion_normalization

    # Closed vocabulary (optional)
    closed_vocab: Optional[List[str]] = None
    vocab_suffix = ""
    if args.closed_vocab_path:
        vocab_info   = json.loads(Path(args.closed_vocab_path).read_text())
        closed_vocab = sorted(vocab_info["restricted_words"])
        vocab_suffix = f"_closed{len(closed_vocab)}"

    print(f"\n{'='*60}")
    print(f"  method            : {args.method}")
    print(f"  eval_scheme       : {args.eval_scheme}")
    print(f"  control           : {args.control}")
    print(f"  fusion_llm        : {fusion_llm}")
    print(f"  fusion_norm       : {norm_tag}")
    if beam_mode:
        print(f"  beam_width        : {args.beam_width}")
        print(f"  top_k             : {args.top_k}")
        print(f"  no_repeat_ngram   : {args.no_repeat_ngram if args.no_repeat_ngram > 0 else 'disabled'}")
    print(f"  vocab             : {'closed (%d words)' % len(closed_vocab) if closed_vocab else 'per-trial'}")
    print(f"  alphas            : {alphas}")
    print(f"{'='*60}\n")

    folds = _iter_folds(args)
    print(f"  n_folds           : {len(folds)}\n")

    # Per-poem LLM cache (teacher-forced only — beam mode recomputes per trial)
    llm_score_cache: Dict[str, torch.Tensor] = {}
    vocab_cache: Dict[str, List[str]] = {}

    per_fold_summary: Dict[str, Dict] = {}
    all_diags: List[Dict] = []

    for fold_info in folds:
        heldout  = fold_info["heldout"]
        fold_k   = fold_info["fold"]
        splits   = fold_info["splits"]
        ckpt_dir = fold_info["ckpt_dir"]

        fold_key = heldout or (f"fold{fold_k}" if fold_k is not None else "stimulus")
        print(f"\n--- Fold: {fold_key}  ckpt: {ckpt_dir}")

        if not ckpt_dir.exists():
            print(f"  [skip] checkpoint dir not found: {ckpt_dir}")
            continue

        test_trials = sorted(set(
            (s, poem, sess)
            for s, poem, sess in splits["test"]["trials"]
        ))

        alpha_trial_metrics: Dict[float, List[Dict]] = {a: [] for a in alphas}
        fold_diags: List[Dict] = []
        trial_records: List[Dict] = []   # beam mode only: per-trial sequences
        n_successful = 0

        for subject, poem, session in test_trials:
            print(f"  {subject} {poem} sess={session} ...", end=" ", flush=True)

            try:
                result = predict(
                    subject   = subject,
                    session   = session,
                    condition = args.condition,
                    poem      = poem,
                    method    = args.method,
                    ckpt_dir  = str(ckpt_dir),
                    device    = str(device),
                )
            except Exception as exc:
                print(f"SKIP ({exc})")
                continue

            word_texts = result["words"]
            valid_mask = result["valid"]

            if closed_vocab is not None:
                meg_scores = _remap_to_closed_vocab(
                    result["scores"], result["vocab"], closed_vocab
                )
                vocab = closed_vocab
            else:
                meg_scores = result["scores"]
                vocab      = result["vocab"]

            if beam_mode:
                # ── Beam search: LLM conditioned on MEG-predicted history ────
                trial_sweep = sweep_alphas_beam(
                    meg_scores, vocab, word_texts, valid_mask,
                    tokenizer, llm_model, device, alphas,
                    beam_width=args.beam_width,
                    top_k=args.top_k,
                    normalization=norm_tag,
                    no_repeat_ngram=args.no_repeat_ngram,
                )
                a_min, a_max = min(alphas), max(alphas)
                acc_lo = trial_sweep[a_min]["option_b"]["word_accuracy"]
                acc_hi = trial_sweep[a_max]["option_b"]["word_accuracy"]
                print(f"acc  alpha={a_min:.2f}:{acc_lo:.3f}  "
                      f"alpha={a_max:.2f}:{acc_hi:.3f}")

                trial_records.append({
                    "subject":      subject,
                    "poem":         poem,
                    "session":      session,
                    "ground_truth": word_texts,
                    "valid_mask":   valid_mask,
                    "predictions":  {
                        str(a): trial_sweep[a]["pred_sequence"] for a in alphas
                    },
                })
            else:
                # ── Teacher-forced: LLM conditioned on ground-truth history ──
                if poem not in llm_score_cache:
                    llm_score_cache[poem] = compute_llm_scores(
                        word_texts, vocab, tokenizer, llm_model, device
                    )
                    vocab_cache[poem] = vocab

                llm_scores = llm_score_cache[poem]

                if vocab != vocab_cache[poem]:
                    print(f"SKIP (vocab mismatch for {poem})")
                    continue

                trial_sweep, trial_diag = sweep_alphas(
                    meg_scores, llm_scores, vocab, word_texts, valid_mask, alphas,
                    normalization=norm_tag,
                )
                fold_diags.append(trial_diag)
                all_diags.append(trial_diag)

                a_min, a_max = min(alphas), max(alphas)
                r1_lo = trial_sweep[a_min]["option_a"]["recall_at_k"][0]
                r1_hi = trial_sweep[a_max]["option_a"]["recall_at_k"][0]
                print(f"R@1  alpha={a_min:.2f}:{r1_lo:.3f}  "
                      f"alpha={a_max:.2f}:{r1_hi:.3f}  "
                      f"scale_ratio={trial_diag['mean_scale_ratio']:.1f}x")

            for alpha, metrics in trial_sweep.items():
                alpha_trial_metrics[alpha].append(metrics)
            n_successful += 1

        # Fold summary
        fold_summary: Dict[str, Dict] = {}
        for alpha in alphas:
            trial_list = alpha_trial_metrics[alpha]
            if trial_list:
                fold_summary[str(alpha)] = (
                    _scalar_summary_beam(trial_list) if beam_mode
                    else _scalar_summary(trial_list)
                )

        if not fold_summary:
            print(f"  [skip] no results collected for fold {fold_key}")
            continue

        per_fold_summary[fold_key] = fold_summary

        if beam_mode:
            best_alpha = max(
                (a for a in alphas if str(a) in fold_summary),
                key=lambda a: fold_summary[str(a)]["mean_accuracy"],
            )
            bs = fold_summary[str(best_alpha)]
            print(f"  fold={fold_key}  best_alpha={best_alpha}"
                  f"  acc={bs['mean_accuracy']:.3f}  BLEU={bs['mean_BLEU1']:.3f}"
                  f"  WER={bs['mean_WER']:.3f}")
        else:
            best_alpha = max(
                (a for a in alphas if str(a) in fold_summary),
                key=lambda a: fold_summary[str(a)]["mean_R@1"],
            )
            bs = fold_summary[str(best_alpha)]
            print(f"  fold={fold_key}  best_alpha={best_alpha}"
                  f"  R@1={bs['mean_R@1']:.3f}  MRR={bs['mean_MRR']:.3f}")

        # Scale diagnostics (teacher-forced only)
        fold_diag_agg: Dict = {}
        if not beam_mode and fold_diags:
            for key in fold_diags[0]:
                fold_diag_agg[key] = float(np.mean([d[key] for d in fold_diags]))
            fold_diag_agg["n_trials"] = len(fold_diags)

        # Output path
        ngram_tag = f"_norep{args.no_repeat_ngram}" if beam_mode and args.no_repeat_ngram > 0 else ""
        beam_tag  = f"_beam{args.beam_width}_top{args.top_k}{ngram_tag}" if beam_mode else ""
        fusion_dir = ckpt_dir / "fusion"
        fusion_dir.mkdir(parents=True, exist_ok=True)
        out_path = fusion_dir / f"fusion{beam_tag}_{fusion_llm_tag}_{norm_tag}{vocab_suffix}.json"

        meta: Dict = {
            "method":               args.method,
            "eval_scheme":          args.eval_scheme,
            "heldout":              heldout,
            "fold":                 fold_k,
            "control":              args.control,
            "fusion_llm":           fusion_llm,
            "fusion_normalization": norm_tag,
            "vocab_type":           f"closed{len(closed_vocab)}" if closed_vocab else "per_trial",
            "closed_vocab_path":    args.closed_vocab_path,
            "training_llm":         args.llm_name,
            "training_bert":        args.bert_name,
            "condition":            args.condition,
            "alphas":               alphas,
            "n_trials":             n_successful,
        }
        if beam_mode:
            meta["beam_width"]       = args.beam_width
            meta["top_k"]            = args.top_k
            meta["no_repeat_ngram"]  = args.no_repeat_ngram

        fold_output: Dict = {"meta": meta, "alphas": alphas, "per_alpha": fold_summary}
        if not beam_mode:
            fold_output["scale_diagnostics"] = fold_diag_agg
        if beam_mode and trial_records:
            fold_output["per_trial"] = trial_records

        out_path.write_text(json.dumps(fold_output, indent=2))
        print(f"  saved → {out_path}")

        # Write a human-readable text file of predicted sequences (beam mode only)
        if beam_mode and trial_records:
            txt_path = out_path.with_suffix(".txt")
            lines = []
            for rec in trial_records:
                lines.append(
                    f"subject={rec['subject']}  poem={rec['poem']}  session={rec['session']}"
                )
                truth_str = " ".join(
                    f"[{w}]" if not v else w
                    for w, v in zip(rec["ground_truth"], rec["valid_mask"])
                )
                lines.append(f"  truth  : {truth_str}")
                for a in alphas:
                    pred_str = " ".join(rec["predictions"][str(a)])
                    lines.append(f"  α={a:.2f} : {pred_str}")
                lines.append("")
            txt_path.write_text("\n".join(lines))
            print(f"  saved → {txt_path}")

    # ---------------------------------------------------------------------------
    #  Aggregate across folds
    # ---------------------------------------------------------------------------
    if not per_fold_summary:
        print("\n[warn] No fold results collected.")
        return

    aggregate: Dict[str, Dict] = {}
    for alpha in alphas:
        a_str = str(alpha)
        if beam_mode:
            fold_accs = [per_fold_summary[fk][a_str]["mean_accuracy"]
                         for fk in per_fold_summary if a_str in per_fold_summary[fk]]
            fold_bleu = [per_fold_summary[fk][a_str]["mean_BLEU1"]
                         for fk in per_fold_summary if a_str in per_fold_summary[fk]]
            fold_wers = [per_fold_summary[fk][a_str]["mean_WER"]
                         for fk in per_fold_summary if a_str in per_fold_summary[fk]]
            if not fold_accs:
                continue
            aggregate[a_str] = {
                "mean_accuracy": float(np.mean(fold_accs)),
                "std_accuracy":  float(np.std(fold_accs)),
                "mean_BLEU1":    float(np.mean(fold_bleu)),
                "mean_WER":      float(np.mean(fold_wers)),
                "n_folds":       len(fold_accs),
            }
        else:
            fold_r1s  = [per_fold_summary[fk][a_str]["mean_R@1"]
                         for fk in per_fold_summary if a_str in per_fold_summary[fk]]
            fold_mrrs = [per_fold_summary[fk][a_str]["mean_MRR"]
                         for fk in per_fold_summary if a_str in per_fold_summary[fk]]
            fold_accs = [per_fold_summary[fk][a_str]["mean_accuracy"]
                         for fk in per_fold_summary if a_str in per_fold_summary[fk]]
            if not fold_r1s:
                continue
            aggregate[a_str] = {
                "mean_R@1":      float(np.mean(fold_r1s)),
                "std_R@1":       float(np.std(fold_r1s)),
                "mean_MRR":      float(np.mean(fold_mrrs)),
                "std_MRR":       float(np.std(fold_mrrs)),
                "mean_accuracy": float(np.mean(fold_accs)),
                "n_folds":       len(fold_r1s),
            }

    if beam_mode:
        print(f"\n{'alpha':>8}  {'acc':>8}  {'±':>6}  {'BLEU-1':>8}  {'WER':>8}")
        print("-" * 50)
        for alpha in alphas:
            a_str = str(alpha)
            if a_str not in aggregate:
                continue
            ag = aggregate[a_str]
            print(f"{alpha:>8.2f}  {ag['mean_accuracy']:>8.4f}  "
                  f"{ag['std_accuracy']:>6.4f}  {ag['mean_BLEU1']:>8.4f}  "
                  f"{ag['mean_WER']:>8.4f}")
        best_alpha = max(aggregate, key=lambda a: aggregate[a]["mean_accuracy"])
        print(f"\nBest alpha = {best_alpha}  acc = {aggregate[best_alpha]['mean_accuracy']:.4f}")
    else:
        print(f"\n{'alpha':>8}  {'R@1':>8}  {'±':>6}  {'MRR':>8}  {'±':>6}")
        print("-" * 45)
        for alpha in alphas:
            a_str = str(alpha)
            if a_str not in aggregate:
                continue
            ag = aggregate[a_str]
            print(f"{alpha:>8.2f}  {ag['mean_R@1']:>8.4f}  "
                  f"{ag['std_R@1']:>6.4f}  {ag['mean_MRR']:>8.4f}  "
                  f"{ag['std_MRR']:>6.4f}")
        best_alpha = max(aggregate, key=lambda a: aggregate[a]["mean_R@1"])
        print(f"\nBest alpha = {best_alpha}  R@1 = {aggregate[best_alpha]['mean_R@1']:.4f}")

        if all_diags:
            print(f"\nScale diagnostics ({norm_tag}):")
            print(f"  mean MEG row std : {np.mean([d['mean_meg_row_std'] for d in all_diags]):.4f}")
            print(f"  mean LLM row std : {np.mean([d['mean_llm_row_std'] for d in all_diags]):.4f}")
            print(f"  mean ratio       : {np.mean([d['mean_scale_ratio'] for d in all_diags]):.1f}x")
            print(f"  median ratio     : {np.median([d['median_scale_ratio'] for d in all_diags]):.1f}x")


if __name__ == "__main__":
    main()
