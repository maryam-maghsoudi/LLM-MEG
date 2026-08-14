"""
evaluate.py — evaluation metrics for all three methods.

Option A — closed-vocabulary ranking
    R@k for k=1…|V| (full recall curve), MRR.
    vocab = all unique words in the test split.

Option B — open-vocabulary text quality
    Word accuracy (exact match), BLEU-1, WER.

Usage
-----
From Python:
    from unified.evaluate import eval_option_a, eval_option_b, run_eval

From CLI:
    python evaluate.py \\
        --method twostage \\
        --eval_scheme loso \\
        --heldout sub-01 \\
        --ckpt_dir out/twostage/.../sub-01/none \\
        --device cuda
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ===========================================================================
#  Option A — Ranking metrics
# ===========================================================================

def eval_option_a(
    scores:     torch.Tensor,      # (N, |V|)  raw scores per position
    vocab:      List[str],         # evaluation vocabulary, sorted
    word_texts: List[str],         # ground-truth word at each position
    valid_mask: Optional[List[bool]] = None,   # if None, evaluate all positions
) -> Dict:
    """
    Compute the full R@k recall curve (k=1…|V|) and MRR over valid positions.

    Only positions where the ground-truth word is in vocab are scored.
    Positions with valid_mask=False are skipped (no usable MEG window).

    Returns
    -------
    {
        'recall_at_k': List[float]   length |V|, recall_at_k[k-1] = R@k
        'mrr':         float
        'n_evaluated': int           number of positions actually scored
        'vocab_size':  int
    }
    """
    V        = len(vocab)
    vocab_idx= {w: i for i, w in enumerate(vocab)}
    N        = len(word_texts)

    if valid_mask is None:
        valid_mask = [True] * N

    ranks: List[int] = []
    for i in range(N):
        if not valid_mask[i]:
            continue
        gt = word_texts[i]
        if gt not in vocab_idx:
            continue                    # word not in eval vocab — skip
        true_idx = vocab_idx[gt]
        row      = scores[i]            # (|V|,)
        # rank of true word (1-indexed, lower = better)
        rank = int((row > row[true_idx]).sum().item()) + 1
        ranks.append(rank)

    if not ranks:
        return {"recall_at_k": [0.0] * V, "mrr": 0.0, "n_evaluated": 0,
                "vocab_size": V}

    ranks_arr   = np.array(ranks)
    recall_at_k = [(ranks_arr <= k).mean().item() for k in range(1, V + 1)]
    mrr         = float(np.mean(1.0 / ranks_arr))

    return {
        "recall_at_k": recall_at_k,
        "mrr":         mrr,
        "n_evaluated": len(ranks),
        "vocab_size":  V,
    }


# ===========================================================================
#  Option B — Text quality metrics
# ===========================================================================

def _bleu1(pred: List[str], ref: List[str]) -> float:
    """Unigram BLEU (precision) with brevity penalty."""
    from collections import Counter
    ref_counts = Counter(ref)
    clipped    = sum(min(cnt, ref_counts[w]) for w, cnt in Counter(pred).items())
    bp = min(1.0, len(pred) / max(len(ref), 1))
    return bp * (clipped / max(len(pred), 1))


def _wer(pred: List[str], ref: List[str]) -> float:
    """Word error rate via dynamic programming edit distance."""
    n, m = len(ref), len(pred)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new = [i] + [0] * m
        for j in range(1, m + 1):
            if ref[i - 1] == pred[j - 1]:
                new[j] = dp[j - 1]
            else:
                new[j] = 1 + min(dp[j], new[j - 1], dp[j - 1])
        dp = new
    return dp[m] / max(n, 1)


def eval_option_b(
    pred_top1:  List[str],
    word_texts: List[str],
    valid_mask: Optional[List[bool]] = None,
) -> Dict:
    """
    Compute word accuracy, BLEU-1, and WER on the predicted sequence.

    valid_mask : if provided, only evaluate at valid (MEG-available) positions.
                 WER and BLEU-1 are still computed over the full sequences.

    Returns
    -------
    {
        'word_accuracy': float   fraction of positions where pred == truth
        'bleu1':         float   unigram BLEU of full sequence
        'wer':           float   word error rate of full sequence
        'n_evaluated':   int     positions used for word_accuracy
    }
    """
    N = len(word_texts)
    if valid_mask is None:
        valid_mask = [True] * N

    correct    = sum(p == r for p, r, v in zip(pred_top1, word_texts, valid_mask) if v)
    n_valid    = sum(valid_mask)
    word_acc   = correct / max(n_valid, 1)

    return {
        "word_accuracy": word_acc,
        "bleu1":         _bleu1(pred_top1, word_texts),
        "wer":           _wer(pred_top1, word_texts),
        "n_evaluated":   n_valid,
    }


# ===========================================================================
#  Per-line breakdown helper
# ===========================================================================

def breakdown_by_line(
    scores:     torch.Tensor,
    vocab:      List[str],
    word_texts: List[str],
    pred_top1:  List[str],
    line_nums:  List[int],
    valid_mask: Optional[List[bool]] = None,
) -> Dict:
    """
    Compute Option A and B metrics separately for each line number (1–12).
    Useful for the stimulus split to compare heldout lines vs. training lines.
    """
    if valid_mask is None:
        valid_mask = [True] * len(word_texts)

    lines = sorted(set(line_nums))
    results = {}
    for ln in lines:
        mask   = [line_nums[i] == ln for i in range(len(word_texts))]
        vmask  = [mask[i] and valid_mask[i] for i in range(len(word_texts))]
        a = eval_option_a(scores, vocab, word_texts, vmask)
        b = eval_option_b(pred_top1, word_texts, vmask)
        results[ln] = {"option_a": a, "option_b": b}
    return results


# ===========================================================================
#  Wilcoxon signed-rank test (for aggregating across subjects/folds)
# ===========================================================================

def wilcoxon_test(
    real_scores:    List[float],
    control_scores: List[float],
    metric:         str = "mrr",
) -> Dict:
    """
    Paired Wilcoxon signed-rank test comparing real vs. control.
    Input: one scalar per subject/fold (e.g., MRR or R@1).
    Returns: {'statistic': float, 'p_value': float, 'n': int}
    """
    from scipy.stats import wilcoxon as _wilcoxon
    diffs = [r - c for r, c in zip(real_scores, control_scores)]
    if all(d == 0 for d in diffs):
        return {"statistic": 0.0, "p_value": 1.0, "n": len(diffs)}
    stat, p = _wilcoxon(diffs)
    return {"statistic": float(stat), "p_value": float(p), "n": len(diffs)}


# ===========================================================================
#  Full evaluation run for one trial
# ===========================================================================

def evaluate_trial(result: Dict) -> Dict:
    """
    Run both Option A and Option B on the output of predict().

    Parameters
    ----------
    result : output dict from predict.predict()

    Returns
    -------
    Combined metrics dict.
    """
    a = eval_option_a(
        scores     = result["scores"],
        vocab      = result["vocab"],
        word_texts = result["words"],
        valid_mask = result["valid"],
    )
    b = eval_option_b(
        pred_top1  = result["pred_top1"],
        word_texts = result["words"],
        valid_mask = result["valid"],
    )
    return {"option_a": a, "option_b": b}


# ===========================================================================
#  CLI entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained model on a test split")
    p.add_argument("--method",      required=True,
                   choices=["inference", "twostage", "interleaved"])
    p.add_argument("--eval_scheme", required=True,
                   choices=["loso", "session_cv", "stimulus"])
    p.add_argument("--ckpt_dir",    required=True,
                   help="Path to the trained model output directory")
    p.add_argument("--heldout",  default=None)
    p.add_argument("--fold",     type=int, default=None)
    p.add_argument("--n_lines",  type=int, default=2, choices=[2, 4])
    p.add_argument("--device",   default="cpu")
    p.add_argument("--condition",default="lis")
    p.add_argument("--out_dir",  default=None,
                   help="Directory for eval output JSON (default: ckpt_dir/eval/)")
    return p.parse_args()


def main():
    args = parse_args()

    from unified.data.splits import (
        make_loso_splits, make_session_cv_splits, make_stimulus_splits, SUBJECTS,
    )
    from unified.predict import predict

    if args.eval_scheme == "loso":
        splits = make_loso_splits(args.heldout)
    elif args.eval_scheme == "session_cv":
        splits = make_session_cv_splits(args.fold)
    else:
        splits = make_stimulus_splits(args.n_lines)

    # Collect unique test trials
    test_trials_set = set()
    for subject, poem, session in splits["test"]["trials"]:
        test_trials_set.add((subject, poem, session))

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for subject, poem, session in sorted(test_trials_set):
        print(f"  evaluating {subject} {poem} session={session} ...", end=" ", flush=True)
        result = predict(
            subject   = subject,
            session   = session,
            condition = args.condition,
            poem      = poem,
            method    = args.method,
            ckpt_dir  = args.ckpt_dir,
            device    = args.device,
        )
        metrics = evaluate_trial(result)
        metrics["subject"] = subject
        metrics["poem"]    = poem
        metrics["session"] = session
        all_results.append(metrics)
        print(f"R@1={metrics['option_a']['recall_at_k'][0]:.3f}  "
              f"MRR={metrics['option_a']['mrr']:.3f}  "
              f"acc={metrics['option_b']['word_accuracy']:.3f}")

    # Aggregate
    r1s  = [r["option_a"]["recall_at_k"][0]      for r in all_results]
    mrrs = [r["option_a"]["mrr"]                  for r in all_results]
    accs = [r["option_b"]["word_accuracy"]         for r in all_results]
    bleu = [r["option_b"]["bleu1"]                 for r in all_results]
    wers = [r["option_b"]["wer"]                   for r in all_results]

    summary = {
        "n_trials":      len(all_results),
        "mean_R@1":      float(np.mean(r1s)),
        "mean_MRR":      float(np.mean(mrrs)),
        "mean_accuracy": float(np.mean(accs)),
        "mean_BLEU1":    float(np.mean(bleu)),
        "mean_WER":      float(np.mean(wers)),
        "trials":        all_results,
    }

    out_path = out_dir / "eval_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary  R@1={summary['mean_R@1']:.3f}  MRR={summary['mean_MRR']:.3f}  "
          f"acc={summary['mean_accuracy']:.3f}  BLEU1={summary['mean_BLEU1']:.3f}  "
          f"WER={summary['mean_WER']:.3f}")
    print(f"Results → {out_path}")


if __name__ == "__main__":
    main()
