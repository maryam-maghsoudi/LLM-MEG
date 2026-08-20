"""
load_data.py — Load all twostage eval_results.json into canonical tables.

Outputs
-------
trials_df : DataFrame, one row per (eval_scheme, fold_id, control, trial).
            Columns: method, eval_scheme, fold_id, control, trial_id,
                     subject, poem, session,
                     mrr, word_acc, bleu1, wer, recall_auc,
                     n_evaluated, vocab_size.

recall_df : DataFrame, long-format recall@k per trial.
            Columns: method, eval_scheme, fold_id, control, trial_id, k, recall.
            'recall' is already averaged over word positions within the trial
            (as produced by evaluate.py); no binary hit reconstruction needed.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

_HERE    = Path(__file__).parent
OUT_ROOT = _HERE.parent / "out" / "twostage" / "HuggingFaceTB_SmolLM2-360M"
METHOD   = "twostage"


def _parse_dir(name: str):
    """
    Parse a checkpoint directory name into (eval_scheme, fold_id, control).

    fold_id:
      loso       → subject string, e.g. "sub-01"
      session_cv → int,           e.g. 0
      stimulus   → string,        e.g. "lines2" or "lines4"
    """
    m = re.search(r'_ctrl_(shuffle_time|zero)$', name)
    if m:
        control = m.group(1)
        base    = name[:m.start()]
    else:
        control = "none"
        base    = name

    if base.startswith("loso_"):
        return "loso", base[5:], control
    if base.startswith("session_cv_fold"):
        return "session_cv", int(base[len("session_cv_fold"):]), control
    if base.startswith("stimulus_lines"):
        return "stimulus", base[len("stimulus_"):], control
    raise ValueError(f"Unrecognised directory: {name!r}")


def _trial_auc(recall_at_k: list) -> float:
    """Normalised AUC of recall@k curve via trapezoidal rule, in [0, 1]."""
    y = np.array(recall_at_k, dtype=float)
    V = len(y)
    if V <= 1:
        return float(y[0]) if V else 0.0
    return float(np.trapz(y) / (V - 1))


def load_tables(out_root: Path = OUT_ROOT):
    """
    Walk out_root, load every eval_results.json, return (trials_df, recall_df).
    """
    trial_rows  = []
    recall_rows = []
    n_files     = 0

    for eval_path in sorted(out_root.glob("*/eval/eval_results.json")):
        dir_name = eval_path.parent.parent.name
        try:
            eval_scheme, fold_id, control = _parse_dir(dir_name)
        except ValueError as e:
            print(f"  [skip] {e}")
            continue

        data    = json.loads(eval_path.read_text())
        n_files += 1

        for t in data["trials"]:
            subj     = t["subject"]
            poem     = t["poem"]
            session  = int(t["session"])
            trial_id = f"{subj}_{poem}_s{session}"
            a        = t["option_a"]
            b        = t["option_b"]

            trial_rows.append({
                "method":      METHOD,
                "eval_scheme": eval_scheme,
                "fold_id":     fold_id,
                "control":     control,
                "trial_id":    trial_id,
                "subject":     subj,
                "poem":        poem,
                "session":     session,
                "mrr":         float(a["mrr"]),
                "word_acc":    float(b["word_accuracy"]),
                "bleu1":       float(b["bleu1"]),
                "wer":         float(b["wer"]),
                "recall_auc":  _trial_auc(a["recall_at_k"]),
                "n_evaluated": int(a["n_evaluated"]),
                "vocab_size":  int(a["vocab_size"]),
            })

            for k, r in enumerate(a["recall_at_k"], start=1):
                recall_rows.append({
                    "method":      METHOD,
                    "eval_scheme": eval_scheme,
                    "fold_id":     fold_id,
                    "control":     control,
                    "trial_id":    trial_id,
                    "k":           k,
                    "recall":      float(r),
                })

    print(f"Loaded {n_files} eval_results.json files → "
          f"{len(trial_rows):,} trial rows, {len(recall_rows):,} recall rows")
    return pd.DataFrame(trial_rows), pd.DataFrame(recall_rows)
