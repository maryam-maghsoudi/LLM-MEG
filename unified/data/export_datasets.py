"""
export_datasets.py — build and save the full MEG datasets (all subjects,
poems, and sessions) so that a collaborator can run training without access
to the raw .fif files.

Run once from the llm_decoder/ parent directory:

    python -m unified.data.export_datasets --out_dir /path/to/cache

This produces two files:
    meg_word_all.pt    — MEGWordDataset items  (word-level,  for M1 / M2 Stage 1)
    meg_trial_all.pt   — MEGTrialDataset items (trial-level, for M2 Stage 2 / M3)

Collaborator usage (no .fif files needed):

    from unified.data.base_dataset import MEGWordDataset, MEGTrialDataset
    from unified.data.splits import make_loso_splits

    word_items  = MEGWordDataset.load_items("meg_word_all.pt")
    trial_items = MEGTrialDataset.load_items("meg_trial_all.pt")

    splits   = make_loso_splits("sub-01")
    ds_train = MEGWordDataset.from_cache(word_items,  splits["train"])
    ds_val   = MEGWordDataset.from_cache(word_items,  splits["val"])
    ds_trial = MEGTrialDataset.from_cache(trial_items, splits["train"])
"""

import argparse
import sys
from itertools import product
from pathlib import Path

_HERE = Path(__file__).parent.parent.parent   # llm_decoder/
sys.path.insert(0, str(_HERE))

from unified.data.base_dataset import MEGWordDataset, MEGTrialDataset
from unified.data.splits import SUBJECTS, POEM_KEYS, N_SESSIONS


def parse_args():
    p = argparse.ArgumentParser(description="Export full MEG datasets to .pt files")
    p.add_argument("--out_dir",   default="unified/data/cache",
                   help="Directory for output .pt files")
    p.add_argument("--condition", default="lis",
                   help="MEG condition suffix (default: lis)")
    p.add_argument("--subjects",  nargs="+", default=None,
                   help="Subset of subjects to export (default: all 13)")
    return p.parse_args()


def main():
    args     = parse_args()
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects or SUBJECTS
    sessions = list(range(N_SESSIONS))
    all_trials = [
        (subj, poem, sess)
        for subj, poem, sess in product(subjects, POEM_KEYS, sessions)
    ]

    print(f"Exporting {len(subjects)} subjects × {len(POEM_KEYS)} poems "
          f"× {len(sessions)} sessions = {len(all_trials)} trials")
    print(f"Condition: {args.condition}")
    print(f"Output:    {out_dir}\n")

    # ── Word-level dataset ────────────────────────────────────────────────────
    print("Building MEGWordDataset (all trials) ...")
    ds_word = MEGWordDataset(all_trials, word_filter=None, condition=args.condition)
    word_path = out_dir / "meg_word_all.pt"
    ds_word.save_items(str(word_path))

    # ── Trial-level dataset ───────────────────────────────────────────────────
    print("\nBuilding MEGTrialDataset (all trials) ...")
    ds_trial = MEGTrialDataset(all_trials, word_filter=None, condition=args.condition)
    trial_path = out_dir / "meg_trial_all.pt"
    ds_trial.save_items(str(trial_path))

    print(f"\nDone.")
    print(f"  {word_path}   ({word_path.stat().st_size / 1e9:.2f} GB)")
    print(f"  {trial_path}  ({trial_path.stat().st_size / 1e9:.2f} GB)")
    print()
    print("Share both files with your collaborator along with the codebase.")
    print("They do NOT need the raw .fif files or onset JSONs to run training.")


if __name__ == "__main__":
    main()
