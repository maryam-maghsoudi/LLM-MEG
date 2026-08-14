"""
controls.py — MEGWordDataset and MEGTrialDataset wrappers for control conditions.

ZeroMEGDataset          Replace all MEG windows with zeros.
                        Tests whether the model learns from word-order / position
                        statistics alone, with no neural signal.

TimeShuffledMEGDataset  Permute MEG windows across word positions within each
                        (subject, poem, session) trial (fixed permutation per trial,
                        drawn once at construction).
                        Tests whether the model uses word-specific MEG content,
                        or only trial-level / positional statistics.

Both wrappers work for MEGWordDataset (word-level) and MEGTrialDataset
(sequence-level). Pass the appropriate base dataset.
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .base_dataset import MEGWordDataset, MEGTrialDataset


# ---------------------------------------------------------------------------
#  Zero-MEG controls
# ---------------------------------------------------------------------------

class ZeroMEGWordDataset(Dataset):
    """Word-level dataset with all MEG windows replaced by zeros."""

    def __init__(self, base: MEGWordDataset):
        self._base = base

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> Dict:
        item = self._base[idx]
        item["meg_window"] = torch.zeros_like(item["meg_window"])
        return item


class ZeroMEGTrialDataset(Dataset):
    """Trial-level dataset with all MEG windows replaced by zeros."""

    def __init__(self, base: MEGTrialDataset):
        self._base = base

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> Dict:
        item = dict(self._base[idx])
        item["meg_windows"] = torch.zeros_like(item["meg_windows"])
        return item


# ---------------------------------------------------------------------------
#  Time-shuffled MEG controls
# ---------------------------------------------------------------------------

class TimeShuffledMEGWordDataset(Dataset):
    """
    Word-level dataset where MEG windows are permuted across word positions
    within each (subject, poem, session) trial.

    The permutation is drawn once at construction (seed-controlled) and fixed
    for the lifetime of the dataset. Text metadata (word_text, word_pos,
    line_num) stays at its true position; only the MEG window is swapped.
    """

    def __init__(self, base: MEGWordDataset, seed: int = 0, augment: bool = False):
        self.augment = augment
        rng = random.Random(seed)

        # Group item indices by trial
        trial_groups: Dict = defaultdict(list)
        for i, item in enumerate(base._items):
            key = (item["subject"], item["poem"], item["session"])
            trial_groups[key].append(i)

        # Build shuffled item list: swap meg_window across positions in each trial
        self._items: List[Dict] = [None] * len(base._items)
        for indices in trial_groups.values():
            src_indices = indices[:]
            rng.shuffle(src_indices)
            for orig_idx, src_idx in zip(indices, src_indices):
                new_item = dict(base._items[orig_idx])         # copy all metadata
                new_item["meg_window"] = base._items[src_idx]["meg_window"].copy()
                self._items[orig_idx]  = new_item

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict:
        item = self._items[idx]
        x = torch.from_numpy(item["meg_window"])
        if self.augment:
            x = x + 0.02 * torch.randn_like(x)
        return {
            "meg_window": x,
            "word_text":  item["word_text"],
            "word_pos":   item["word_pos"],
            "line_num":   item["line_num"],
            "poem":       item["poem"],
            "subject":    item["subject"],
            "session":    item["session"],
        }


class TimeShuffledMEGTrialDataset(Dataset):
    """
    Trial-level dataset where, within each trial, the MEG windows are
    permuted across word positions. The text sequence (word_texts, word_poses)
    remains in its true order; only the neural signal is shuffled.

    Permutation is drawn once per trial at construction (seed-controlled).
    """

    def __init__(self, base: MEGTrialDataset, seed: int = 0):
        rng = random.Random(seed)
        self._items: List[Dict] = []

        for item in base._items:
            windows    = item["meg_windows"].clone()   # (N, C, T)
            valid_mask = item["valid_mask"]
            N          = windows.shape[0]

            # Only permute positions that have valid MEG windows
            valid_pos = [i for i in range(N) if valid_mask[i]]
            if len(valid_pos) > 1:
                src_pos = valid_pos[:]
                rng.shuffle(src_pos)
                orig_windows = windows.clone()
                for dst, src in zip(valid_pos, src_pos):
                    windows[dst] = orig_windows[src]

            new_item = dict(item)
            new_item["meg_windows"] = windows
            self._items.append(new_item)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict:
        return self._items[idx]


# ---------------------------------------------------------------------------
#  Factory helpers
# ---------------------------------------------------------------------------

def make_control(base: Dataset, control: str, seed: int = 0, augment: bool = False) -> Dataset:
    """
    Wrap a base dataset with a control condition.

    Parameters
    ----------
    base    : MEGWordDataset or MEGTrialDataset
    control : 'none'         → return base unchanged
              'zero'         → ZeroMEG wrapper
              'shuffle_time' → TimeShuffledMEG wrapper
    seed    : random seed for shuffle controls
    augment : passed to TimeShuffled word-level wrapper
    """
    if control == "none":
        return base
    if isinstance(base, MEGWordDataset):
        if control == "zero":
            return ZeroMEGWordDataset(base)
        if control == "shuffle_time":
            return TimeShuffledMEGWordDataset(base, seed=seed, augment=augment)
    if isinstance(base, MEGTrialDataset):
        if control == "zero":
            return ZeroMEGTrialDataset(base)
        if control == "shuffle_time":
            return TimeShuffledMEGTrialDataset(base, seed=seed)
    raise ValueError(f"Unknown control {control!r} or unrecognised base type {type(base)}")
