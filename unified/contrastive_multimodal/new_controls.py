"""
controls.py — control-condition dataset wrappers for MEGContinuousTrialDataset.

REWRITTEN for the continuous-encoder pipeline. The original version wrapped
the OLD fixed-window MEGWordDataset/MEGTrialDataset (base_dataset.py),
which are no longer part of the active pipeline. Since there's no longer a
pre-extracted (N, C, T) window array to permute, "shuffle the MEG content
across word positions" now means shuffling which (onset_sample,
offset_sample) PAIR each word position gets — meg_trial itself and the
text sequence stay untouched; only which time window gets pooled for which
word changes.

ZeroMEGTrialDataset          Replace meg_trial entirely with zeros.
                              Tests how much of a result survives with NO
                              neural signal at all.
TimeShuffledMEGTrialDataset  Permute (onset_sample, offset_sample) across
                              a trial's own VALID word positions, leaving
                              meg_trial, word_texts, and word_poses in
                              their true order/content. Direct continuous-
                              pipeline equivalent of the original design's
                              "permute windows, keep text order" control.

WHY THIS MATTERS: subject-grouped CV alone doesn't rule out a model
exploiting fixed per-poem structure (text/position) with the MEG signal
contributing little or nothing. These give the actual baselines to check
against — real-vs-zero and real-vs-shuffled top-1/top-5 — ideally via a
proper permutation-null distribution (many shuffle seeds run through
eval_stage1.py / eval_stage2.py), not a single shuffled run. Building that
null-distribution LOOP belongs in the eval scripts, not here — this file
only provides the datasets to shuffle.
"""

import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from new_dataset import MEGContinuousTrialDataset


class ZeroMEGTrialDataset(Dataset):
    """meg_trial replaced entirely by zeros; everything else untouched."""

    def __init__(self, base: MEGContinuousTrialDataset):
        self._base = base

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> Dict:
        item = dict(self._base[idx])
        item["meg_trial"] = torch.zeros_like(item["meg_trial"])
        return item


class TimeShuffledMEGTrialDataset(Dataset):
    """
    Permutes (onset_sample, offset_sample) across a trial's own VALID word
    positions. Only permutes among a trial's OWN valid positions — never
    into an invalid slot (would fabricate a spuriously "valid" word out of
    the -1 sentinel) and never across different trials (would leak one
    trial's real MEG timing into another, a different and stronger
    corruption than intended). valid_mask itself is therefore unchanged:
    the SET of valid positions is identical before and after — only which
    onset/offset each valid position holds moves.

    Permutation is drawn once per trial at construction (seed-controlled),
    fixed for the dataset's lifetime.
    """

    def __init__(self, base: MEGContinuousTrialDataset, seed: int = 0):
        rng = random.Random(seed)
        self._items: List[Dict] = []

        for item in base._items:
            onset  = item["onset_samples"].clone()
            offset = item["offset_samples"].clone()
            valid  = item["valid_mask"]
            N = onset.shape[0]

            valid_pos = [i for i in range(N) if valid[i]]
            if len(valid_pos) > 1:
                src_pos = valid_pos[:]
                rng.shuffle(src_pos)
                orig_onset, orig_offset = onset.clone(), offset.clone()
                for dst, src in zip(valid_pos, src_pos):
                    onset[dst]  = orig_onset[src]
                    offset[dst] = orig_offset[src]

            new_item = dict(item)   # shallow copy: meg_trial, valid_mask, word_texts,
            new_item["onset_samples"]  = onset   # word_poses, poem, subject, session all
            new_item["offset_samples"] = offset  # unchanged, safe to share (never mutated)
            self._items.append(new_item)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict:
        return self._items[idx]


def make_control(base: MEGContinuousTrialDataset, control: str, seed: int = 0) -> Dataset:
    """
    control : 'none'         -> base unchanged
              'zero'         -> ZeroMEGTrialDataset
              'shuffle_time' -> TimeShuffledMEGTrialDataset
    """
    if control == "none":
        return base
    if control == "zero":
        return ZeroMEGTrialDataset(base)
    if control == "shuffle_time":
        return TimeShuffledMEGTrialDataset(base, seed=seed)
    raise ValueError(f"Unknown control {control!r}")


if __name__ == "__main__":
    print("=== controls.py sanity check ===\n")

    # Fake base dataset via __new__ + manual _items — the SAME pattern
    # MEGContinuousTrialDataset.from_cache already uses to skip __init__'s
    # .fif/onset-JSON loading. No real files needed for this check.
    fake_base = MEGContinuousTrialDataset.__new__(MEGContinuousTrialDataset)
    fake_base._items = [{
        "meg_trial":      torch.arange(30.0).reshape(3, 10),   # (C=3, T=10), distinctive values
        "onset_samples":  torch.tensor([1, 4, 7, -1]),
        "offset_samples": torch.tensor([2, 5, 8, -1]),
        "valid_mask":     torch.tensor([True, True, True, False]),
        "word_texts": ["a", "b", "c", "d"],
        "word_poses": [0, 1, 2, 3],
        "poem": "poem1", "subject": "sub-fake", "session": 0,
    }]
    base_item = fake_base._items[0]

    # ------------------------------------------------------------------
    # 1. ZeroMEGTrialDataset
    # ------------------------------------------------------------------
    zero_item = ZeroMEGTrialDataset(fake_base)[0]
    assert torch.all(zero_item["meg_trial"] == 0), "meg_trial must be all zeros"
    assert torch.equal(zero_item["onset_samples"], base_item["onset_samples"]), \
        "onset_samples must be unchanged by the zero control"
    assert torch.equal(zero_item["valid_mask"], base_item["valid_mask"])
    assert zero_item["word_texts"] == base_item["word_texts"]
    print("[OK] ZeroMEGTrialDataset: meg_trial zeroed, everything else untouched")

    # ------------------------------------------------------------------
    # 2. TimeShuffledMEGTrialDataset
    # ------------------------------------------------------------------
    shuf_item = TimeShuffledMEGTrialDataset(fake_base, seed=1)[0]

    assert torch.equal(shuf_item["meg_trial"], base_item["meg_trial"]), \
        "meg_trial itself must be untouched — only onset/offset ASSIGNMENT shuffles"
    assert shuf_item["word_texts"] == base_item["word_texts"], "text sequence must stay in TRUE order"
    assert shuf_item["word_poses"] == base_item["word_poses"]
    assert torch.equal(shuf_item["valid_mask"], base_item["valid_mask"]), \
        "the SET of valid positions must be unchanged — only which onset each valid position holds"

    valid = base_item["valid_mask"]
    orig_onsets = sorted(base_item["onset_samples"][valid].tolist())
    shuf_onsets = sorted(shuf_item["onset_samples"][valid].tolist())
    assert orig_onsets == shuf_onsets, "shuffled onsets must be a PERMUTATION of the original valid onsets"
    print("[OK] TimeShuffledMEGTrialDataset: meg_trial untouched, text order preserved, "
          "onset assignment is a genuine permutation of the original valid onsets")

    moved = not torch.equal(shuf_item["onset_samples"][valid], base_item["onset_samples"][valid])
    print(f"[{'OK' if moved else 'NOTE'}] seed=1 {'moved' if moved else 'happened to equal'} the identity "
          f"permutation — the multiset check above is the real correctness test either way.")

    shuf_again = TimeShuffledMEGTrialDataset(fake_base, seed=1)[0]
    assert torch.equal(shuf_again["onset_samples"], shuf_item["onset_samples"]), "same seed must reproduce identically"
    print("[OK] same seed reproduces the identical permutation")

    # ------------------------------------------------------------------
    # 3. make_control dispatcher
    # ------------------------------------------------------------------
    assert make_control(fake_base, "none") is fake_base
    assert isinstance(make_control(fake_base, "zero"), ZeroMEGTrialDataset)
    assert isinstance(make_control(fake_base, "shuffle_time", seed=2), TimeShuffledMEGTrialDataset)
    try:
        make_control(fake_base, "bogus")
        raise AssertionError("expected ValueError for an unknown control name")
    except ValueError:
        pass
    print("[OK] make_control dispatches correctly and rejects unknown control names")

    print("\n=== ALL CHECKS PASSED ===")
