"""
new_dataset.py — continuous-trial MEG dataset for the continuous encoder pipeline.

"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import json


DS_FACTOR    = 10
SFREQ_DS     = 100.0
N_CHANNELS   = 155
EPOCH_TMIN_S = 0.0
MEG_BASE = Path("/fs/nexus-projects/brain_project/maryam_meg_dataset/icaed_Sai")
ONSET_DIR = Path("/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/contrastive_learning/onset_out")

def _load_meg_trial(
    subject:   str,
    condition: str,
    session:   int,
    meg_base:  Optional[Path] = None,
) -> Optional[np.ndarray]:
    """
    Load one MEG trial → (N_CHANNELS, T_ds) float32, or None if unavailable.

    If a preprocessed .npy file exists in meg_base (e.g. icaed_Sai), it is
    loaded directly — no MNE, no preprocessing.  Otherwise falls through to
    the raw .fif pipeline:
      1. Collapse epoch repetitions: mean(axis=0)
      2. Downsample 10× with scipy.signal.resample
      3. Scale by 1e12 (Tesla → picoTesla range) before z-scoring so that
         float32 std does not underflow to zero for raw Tesla-scale values
      4. Z-score per channel in float64, then cast to float32

    Parameters
    ----------
    meg_base : override for MEG_BASE (e.g. Path to icaed_Sai).  None → MEG_BASE.
    """
    base    = Path(meg_base) if meg_base is not None else MEG_BASE
    meg_dir = base / subject / f"ses-{session}" / "meg"

    # Fast path: preprocessed .npy (produced by export_preprocessed.py)
    npy_path = meg_dir / f"{subject}_sess-{session}_task-{condition}.npy"
    if npy_path.exists():
        return np.load(str(npy_path))

    # Slow path: raw .fif → preprocess
    fpath = meg_dir / f"{subject}_sess-{session}_task-{condition}_meg-epo.fif"
    if not fpath.exists():
        return None
    try:
        epochs = mne.read_epochs(str(fpath), preload=True)
    except Exception:
        return None
    raw  = epochs.get_data().mean(axis=0)                        # (C, T_raw)
    data = resample(raw, raw.shape[1] // DS_FACTOR, axis=1) * 1e12  # float64
    mu   = data.mean(axis=1, keepdims=True)
    sd   = np.maximum(data.std(axis=1, keepdims=True), 1e-12)
    return ((data - mu) / sd).astype(np.float32)


def _load_onsets(poem: str) -> List[Dict]:
    with open(ONSET_DIR / f"{poem}_word_onsets.json", 'r') as f:
        onsets = json.load(f)
#     return json.loads((ONSET_DIR / f"{poem}_word_onsets.json").read_text())
    return onsets


def _onset_offset_to_samples(onset_s: float, offset_s: float, n_t: int) -> Optional[Tuple[int, int]]:
    """
    Convert a word's [onset_s, offset_s] (seconds, trial-relative) to
    [onset_sample, offset_sample) in the downsampled trial. Unlike
    _onset_to_window (base_dataset.py), this does NOT add a fixed pre/post
    margin — the continuous encoder sees the whole trial, so word
    boundaries only need to be recorded, not cut out.

    Returns None if either boundary falls outside the trial or the
    resulting span is degenerate (offset <= onset).
    """
    onset_samp  = int(round((onset_s  - EPOCH_TMIN_S) * SFREQ_DS))
    offset_samp = int(round((offset_s - EPOCH_TMIN_S) * SFREQ_DS))
    if onset_samp < 0 or offset_samp > n_t or offset_samp <= onset_samp:
        return None
    return onset_samp, offset_samp


class MEGContinuousTrialDataset(Dataset):
    """
    One item per (subject, poem, session) trial: the FULL continuous MEG
    trial (no fixed-window epoching), plus per-word onset/offset sample
    indices for downstream pooling.

    Item keys
    ---------
    meg_trial      : Tensor(N_CHANNELS, T)  float32 — full continuous trial
    onset_samples  : Tensor(N_words,) long  — word onset, in samples (-1 if invalid)
    offset_samples : Tensor(N_words,) long  — word offset, in samples (-1 if invalid)
    valid_mask     : Tensor(N_words,) bool  — True where onset/offset both
                      land inside the trial and offset > onset
    word_texts     : List[str]
    word_poses     : List[int]
    poem           : str
    subject        : str
    session        : int
    """

    def __init__(
        self,
        trials:      List[Tuple[str, str, int]],
        word_filter: Optional[Dict[str, List[int]]] = None,
        condition:   str = "lis",
        meg_base:    Optional[Path] = None,
    ):
        self._items: List[Dict] = []
        onsets = {p: _load_onsets(p) for p in ["poem1", "poem2"]}

        allowed: Dict[str, Optional[List[int]]] = {}
        for poem in ["poem1", "poem2"]:
            if word_filter is None or poem not in word_filter:
                allowed[poem] = None
            else:
                allowed[poem] = sorted(word_filter[poem])

        n_missing = 0
        seen: Set[Tuple] = set()

        for subject, poem, session in trials:
            key = (subject, poem, session)
            if key in seen:
                continue
            seen.add(key)

            data = _load_meg_trial(subject, f"{poem}{condition}", session, meg_base=meg_base)
            if data is None:
                n_missing += 1
                continue

            onset_list = onsets[poem]
            pos_list   = allowed[poem] if allowed[poem] is not None \
                         else list(range(len(onset_list)))
            N          = len(pos_list)
            n_t        = data.shape[1]

            onset_samples  = np.full(N, -1, dtype=np.int64)
            offset_samples = np.full(N, -1, dtype=np.int64)
            valid_mask     = np.zeros(N, dtype=bool)
            word_texts     = []

            for i, pos in enumerate(pos_list):
                entry = onset_list[pos]
                word_texts.append(entry["word"].strip().lower())
                span = _onset_offset_to_samples(entry["start"], entry["end"], n_t)
                if span is None:
                    continue
                onset_samples[i], offset_samples[i] = span
                valid_mask[i] = True

            self._items.append({
                "meg_trial":      torch.from_numpy(data),
                "onset_samples":  torch.from_numpy(onset_samples),
                "offset_samples": torch.from_numpy(offset_samples),
                "valid_mask":     torch.from_numpy(valid_mask),
                "word_texts":     word_texts,
                "word_poses":     pos_list,
                "poem":           poem,
                "subject":        subject,
                "session":        session,
            })

        print(
            f"  MEGContinuousTrialDataset  trials={len(self._items):,}  missing={n_missing}"
        )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict:
        return self._items[idx]

    # ------------------------------------------------------------------
    #  Cache helpers — same conventions as MEGTrialDataset (base_dataset.py)
    # ------------------------------------------------------------------

    def save_items(self, path: str) -> None:
        torch.save(self._items, path)
        print(f"  MEGContinuousTrialDataset  saved {len(self._items):,} items → {path}")

    @staticmethod
    def load_items(path: str) -> List[Dict]:
        return torch.load(path, weights_only=False)

    @classmethod
    def from_cache(cls, all_items: List[Dict], split: Dict) -> "MEGContinuousTrialDataset":
        """
        Reconstruct a split-specific dataset from pre-cached items, without
        reading any .fif files. word_filter restricts which word positions
        are kept per trial — meg_trial itself is NEVER sliced here (it's
        the full continuous trial, shared across every word position in it);
        only the per-word arrays (onset/offset/valid_mask/texts/poses) are
        filtered down to the allowed positions.
        """
        trials      = {(s, p, sess) for s, p, sess in split["trials"]}
        word_filter = split.get("word_filter")

        allowed: Dict[str, Optional[Set[int]]] = {}
        for poem in ["poem1", "poem2"]:
            if word_filter is None or poem not in word_filter:
                allowed[poem] = None
            else:
                allowed[poem] = set(word_filter[poem])

        obj = cls.__new__(cls)
        obj._items = []

        for item in all_items:
            key = (item["subject"], item["poem"], item["session"])
            if key not in trials:
                continue

            allow_set = allowed[item["poem"]]
            if allow_set is None:
                obj._items.append(item)
                continue

            poses = item["word_poses"]
            keep  = [i for i, p in enumerate(poses) if p in allow_set]
            if not keep:
                continue
            new_item = {
                "meg_trial":      item["meg_trial"],
                "onset_samples":  item["onset_samples"][keep],
                "offset_samples": item["offset_samples"][keep],
                "valid_mask":     item["valid_mask"][keep],
                "word_texts":     [item["word_texts"][i] for i in keep],
                "word_poses":     [poses[i]               for i in keep],
                "poem":           item["poem"],
                "subject":        item["subject"],
                "session":        item["session"],
            }
            obj._items.append(new_item)

        print(f"  MEGContinuousTrialDataset.from_cache  trials={len(obj._items):,}")
        return obj


def collate_continuous_trials(batch: List[Dict]) -> Dict:
    """
    Right-pads variable-length CONTINUOUS trials (T differs per trial —
    subject/session-dependent recording length) into a batched
    (B, C, T_max) tensor, plus trial_mask over TIME so the encoder can
    tell real samples from padding. Per-word onset/offset/valid_mask
    arrays are padded separately over N (word count), same pattern as
    base_dataset.py's collate_trials — but note this now pads over TWO
    different axes (time for meg_trial, word count for the onset/offset/
    valid arrays), where collate_trials only ever padded over word count.
    """
    B     = len(batch)
    C     = N_CHANNELS
    T_max = max(item["meg_trial"].shape[1] for item in batch)
    N_max = max(item["onset_samples"].shape[0] for item in batch)

    meg        = torch.zeros(B, C, T_max)
    trial_mask = torch.zeros(B, T_max, dtype=torch.bool)   # True = real sample, False = pad
    onset      = torch.zeros(B, N_max, dtype=torch.long)
    offset     = torch.zeros(B, N_max, dtype=torch.long)
    word_mask  = torch.zeros(B, N_max, dtype=torch.bool)

    for b, item in enumerate(batch):
        T = item["meg_trial"].shape[1]
        N = item["onset_samples"].shape[0]
        meg[b, :, :T]     = item["meg_trial"]
        trial_mask[b, :T] = True
        onset[b, :N]      = item["onset_samples"]
        offset[b, :N]     = item["offset_samples"]
        word_mask[b, :N]  = item["valid_mask"]

    return {
        "meg_trial":      meg,
        "trial_mask":     trial_mask,
        "onset_samples":  onset,
        "offset_samples": offset,
        "valid_mask":     word_mask,
        "word_texts":     [item["word_texts"] for item in batch],
        "word_poses":     [item["word_poses"] for item in batch],
        "poem":           [item["poem"]       for item in batch],
        "subject":        [item["subject"]    for item in batch],
        "session":        [item["session"]    for item in batch],
    }
