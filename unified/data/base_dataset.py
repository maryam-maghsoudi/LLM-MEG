"""
base_dataset.py — core MEG dataset shared by all three methods.

MEGWordDataset   One item per word occurrence (word-level, for Stage 1 / Method 1).
MEGTrialDataset  One item per (subject, poem, session) trial (sequence-level,
                 for Stage 2 / Method 3).
collate_trials   Pads variable-length trials into batched tensors.

Sharing without .fif files
--------------------------
Build the full datasets once (all subjects / poems / sessions), save to disk,
then reconstruct any split on the collaborator's machine without .fif access:

    # One-time export (requires .fif files):
    python -m unified.data.export_datasets --out_dir /path/to/cache

    # Collaborator usage (no .fif files needed):
    from unified.data.base_dataset import MEGWordDataset, MEGTrialDataset
    from unified.data.splits import make_loso_splits

    word_items  = MEGWordDataset.load_items("/path/to/cache/meg_word_all.pt")
    trial_items = MEGTrialDataset.load_items("/path/to/cache/meg_trial_all.pt")

    splits   = make_loso_splits("sub-01")
    ds_train = MEGWordDataset.from_cache(word_items,  splits["train"])
    ds_val   = MEGWordDataset.from_cache(word_items,  splits["val"])
    ds_trial = MEGTrialDataset.from_cache(trial_items, splits["train"])
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import mne
import numpy as np
import torch
from scipy.signal import resample
from torch.utils.data import Dataset

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------
_HERE     = Path(__file__).parent.parent          # unified/
MEG_BASE  = Path("/fs/nexus-projects/brain_project/maryam_meg_dataset/icaed")
ONSET_DIR = _HERE.parent.parent / "contrastive_learning" / "onset_out"

# ---------------------------------------------------------------------------
#  MEG constants  (match llm_twostage)
# ---------------------------------------------------------------------------
DS_FACTOR    = 10
SFREQ_DS     = 100.0
N_CHANNELS   = 155
EPOCH_TMIN_S = 0.0
WIN_PRE      = int(100 * SFREQ_DS / 1000)   # 10 samples  (100 ms before onset)
WIN_POST     = int(300 * SFREQ_DS / 1000)   # 30 samples  (300 ms after onset)
WIN_SIZE     = WIN_PRE + WIN_POST            # 40 samples total

# ---------------------------------------------------------------------------
#  Poem line map  (1-indexed lines → word positions)
# ---------------------------------------------------------------------------
POEM_LINES: Dict[str, List[List[int]]] = {
    "poem1": [
        list(range(0,  5)),   list(range(5,  10)),
        list(range(10, 15)),  list(range(15, 21)),
        list(range(21, 25)),  list(range(25, 30)),
        list(range(30, 34)),  list(range(34, 39)),
        list(range(39, 44)),  list(range(44, 48)),
        list(range(48, 53)),  list(range(53, 56)),
    ],
    "poem2": [
        list(range(0,  6)),   list(range(6,  12)),
        list(range(12, 18)),  list(range(18, 22)),
        list(range(22, 26)),  list(range(26, 32)),
        list(range(32, 38)),  list(range(38, 42)),
        list(range(42, 47)),  list(range(47, 51)),
        list(range(51, 56)),  list(range(56, 61)),
    ],
}

# word_pos → line number (1-indexed)
_WORD_TO_LINE: Dict[str, Dict[int, int]] = {}
for _poem, _lines in POEM_LINES.items():
    _WORD_TO_LINE[_poem] = {}
    for _ln, _words in enumerate(_lines, start=1):
        for _w in _words:
            _WORD_TO_LINE[_poem][_w] = _ln


# ---------------------------------------------------------------------------
#  MEG loading helpers
# ---------------------------------------------------------------------------

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


def _onset_to_window(onset_s: float, n_t: int) -> Optional[Tuple[int, int]]:
    center = int(round((onset_s - EPOCH_TMIN_S) * SFREQ_DS))
    s, e   = center - WIN_PRE, center + WIN_POST
    return (s, e) if (s >= 0 and e <= n_t) else None


def _load_onsets(poem: str) -> List[Dict]:
    return json.loads((ONSET_DIR / f"{poem}_word_onsets.json").read_text())


# ---------------------------------------------------------------------------
#  MEGWordDataset — word-level, for Stage 1 and Method 1
# ---------------------------------------------------------------------------

class MEGWordDataset(Dataset):
    """
    One item per valid word occurrence.

    Parameters
    ----------
    trials      : list of (subject, poem, session) tuples from splits.py
    word_filter : {poem: [word_positions]} to include, or None for all positions
    augment     : if True, add Gaussian noise (σ=0.02) to meg_window at fetch time
    condition   : MEG condition suffix, default 'lis'

    Item keys
    ---------
    meg_window : Tensor(N_CHANNELS, WIN_SIZE)  float32
    word_text  : str    — lowercased word
    word_pos   : int    — 0-indexed position in poem
    line_num   : int    — 1-indexed line number (1–12)
    poem       : str
    subject    : str
    session    : int
    """

    def __init__(
        self,
        trials:      List[Tuple[str, str, int]],
        word_filter: Optional[Dict[str, List[int]]] = None,
        augment:     bool = False,
        condition:   str  = "lis",
        meg_base:    Optional[Path] = None,
    ):
        self.augment = augment
        self._items: List[Dict] = []

        onsets = {p: _load_onsets(p) for p in ["poem1", "poem2"]}

        allowed: Dict[str, Optional[Set[int]]] = {}
        for poem in ["poem1", "poem2"]:
            if word_filter is None or poem not in word_filter:
                allowed[poem] = None
            else:
                allowed[poem] = set(word_filter[poem])

        n_missing = n_invalid = 0
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

            n_t       = data.shape[1]
            allow_set = allowed[poem]

            for pos, entry in enumerate(onsets[poem]):
                if allow_set is not None and pos not in allow_set:
                    continue
                idx = _onset_to_window(entry["start"], n_t)
                if idx is None:
                    n_invalid += 1
                    continue
                s, e = idx
                win  = data[:, s:e]
                if win.shape[-1] != WIN_SIZE:
                    continue
                self._items.append({
                    "meg_window": win.copy(),
                    "word_text":  entry["word"].strip().lower(),
                    "word_pos":   pos,
                    "line_num":   _WORD_TO_LINE[poem][pos],
                    "poem":       poem,
                    "subject":    subject,
                    "session":    session,
                })

        print(
            f"  MEGWordDataset  items={len(self._items):,}  "
            f"missing_trials={n_missing}  invalid_windows={n_invalid}"
        )

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

    # ------------------------------------------------------------------
    #  Cache helpers (no .fif files required on load)
    # ------------------------------------------------------------------

    def save_items(self, path: str) -> None:
        """Save all items to a .pt file for sharing without .fif access."""
        torch.save(self._items, path)
        print(f"  MEGWordDataset  saved {len(self._items):,} items → {path}")

    @staticmethod
    def load_items(path: str) -> List[Dict]:
        """Load raw items saved by save_items()."""
        return torch.load(path, weights_only=False)

    @classmethod
    def from_cache(
        cls,
        all_items:   List[Dict],
        split:       Dict,
        augment:     bool = False,
    ) -> "MEGWordDataset":
        """
        Reconstruct a split-specific dataset from pre-cached items.

        Parameters
        ----------
        all_items : full item list returned by load_items()
        split     : one entry from make_*_splits() — a dict with keys
                    'trials' and 'word_filter'
        augment   : passed through to __getitem__

        Returns a MEGWordDataset without reading any .fif files.
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
        obj.augment = augment
        obj._items  = [
            item for item in all_items
            if (item["subject"], item["poem"], item["session"]) in trials
            and (allowed[item["poem"]] is None
                 or item["word_pos"] in allowed[item["poem"]])
        ]
        print(f"  MEGWordDataset.from_cache  items={len(obj._items):,}")
        return obj


# ---------------------------------------------------------------------------
#  MEGTrialDataset — sequence-level, for Stage 2 (GRU) and Method 3 (interleaved)
# ---------------------------------------------------------------------------

class MEGTrialDataset(Dataset):
    """
    One item per (subject, poem, session) trial, returning the full word
    sequence in order. Used by Method 2 Stage 2 (GRU) and Method 3 (interleaved).

    Item keys
    ---------
    meg_windows : Tensor(N_words, N_CHANNELS, WIN_SIZE)  — zero where invalid
    valid_mask  : Tensor(N_words,) bool  — True where MEG window is usable
    word_texts  : List[str]
    word_poses  : List[int]             — [0, 1, 2, ..., N_words-1]
    poem        : str
    subject     : str
    session     : int
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

            onset_list  = onsets[poem]
            pos_list    = allowed[poem] if allowed[poem] is not None \
                          else list(range(len(onset_list)))
            N           = len(pos_list)
            n_t         = data.shape[1]
            windows     = np.zeros((N, N_CHANNELS, WIN_SIZE), dtype=np.float32)
            valid_mask  = np.zeros(N, dtype=bool)
            word_texts  = []

            for i, pos in enumerate(pos_list):
                entry = onset_list[pos]
                word_texts.append(entry["word"].strip().lower())
                idx = _onset_to_window(entry["start"], n_t)
                if idx is None:
                    continue
                s, e = idx
                win  = data[:, s:e]
                if win.shape[-1] != WIN_SIZE:
                    continue
                windows[i]    = win
                valid_mask[i] = True

            self._items.append({
                "meg_windows": torch.from_numpy(windows),
                "valid_mask":  torch.from_numpy(valid_mask),
                "word_texts":  word_texts,
                "word_poses":  pos_list,
                "poem":        poem,
                "subject":     subject,
                "session":     session,
            })

        print(
            f"  MEGTrialDataset  trials={len(self._items):,}  missing={n_missing}"
        )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict:
        return self._items[idx]

    # ------------------------------------------------------------------
    #  Cache helpers (no .fif files required on load)
    # ------------------------------------------------------------------

    def save_items(self, path: str) -> None:
        """Save all items to a .pt file for sharing without .fif access."""
        torch.save(self._items, path)
        print(f"  MEGTrialDataset  saved {len(self._items):,} items → {path}")

    @staticmethod
    def load_items(path: str) -> List[Dict]:
        """Load raw items saved by save_items()."""
        return torch.load(path, weights_only=False)

    @classmethod
    def from_cache(
        cls,
        all_items:   List[Dict],
        split:       Dict,
    ) -> "MEGTrialDataset":
        """
        Reconstruct a split-specific dataset from pre-cached items.

        Parameters
        ----------
        all_items : full item list returned by load_items()
        split     : one entry from make_*_splits() — a dict with keys
                    'trials' and 'word_filter'

        For the stimulus split, word_filter restricts which word positions
        are included within each trial's sequence. Positions outside the
        filter are zeroed out and marked invalid in valid_mask.

        Returns a MEGTrialDataset without reading any .fif files.
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
            else:
                # Keep only word positions in the filter; zero-out the rest
                poses      = item["word_poses"]
                keep       = [i for i, p in enumerate(poses) if p in allow_set]
                if not keep:
                    continue
                new_item = {
                    "meg_windows": item["meg_windows"][keep],
                    "valid_mask":  item["valid_mask"][keep],
                    "word_texts":  [item["word_texts"][i] for i in keep],
                    "word_poses":  [poses[i]              for i in keep],
                    "poem":        item["poem"],
                    "subject":     item["subject"],
                    "session":     item["session"],
                }
                obj._items.append(new_item)

        print(f"  MEGTrialDataset.from_cache  trials={len(obj._items):,}")
        return obj


def collate_trials(batch: List[Dict]) -> Dict:
    """
    Pad trials of different lengths into batched tensors.
    Preserves all scalar and list fields; zero-pads meg_windows and valid_mask.
    """
    max_N = max(item["meg_windows"].shape[0] for item in batch)
    B     = len(batch)
    C, T  = N_CHANNELS, WIN_SIZE

    meg  = torch.zeros(B, max_N, C, T)
    mask = torch.zeros(B, max_N, dtype=torch.bool)

    for b, item in enumerate(batch):
        N = item["meg_windows"].shape[0]
        meg[b, :N]  = item["meg_windows"]
        mask[b, :N] = item["valid_mask"]

    return {
        "meg_windows": meg,
        "valid_mask":  mask,
        "word_texts":  [item["word_texts"] for item in batch],
        "word_poses":  [item["word_poses"] for item in batch],
        "poem":        [item["poem"]       for item in batch],
        "subject":     [item["subject"]    for item in batch],
        "session":     [item["session"]    for item in batch],
    }
