"""
dataset.py
==========
Poem-level MEG dataset for the LLM-guided imagined speech decoder.

Each item represents one complete poem trial:
  subject × session × poem × condition (listened or imagined)

Returns per trial:
  meg_windows   : (N_words, C, WIN_SIZE) float32
                  One z-scored, downsampled MEG window per poem word.
                  Rows where valid_mask is False are zero-padded.
  valid_mask    : (N_words,) bool
                  True for words whose window fell within the trial and
                  did not overlap a boundary.
  word_token_ids: List[List[int]]  (length N_words)
                  GPT-2 token IDs for each word (leading-space tokenisation
                  to match how words appear in natural text).
  word_texts    : List[str]  (length N_words)
                  Raw word strings from the onset file.
  meta          : dict  {subject, poem, session, condition}

The dataset is design-agnostic: it does not build the interleaved sequence
(that is the model's job). It simply provides the per-word MEG windows and
the corresponding token IDs that the model needs for both Design A and B.

Usage
-----
  from dataset import PoemTrialDataset, collate_trials
  from torch.utils.data import DataLoader

  ds = PoemTrialDataset(subjects=SUBJECTS, poems=["poem1"], sessions=range(8))
  loader = DataLoader(ds, batch_size=4, collate_fn=collate_trials, shuffle=True)

  for batch in loader:
      # batch["meg_windows"]    : (B, N_words, C, WIN_SIZE)
      # batch["valid_mask"]     : (B, N_words)
      # batch["word_token_ids"] : List[B × List[N_words × List[int]]]
      # batch["word_texts"]     : List[B × List[str]]
      # batch["meta"]           : List[B × dict]
      ...
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import torch
from scipy.signal import resample
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import (
    MEG_BASE, ONSET_DIR, SUBJECTS, POEM_KEYS, N_SESSIONS,
    DS_FACTOR, SFREQ_DS, EPOCH_TMIN_S,
    WIN_PRE, WIN_POST, WIN_SIZE,
    LLM_NAME,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
#  MEG LOADING  (mirrors contrastive_word_meg.py, kept self-contained here)
# =============================================================================

def _load_meg_trial(
    subject: str,
    condition: str,
    session: int,
) -> Optional[np.ndarray]:
    """
    Load one MEG trial, downsample, z-score.

    Parameters
    ----------
    subject   : e.g. "sub-01"
    condition : e.g. "poem1lis" or "poem1img"
    session   : integer 0–9

    Returns
    -------
    data : (C, T_ds) float32, or None if the file cannot be read.
    """
    fname = f"{subject}_sess-{session}_task-{condition}_meg-epo.fif"
    fpath = MEG_BASE / subject / f"ses-{session}" / "meg" / fname

    if not fpath.exists():
        return None

    try:
        epochs = mne.read_epochs(str(fpath), preload=True)
    except Exception:
        return None

    raw = epochs.get_data().mean(axis=0)                        # (C, T_raw)
    new_T = raw.shape[1] // DS_FACTOR
    data  = resample(raw, new_T, axis=1).astype(np.float32)    # (C, T_ds)

    # z-score per channel
    mu  = data.mean(axis=1, keepdims=True)
    sd  = np.maximum(data.std(axis=1, keepdims=True), 1e-12)
    return (data - mu) / sd


def _onset_to_window(
    onset_s: float,
    n_t: int,
) -> Optional[Tuple[int, int]]:
    """
    Convert an audio onset time (seconds) to a sample index window.

    Returns (start, end) exclusive, or None if the window goes out of bounds.
    """
    center = int(round((onset_s - EPOCH_TMIN_S) * SFREQ_DS))
    start  = center - WIN_PRE
    end    = center + WIN_POST
    if start < 0 or end > n_t:
        return None
    return start, end


# =============================================================================
#  ONSET LOADING
# =============================================================================

def _load_onsets(poem: str) -> List[Dict]:
    """
    Load word onset list for a poem.

    Returns a list of dicts: [{word, start, end}, ...]
    """
    path = Path(ONSET_DIR) / f"{poem}_word_onsets.json"
    if not path.exists():
        raise FileNotFoundError(f"Onset file not found: {path}")
    with open(path) as f:
        return json.load(f)


# =============================================================================
#  TOKENIZER (initialised once, shared across dataset instances)
# =============================================================================

def _make_tokenizer(llm_name: str = LLM_NAME) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(llm_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# =============================================================================
#  TRIAL BUILDER
# =============================================================================

@dataclass
class _Trial:
    """Identifies one poem trial."""
    subject:   str
    poem:      str
    session:   int
    condition: str   # "lis" or "img"


def _build_trials(
    subjects:   List[str],
    poems:      List[str],
    sessions:   List[int],
    condition:  str,
) -> List[_Trial]:
    return [
        _Trial(subject=s, poem=p, session=ses, condition=condition)
        for s   in subjects
        for p   in poems
        for ses in sessions
    ]


# =============================================================================
#  DATASET
# =============================================================================

class PoemTrialDataset(Dataset):
    """
    Poem-level dataset: one item = one full poem trial.

    Parameters
    ----------
    subjects   : list of subject IDs to include
    poems      : list of poem keys, e.g. ["poem1"] or ["poem1", "poem2"]
    sessions   : list of session indices (0–9) to include
    condition  : "lis" (listened) or "img" (imagined)
    llm_name   : HuggingFace model name for the tokenizer (must match the LLM
                 used in model.py so token IDs are consistent)
    cache      : if True, pre-load all MEG data into RAM during __init__
    min_word_len : words shorter than this are still included in the sequence
                   (they occupy a window slot with valid_mask=True/False) but
                   their token IDs are still computed; set to 1 to include all
    """

    def __init__(
        self,
        subjects:     List[str] = SUBJECTS,
        poems:        List[str] = POEM_KEYS,
        sessions:     Optional[List[int]] = None,
        condition:    str = "lis",
        llm_name:     str = LLM_NAME,
        cache:        bool = False,
        min_word_len: int = 1,
    ):
        if sessions is None:
            sessions = list(range(N_SESSIONS))

        self.condition    = condition
        self.min_word_len = min_word_len

        # Build (subject, poem, session) trial index
        self._trials: List[_Trial] = _build_trials(
            subjects, poems, sessions, condition
        )

        # Pre-load onset lists (one per poem, shared across all subjects/sessions)
        self._onsets: Dict[str, List[Dict]] = {
            poem: _load_onsets(poem) for poem in poems
        }

        # Shared tokenizer (loaded once)
        self._tokenizer = _make_tokenizer(llm_name)

        # Per-poem token-ID sequences (same words every trial; only MEG varies)
        self._poem_token_ids: Dict[str, List[List[int]]] = {
            poem: self._tokenize_onsets(self._onsets[poem])
            for poem in poems
        }
        self._poem_texts: Dict[str, List[str]] = {
            poem: [w["word"].strip().lower() for w in self._onsets[poem]]
            for poem in poems
        }

        # Optional RAM cache
        self._cache: Optional[List[Optional[Dict]]] = (
            [None] * len(self._trials) if cache else None
        )
        if cache:
            print(f"  Caching {len(self._trials)} trials into RAM...")
            for i in range(len(self._trials)):
                self._cache[i] = self._load_item(i)
            n_ok = sum(1 for x in self._cache if x is not None)
            print(f"  Cached {n_ok}/{len(self._trials)} trials successfully.")

        print(
            f"PoemTrialDataset: {len(self._trials)} trials "
            f"({len(subjects)} subjects × {len(poems)} poems × "
            f"{len(sessions)} sessions, condition={condition!r})"
        )

    # ------------------------------------------------------------------
    #  Tokenisation helpers
    # ------------------------------------------------------------------

    def _tokenize_word(self, word: str) -> List[int]:
        """
        Tokenise a single word, prepending a space so BPE matches how the
        word appears mid-sentence (e.g. " dressed" not "dressed").
        Returns a non-empty list of token IDs (at least one token).
        """
        ids = self._tokenizer.encode(
            " " + word,
            add_special_tokens=False,
        )
        if len(ids) == 0:
            # Fallback: encode without space
            ids = self._tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 0:
            # Ultimate fallback: unknown token
            ids = [self._tokenizer.unk_token_id or 0]
        return ids

    def _tokenize_onsets(self, onsets: List[Dict]) -> List[List[int]]:
        return [
            self._tokenize_word(w["word"].strip().lower()) for w in onsets
        ]

    # ------------------------------------------------------------------
    #  MEG window extraction
    # ------------------------------------------------------------------

    def _extract_windows(
        self,
        trial: _Trial,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the MEG trial and extract per-word windows.

        Returns
        -------
        windows    : (N_words, C, WIN_SIZE) float32  (zero where invalid)
        valid_mask : (N_words,) bool
        """
        onsets = self._onsets[trial.poem]
        N      = len(onsets)

        cond = f"{trial.poem}{trial.condition}"
        data = _load_meg_trial(trial.subject, cond, trial.session)

        C   = data.shape[0] if data is not None else 0
        windows    = np.zeros((N, C or 1, WIN_SIZE), dtype=np.float32)
        valid_mask = np.zeros(N, dtype=bool)

        if data is None:
            return windows, valid_mask

        windows = np.zeros((N, C, WIN_SIZE), dtype=np.float32)
        n_t = data.shape[1]

        for i, onset_entry in enumerate(onsets):
            word = onset_entry["word"].strip().lower()
            if len(word) < self.min_word_len:
                continue
            idx = _onset_to_window(onset_entry["start"], n_t)
            if idx is None:
                continue
            start, end = idx
            window = data[:, start:end]
            if window.shape[-1] != WIN_SIZE:
                continue
            windows[i]    = window
            valid_mask[i] = True

        return windows, valid_mask

    # ------------------------------------------------------------------
    #  Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._trials)

    def _load_item(self, idx: int) -> Dict:
        trial = self._trials[idx]
        windows, valid_mask = self._extract_windows(trial)

        return {
            "meg_windows":    torch.from_numpy(windows),             # (N, C, T)
            "valid_mask":     torch.from_numpy(valid_mask),          # (N,)
            "word_token_ids": self._poem_token_ids[trial.poem],      # List[List[int]]
            "word_texts":     self._poem_texts[trial.poem],          # List[str]
            "meta": {
                "subject":   trial.subject,
                "poem":      trial.poem,
                "session":   trial.session,
                "condition": trial.condition,
            },
        }

    def __getitem__(self, idx: int) -> Dict:
        if self._cache is not None:
            if self._cache[idx] is None:
                self._cache[idx] = self._load_item(idx)
            return self._cache[idx]
        return self._load_item(idx)

    # ------------------------------------------------------------------
    #  Convenience: vocab of all unique words across loaded poems
    # ------------------------------------------------------------------

    @property
    def vocab(self) -> List[str]:
        words = []
        seen  = set()
        for texts in self._poem_texts.values():
            for w in texts:
                if w not in seen:
                    seen.add(w)
                    words.append(w)
        return words


# =============================================================================
#  COLLATE FUNCTION
# =============================================================================

def collate_trials(batch: List[Dict]) -> Dict:
    """
    Collate a list of trial dicts into a batch.

    MEG windows and valid masks are stacked into tensors.
    word_token_ids and word_texts remain as nested lists (they are ragged
    across words and sub-tokens, so tensors are not appropriate here).

    Returns
    -------
    {
      "meg_windows"    : (B, N_words, C, WIN_SIZE)   float32
      "valid_mask"     : (B, N_words)                bool
      "word_token_ids" : List[B]  each element is List[N_words × List[int]]
      "word_texts"     : List[B]  each element is List[str]
      "meta"           : List[B]  each element is dict
    }

    Note: trials in a batch may have different N_words if different poems are
    mixed. In that case meg_windows and valid_mask are zero-padded to the
    maximum N_words in the batch, with a "n_words" key indicating true lengths.
    """
    # Find max N_words in the batch (usually all same poem → same N)
    n_words_list = [item["meg_windows"].shape[0] for item in batch]
    max_n = max(n_words_list)

    # Pad and stack MEG windows
    meg_list  = []
    mask_list = []
    for item in batch:
        n = item["meg_windows"].shape[0]
        if n < max_n:
            # Pad with zeros along the word dimension
            pad_w = torch.zeros(max_n - n, *item["meg_windows"].shape[1:])
            pad_m = torch.zeros(max_n - n, dtype=torch.bool)
            meg_list.append(torch.cat([item["meg_windows"], pad_w], dim=0))
            mask_list.append(torch.cat([item["valid_mask"], pad_m], dim=0))
        else:
            meg_list.append(item["meg_windows"])
            mask_list.append(item["valid_mask"])

    return {
        "meg_windows":    torch.stack(meg_list,  dim=0),   # (B, N, C, T)
        "valid_mask":     torch.stack(mask_list, dim=0),   # (B, N)
        "n_words":        torch.tensor(n_words_list),      # (B,)
        "word_token_ids": [item["word_token_ids"] for item in batch],
        "word_texts":     [item["word_texts"]     for item in batch],
        "meta":           [item["meta"]           for item in batch],
    }


# =============================================================================
#  SPLIT HELPERS
# =============================================================================

def make_splits(
    train_subjects: List[str],
    train_poems:    List[str],
    train_sessions: List[int],
    val_subjects:   List[str],
    val_poems:      List[str],
    val_sessions:   List[int],
    test_subjects:  List[str],
    test_poems:     List[str],
    test_sessions:  List[int],
    condition:      str = "lis",
    llm_name:       str = LLM_NAME,
    cache:          bool = False,
) -> Tuple["PoemTrialDataset", "PoemTrialDataset", "PoemTrialDataset"]:
    """
    Build train / val / test datasets from explicit subject/poem/session lists.

    Recommended split (from Architecture.md):
      train  : all subjects, poem1, sessions 0-7
      val    : all subjects, poem1, sessions 8-9
      test   : all subjects, poem2, all sessions

    This enforces poem-level generalisation: the model never sees poem2
    during training, so it cannot memorise that poem's word sequence.
    """
    train_ds = PoemTrialDataset(train_subjects, train_poems, train_sessions,
                                condition, llm_name, cache)
    val_ds   = PoemTrialDataset(val_subjects,   val_poems,   val_sessions,
                                condition, llm_name, cache)
    test_ds  = PoemTrialDataset(test_subjects,  test_poems,  test_sessions,
                                condition, llm_name, cache)
    return train_ds, val_ds, test_ds


# =============================================================================
#  QUICK SANITY CHECK
# =============================================================================

if __name__ == "__main__":
    from config import SUBJECTS, TRAIN_POEMS, TEST_POEMS, TRAIN_SESSIONS, VAL_SESSIONS

    print("=" * 60)
    print("Sanity check: building train split (poem1, sessions 0-7)")
    print("=" * 60)

    ds = PoemTrialDataset(
        subjects=SUBJECTS[:2],      # first 2 subjects for speed
        poems=TRAIN_POEMS,
        sessions=TRAIN_SESSIONS[:2],
        condition="lis",
        cache=False,
    )

    print(f"\nDataset length: {len(ds)}")

    item = ds[0]
    print(f"\nFirst item:")
    print(f"  meg_windows  : {item['meg_windows'].shape}  dtype={item['meg_windows'].dtype}")
    print(f"  valid_mask   : {item['valid_mask'].shape}  n_valid={item['valid_mask'].sum().item()}")
    print(f"  n_words      : {len(item['word_texts'])}")
    print(f"  meta         : {item['meta']}")
    print(f"\nFirst 5 words and token IDs:")
    for i, (word, ids) in enumerate(
        zip(item["word_texts"][:5], item["word_token_ids"][:5])
    ):
        print(f"  {word!r:15s} → {ids}")

    print(f"\nVocab size: {len(ds.vocab)}")
    print(f"Vocab (first 10): {ds.vocab[:10]}")

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_trials, shuffle=False)
    batch  = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  meg_windows : {batch['meg_windows'].shape}")
    print(f"  valid_mask  : {batch['valid_mask'].shape}")
    print(f"  n_words     : {batch['n_words']}")
    print(f"  meta[0]     : {batch['meta'][0]}")
    print("\nSanity check passed.")
