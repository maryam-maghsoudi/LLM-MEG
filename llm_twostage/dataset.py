"""
dataset.py — Steps 3, 5, 9
============================
MEGWordDatasetLLM   (Stage 1)
    One item = one word occurrence from any (subject, session, poem) trial.
    Returns: meg_window (C, WIN_SIZE), hmid_t (d_model,)
    The hmid_t comes from the precomputed LLM hidden-state cache (Step 1).

MEGSequenceDataset  (Stage 2)
    One item = one full (subject, session, poem) trial in word order.
    Returns: meg_windows (N, C, WIN_SIZE), p_t_restricted (N, R), valid_mask (N,)

make_loso_splits
    Subject-primary / session-secondary LOSO.
    train : non-heldout subjects, sessions 0-7, both poems
    val   : non-heldout subjects, sessions 8-9, both poems
    test  : heldout subject, all sessions, both poems

MEG loading mirrors contrastive_word_meg.py (REMOVE_FLASHES=False,
WIN=[-200ms, +800ms], downsample 10×, z-score per channel).
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
_HERE      = Path(__file__).parent
MEG_BASE   = Path("/fs/nexus-projects/brain_project/maryam_meg_dataset/icaed")
ONSET_DIR  = _HERE.parent.parent / "contrastive_learning" / "onset_out"
CACHE_ROOT = _HERE / "cache"

# ---------------------------------------------------------------------------
#  MEG constants  (must match cache_llm_hiddens.py and llm_decoder/config.py)
# ---------------------------------------------------------------------------
DS_FACTOR    = 10
SFREQ_DS     = 100.0
N_CHANNELS   = 155
EPOCH_TMIN_S = 0.0
WIN_PRE      = int(200 * SFREQ_DS / 1000)    # 20 samples
WIN_POST     = int(800 * SFREQ_DS / 1000)    # 80 samples
WIN_SIZE     = WIN_PRE + WIN_POST             # 100 samples

SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]
POEM_KEYS      = ["poem1", "poem2"]
N_SESSIONS     = 10
TRAIN_SESSIONS = list(range(8))
VAL_SESSIONS   = [8, 9]


# ===========================================================================
#  MEG helpers
# ===========================================================================

def _load_meg_trial(subject: str, condition: str, session: int) -> Optional[np.ndarray]:
    """
    Load one MEG trial, downsample 10×, z-score per channel.
    condition: e.g. "poem1lis", "poem2lis".
    Returns (N_CHANNELS, T_ds) float32, or None if file is missing/unreadable.
    """
    fname = f"{subject}_sess-{session}_task-{condition}_meg-epo.fif"
    fpath = MEG_BASE / subject / f"ses-{session}" / "meg" / fname
    if not fpath.exists():
        return None
    try:
        epochs = mne.read_epochs(str(fpath), preload=True)
    except Exception:
        return None
    raw  = epochs.get_data().mean(axis=0)
    data = resample(raw, raw.shape[1] // DS_FACTOR, axis=1).astype(np.float32)
    mu   = data.mean(axis=1, keepdims=True)
    sd   = np.maximum(data.std(axis=1, keepdims=True), 1e-12)
    return (data - mu) / sd


def _onset_to_window(onset_s: float, n_t: int) -> Optional[Tuple[int, int]]:
    center = int(round((onset_s - EPOCH_TMIN_S) * SFREQ_DS))
    s, e   = center - WIN_PRE, center + WIN_POST
    return (s, e) if (s >= 0 and e <= n_t) else None


def _load_onsets(poem: str) -> List[Dict]:
    return json.loads((ONSET_DIR / f"{poem}_word_onsets.json").read_text())


# ===========================================================================
#  Cache helpers
# ===========================================================================

def model_tag(llm_name: str) -> str:
    return llm_name.replace("/", "_")


def load_poem_cache(poem: str, llm_name: str) -> Dict:
    path = CACHE_ROOT / model_tag(llm_name) / f"{poem}_hiddens.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Cache not found: {path}\n"
            f"Run: python cache_llm_hiddens.py --llm_name {llm_name}"
        )
    return torch.load(path, map_location="cpu")


def load_vocab_info(llm_name: str) -> Dict:
    path = CACHE_ROOT / model_tag(llm_name) / "vocab_info.json"
    return json.loads(path.read_text())


# ===========================================================================
#  Stage 1 dataset — occurrence-level
# ===========================================================================

class MEGWordDatasetLLM(Dataset):
    """
    One item = (meg_window, hmid_t) for one word occurrence.

    meg_window : (C, WIN_SIZE)  float32 — from raw MEG
    hmid_t     : (d_model,)    float32 — LLM hidden state at hmid_layer
    """

    def __init__(
        self,
        subjects:   List[str],
        poems:      List[str],
        sessions:   List[int],
        llm_name:   str,
        hmid_layer: int,
        condition:  str  = "lis",
        augment:    bool = False,
    ):
        self.augment = augment
        self._pairs: List[Tuple[np.ndarray, torch.Tensor]] = []

        caches = {p: load_poem_cache(p, llm_name) for p in poems}
        onsets = {p: _load_onsets(p) for p in poems}

        n_missing = n_invalid = 0

        for poem in poems:
            hmid_all   = caches[poem]["hidden_all_layers"][hmid_layer]  # (N, d)
            onset_list = onsets[poem]
            cond       = f"{poem}{condition}"

            for subject in subjects:
                for session in sessions:
                    data = _load_meg_trial(subject, cond, session)
                    if data is None:
                        n_missing += 1
                        continue
                    n_t = data.shape[1]
                    for pos, entry in enumerate(onset_list):
                        idx = _onset_to_window(entry["start"], n_t)
                        if idx is None:
                            n_invalid += 1
                            continue
                        s, e = idx
                        win  = data[:, s:e]
                        if win.shape[-1] != WIN_SIZE:
                            continue
                        self._pairs.append((win.copy(), hmid_all[pos]))

        print(
            f"  MEGWordDatasetLLM  cond={condition!r}  layer={hmid_layer}  "
            f"pairs={len(self._pairs):,}  "
            f"(missing_trials={n_missing}, invalid_windows={n_invalid})"
        )

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        window, hmid = self._pairs[idx]
        x = torch.from_numpy(window)
        if self.augment:
            x = x + 0.02 * torch.randn_like(x)
        return x, hmid


# ===========================================================================
#  Stage 2 dataset — trial-level (full poem sequence)
# ===========================================================================

class MEGSequenceDataset(Dataset):
    """
    One item = one (subject, session, poem) trial.

    meg_windows    : (N, C, WIN_SIZE)  float32  — zero where invalid
    p_t_restricted : (N, R)            float32  — teacher logits from cache
    valid_mask     : (N,)              bool     — True where MEG window is usable
    word_texts     : List[str]                  — word strings for readability
    meta           : dict
    """

    def __init__(
        self,
        subjects:  List[str],
        poems:     List[str],
        sessions:  List[int],
        llm_name:  str,
        condition: str = "lis",
    ):
        caches = {p: load_poem_cache(p, llm_name) for p in poems}
        onsets = {p: _load_onsets(p) for p in poems}

        self._items: List[Dict] = []
        n_missing = 0

        for poem in poems:
            p_t_all    = caches[poem]["lm_logits_restricted"]  # (N, R)
            onset_list = onsets[poem]
            word_texts = [e["word"].strip().lower() for e in onset_list]
            N          = len(onset_list)
            cond       = f"{poem}{condition}"

            for subject in subjects:
                for session in sessions:
                    data = _load_meg_trial(subject, cond, session)
                    if data is None:
                        n_missing += 1
                        continue

                    C          = data.shape[0]
                    n_t        = data.shape[1]
                    windows    = np.zeros((N, C, WIN_SIZE), dtype=np.float32)
                    valid_mask = np.zeros(N, dtype=bool)

                    for pos, entry in enumerate(onset_list):
                        idx = _onset_to_window(entry["start"], n_t)
                        if idx is None:
                            continue
                        s, e = idx
                        win  = data[:, s:e]
                        if win.shape[-1] != WIN_SIZE:
                            continue
                        windows[pos]    = win
                        valid_mask[pos] = True

                    self._items.append({
                        "meg_windows":    torch.from_numpy(windows),
                        "p_t_restricted": p_t_all,
                        "valid_mask":     torch.from_numpy(valid_mask),
                        "word_texts":     word_texts,
                        "meta": {
                            "subject": subject,
                            "poem":    poem,
                            "session": session,
                        },
                    })

        print(
            f"  MEGSequenceDataset cond={condition!r}  "
            f"trials={len(self._items):,}  missing={n_missing}"
        )

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def collate_sequences(batch: List[Dict]) -> Dict:
    """
    Pad trials of different poem lengths (56 vs 61 words) into batched tensors.
    """
    max_N = max(item["meg_windows"].shape[0] for item in batch)
    B     = len(batch)
    C     = batch[0]["meg_windows"].shape[1]
    R     = batch[0]["p_t_restricted"].shape[-1]

    meg  = torch.zeros(B, max_N, C, WIN_SIZE)
    p_t  = torch.zeros(B, max_N, R)
    mask = torch.zeros(B, max_N, dtype=torch.bool)

    for b, item in enumerate(batch):
        N = item["meg_windows"].shape[0]
        meg[b, :N]  = item["meg_windows"]
        p_t[b, :N]  = item["p_t_restricted"]
        mask[b, :N] = item["valid_mask"]

    return {
        "meg_windows":    meg,
        "p_t_restricted": p_t,
        "valid_mask":     mask,
        "word_texts":     [item["word_texts"] for item in batch],
        "meta":           [item["meta"]       for item in batch],
    }


# ===========================================================================
#  LOSO split
# ===========================================================================

def make_loso_splits(
    heldout_subject: str,
    all_subjects:    List[str] = SUBJECTS,
    poems:           List[str] = POEM_KEYS,
    train_sessions:  List[int] = TRAIN_SESSIONS,
    val_sessions:    List[int] = VAL_SESSIONS,
) -> Dict:
    """
    Subject-primary, session-secondary LOSO split.
    Returns a dict with keys 'train', 'val', 'test', each containing
    {subjects, poems, sessions}.
    """
    if heldout_subject not in all_subjects:
        raise ValueError(f"{heldout_subject!r} not in SUBJECTS list.\n"
                         f"Valid: {all_subjects}")
    train_subs = [s for s in all_subjects if s != heldout_subject]

    print(f"\nLOSO split  (heldout={heldout_subject})")
    print(f"  train : {len(train_subs)} subjects × {len(poems)} poems "
          f"× {len(train_sessions)} sessions "
          f"= ~{len(train_subs)*len(poems)*len(train_sessions)} trials")
    print(f"  val   : {len(train_subs)} subjects × {len(poems)} poems "
          f"× {len(val_sessions)} sessions "
          f"= ~{len(train_subs)*len(poems)*len(val_sessions)} trials")
    print(f"  test  : 1 subject × {len(poems)} poems "
          f"× {N_SESSIONS} sessions "
          f"= ~{len(poems)*N_SESSIONS} trials")

    return {
        "train": {"subjects": train_subs,         "poems": poems, "sessions": train_sessions},
        "val":   {"subjects": train_subs,         "poems": poems, "sessions": val_sessions},
        "test":  {"subjects": [heldout_subject],  "poems": poems,
                  "sessions": list(range(N_SESSIONS))},
    }
