"""
trial_decode.py
===============
Run the full decoding pipeline for a single subject + session + poem.

For listened (lis):
  Load MEG trial → extract word windows → MEG encoder → cosine sim vs BERT

For imagined (img):
  Load MEG trial → img→lis mapping → extract word windows → MEG encoder → cosine sim vs BERT

Output
------
  (N_words, V) cosine similarity matrix   — rows = words in trial order
  (N_words,)   true word labels            — the ground-truth word for each row

Usage
-----
  # Listened trial
  python trial_decode.py --subject sub-01 --session 3 --poem poem1 --cond lis

  # Imagined trial (needs img→lis checkpoint)
  python trial_decode.py --subject sub-01 --session 3 --poem poem1 --cond img \\
      --img_lis_ckpt /path/to/CNN1D_full.pt --img_lis_arch CNN1D

  # Use 400ms window decoder instead of 800ms
  python trial_decode.py --subject sub-01 --session 3 --poem poem1 --cond lis \\
      --window 400ms
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample

import mne
mne.set_log_level("ERROR")

# =============================================================================
#  CONFIG — edit here to change defaults
# =============================================================================

BASE_PATH  = "/fs/nexus-projects/brain_project/maryam_meg_dataset/icaed"
ONSET_DIR  = "./onset_out"

# Decoder checkpoints — keyed by window size
DECODER_DIRS = {
    "800ms": "./compare_out/models/bert",
    "400ms": "./compare_out_400ms/models/bert",
}

# img→lis checkpoint directory (LOSO models, one per heldout subject)
IMG_LIS_DIR = (
    "/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis"
    "/benchmark/no_flash_removal/loso_out/models"
)

# MEG preprocessing
DS_FACTOR    = 10
SFREQ_DS     = 100.0
EPOCH_TMIN_S = 0.0

# Window configs
WINDOW_CONFIGS = {
    "800ms": {"pre_ms": 200, "post_ms": 800},
    "400ms": {"pre_ms": 200, "post_ms": 400},
}

EMB_DIM  = 128
SEED     = 42

def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = _get_device()


# =============================================================================
#  MEG LOADING
# =============================================================================

def load_trial(subject: str, session: int, cond: str) -> np.ndarray:
    """
    Load one MEG trial (subject, session, condition string like 'poem1lis').
    Returns (C, T) float32, z-scored per channel, downsampled to SFREQ_DS.
    """
    fname = f"{subject}_sess-{session}_task-{cond}_meg-epo.fif"
    fpath = os.path.join(BASE_PATH, subject, f"ses-{session}", "meg", fname)
    epochs  = mne.read_epochs(fpath, preload=True)
    raw     = epochs.get_data().mean(axis=0)           # (C, T_raw)
    new_T   = raw.shape[1] // DS_FACTOR
    data    = resample(raw, new_T, axis=1).astype(np.float32)
    mu = data.mean(axis=1, keepdims=True)
    sd = np.maximum(data.std(axis=1, keepdims=True), 1e-12)
    return (data - mu) / sd                            # (C, T_ds)


# =============================================================================
#  IMG→LIS MAPPING
# =============================================================================

def apply_img_lis(data: np.ndarray, arch: str, ckpt_path: str) -> np.ndarray:
    """
    Apply a trained img→lis mapping model to a full trial.
    data : (C, T) imagined MEG
    Returns (C, T) predicted listened MEG, same shape.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark" / "no_flash_removal"))
    from benchmark_loso import CNN1D, ShallowMLP, UNet1D, RNN, TCN, TARGET_HIDDEN

    arch_map = {
        "CNN1D":      lambda C: CNN1D(C, TARGET_HIDDEN),
        "ShallowMLP": lambda C: ShallowMLP(C, TARGET_HIDDEN),
        "UNet1D":     lambda C: UNet1D(C, TARGET_HIDDEN // 2),
        "RNN":        lambda C: RNN(C, TARGET_HIDDEN),
        "TCN":        lambda C: TCN(C, TARGET_HIDDEN // 2),
    }
    if arch not in arch_map:
        raise ValueError(f"Unknown arch {arch!r}. Choose from {list(arch_map)}")

    C = data.shape[0]
    model = arch_map[arch](C)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        x     = torch.from_numpy(data).unsqueeze(0)   # (1, C, T)
        x_hat = model(x).squeeze(0).numpy()           # (C, T)

    print(f"  img→lis [{arch}] applied: {data.shape} → {x_hat.shape}")
    return x_hat


# =============================================================================
#  WORD WINDOW EXTRACTION
# =============================================================================

def extract_word_windows(
    data:    np.ndarray,    # (C, T)
    onsets:  List[dict],    # list of {"word": str, "start": float}
    pre_ms:  int,
    post_ms: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Slice word windows from a trial.

    Returns
    -------
    windows    : (N_words, C, WIN_SIZE) float32
    word_labels: (N_words,) list of word strings, in onset order
    """
    pre  = int(pre_ms  * SFREQ_DS / 1000)
    post = int(post_ms * SFREQ_DS / 1000)
    win_size = pre + post
    n_t = data.shape[-1]

    windows = []
    labels  = []

    for w in onsets:
        word  = w["word"].strip().lower()
        onset = int(round((w["start"] - EPOCH_TMIN_S) * SFREQ_DS))
        start = onset - pre
        end   = onset + post

        if start < 0 or end > n_t:
            print(f"  SKIP '{word}' (onset {w['start']:.3f}s): out of bounds")
            continue

        window = data[:, start:end]
        if window.shape[-1] != win_size:
            continue

        windows.append(window)
        labels.append(word)

    if not windows:
        raise RuntimeError("No valid word windows found in this trial.")

    return np.stack(windows, axis=0).astype(np.float32), labels


# =============================================================================
#  MODEL LOADING
# =============================================================================

def load_decoder(decoder_dir: str, n_channels: int, window_size: str):
    """Load MEG encoder + text encoder from a compare_out/models/bert dir."""
    # Import only the model classes — window size doesn't affect model architecture,
    # only which checkpoint directory we load from (already handled by DECODER_DIRS).
    import contrastive_word_meg_compare as cmp

    ckpt_meg = os.path.join(decoder_dir, "meg_encoder.pt")
    ckpt_txt = os.path.join(decoder_dir, "text_encoder.pt")

    if not os.path.exists(ckpt_meg):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_meg}")

    # The text encoder checkpoint includes the frozen embeddings as a buffer
    txt_state = torch.load(ckpt_txt, map_location="cpu")
    raw_emb   = txt_state["embeddings"]              # (V, raw_dim)
    V         = raw_emb.shape[0]

    meg_enc = cmp.make_meg_encoder(n_channels, model_size="small").to(DEVICE)
    txt_enc = cmp.TextEncoder(raw_emb).to(DEVICE)

    meg_enc.load_state_dict(torch.load(ckpt_meg, map_location="cpu"))
    txt_enc.load_state_dict(txt_state)
    meg_enc.eval()
    txt_enc.eval()

    print(f"  Loaded decoder: {decoder_dir}  (V={V})")
    return meg_enc, txt_enc


# =============================================================================
#  COSINE SIMILARITY MATRIX
# =============================================================================

@torch.no_grad()
def compute_cosine_matrix(
    meg_enc,
    txt_enc,
    windows: np.ndarray,    # (N_words, C, WIN_SIZE)
) -> np.ndarray:
    """
    Returns (N_words, V) cosine similarity matrix.
    Each row is one word window's similarity against every vocabulary word.
    """
    all_text = txt_enc.get_all().to(DEVICE)    # (V, D) normalised

    x = torch.from_numpy(windows).to(DEVICE)   # (N_words, C, WIN_SIZE)
    z = meg_enc(x)                             # (N_words, D) normalised
    sim = (z @ all_text.T).cpu().numpy()       # (N_words, V)
    return sim


# =============================================================================
#  VOCABULARY — recover word list from text encoder
# =============================================================================

def get_vocab_words(onset_dir: str) -> List[str]:
    """
    Reconstruct vocabulary from onset JSON files (same order as during training).
    This mirrors what MEGWordDataset builds.
    """
    vocab = {}
    for poem_key in ["poem1", "poem2"]:
        onset_file = os.path.join(onset_dir, f"{poem_key}_word_onsets.json")
        if not os.path.exists(onset_file):
            continue
        with open(onset_file) as f:
            word_onsets = json.load(f)
        for w in word_onsets:
            word = w["word"].strip().lower()
            if word not in vocab:
                vocab[word] = len(vocab)
    return sorted(vocab, key=vocab.get)


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def run_pipeline(
    subject:      str,
    session:      int,
    poem:         str,
    cond:         str,           # "lis" or "img"
    window_size:  str = "800ms", # "800ms" or "400ms"
    img_lis_arch: str = "CNN1D",
    img_lis_ckpt: Optional[str] = None,
    save_dir:     Optional[str] = None,
    verbose:      bool = True,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Full decoding pipeline for one trial.

    Returns
    -------
    sim_matrix  : (N_words, V) cosine similarities
    word_labels : (N_words,) true word for each row (trial onset order)
    vocab_words : (V,) vocabulary word for each column
    """
    cfg = WINDOW_CONFIGS[window_size]
    pre_ms, post_ms = cfg["pre_ms"], cfg["post_ms"]

    # ---- Load onset timestamps ----
    onset_file = os.path.join(ONSET_DIR, f"{poem}_word_onsets.json")
    if not os.path.exists(onset_file):
        raise FileNotFoundError(f"Onset file not found: {onset_file}")
    with open(onset_file) as f:
        onsets = json.load(f)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Subject : {subject}")
        print(f"  Session : {session}")
        print(f"  Poem    : {poem}")
        print(f"  Cond    : {cond}")
        print(f"  Window  : -{pre_ms}ms / +{post_ms}ms")
        print(f"  Device  : {DEVICE}")
        print(f"{'='*60}")

    # ---- Load MEG ----
    task_cond = f"{poem}{cond}"
    if verbose:
        print(f"\n  Loading MEG: {subject}/ses-{session}/task-{task_cond}...")
    data = load_trial(subject, session, task_cond)
    if verbose:
        print(f"  Trial shape: {data.shape}  ({data.shape[1]/SFREQ_DS:.1f}s)")

    n_channels = data.shape[0]

    # ---- img→lis mapping (imagined only) ----
    if cond == "img":
        if img_lis_ckpt is None:
            # Auto-resolve LOSO checkpoint for this subject
            img_lis_ckpt = os.path.join(
                IMG_LIS_DIR, f"heldout_{subject}", f"{img_lis_arch}_full.pt"
            )
            if verbose:
                print(f"  Auto-resolved img→lis ckpt: {img_lis_ckpt}")
        if not os.path.exists(img_lis_ckpt):
            raise FileNotFoundError(f"img→lis checkpoint not found: {img_lis_ckpt}")
        if verbose:
            print(f"  Applying img→lis mapping [{img_lis_arch}]...")
        data = apply_img_lis(data, img_lis_arch, img_lis_ckpt)

    # ---- Extract word windows ----
    if verbose:
        print(f"  Extracting word windows ({len(onsets)} onsets)...")
    windows, word_labels = extract_word_windows(data, onsets, pre_ms, post_ms)
    if verbose:
        print(f"  Windows: {windows.shape}  ({len(word_labels)} words)")

    # ---- Load decoder ----
    decoder_dir = DECODER_DIRS[window_size]
    if verbose:
        print(f"  Loading decoder from: {decoder_dir}")
    meg_enc, txt_enc = load_decoder(decoder_dir, n_channels, window_size)

    # ---- Cosine similarity matrix ----
    if verbose:
        print(f"  Computing cosine similarities...")
    sim_matrix = compute_cosine_matrix(meg_enc, txt_enc, windows)

    # Vocabulary (column labels)
    vocab_words = get_vocab_words(ONSET_DIR)
    # Pad vocab to match text encoder size if needed
    V_enc = txt_enc.get_all().shape[0]
    if len(vocab_words) < V_enc:
        vocab_words += [f"<unk_{i}>" for i in range(V_enc - len(vocab_words))]
    vocab_words = vocab_words[:V_enc]

    if verbose:
        print(f"\n  sim_matrix shape: {sim_matrix.shape}  "
              f"(N_words={len(word_labels)}, V={len(vocab_words)})")
        print(f"\n  Top-1 predictions per word:")
        print(f"  {'True word':20s}  {'Pred (rank of true)':20s}  {'Top-3 preds'}")
        print(f"  {'-'*70}")
        for i, true_word in enumerate(word_labels):
            row   = sim_matrix[i]
            order = np.argsort(-row)
            top3  = [vocab_words[j] for j in order[:3]]
            if true_word in vocab_words:
                true_idx  = vocab_words.index(true_word)
                true_rank = int(np.where(order == true_idx)[0][0]) + 1
            else:
                true_rank = -1
            print(f"  {true_word:20s}  rank={true_rank:<5d}               {top3}")

    # ---- Save ----
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        tag = f"{subject}_ses{session}_{poem}_{cond}_{window_size}"
        np.save(os.path.join(save_dir, f"sim_matrix_{tag}.npy"), sim_matrix)
        with open(os.path.join(save_dir, f"word_labels_{tag}.json"), "w") as f:
            json.dump({"word_labels": word_labels, "vocab_words": vocab_words}, f, indent=2)
        if verbose:
            print(f"\n  [saved] {save_dir}/sim_matrix_{tag}.npy")
            print(f"  [saved] {save_dir}/word_labels_{tag}.json")

    return sim_matrix, word_labels, vocab_words


# =============================================================================
#  CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Decode a single MEG trial → (N_words, V) cosine similarity matrix",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--subject",  required=True, help="e.g. sub-01")
    parser.add_argument("--session",  required=True, type=int, help="0–9")
    parser.add_argument("--poem",     required=True, choices=["poem1", "poem2"])
    parser.add_argument("--cond",     required=True, choices=["lis", "img"],
                        help="lis = listened  |  img = imagined")
    parser.add_argument("--window",   default="800ms", choices=["800ms", "400ms"],
                        help="Post-onset window size (default: 800ms)")
    parser.add_argument("--img_lis_arch", default="CNN1D",
                        choices=["CNN1D", "ShallowMLP", "UNet1D", "RNN", "TCN"],
                        help="img→lis mapping architecture (only used if --cond img)")
    parser.add_argument("--img_lis_ckpt", default=None,
                        help="Path to img→lis .pt checkpoint. "
                             "If omitted, auto-resolves LOSO ckpt for this subject.")
    parser.add_argument("--save_dir", default="./trial_decode_out",
                        help="Directory to save sim_matrix and word_labels (default: ./trial_decode_out)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    run_pipeline(
        subject=args.subject,
        session=args.session,
        poem=args.poem,
        cond=args.cond,
        window_size=args.window,
        img_lis_arch=args.img_lis_arch,
        img_lis_ckpt=args.img_lis_ckpt,
        save_dir=args.save_dir,
        verbose=True,
    )


if __name__ == "__main__":
    main()
