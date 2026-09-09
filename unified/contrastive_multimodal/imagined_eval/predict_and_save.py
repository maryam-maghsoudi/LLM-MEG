"""
imagined_eval/predict_and_save.py
=================================
STAGE A of the imagined -> predicted-listened -> contrastive-decoder pipeline.

Runs a trained img->lis mapping model (from the no-flash-removal LOSO
benchmark) over each held-out subject's IMAGINED MEG, and saves the
predicted-listened signal as .npy files in the EXACT format the contrastive
Stage 1 decoder reads (icaed_Sai layout + naming), so eval_imagined.py can
feed them straight into MEGContinuousTrialDataset's fast .npy path with zero
changes to new_dataset.py.

Why the no-flash-removal mapping models
---------------------------------------
The contrastive decoder was trained on FULL (no-flash-removed) listened MEG
(icaed_Sai, shape (155, 2700), 100 Hz, per-channel z-scored). The mapping
models under
    benchmark/no_flash_removal/loso_out/models/heldout_{subj}/{key}.pt
are likewise trained on the full time series, so their predicted-listened
output is already in the same time base and preprocessing as the decoder's
training data. No flash-index reconstruction is needed.

Format contract (must match icaed_Sai / new_dataset.py._load_meg_trial)
----------------------------------------------------------------------
  path : {out_root}/{subject}/ses-{s}/meg/{subject}_sess-{s}_task-{poem}lis.npy
  array: (N_CHANNELS=155, T=2700) float32, per-channel z-scored

We deliberately save under the "...lis.npy" name (not "img"): the file holds
predicted-LISTENED signal, and the contrastive dataset requests condition
"lis" by default — so the predicted signal transparently stands in for the
real listened trial the decoder expects.

Preprocessing (identical to no_flash_removal/benchmark_loso.py's img path)
-------------------------------------------------------------------------
  mean over epochs -> downsample x10 (scipy.resample) -> per-channel z-score
  -> model forward -> per-channel z-score of the prediction (so the saved
     array matches the exactly-z-scored distribution the encoder trained on).

Usage
-----
    python predict_and_save.py                          # RNN_full, all 13 subjects
    python predict_and_save.py --mapping_key CNN1D_full
    python predict_and_save.py --subjects sub-01 sub-03
    python predict_and_save.py --mapping_key LinearLag_full   # ridge baseline

Output goes to  ./predicted_npy/{mapping_key}/  by default.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

import mne
mne.set_log_level("ERROR")
from scipy.signal import resample

# ---------------------------------------------------------------------------
#  Paths / constants
# ---------------------------------------------------------------------------
ICAED_DIR   = "/fs/nexus-projects/brain_project/maryam_meg_dataset/icaed"
BENCH_DIR   = "/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/benchmark/no_flash_removal"
LOSO_MODELS = os.path.join(BENCH_DIR, "loso_out", "models")

SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]
POEMS      = ["poem1", "poem2"]
N_SESSIONS = 10
DS_FACTOR  = 10

# Import the mapping architectures + ridge helpers straight from the benchmark
# that trained them (same pattern contrastive_word_meg.py already uses). This
# guarantees the module structure of the checkpoints matches exactly.
sys.path.insert(0, BENCH_DIR)
from benchmark_loso import (            # noqa: E402
    make_models,
    build_lagged_features, predict_ridge, ms_to_samples,
    LAG_BEFORE_MS, LAG_AFTER_MS,
)


# ---------------------------------------------------------------------------
#  Preprocessing (mirrors no_flash_removal/benchmark_loso.py exactly)
# ---------------------------------------------------------------------------

def zscore_ch(x: np.ndarray) -> np.ndarray:
    """x: (C, T) — per-channel z-score over time."""
    mu = x.mean(axis=1, keepdims=True)
    sd = np.maximum(x.std(axis=1, keepdims=True), 1e-12)
    return (x - mu) / sd


def load_img_trial(subject: str, poem: str, session: int) -> "np.ndarray | None":
    """
    Load one imagined trial -> (C, T_ds) float32, per-channel z-scored, or
    None if the .fif is missing. Matches benchmark_loso.py's load path
    (mean over epochs -> resample x10) followed by get_xy_full's z-score.
    """
    fpath = os.path.join(
        ICAED_DIR, subject, f"ses-{session}", "meg",
        f"{subject}_sess-{session}_task-{poem}img_meg-epo.fif",
    )
    if not os.path.exists(fpath):
        return None
    epochs  = mne.read_epochs(fpath, preload=True)
    data    = epochs.get_data().mean(axis=0)                    # (C, T_raw)
    new_T   = data.shape[1] // DS_FACTOR
    data_ds = resample(data, new_T, axis=1).astype(np.float32)  # (C, T_ds)
    return zscore_ch(data_ds).astype(np.float32)


# ---------------------------------------------------------------------------
#  Mapping model loading + inference
# ---------------------------------------------------------------------------

def load_neural_mapping(arch: str, ckpt_path: str, C: int, device) -> torch.nn.Module:
    model = make_models(C)[arch]
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def predict_neural(model: torch.nn.Module, x: np.ndarray, device) -> np.ndarray:
    """x: (C, T) -> y_hat: (C, T) via a single full-trial forward pass."""
    with torch.no_grad():
        xt   = torch.from_numpy(x).unsqueeze(0).to(device)   # (1, C, T)
        yhat = model(xt).squeeze(0).cpu().numpy()            # (C, T)
    return yhat


def predict_linearlag(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Ridge baseline: x (C, T) -> y_hat (C, T)."""
    lb = ms_to_samples(LAG_BEFORE_MS)
    la = ms_to_samples(LAG_AFTER_MS)
    return predict_ridge(x[None, ...], W, lb, la)[0]          # (C, T)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_ridge = args.mapping_key.startswith("LinearLag")
    arch     = args.mapping_key.rsplit("_", 1)[0]   # "RNN_full" -> "RNN"

    out_root = Path(args.out_root) / args.mapping_key
    print(f"=== predict_and_save  mapping_key={args.mapping_key}  device={device} ===")
    print(f"    out_root = {out_root}")

    subjects = args.subjects if args.subjects else SUBJECTS
    n_saved = n_missing = 0

    for subj in subjects:
        # loading the mapping model
        ckpt_dir = os.path.join(LOSO_MODELS, f"heldout_{subj}")
        if is_ridge:
            ckpt_path = os.path.join(ckpt_dir, "LinearLag_W.npy")
        else:
            ckpt_path = os.path.join(ckpt_dir, f"{args.mapping_key}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [skip] {subj}: checkpoint not found ({ckpt_path})")
            continue

        W = np.load(ckpt_path) if is_ridge else None
        model = None   # lazy-built once C is known from the first trial

        for poem in POEMS:
            for session in range(N_SESSIONS):
                x = load_img_trial(subj, poem, session)
                if x is None:
                    n_missing += 1
                    continue

                if is_ridge:
                    yhat = predict_linearlag(W, x)
                else:
                    if model is None:
                        model = load_neural_mapping(arch, ckpt_path, x.shape[0], device)
                    yhat = predict_neural(model, x, device)

                # Re-z-score so the saved array matches the exactly-z-scored
                # distribution the contrastive encoder was trained on.
                yhat = zscore_ch(yhat).astype(np.float32)

                out_dir = out_root / subj / f"ses-{session}" / "meg"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{subj}_sess-{session}_task-{poem}lis.npy"
                np.save(str(out_path), yhat)
                n_saved += 1

        print(f"  [done] {subj}")
        if not is_ridge and device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nSaved {n_saved} predicted-listened trials  "
          f"({n_missing} imagined trials missing)  -> {out_root}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Map imagined MEG -> predicted listened, save as .npy.")
    p.add_argument("--mapping_key", type=str, default="RNN_full",
                   help="Mapping model key, e.g. RNN_full, CNN1D_full, TCN_full, "
                        "UNet1D_full, ShallowMLP_full, or LinearLag_full.")
    p.add_argument("--subjects", type=str, nargs="*", default=None,
                   help="Subset of subjects (default: all 13).")
    p.add_argument("--out_root", type=str,
                   default=str(Path(__file__).parent / "predicted_npy"),
                   help="Root dir for predicted .npy (a per-mapping_key subfolder is created).")
    return p


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
