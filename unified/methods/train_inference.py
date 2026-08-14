"""
train_inference.py — Method 1: Contrastive MEG encoder (InfoNCE vs BERT).

Training
--------
1. Compute BERT contextual embeddings for every word position in both poems
   (run once, cached in memory — BERT processes the full poem as context).
2. Train MEGEncoder + BERTTextProjection with InfoNCE loss.
3. Save best MEGEncoder weights (val InfoNCE, early stopping).

The LLM alpha-fusion step happens at inference time in predict.py, not here.

Output files
------------
out_dir/
    meg_encoder_best.pt       MEGEncoder state dict at best val loss
    meg_encoder_final.pt      MEGEncoder state dict at last epoch
    bert_proj_best.pt         BERTTextProjection state dict (for diagnostics)
    history.json              per-epoch train/val loss
    run_config.json           hyperparameters
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..data.base_dataset import MEGWordDataset, ONSET_DIR
from .models import MEGEncoder, BERTTextProjection, load_bert_hiddens

# ---------------------------------------------------------------------------
#  Hyperparameters
# ---------------------------------------------------------------------------
LR       = 3e-4
EPOCHS   = 60
BS       = 64
PATIENCE = 10
TEMP     = 0.07
DROPOUT  = 0.3


# ---------------------------------------------------------------------------
#  BERT-augmented dataset wrapper
# ---------------------------------------------------------------------------

class BERTAugmentedDataset(Dataset):
    """
    Wraps MEGWordDataset, attaching the precomputed BERT hidden state for
    each word occurrence. BERT hiddens depend only on (poem, word_pos) —
    the same vector is used for all subjects and sessions at that position.
    """

    def __init__(
        self,
        base:         MEGWordDataset,
        bert_hiddens: Dict[str, torch.Tensor],   # {poem: (N_words, 768)}
    ):
        self._base   = base
        self._hiddens = bert_hiddens

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self._base[idx]
        h    = self._hiddens[item["poem"]][item["word_pos"]]   # (768,)
        return item["meg_window"], h


# ---------------------------------------------------------------------------
#  InfoNCE loss
# ---------------------------------------------------------------------------

def info_nce(z_meg: torch.Tensor, z_text: torch.Tensor,
             temperature: float = TEMP) -> torch.Tensor:
    """Single-direction InfoNCE (MEG as query). (N, d) × (N, d) → scalar."""
    N   = z_meg.shape[0]
    sim = z_meg @ z_text.T / temperature         # (N, N)
    return F.cross_entropy(sim, torch.arange(N, device=z_meg.device))


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------

def train(
    splits:       Dict,
    out_dir:      Path,
    device:       torch.device,
    bert_name:    str   = "bert-base-uncased",
    bert_layer:   int   = -1,
    lr:           float = LR,
    epochs:       int   = EPOCHS,
    batch_size:   int   = BS,
    patience:     int   = PATIENCE,
    temperature:  float = TEMP,
    dropout:      float = DROPOUT,
    control:      str   = "none",
    augment:      bool  = True,
) -> Dict:
    """
    Full Method 1 training run.

    Parameters
    ----------
    splits   : output of any make_*_splits() function
    out_dir  : directory for checkpoints and logs
    device   : torch.device
    control  : 'none' | 'zero' | 'shuffle_time'  (applies to base datasets)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── BERT hiddens (computed once, shared across train/val) ────────────────
    bert_device = "cpu"   # BERT inference on CPU to save GPU memory during training
    bert_hiddens = load_bert_hiddens(ONSET_DIR, bert_name, device=bert_device,
                                     layer=bert_layer)

    # ── Datasets ─────────────────────────────────────────────────────────────
    from ..data.controls import make_control

    print("\nBuilding datasets ...")
    t0 = time.time()
    base_train = MEGWordDataset(
        splits["train"]["trials"],
        word_filter=splits["train"]["word_filter"],
        augment=augment,
    )
    base_val = MEGWordDataset(
        splits["val"]["trials"],
        word_filter=splits["val"]["word_filter"],
        augment=False,
    )
    base_train = make_control(base_train, control, augment=augment)
    base_val   = make_control(base_val,   control, augment=False)

    ds_train = BERTAugmentedDataset(base_train, bert_hiddens)
    ds_val   = BERTAugmentedDataset(base_val,   bert_hiddens)
    print(f"  built in {time.time() - t0:.1f}s")

    if len(ds_train) == 0:
        raise RuntimeError("Training set is empty — check MEG files and trial list.")

    pin = device.type == "cuda"
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=pin)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=pin)

    # ── Models ───────────────────────────────────────────────────────────────
    meg_enc   = MEGEncoder(dropout=dropout).to(device)
    bert_proj = BERTTextProjection(dropout=dropout).to(device)

    n_enc  = sum(p.numel() for p in meg_enc.parameters())
    n_proj = sum(p.numel() for p in bert_proj.parameters())
    print(f"\nMEGEncoder         {n_enc:,} params")
    print(f"BERTTextProjection {n_proj:,} params")

    # ── Optimiser ────────────────────────────────────────────────────────────
    opt   = torch.optim.AdamW(
        list(meg_enc.parameters()) + list(bert_proj.parameters()), lr=lr
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history   = {"train_loss": [], "val_loss": []}
    best_val  = float("inf")
    wait      = 0
    best_state: Optional[Dict] = None

    print(f"\n{'='*60}")
    print(f"Method 1 training  bert={bert_name}  control={control}")
    print(f"  lr={lr}  bs={batch_size}  epochs={epochs}  patience={patience}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        # ── train ────────────────────────────────────────────────────────────
        meg_enc.train()
        bert_proj.train()
        t_losses = []
        for x, h in dl_train:
            x = x.to(device)
            h = h.to(device)
            z = meg_enc(x)
            t = bert_proj(h)
            loss = info_nce(z, t, temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            t_losses.append(loss.item())
        sched.step()

        # ── val ──────────────────────────────────────────────────────────────
        meg_enc.eval()
        bert_proj.eval()
        v_losses = []
        with torch.no_grad():
            for x, h in dl_val:
                x = x.to(device)
                h = h.to(device)
                z = meg_enc(x)
                t = bert_proj(h)
                v_losses.append(info_nce(z, t, temperature).item())

        t_loss = float(np.mean(t_losses))
        v_loss = float(np.mean(v_losses))
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        improved = v_loss < best_val
        if improved:
            best_val   = v_loss
            wait       = 0
            best_state = {
                "meg_encoder": {k: v.cpu() for k, v in meg_enc.state_dict().items()},
                "bert_proj":   {k: v.cpu() for k, v in bert_proj.state_dict().items()},
                "epoch":       epoch,
                "val_loss":    v_loss,
            }
        else:
            wait += 1

        marker = " ✓" if improved else f"  (wait {wait}/{patience})"
        print(f"  epoch {epoch:3d}/{epochs}  "
              f"train={t_loss:.4f}  val={v_loss:.4f}{marker}")

        if wait >= patience:
            print(f"  → early stop at epoch {epoch}")
            break

    # ── save ─────────────────────────────────────────────────────────────────
    meg_enc.load_state_dict(best_state["meg_encoder"])
    torch.save(best_state["meg_encoder"], out_dir / "meg_encoder_best.pt")
    torch.save({k: v.cpu() for k, v in meg_enc.state_dict().items()},
               out_dir / "meg_encoder_final.pt")
    torch.save(best_state["bert_proj"], out_dir / "bert_proj_best.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    metrics = {
        "best_val_loss": best_val,
        "best_epoch":    best_state["epoch"],
        "n_epochs_run":  len(history["train_loss"]),
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nBest val={best_val:.4f} at epoch {best_state['epoch']}")
    print(f"Checkpoints → {out_dir}")
    return metrics
