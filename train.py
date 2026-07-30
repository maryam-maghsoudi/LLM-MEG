"""
train.py
========
Training loop for the LLM-guided MEG decoder.

Data split (poem-level, critical for preventing LLM memorisation):
  train : all subjects, poem1, sessions 0-7   (~104 trials)
  val   : all subjects, poem1, sessions 8-9   (~26 trials)
  test  : all subjects, poem2, all sessions   (evaluate.py)

Only the adapter weights are updated (Option A, frozen encoder + frozen LLM).
For Option B (joint encoder training) pass --unfreeze_encoder.

Usage
-----
  python train.py                                  # defaults from config.py
  python train.py --design upfront                 # Design B baseline
  python train.py --unfreeze_encoder               # Option B
  python train.py --llm_name gpt2-medium           # larger LLM
  python train.py --out_dir out/run_01 --seed 0
"""

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    SUBJECTS, TRAIN_POEMS, TEST_POEMS, TRAIN_SESSIONS, VAL_SESSIONS,
    LLM_NAME, SEQUENCE_DESIGN,
    BATCH_SIZE, LR, WARMUP_STEPS, N_EPOCHS, PATIENCE, WEIGHT_DECAY, SEED,
    OUT_DIR,
)
from dataset import PoemTrialDataset, collate_trials
from model import build_model


# =============================================================================
#  HELPERS
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        if cap[0] >= 7:
            return torch.device("cuda")
        print(f"  GPU compute capability {cap[0]}.{cap[1]} < 7.0 — using CPU")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then constant LR for the remainder of training."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "meg_windows":    batch["meg_windows"].to(device),
        "valid_mask":     batch["valid_mask"].to(device),
        "word_token_ids": batch["word_token_ids"],   # stays as list-of-lists
        "word_texts":     batch["word_texts"],
        "meta":           batch["meta"],
    }


# =============================================================================
#  EPOCH RUNNERS
# =============================================================================

def run_train_epoch(model, loader, optimizer, scheduler, device, clip_norm: float = 1.0):
    model.train()
    losses = []
    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad()
        out = model(
            meg_windows    = batch["meg_windows"],
            valid_mask     = batch["valid_mask"],
            word_token_ids = batch["word_token_ids"],
        )
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), clip_norm)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def run_val_epoch(model, loader, device):
    model.eval()
    losses = []
    for batch in loader:
        batch = _move_batch(batch, device)
        out = model(
            meg_windows    = batch["meg_windows"],
            valid_mask     = batch["valid_mask"],
            word_token_ids = batch["word_token_ids"],
        )
        losses.append(out.loss.item())
    return float(np.mean(losses))


# =============================================================================
#  PLOT
# =============================================================================

def save_training_curve(history: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(history["train"]) + 1)
    ax.plot(epochs, history["train"], label="train")
    ax.plot(epochs, history["val"],   label="val", linestyle="--")
    if history.get("best_epoch"):
        ax.axvline(history["best_epoch"], color="gray", linestyle=":", alpha=0.7,
                   label=f"best (epoch {history['best_epoch']})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("LLM decoder training curve")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "training_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# =============================================================================
#  MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train LLM-guided MEG decoder")
    p.add_argument("--llm_name",         default=LLM_NAME)
    p.add_argument("--design",           default=SEQUENCE_DESIGN,
                   choices=["interleaved", "upfront"])
    p.add_argument("--unfreeze_encoder", action="store_true",
                   help="Option B: train MEG encoder jointly with adapter")
    p.add_argument("--batch_size",       type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",               type=float, default=LR)
    p.add_argument("--warmup_steps",     type=int,   default=WARMUP_STEPS)
    p.add_argument("--n_epochs",         type=int,   default=N_EPOCHS)
    p.add_argument("--patience",         type=int,   default=PATIENCE)
    p.add_argument("--weight_decay",     type=float, default=WEIGHT_DECAY)
    p.add_argument("--seed",             type=int,   default=SEED)
    p.add_argument("--out_dir",          type=Path,  default=OUT_DIR / "train")
    p.add_argument("--cache_data",       action="store_true",
                   help="Pre-load all MEG into RAM before training")
    p.add_argument("--num_workers",      type=int,   default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    set_seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  LLM decoder training")
    print(f"  design={args.design!r}  llm={args.llm_name!r}  device={device}")
    print(f"  lr={args.lr}  batch={args.batch_size}  epochs={args.n_epochs}")
    print(f"  out_dir={args.out_dir}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Building datasets...")
    train_ds = PoemTrialDataset(
        subjects  = SUBJECTS,
        poems     = TRAIN_POEMS,
        sessions  = TRAIN_SESSIONS,
        condition = "lis",
        llm_name  = args.llm_name,
        cache     = args.cache_data,
    )
    val_ds = PoemTrialDataset(
        subjects  = SUBJECTS,
        poems     = TRAIN_POEMS,
        sessions  = VAL_SESSIONS,
        condition = "lis",
        llm_name  = args.llm_name,
        cache     = args.cache_data,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_trials, num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_trials, num_workers=args.num_workers,
    )
    print(f"  train: {len(train_ds)} trials ({len(train_loader)} batches)")
    print(f"  val  : {len(val_ds)} trials ({len(val_loader)} batches)\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        device          = device,
        llm_name        = args.llm_name,
        freeze_encoder  = not args.unfreeze_encoder,
        sequence_design = args.design,
    )

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = args.n_epochs * len(train_loader)
    scheduler   = make_lr_scheduler(optimizer, args.warmup_steps, total_steps)

    # ── Training loop ─────────────────────────────────────────────────────────
    history    = {"train": [], "val": [], "best_epoch": None}
    best_val   = math.inf
    no_improve = 0
    t0         = time.time()

    # Sanity-check that frozen components stayed in eval mode after model.train()
    model.train()
    print(f"  Adapter training mode     : {model.adapter.training}")       # should be True
    print(f"  MEG encoder training mode : {model.meg_encoder.training}")   # should be False if frozen
    print(f"  LLM training mode         : {model.frozen_llm.training}\n")  # should be False

    print(f"{'Epoch':>6}  {'Train loss':>10}  {'Val loss':>10}  {'Perplexity':>10}  {'LR':>10}  {'No-imp':>6}")
    print("-" * 66)

    for epoch in range(1, args.n_epochs + 1):
        train_loss = run_train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        val_loss = run_val_epoch(model, val_loader, device)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        current_lr = scheduler.get_last_lr()[0]
        ppl        = math.exp(min(val_loss, 20))   # cap to avoid overflow display

        print(
            f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}"
            f"  {ppl:>10.2f}  {current_lr:>10.2e}  {no_improve:>6}"
        )

        if val_loss < best_val - 1e-6:
            best_val             = val_loss
            no_improve           = 0
            history["best_epoch"] = epoch
            model.save_adapter(args.out_dir / "best_adapter.pt")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}  (best val={best_val:.4f})")
                break

    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed/60:.1f} min")
    print(f"  Best val loss: {best_val:.4f} (epoch {history['best_epoch']})")

    # ── Save final artefacts ──────────────────────────────────────────────────
    model.save_adapter(args.out_dir / "final_adapter.pt")

    run_cfg = {
        "llm_name":         args.llm_name,
        "design":           args.design,
        "unfreeze_encoder": args.unfreeze_encoder,
        "batch_size":       args.batch_size,
        "lr":               args.lr,
        "warmup_steps":     args.warmup_steps,
        "n_epochs":         args.n_epochs,
        "patience":         args.patience,
        "weight_decay":     args.weight_decay,
        "seed":             args.seed,
        "train_subjects":   SUBJECTS,
        "train_poems":      TRAIN_POEMS,
        "train_sessions":   TRAIN_SESSIONS,
        "val_sessions":     VAL_SESSIONS,
        "best_val_loss":    best_val,
        "best_epoch":       history["best_epoch"],
        "elapsed_s":        round(elapsed),
        "shuffled_meg":     False,   # set True in shuffle-ablation runs
    }
    with open(args.out_dir / "run_config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

    save_training_curve(history, args.out_dir)
    print(f"\n  Outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
