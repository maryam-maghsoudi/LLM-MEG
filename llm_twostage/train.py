"""
train.py — Steps 4, 6, 7, 8, 9
=================================
Two-stage LOSO training script.

Stage 1: Contrastive alignment
    MEGEncoder + LLMTextProjection trained with InfoNCE against precomputed
    LLM middle-layer hidden states (hmid_t).  Early stopping on val InfoNCE.

Stage 2: KL next-word distillation
    Frozen Stage 1 MEGEncoder + trainable GRUHead. KL(p_t || q_t) loss where
    p_t is the teacher LLM's restricted-vocab distribution (from cache) and
    q_t = lm_head(GRUHead(z_1..z_t)).  Early stopping on val KL.

Usage
-----
python train.py --heldout sub-01
python train.py --heldout sub-01 --llm_name HuggingFaceTB/SmolLM2-360M
python train.py --heldout sub-01 --loss nt_xent  # symmetric loss instead
python train.py --heldout sub-01 --skip_stage2   # Stage 1 only

Results saved to:
    llm_twostage/out/<model_tag>/<heldout>/
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# import pdb

# ── local imports ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))                 # llm_twostage/ first (avoids shadowing by parent dataset.py)
sys.path.insert(1, str(_HERE.parent))          # llm_decoder/ (not needed now but harmless)

from dataset import (
    MEGWordDatasetLLM,
    MEGSequenceDataset,
    collate_sequences,
    load_vocab_info,
    make_loso_splits,
    model_tag,
    POEM_KEYS,
    SUBJECTS,
)
from models import MEGEncoder, LLMTextProjection, GRUHead, load_lm_head

# ===========================================================================
#  Per-model config  (d_model, default hmid_layer)
# ===========================================================================

MODEL_CONFIGS = {
    "HuggingFaceTB/SmolLM2-360M": {"d_model": 960,  "hmid_layer": 11},
    "HuggingFaceTB/SmolLM2-1.7B": {"d_model": 2048, "hmid_layer": 8},
    "gpt2":                        {"d_model": 768,  "hmid_layer": 4},
    "Qwen/Qwen2-0.5B":             {"d_model": 896,  "hmid_layer": 8},
}

# ===========================================================================
#  Training hyper-parameters  (can be overridden via argparse)
# ===========================================================================

S1_EPOCHS  = 60
S1_LR      = 3e-4
S1_BS      = 64
S1_PATIENCE = 10

S2_EPOCHS  = 60
S2_LR      = 1e-4
S2_BS      = 4    # full poem trials per batch
S2_PATIENCE = 10

TEMPERATURE = 0.07
GRU_HIDDEN  = 256
MEG_EMB     = 128


# ===========================================================================
#  Losses
# ===========================================================================

def info_nce_loss(z_meg, z_text, temperature=TEMPERATURE):
    """InfoNCE, single direction (MEG as query).  (N, d) × (N, d) → scalar"""
    N   = z_meg.shape[0]
    sim = z_meg @ z_text.T / temperature     # (N, N)
    return F.cross_entropy(sim, torch.arange(N, device=z_meg.device))


def nt_xent_loss_symmetric(z_meg, z_text, temperature=TEMPERATURE):
    """Symmetric NT-Xent (both directions averaged).  Available behind --loss flag."""
    N    = z_meg.shape[0]
    sim  = z_meg @ z_text.T / temperature
    labs = torch.arange(N, device=z_meg.device)
    return (F.cross_entropy(sim, labs) + F.cross_entropy(sim.T, labs)) / 2.0


def kl_loss(q_logits_r, p_logits_r, valid_mask):
    """
    KL(p_t || q_t) averaged over valid positions 0..T-2.

    q_logits_r : (B, T, R) — student logits (not softmaxed)
    p_logits_r : (B, T, R) — teacher logits (from cache)
    valid_mask  : (B, T)   — True where position is a real word (not padding)
                             AND has a valid MEG window

    Mask condition: valid_mask[:, 1:]  — does word t+1 exist?
    This is the correct condition because at position t we predict word t+1.
    valid_mask[:, :-1] would ask "does word t exist?" — wrong direction.
    Critically, at the last real word of a trial (position N-1), valid_mask[N]
    is False (padding), so that position is excluded. With valid_mask[:, :-1]
    valid_mask[N-1] would be True and we'd score against p_t_r[N-1] which
    predicts the non-existent word N — a meaningless target.
    """
    q = F.log_softmax(q_logits_r[:, :-1, :], dim=-1)    # (B, T-1, R)
    p = F.softmax(   p_logits_r[:, :-1, :], dim=-1)     # (B, T-1, R)
    mask = valid_mask[:, 1:].float()                     # (B, T-1): is word t+1 real?

    # element-wise KL then sum over vocab, then mask-average over time+batch
    kl_per = F.kl_div(q, p, reduction="none").sum(-1)   # (B, T-1)
    denom  = mask.sum().clamp(min=1)
    return (kl_per * mask).sum() / denom


def shuffle_time_per_trial(z: torch.Tensor, p_t_r: torch.Tensor) -> torch.Tensor:
    """
    Permute MEG positions independently within each trial while keeping the
    target sequence p_1..p_T in its true order.

    At step t the GRU sees some other word's MEG from the same trial instead
    of word t's MEG. If the model just runs a step-counter that reproduces poem
    statistics regardless of content, performance is unchanged. If it actually
    uses the content of z_t for its position, performance degrades.

    z     : (B, N, D)
    p_t_r : (B, N, R) — infers each trial's real (non-padded) length via
            non-zero rows (padded positions are all-zero from collate_sequences)
    """
    B, N, D = z.shape
    z_out = z.clone()
    valid_lengths = (p_t_r.abs().sum(-1) > 0).sum(dim=1)  # (B,) real word counts
    for b in range(B):
        L = int(valid_lengths[b].item())
        if L > 1:
            perm = torch.randperm(L, device=z.device)
            z_out[b, :L] = z[b, perm]
    return z_out


def _within_poem_perm(B: int, metas: list, device) -> torch.Tensor:
    """
    Random permutation restricted to within-poem pairs.
    Trials sharing the same poem are shuffled among themselves; others stay fixed.
    Using within-poem shuffle (rather than across-poem) avoids confounding
    sequence-length and vocabulary-distribution differences between poems.
    Draw a fresh permutation every call — never reuse a fixed pairing.
    """
    perm = list(range(B))
    poem_groups: dict = {}
    for i, m in enumerate(metas):
        poem_groups.setdefault(m["poem"], []).append(i)
    for indices in poem_groups.values():
        if len(indices) > 1:
            shuffled = indices[:]
            random.shuffle(shuffled)
            for orig, new in zip(indices, shuffled):
                perm[orig] = new
    return torch.tensor(perm, device=device)


# ===========================================================================
#  Stage 1 training loop
# ===========================================================================

def train_stage1(
    meg_enc: MEGEncoder,
    text_proj: LLMTextProjection,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device,
    loss_fn: str = "info_nce",
    lr:      float = S1_LR,
    n_epochs: int  = S1_EPOCHS,
    patience: int  = S1_PATIENCE,
    out_dir:  Path = None,
) -> dict:
    """
    Returns history dict and saves best checkpoint to out_dir/stage1_best.pt.
    """
    loss_func = info_nce_loss if loss_fn == "info_nce" else nt_xent_loss_symmetric
    opt  = torch.optim.AdamW(
        list(meg_enc.parameters()) + list(text_proj.parameters()), lr=lr
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    wait      = 0
    best_state = None

    print(f"\n{'='*60}")
    print(f"STAGE 1  loss={loss_fn}  lr={lr}  bs={train_loader.batch_size}")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        # ── train ────────────────────────────────────────────────────────
        meg_enc.train()
        text_proj.train()
        t_losses = []
        for x, h in train_loader:
            x = x.to(device)    #torch.Size([64, 155, 100])
            h = h.to(device)    #torch.Size([64, 960]): 960 is the d-model of HuggingFace/SmolLM2-360M
            z = meg_enc(x)      #torch.Size([64, 128])
            t = text_proj(h)    #torch.Size([64, 128])
            loss = loss_func(z, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            t_losses.append(loss.item())
        sched.step()

        # ── val ──────────────────────────────────────────────────────────
        meg_enc.eval()
        text_proj.eval()
        v_losses = []
        with torch.no_grad():
            for x, h in val_loader:
                x = x.to(device)
                h = h.to(device)
                z = meg_enc(x)
                t = text_proj(h)
                v_losses.append(loss_func(z, t).item())

        t_loss = float(np.mean(t_losses))
        v_loss = float(np.mean(v_losses))
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        improved = v_loss < best_val
        if improved:
            best_val  = v_loss
            wait      = 0
            best_state = {
                "meg_encoder": {k: v.cpu() for k, v in meg_enc.state_dict().items()},
                "text_proj":   {k: v.cpu() for k, v in text_proj.state_dict().items()},
                "epoch":       epoch,
                "val_loss":    v_loss,
            }
        else:
            wait += 1

        marker = " ✓" if improved else f"  (wait {wait}/{patience})"
        print(
            f"  epoch {epoch:3d}/{n_epochs}  "
            f"train={t_loss:.4f}  val={v_loss:.4f}{marker}"
        )

        if wait >= patience:
            print(f"  → early stop at epoch {epoch} (best val={best_val:.4f})")
            break

    if out_dir is not None and best_state is not None:
        torch.save(best_state, out_dir / "stage1_best.pt")
        print(f"  Stage 1 best saved → {out_dir / 'stage1_best.pt'}")

    return {"history": history, "best_val": best_val,
            "best_epoch": best_state["epoch"] if best_state else n_epochs,
            "best_state": best_state}


# ===========================================================================
#  Stage 2 training loop
# ===========================================================================

def train_stage2(
    meg_enc:   MEGEncoder,
    gru_head:  GRUHead,
    lm_head:   torch.nn.Module,
    r_ids:     torch.Tensor,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device,
    lr:        float = S2_LR,
    n_epochs:  int   = S2_EPOCHS,
    patience:  int   = S2_PATIENCE,
    out_dir:   Path  = None,
    z_control: str   = "none",
) -> dict:
    """
    MEGEncoder is frozen throughout (set by caller before this function).
    Trains GRUHead only.  Saves best checkpoint to out_dir/stage2_best.pt.

    z_control: 'none' = real MEG (default)
               'zero' = replace z_t with zeros
               'shuffle' = within-poem random trial permutation each batch
    """
    opt   = torch.optim.AdamW(gru_head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    wait      = 0
    best_state = None

    print(f"\n{'='*60}")
    print(f"STAGE 2  KL distillation  lr={lr}  bs={train_loader.batch_size}")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        # ── train ────────────────────────────────────────────────────────
        gru_head.train()
        t_losses = []
        for batch in train_loader:
            meg_seq    = batch["meg_windows"].to(device)     # (B, N, C, T)
            p_t_r      = batch["p_t_restricted"].to(device)  # (B, N, R)
            valid_mask = batch["valid_mask"].to(device)       # (B, N)

            B, N, C, T = meg_seq.shape
            # Encode every word window with the frozen encoder
            with torch.no_grad():
                z = meg_enc(meg_seq.view(B * N, C, T))
            z = z.view(B, N, -1)                             # (B, N, emb)
            if z_control == "zero":
                z = torch.zeros_like(z)
            elif z_control == "shuffle":
                z = z[_within_poem_perm(B, batch["meta"], device)]
            elif z_control == "shuffle_time":
                z = shuffle_time_per_trial(z, p_t_r)

            y        = gru_head(z)                           # (B, N, d_model)
            all_logits = lm_head(y)                          # (B, N, V)
            q_logits_r = all_logits[:, :, r_ids]            # (B, N, R)

            loss = kl_loss(q_logits_r, p_t_r, valid_mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            t_losses.append(loss.item())
        sched.step()

        # ── val ──────────────────────────────────────────────────────────
        gru_head.eval()
        v_losses = []
        with torch.no_grad():
            for batch in val_loader:
                meg_seq    = batch["meg_windows"].to(device)
                p_t_r      = batch["p_t_restricted"].to(device)
                valid_mask = batch["valid_mask"].to(device)

                B, N, C, T = meg_seq.shape
                z  = meg_enc(meg_seq.view(B * N, C, T)).view(B, N, -1)
                if z_control == "zero":
                    z = torch.zeros_like(z)
                elif z_control == "shuffle":
                    z = z[_within_poem_perm(B, batch["meta"], device)]
                elif z_control == "shuffle_time":
                    z = shuffle_time_per_trial(z, p_t_r)
                y  = gru_head(z)
                q  = lm_head(y)[:, :, r_ids]
                v_losses.append(kl_loss(q, p_t_r, valid_mask).item())

        t_loss = float(np.mean(t_losses))
        v_loss = float(np.mean(v_losses))
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        improved = v_loss < best_val
        if improved:
            best_val  = v_loss
            wait      = 0
            best_state = {
                "gru_head": {k: v.cpu() for k, v in gru_head.state_dict().items()},
                "epoch":    epoch,
                "val_loss": v_loss,
            }
        else:
            wait += 1

        marker = " ✓" if improved else f"  (wait {wait}/{patience})"
        print(
            f"  epoch {epoch:3d}/{n_epochs}  "
            f"train={t_loss:.4f}  val={v_loss:.4f}{marker}"
        )

        if wait >= patience:
            print(f"  → early stop at epoch {epoch} (best val={best_val:.4f})")
            break

    if out_dir is not None and best_state is not None:
        torch.save(best_state, out_dir / "stage2_best.pt")
        print(f"  Stage 2 best saved → {out_dir / 'stage2_best.pt'}")

    return {"history": history, "best_val": best_val,
            "best_epoch": best_state["epoch"] if best_state else n_epochs}


# ===========================================================================
#  Curve plotting
# ===========================================================================

def plot_curves(s1_hist, s2_hist, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, hist, title in zip(
        axes,
        [s1_hist, s2_hist],
        ["Stage 1 — InfoNCE loss", "Stage 2 — KL loss"],
    ):
        ax.plot(hist["train_loss"], label="train")
        ax.plot(hist["val_loss"],   label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"Training curves → {out_path}")


# ===========================================================================
#  Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Two-stage MEG decoder training (LOSO)")
    p.add_argument("--heldout",    required=True,
                   help="Held-out subject for LOSO, e.g. sub-01")
    p.add_argument("--llm_name",   default="HuggingFaceTB/SmolLM2-360M",
                   choices=list(MODEL_CONFIGS.keys()),
                   help="LLM whose cache / hidden states to use")
    p.add_argument("--hmid_layer", type=int, default=None,
                   help="Layer index for Stage 1 targets (default: per-model best)")
    p.add_argument("--loss",       default="info_nce",
                   choices=["info_nce", "nt_xent"],
                   help="Stage 1 loss: InfoNCE (single-dir) or NT-Xent (symmetric)")
    p.add_argument("--s1_epochs",  type=int, default=S1_EPOCHS)
    p.add_argument("--s2_epochs",  type=int, default=S2_EPOCHS)
    p.add_argument("--s1_lr",      type=float, default=S1_LR)
    p.add_argument("--s2_lr",      type=float, default=S2_LR)
    p.add_argument("--s1_bs",      type=int, default=S1_BS)
    p.add_argument("--s2_bs",      type=int, default=S2_BS)
    p.add_argument("--patience",   type=int, default=None,
                   help="Override both S1 and S2 patience")
    p.add_argument("--gru_hidden", type=int, default=GRU_HIDDEN)
    p.add_argument("--skip_stage2", action="store_true",
                   help="Run Stage 1 only (no KL distillation)")
    p.add_argument("--out_root",   default=str(_HERE / "out"),
                   help="Root directory for outputs (default: llm_twostage/out)")
    p.add_argument("--device",     default=None,
                   help="cuda / cpu (default: cuda if available)")
    p.add_argument("--z_control",  default="none",
                   choices=["none", "zero", "shuffle", "shuffle_time"],
                   help="Stage 2 z_t corruption for ablation retraining: "
                        "'zero' replaces z_t with zeros; "
                        "'shuffle' does within-poem trial permutation each batch")
    p.add_argument("--load_stage1", default=None,
                   help="Path to existing stage1_best.pt; skips Stage 1 training entirely")
    return p.parse_args()


def _resolve_device(requested: str) -> torch.device:
    """
    Pick a compute device, validating that CUDA actually works on this node.
    Falls back to CPU when the GPU exists but is architecturally incompatible
    with the installed PyTorch (e.g. sm_35 / old Tesla on PyTorch ≥ 2.8).
    """
    if requested:
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.tensor([1.0]).cuda()          # tiny smoke test
        return torch.device("cuda")
    except RuntimeError as e:
        print(f"[warn] CUDA available but unusable ({e.__class__.__name__}): {e}")
        print("[warn] Falling back to CPU.")
        return torch.device("cpu")


def main():
    args   = parse_args()
    device = _resolve_device(args.device)

    # ── model config ─────────────────────────────────────────────────────
    cfg        = MODEL_CONFIGS[args.llm_name]
    d_model    = cfg["d_model"]
    hmid_layer = args.hmid_layer if args.hmid_layer is not None else cfg["hmid_layer"]
    s1_pat     = args.patience if args.patience is not None else S1_PATIENCE
    s2_pat     = args.patience if args.patience is not None else S2_PATIENCE

    tag         = model_tag(args.llm_name)
    ctrl_suffix = f"_ctrl_{args.z_control}" if args.z_control != "none" else ""
    out_dir     = Path(args.out_root) / tag / f"{args.heldout}{ctrl_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"LLM two-stage training")
    print(f"  heldout   : {args.heldout}")
    print(f"  llm       : {args.llm_name}")
    print(f"  d_model   : {d_model}")
    print(f"  hmid_layer: {hmid_layer}")
    print(f"  device    : {device}")
    print(f"  out_dir   : {out_dir}")
    print(f"{'='*60}")

    # ── save run config ──────────────────────────────────────────────────
    run_cfg = vars(args)
    run_cfg.update({"d_model": d_model, "hmid_layer_used": hmid_layer,
                    "device": str(device)})
    (out_dir / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2, default=str)
    )

    # ── LOSO split ───────────────────────────────────────────────────────
    splits = make_loso_splits(args.heldout)

    # ── vocab info (for restricted IDs) ─────────────────────────────────
    vocab_info = load_vocab_info(args.llm_name)
    r_ids      = torch.tensor(vocab_info["restricted_first_token_ids"],
                              dtype=torch.long, device=device)
    R          = r_ids.shape[0]
    print(f"\nRestricted vocab size R={R}")

    # ── Stage 1 ──────────────────────────────────────────────────────────
    meg_enc = MEGEncoder().to(device)

    if args.load_stage1:
        # Skip training; load an existing Stage 1 checkpoint directly.
        print(f"\nSkipping Stage 1 training — loading {args.load_stage1}")
        ckpt = torch.load(args.load_stage1, map_location="cpu")
        meg_enc.load_state_dict(ckpt["meg_encoder"])
        meg_enc.freeze()
        s1_history = {"train_loss": [], "val_loss": []}
        (out_dir / "stage1_val_metrics.json").write_text(
            json.dumps({"loaded_from": str(args.load_stage1), "skipped": True}, indent=2)
        )
        print(f"MEGEncoder frozen  (from {args.load_stage1})")
    else:
        # Normal path: build datasets, train, freeze.
        print("\nBuilding Stage 1 datasets ...")
        t0 = time.time()
        ds_s1_train = MEGWordDatasetLLM(
            **splits["train"], llm_name=args.llm_name,
            hmid_layer=hmid_layer, augment=True,
        )
        ds_s1_val = MEGWordDatasetLLM(
            **splits["val"],   llm_name=args.llm_name,
            hmid_layer=hmid_layer, augment=False,
        )
        print(f"  (built in {time.time()-t0:.1f}s)")

        if len(ds_s1_train) == 0:
            raise RuntimeError("Stage 1 training set is empty — check MEG files and cache.")

        dl_s1_train = DataLoader(ds_s1_train, batch_size=args.s1_bs, shuffle=True,
                                 num_workers=2, pin_memory=(device.type == "cuda"))
        dl_s1_val   = DataLoader(ds_s1_val,   batch_size=args.s1_bs, shuffle=False,
                                 num_workers=2, pin_memory=(device.type == "cuda"))

        text_proj = LLMTextProjection(d_model=d_model).to(device)
        n_enc  = sum(p.numel() for p in meg_enc.parameters())
        n_proj = sum(p.numel() for p in text_proj.parameters())
        print("\nInitializing models ...")
        print(f"  MEGEncoder      {n_enc:,} params")
        print(f"  TextProjection  {n_proj:,} params  (d_model={d_model} → 128)")

        s1_result = train_stage1(
            meg_enc      = meg_enc,
            text_proj    = text_proj,
            train_loader = dl_s1_train,
            val_loader   = dl_s1_val,
            device       = device,
            loss_fn      = args.loss,
            lr           = args.s1_lr,
            n_epochs     = args.s1_epochs,
            patience     = s1_pat,
            out_dir      = out_dir,
        )

        s1_metrics = {
            "best_val_loss": s1_result["best_val"],
            "best_epoch":    s1_result["best_epoch"],
            "n_epochs_run":  len(s1_result["history"]["train_loss"]),
        }
        (out_dir / "stage1_val_metrics.json").write_text(
            json.dumps(s1_metrics, indent=2)
        )
        print(f"\nStage 1 summary: best_val={s1_result['best_val']:.4f}  "
              f"at epoch {s1_result['best_epoch']}")

        if args.skip_stage2:
            plot_curves(s1_result["history"],
                        {"train_loss": [], "val_loss": []},
                        out_dir / "training_curve.png")
            print("\nDone (Stage 1 only).")
            return

        best_s1 = s1_result["best_state"]
        meg_enc.load_state_dict(best_s1["meg_encoder"])
        meg_enc.freeze()
        print(f"\nMEGEncoder frozen at Stage 1 best (epoch {best_s1['epoch']}).")
        s1_history = s1_result["history"]

    # ── Stage 2 datasets ─────────────────────────────────────────────────
    print("\nBuilding Stage 2 datasets ...")
    t0 = time.time()
    ds_s2_train = MEGSequenceDataset(**splits["train"], llm_name=args.llm_name)
    ds_s2_val   = MEGSequenceDataset(**splits["val"],   llm_name=args.llm_name)
    print(f"  (built in {time.time()-t0:.1f}s)")

    dl_s2_train = DataLoader(ds_s2_train, batch_size=args.s2_bs, shuffle=True,
                             collate_fn=collate_sequences, num_workers=2,
                             pin_memory=(device.type == "cuda"))
    dl_s2_val   = DataLoader(ds_s2_val,   batch_size=args.s2_bs, shuffle=False,
                             collate_fn=collate_sequences, num_workers=2,
                             pin_memory=(device.type == "cuda"))

    # ── Load lm_head + GRUHead ────────────────────────────────────────────
    lm_head  = load_lm_head(args.llm_name, device=device)
    gru_head = GRUHead(
        meg_emb    = MEG_EMB,
        gru_hidden = args.gru_hidden,
        d_model    = d_model,
    ).to(device)
    n_gru = sum(p.numel() for p in gru_head.parameters())
    print(f"GRUHead  {n_gru:,} params  (hidden={args.gru_hidden})")
    if args.z_control != "none":
        print(f"[control] z_control={args.z_control!r} — z_t will be corrupted during Stage 2")

    # ── Stage 2 ──────────────────────────────────────────────────────────
    s2_result = train_stage2(
        meg_enc      = meg_enc,
        gru_head     = gru_head,
        lm_head      = lm_head,
        r_ids        = r_ids,
        train_loader = dl_s2_train,
        val_loader   = dl_s2_val,
        device       = device,
        lr           = args.s2_lr,
        n_epochs     = args.s2_epochs,
        patience     = s2_pat,
        out_dir      = out_dir,
        z_control    = args.z_control,
    )

    s2_metrics = {
        "best_val_loss": s2_result["best_val"],
        "best_epoch":    s2_result["best_epoch"],
        "n_epochs_run":  len(s2_result["history"]["train_loss"]),
    }
    (out_dir / "stage2_val_metrics.json").write_text(
        json.dumps(s2_metrics, indent=2)
    )
    print(f"\nStage 2 summary: best_val={s2_result['best_val']:.4f}  "
          f"at epoch {s2_result['best_epoch']}")

    # ── Training curve ───────────────────────────────────────────────────
    plot_curves(s1_history, s2_result["history"],
                out_dir / "training_curve.png")

    print(f"\nAll results in {out_dir}")


if __name__ == "__main__":
    main()
