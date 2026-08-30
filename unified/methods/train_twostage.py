"""
train_twostage.py — Method 2: Two-stage MEG decoder.

Stage 1  InfoNCE alignment of MEGEncoder to LLM contextual hidden states.
Stage 2  KL distillation: frozen MEGEncoder + trainable GRUHead predicts
         the frozen LLM's next-word distribution.

Prerequisites
-------------
Run cache_llm_hiddens.py before training to populate the LLM cache.
The LLM itself is never loaded during training — all targets come from cache.

Output files
------------
out_dir/
    stage1_best.pt      {meg_encoder, text_proj, epoch, val_loss}
    stage2_best.pt      {gru_head, epoch, val_loss}
    stage1_metrics.json
    stage2_metrics.json
    history.json
    run_config.json
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..data.base_dataset import (
    MEGWordDataset, MEGTrialDataset, collate_trials, ONSET_DIR,
)
from .models import (
    MEGEncoder, LLMTextProjection, GRUHead, load_lm_head,
)

# ---------------------------------------------------------------------------
#  LLM model configs (d_model, best hmid_layer from probe_layers.py)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "HuggingFaceTB/SmolLM2-360M": {"d_model": 960,  "hmid_layer": 11},
    "HuggingFaceTB/SmolLM2-1.7B": {"d_model": 2048, "hmid_layer": 8},
    "gpt2":                        {"d_model": 768,  "hmid_layer": 4},
    "Qwen/Qwen2-0.5B":             {"d_model": 896,  "hmid_layer": 8},
}

# ---------------------------------------------------------------------------
#  Hyperparameters
# ---------------------------------------------------------------------------
S1_LR, S1_EPOCHS, S1_BS, S1_PATIENCE = 3e-4, 60, 64, 10
S2_LR, S2_EPOCHS, S2_BS, S2_PATIENCE = 1e-4, 60, 4,  10
TEMP       = 0.07
GRU_HIDDEN = 256
MEG_EMB    = 128


# ---------------------------------------------------------------------------
#  Cache helpers
# ---------------------------------------------------------------------------

def _model_tag(llm_name: str) -> str:
    return llm_name.replace("/", "_")


def _load_cache(llm_name: str, poem: str, cache_root: Path) -> Dict:
    path = cache_root / _model_tag(llm_name) / f"{poem}_hiddens.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"LLM cache not found: {path}\n"
            f"Run: python cache_llm_hiddens.py --llm_name {llm_name}"
        )
    return torch.load(path, map_location="cpu")


def _load_vocab_info(llm_name: str, cache_root: Path) -> Dict:
    path = cache_root / _model_tag(llm_name) / "vocab_info.json"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
#  Stage 1 dataset wrapper
# ---------------------------------------------------------------------------

class LLMAugmentedDataset(Dataset):
    """
    Wraps MEGWordDataset, attaching the precomputed LLM hidden state
    at hmid_layer for each word occurrence.
    hmid_t depends only on (poem, word_pos) — same across all subjects/sessions.
    """

    def __init__(
        self,
        base:       MEGWordDataset,
        llm_caches: Dict[str, Dict],   # {poem: cache dict}
        hmid_layer: int,
    ):
        self._base   = base
        self._hiddens = {
            poem: cache["hidden_all_layers"][hmid_layer]   # (N_words, d_model)
            for poem, cache in llm_caches.items()
        }

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self._base[idx]
        h    = self._hiddens[item["poem"]][item["word_pos"]]
        return item["meg_window"], h


# ---------------------------------------------------------------------------
#  Stage 2 dataset wrapper (attaches teacher logits to MEGTrialDataset)
# ---------------------------------------------------------------------------

class LLMSequenceDataset(Dataset):
    """
    Wraps MEGTrialDataset, adding the teacher's restricted-vocab logits
    (p_t_restricted) from the LLM cache.
    """

    def __init__(
        self,
        base:       MEGTrialDataset,
        llm_caches: Dict[str, Dict],   # {poem: cache dict}
    ):
        self._base   = base
        self._p_t    = {
            poem: cache["lm_logits_restricted"]   # (N_words, R)
            for poem, cache in llm_caches.items()
        }

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> Dict:
        item  = self._base[idx]
        poem  = item["poem"]
        poses = item["word_poses"]
        p_t   = self._p_t[poem][poses]            # (N, R) — index by word positions
        return {**item, "p_t_restricted": p_t}


def _collate_llm_sequences(batch: List[Dict]) -> Dict:
    """Extends collate_trials to also pad p_t_restricted."""
    from ..data.base_dataset import collate_trials as _base_collate
    base = _base_collate(batch)
    max_N = base["meg_windows"].shape[1]
    B     = len(batch)
    R     = batch[0]["p_t_restricted"].shape[-1]

    p_t = torch.zeros(B, max_N, R)
    for b, item in enumerate(batch):
        N = item["p_t_restricted"].shape[0]
        p_t[b, :N] = item["p_t_restricted"]

    base["p_t_restricted"] = p_t
    return base


# ---------------------------------------------------------------------------
#  Losses
# ---------------------------------------------------------------------------

def info_nce(z_meg, z_text, temperature=TEMP):
    N   = z_meg.shape[0]
    sim = z_meg @ z_text.T / temperature
    return F.cross_entropy(sim, torch.arange(N, device=z_meg.device))


def kl_loss(q_logits_r, p_logits_r, valid_mask):
    """
    KL(p_t || q_t) averaged over valid positions 0..T-2.
    Mask is valid_mask[:, 1:] — does word t+1 exist?
    At position t the GRU predicts word t+1, so we test whether the next
    word is real (not the current word). See DESIGN.md §5.2.
    """
    q    = F.log_softmax(q_logits_r[:, :-1, :], dim=-1)
    p    = F.softmax(   p_logits_r[:, :-1, :], dim=-1)
    mask = valid_mask[:, 1:].float()
    kl   = F.kl_div(q, p, reduction="none").sum(-1)   # (B, T-1)
    return (kl * mask).sum() / mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
#  Stage 1 training loop
# ---------------------------------------------------------------------------

def _train_stage1(
    meg_enc, text_proj, dl_train, dl_val, device,
    lr, epochs, patience, temperature, out_dir,
):
    opt   = torch.optim.AdamW(
        list(meg_enc.parameters()) + list(text_proj.parameters()), lr=lr
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history  = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    wait     = 0
    best_state: Optional[Dict] = None

    print(f"\n{'='*60}")
    print(f"STAGE 1  lr={lr}  bs={dl_train.batch_size}  patience={patience}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        meg_enc.train(); text_proj.train()
        t_losses = []
        for x, h in dl_train:
            x, h = x.to(device), h.to(device)
            loss  = info_nce(meg_enc(x), text_proj(h), temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            t_losses.append(loss.item())
        sched.step()

        meg_enc.eval(); text_proj.eval()
        v_losses = []
        with torch.no_grad():
            for x, h in dl_val:
                x, h = x.to(device), h.to(device)
                v_losses.append(info_nce(meg_enc(x), text_proj(h), temperature).item())

        t_loss = float(np.mean(t_losses))
        v_loss = float(np.mean(v_losses))
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        if v_loss < best_val:
            best_val = v_loss; wait = 0
            best_state = {
                "meg_encoder": {k: v.cpu() for k, v in meg_enc.state_dict().items()},
                "text_proj":   {k: v.cpu() for k, v in text_proj.state_dict().items()},
                "epoch": epoch, "val_loss": v_loss,
            }
            marker = " ✓"
        else:
            wait += 1
            marker = f"  (wait {wait}/{patience})"

        print(f"  epoch {epoch:3d}/{epochs}  train={t_loss:.4f}  val={v_loss:.4f}{marker}")
        if wait >= patience:
            print(f"  → early stop at epoch {epoch}")
            break

    torch.save(best_state, out_dir / "stage1_best.pt")
    return history, best_val, best_state


# ---------------------------------------------------------------------------
#  Stage 2 training loop
# ---------------------------------------------------------------------------

def _train_stage2(
    meg_enc, gru_head, lm_head, r_ids,
    dl_train, dl_val, device,
    lr, epochs, patience, out_dir, z_control="none",
):
    opt   = torch.optim.AdamW(gru_head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history  = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    wait     = 0
    best_state: Optional[Dict] = None

    print(f"\n{'='*60}")
    print(f"STAGE 2  lr={lr}  bs={dl_train.batch_size}  z_control={z_control}")
    print(f"{'='*60}")

    def _encode_batch(batch):
        meg_seq    = batch["meg_windows"].to(device)   # (B, N, C, T)
        valid_mask = batch["valid_mask"].to(device)
        p_t_r      = batch["p_t_restricted"].to(device)
        B, N, C, T = meg_seq.shape
        with torch.no_grad():
            z = meg_enc(meg_seq.view(B * N, C, T)).view(B, N, -1)
        if z_control == "zero":
            z = torch.zeros_like(z)
        elif z_control == "shuffle_time":
            # permute within each trial
            for b in range(B):
                L = int(valid_mask[b].sum().item())
                if L > 1:
                    perm = torch.randperm(L, device=device)
                    z[b, :L] = z[b, perm]
        return z, p_t_r, valid_mask

    for epoch in range(1, epochs + 1):
        gru_head.train()
        t_losses = []
        for batch in dl_train:
            z, p_t_r, valid_mask = _encode_batch(batch)
            y         = gru_head(z)
            q_logits  = lm_head(y)[:, :, r_ids]
            loss      = kl_loss(q_logits, p_t_r, valid_mask)
            opt.zero_grad(); loss.backward(); opt.step()
            t_losses.append(loss.item())
        sched.step()

        gru_head.eval()
        v_losses = []
        with torch.no_grad():
            for batch in dl_val:
                z, p_t_r, valid_mask = _encode_batch(batch)
                y = gru_head(z)
                q = lm_head(y)[:, :, r_ids]
                v_losses.append(kl_loss(q, p_t_r, valid_mask).item())

        t_loss = float(np.mean(t_losses))
        v_loss = float(np.mean(v_losses))
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        if v_loss < best_val:
            best_val = v_loss; wait = 0
            best_state = {
                "gru_head": {k: v.cpu() for k, v in gru_head.state_dict().items()},
                "epoch": epoch, "val_loss": v_loss,
            }
            marker = " ✓"
        else:
            wait += 1
            marker = f"  (wait {wait}/{patience})"

        print(f"  epoch {epoch:3d}/{epochs}  train={t_loss:.4f}  val={v_loss:.4f}{marker}")
        if wait >= patience:
            print(f"  → early stop at epoch {epoch}")
            break

    torch.save(best_state, out_dir / "stage2_best.pt")
    return history, best_val, best_state


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def train(
    splits:       Dict,
    out_dir:      Path,
    device:       torch.device,
    llm_name:     str   = "HuggingFaceTB/SmolLM2-360M",
    hmid_layer:   Optional[int] = None,
    cache_root:   Optional[Path] = None,
    skip_stage2:  bool  = False,
    s1_lr:        float = S1_LR,
    s1_epochs:    int   = S1_EPOCHS,
    s1_bs:        int   = S1_BS,
    s1_patience:  int   = S1_PATIENCE,
    s2_lr:        float = S2_LR,
    s2_epochs:    int   = S2_EPOCHS,
    s2_bs:        int   = S2_BS,
    s2_patience:  int   = S2_PATIENCE,
    gru_hidden:   int   = GRU_HIDDEN,
    temperature:  float = TEMP,
    control:      str   = "none",
    load_stage1:  Optional[str] = None,
    cache_dir:    Optional[str] = None,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    if cache_root is None:
        cache_root = Path(__file__).parent.parent.parent / "llm_twostage" / "cache"

    cfg        = MODEL_CONFIGS[llm_name]
    d_model    = cfg["d_model"]
    hmid_layer = hmid_layer if hmid_layer is not None else cfg["hmid_layer"]

    print(f"\nMethod 2 (twostage)  llm={llm_name}  d_model={d_model}  "
          f"hmid_layer={hmid_layer}  control={control}")

    # ── Load caches ──────────────────────────────────────────────────────────
    llm_caches = {p: _load_cache(llm_name, p, cache_root) for p in ["poem1", "poem2"]}
    vocab_info = _load_vocab_info(llm_name, cache_root)
    r_ids      = torch.tensor(vocab_info["restricted_first_token_ids"],
                               dtype=torch.long, device=device)

    meg_enc = MEGEncoder().to(device)

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    if load_stage1:
        print(f"\nSkipping Stage 1 — loading {load_stage1}")
        ckpt = torch.load(load_stage1, map_location="cpu")
        meg_enc.load_state_dict(ckpt["meg_encoder"])
        meg_enc.freeze()
        s1_history = {"train_loss": [], "val_loss": []}
        s1_best_val = float("nan")
    else:
        print("\nBuilding Stage 1 datasets ...")
        t0 = time.time()
        if cache_dir is not None:
            from pathlib import Path as _Path
            _word_items = MEGWordDataset.load_items(str(_Path(cache_dir) / "meg_word_all.pt"))
            base_tr = MEGWordDataset.from_cache(_word_items, splits["train"], augment=False)
            base_vl = MEGWordDataset.from_cache(_word_items, splits["val"],   augment=False)
        else:
            base_tr = MEGWordDataset(splits["train"]["trials"],
                                      splits["train"]["word_filter"], augment=False)
            base_vl = MEGWordDataset(splits["val"]["trials"],
                                      splits["val"]["word_filter"],   augment=False)

        ds_tr = LLMAugmentedDataset(base_tr, llm_caches, hmid_layer)
        ds_vl = LLMAugmentedDataset(base_vl, llm_caches, hmid_layer)
        print(f"  built in {time.time() - t0:.1f}s")

        pin = device.type == "cuda"
        dl_tr = DataLoader(ds_tr, batch_size=s1_bs, shuffle=True,
                           num_workers=4, pin_memory=pin)
        dl_vl = DataLoader(ds_vl, batch_size=s1_bs, shuffle=False,
                           num_workers=2, pin_memory=pin)

        text_proj = LLMTextProjection(d_model=d_model).to(device)
        s1_history, s1_best_val, s1_best = _train_stage1(
            meg_enc, text_proj, dl_tr, dl_vl, device,
            s1_lr, s1_epochs, s1_patience, temperature, out_dir,
        )
        meg_enc.load_state_dict(s1_best["meg_encoder"])
        meg_enc.freeze()
        print(f"\nMEGEncoder frozen at Stage 1 best (epoch {s1_best['epoch']})")

    (out_dir / "stage1_metrics.json").write_text(json.dumps({
        "best_val_loss": s1_best_val,
        "n_epochs":      len(s1_history["train_loss"]),
    }, indent=2))

    if skip_stage2:
        return {"stage1_val": s1_best_val}

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    print("\nBuilding Stage 2 datasets ...")
    t0 = time.time()
    if cache_dir is not None:
        from pathlib import Path as _Path
        _trial_items = MEGTrialDataset.load_items(str(_Path(cache_dir) / "meg_trial_all.pt"))
        base_seq_tr = MEGTrialDataset.from_cache(_trial_items, splits["train"])
        base_seq_vl = MEGTrialDataset.from_cache(_trial_items, splits["val"])
    else:
        base_seq_tr = MEGTrialDataset(splits["train"]["trials"],
                                       splits["train"]["word_filter"])
        base_seq_vl = MEGTrialDataset(splits["val"]["trials"],
                                       splits["val"]["word_filter"])

    ds_seq_tr = LLMSequenceDataset(base_seq_tr, llm_caches)
    ds_seq_vl = LLMSequenceDataset(base_seq_vl, llm_caches)
    print(f"  built in {time.time() - t0:.1f}s")

    pin = device.type == "cuda"
    dl_seq_tr = DataLoader(ds_seq_tr, batch_size=s2_bs, shuffle=True,
                           collate_fn=_collate_llm_sequences,
                           num_workers=2, pin_memory=pin)
    dl_seq_vl = DataLoader(ds_seq_vl, batch_size=s2_bs, shuffle=False,
                           collate_fn=_collate_llm_sequences,
                           num_workers=2, pin_memory=pin)

    lm_head  = load_lm_head(llm_name, device)
    gru_head = GRUHead(meg_emb=MEG_EMB, gru_hidden=gru_hidden,
                       d_model=d_model).to(device)
    n_gru = sum(p.numel() for p in gru_head.parameters())
    print(f"GRUHead  {n_gru:,} params")

    s2_history, s2_best_val, _ = _train_stage2(
        meg_enc, gru_head, lm_head, r_ids,
        dl_seq_tr, dl_seq_vl, device,
        s2_lr, s2_epochs, s2_patience, out_dir,
        z_control=control if control in ("zero", "shuffle_time") else "none",
    )

    (out_dir / "stage2_metrics.json").write_text(json.dumps({
        "best_val_loss": s2_best_val,
        "n_epochs":      len(s2_history["train_loss"]),
    }, indent=2))

    history = {"stage1": s1_history, "stage2": s2_history}
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nAll outputs → {out_dir}")
    return {"stage1_val": s1_best_val, "stage2_val": s2_best_val}
