"""
train_interleaved.py — Method 3: Soft-token injection into frozen LLM.

The MEGEncoder (from Method 1's best checkpoint) is frozen throughout.
Only the Adapter MLP is trained.

Sequence design (interleaved):
    [soft(w1)] [tok(w1,0)..tok(w1,k)] [soft(w2)] [tok(w2,0)..] ...

Loss: cross-entropy at text-token positions only (soft positions get label=-100).

Output files
------------
out_dir/
    adapter_best.pt     Adapter state dict at best val loss
    adapter_final.pt
    history.json
    train_metrics.json
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.base_dataset import MEGTrialDataset, ONSET_DIR
from .models import MEGEncoder, Adapter, load_meg_encoder

# ---------------------------------------------------------------------------
#  Hyperparameters
# ---------------------------------------------------------------------------
LR       = 1e-4
EPOCHS   = 60
BS       = 4        # full trials per batch
PATIENCE = 10
N_SOFT   = 1        # soft tokens per word


# ---------------------------------------------------------------------------
#  Tokenise poems once (token IDs don't vary by subject/session)
# ---------------------------------------------------------------------------

def _tokenise_poems(
    tokenizer,
    onset_dir: Path,
) -> Dict[str, List[List[int]]]:
    """
    Returns {poem: [[tok_ids_word0], [tok_ids_word1], ...]} for both poems.
    Words are tokenised with a leading space (mid-sentence BPE convention).
    """
    import json as _json
    poem_token_ids: Dict[str, List[List[int]]] = {}
    for poem in ["poem1", "poem2"]:
        onsets = _json.loads((onset_dir / f"{poem}_word_onsets.json").read_text())
        ids_per_word = []
        for entry in onsets:
            word = entry["word"].strip().lower()
            ids  = tokenizer.encode(" " + word, add_special_tokens=False)
            if not ids:
                ids = tokenizer.encode(word, add_special_tokens=False)
            if not ids:
                ids = [tokenizer.unk_token_id or 0]
            ids_per_word.append(ids)
        poem_token_ids[poem] = ids_per_word
    return poem_token_ids


# ---------------------------------------------------------------------------
#  Sequence dataset wrapper
# ---------------------------------------------------------------------------

class InterleavedSequenceDataset(Dataset):
    """
    Wraps MEGTrialDataset, attaching per-word token ID lists.
    One item = one full trial.

    Item keys (in addition to MEGTrialDataset keys)
    ------------------------------------------------
    token_ids_per_word : List[List[int]]  — LLM token IDs for each word
    """

    def __init__(
        self,
        base:            MEGTrialDataset,
        poem_token_ids:  Dict[str, List[List[int]]],
    ):
        self._base           = base
        self._poem_token_ids = poem_token_ids

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> Dict:
        item  = self._base[idx]
        poem  = item["poem"]
        poses = item["word_poses"]
        all_ids = self._poem_token_ids[poem]
        item["token_ids_per_word"] = [all_ids[p] for p in poses]
        return item


# ---------------------------------------------------------------------------
#  Build interleaved inputs_embeds + labels for one trial
# ---------------------------------------------------------------------------

def _build_interleaved(
    meg_windows:       torch.Tensor,   # (N, C, T)
    valid_mask:        torch.Tensor,   # (N,) bool
    token_ids_per_word: List[List[int]],
    meg_enc:           MEGEncoder,
    adapter:           Adapter,
    token_embedding:   nn.Embedding,   # LLM wte
    device:            torch.device,
    n_soft:            int = N_SOFT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the interleaved embedding sequence and corresponding labels for
    one trial.

    Returns
    -------
    embeds : (1, L, d_model)  — full sequence via inputs_embeds
    labels : (1, L)           — -100 at soft positions, token IDs at text positions
    """
    embed_parts: List[torch.Tensor] = []
    label_parts: List[torch.Tensor] = []

    for i, (tok_ids, is_valid) in enumerate(zip(token_ids_per_word, valid_mask)):
        # Soft token(s) for this word
        if is_valid:
            win = meg_windows[i].unsqueeze(0).to(device)    # (1, C, T)
            with torch.no_grad():
                z = meg_enc(win)                             # (1, 128)
            soft = adapter(z)                                # (1, n_soft, d_model)
        else:
            d   = adapter.d_model
            soft = torch.zeros(1, n_soft, d, device=device)

        embed_parts.append(soft.squeeze(0))                  # (n_soft, d_model)
        label_parts.append(torch.full((n_soft,), -100, dtype=torch.long, device=device))

        # Text token embeddings for this word
        ids_t  = torch.tensor(tok_ids, dtype=torch.long, device=device)   # (k,)
        t_emb  = token_embedding(ids_t)                                    # (k, d_model)
        embed_parts.append(t_emb)
        label_parts.append(ids_t)

    embeds = torch.cat(embed_parts, dim=0).unsqueeze(0)   # (1, L, d_model)
    labels = torch.cat(label_parts, dim=0).unsqueeze(0)   # (1, L)
    return embeds, labels


# ---------------------------------------------------------------------------
#  Collate function for variable-length trials
# ---------------------------------------------------------------------------

def _collate_interleaved(batch: List[Dict]) -> List[Dict]:
    """Return the batch as-is; padding is handled inside the training loop."""
    return batch


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------

def train(
    splits:         Dict,
    out_dir:        Path,
    device:         torch.device,
    meg_enc_ckpt:   str,                             # Method 1 checkpoint
    llm_name:       str  = "gpt2",
    n_soft:         int  = N_SOFT,
    lr:             float = LR,
    epochs:         int   = EPOCHS,
    batch_size:     int   = BS,
    patience:       int   = PATIENCE,
    control:        str   = "none",
    cache_dir:      Optional[str] = None,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Frozen MEGEncoder (from Method 1) ────────────────────────────────────
    meg_enc = load_meg_encoder(meg_enc_ckpt, device, freeze=True)

    # ── Frozen LLM ───────────────────────────────────────────────────────────
    print(f"Loading {llm_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(llm_name).to(device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    token_embedding = llm.get_input_embeddings()   # nn.Embedding (frozen)
    d_model = token_embedding.embedding_dim

    # ── Tokenise poems ───────────────────────────────────────────────────────
    poem_token_ids = _tokenise_poems(tokenizer, ONSET_DIR)

    # ── Adapter (only trainable component) ───────────────────────────────────
    adapter = Adapter(n_soft=n_soft, d_model=d_model).to(device)
    n_adapt = sum(p.numel() for p in adapter.parameters())
    print(f"Adapter  {n_adapt:,} params  n_soft={n_soft}  d_model={d_model}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    from ..data.controls import make_control

    print("\nBuilding datasets ...")
    t0 = time.time()
    if cache_dir is not None:
        from pathlib import Path as _Path
        _trial_items = MEGTrialDataset.load_items(str(_Path(cache_dir) / "meg_trial_all.pt"))
        base_tr = MEGTrialDataset.from_cache(_trial_items, splits["train"])
        base_vl = MEGTrialDataset.from_cache(_trial_items, splits["val"])
    else:
        base_tr = MEGTrialDataset(splits["train"]["trials"],
                                   splits["train"]["word_filter"])
        base_vl = MEGTrialDataset(splits["val"]["trials"],
                                   splits["val"]["word_filter"])
    base_tr = make_control(base_tr, control)
    base_vl = make_control(base_vl, control)

    ds_tr = InterleavedSequenceDataset(base_tr, poem_token_ids)
    ds_vl = InterleavedSequenceDataset(base_vl, poem_token_ids)
    print(f"  built in {time.time() - t0:.1f}s")

    pin = device.type == "cuda"
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       collate_fn=_collate_interleaved, num_workers=0, pin_memory=pin)
    dl_vl = DataLoader(ds_vl, batch_size=batch_size, shuffle=False,
                       collate_fn=_collate_interleaved, num_workers=0, pin_memory=pin)

    # ── Optimiser ────────────────────────────────────────────────────────────
    opt   = torch.optim.AdamW(adapter.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history  = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    wait     = 0
    best_state: Optional[Dict] = None

    print(f"\n{'='*60}")
    print(f"Method 3 (interleaved)  llm={llm_name}  control={control}")
    print(f"  lr={lr}  bs={batch_size}  epochs={epochs}  patience={patience}")
    print(f"{'='*60}")

    def _batch_loss(batch_list):
        losses = []
        for item in batch_list:
            embeds, labels = _build_interleaved(
                item["meg_windows"], item["valid_mask"],
                item["token_ids_per_word"],
                meg_enc, adapter, token_embedding, device, n_soft,
            )
            out  = llm(inputs_embeds=embeds, labels=labels)
            losses.append(out.loss)
        return torch.stack(losses).mean()

    for epoch in range(1, epochs + 1):
        adapter.train()
        t_losses = []
        for batch in dl_tr:
            loss = _batch_loss(batch)
            opt.zero_grad(); loss.backward(); opt.step()
            t_losses.append(loss.item())
        sched.step()

        adapter.eval()
        v_losses = []
        with torch.no_grad():
            for batch in dl_vl:
                v_losses.append(_batch_loss(batch).item())

        t_loss = float(np.mean(t_losses))
        v_loss = float(np.mean(v_losses))
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        if v_loss < best_val:
            best_val = v_loss; wait = 0
            best_state = {"adapter": {k: v.cpu()
                                      for k, v in adapter.state_dict().items()},
                          "epoch": epoch, "val_loss": v_loss}
            marker = " ✓"
        else:
            wait += 1
            marker = f"  (wait {wait}/{patience})"

        print(f"  epoch {epoch:3d}/{epochs}  train={t_loss:.4f}  val={v_loss:.4f}{marker}")
        if wait >= patience:
            print(f"  → early stop at epoch {epoch}")
            break

    torch.save(best_state["adapter"], out_dir / "adapter_best.pt")
    torch.save({k: v.cpu() for k, v in adapter.state_dict().items()},
               out_dir / "adapter_final.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    metrics = {"best_val_loss": best_val, "best_epoch": best_state["epoch"],
               "n_epochs_run": len(history["train_loss"])}
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nBest val={best_val:.4f}  → {out_dir}")
    return metrics
