"""
models.py — neural network modules for the two-stage MEG decoder

MEGEncoder          Spatial-temporal CNN (mirrors MEGWordEncoderSmall).
                    Stand-alone copy of llm_decoder/meg_encoder.py so
                    llm_twostage/ has no circular config dependency.

LLMTextProjection   Trainable head for Stage 1: maps precomputed LLM hidden
                    states (d_model) → MEG embedding space (emb_dim=128).

GRUHead             Causal GRU head for Stage 2: reads z_t from frozen MEG
                    encoder and outputs y_t ∈ R^d_model for the frozen lm_head.
"""

import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
#  Defaults
# ---------------------------------------------------------------------------
N_CHANNELS = 155
WIN_SIZE   = 40    # [-100ms, +300ms] @ 100 Hz
MEG_EMB    = 128


# ===========================================================================
#  MEGEncoder (matches MEGWordEncoderSmall from contrastive pipeline)
# ===========================================================================

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation=1, dropout=0.3):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class MEGEncoder(nn.Module):
    """
    Spatial-temporal conv encoder.
    Input : (B, C, T) — C=155, T=100
    Output: (B, emb_dim) — L2-normalized
    """

    def __init__(self, n_channels=N_CHANNELS, win_size=WIN_SIZE,
                 emb_dim=MEG_EMB, dropout=0.3):
        super().__init__()
        self._frozen = False
        self.emb_dim = emb_dim
        self.spatial  = nn.Sequential(
            nn.Conv1d(n_channels, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            _ConvBlock(32,  64,  7, dilation=1, dropout=dropout),
            _ConvBlock(64,  128, 5, dilation=2, dropout=dropout),
            _ConvBlock(128, 128, 3, dilation=4, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(128, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, emb_dim),
        )

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self._frozen = True
        self.eval()

    def train(self, mode=True):
        if self._frozen:
            return super().train(False)
        return super().train(mode)

    def forward(self, x):
        x = self.spatial(x)
        x = self.temporal(x)
        x = self.pool(x).squeeze(-1)
        return F.normalize(self.proj(x), dim=-1)


def load_meg_encoder(ckpt_path=None, freeze=True) -> MEGEncoder:
    """
    Instantiate MEGEncoder, optionally loading pretrained contrastive weights.
    ckpt_path=None → train from scratch.
    """
    enc = MEGEncoder()
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu")
        enc.load_state_dict(state)
        src = f"checkpoint  {ckpt_path}"
    else:
        src = "random init"
    if freeze:
        enc.freeze()
    status = "frozen" if freeze else "trainable"
    n = sum(p.numel() for p in enc.parameters())
    print(f"MEGEncoder  ({src})  {n:,} params  {status}")
    return enc


# ===========================================================================
#  Stage 1 — text projection head
# ===========================================================================

class LLMTextProjection(nn.Module):
    """
    Maps precomputed LLM hidden states → MEG embedding space for InfoNCE.

    Architecture mirrors the TextEncoder.proj from contrastive_word_meg.py:
        Linear(d_model, 256) → GELU → Dropout → Linear(256, emb_dim)
    Output is L2-normalized (same as MEGEncoder output).

    Trainable during Stage 1; discarded after Stage 1 finishes (its purpose
    was only to align training, not to generate embeddings at inference time).
    """

    def __init__(self, d_model: int, emb_dim: int = MEG_EMB, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, emb_dim),
        )

    def forward(self, hmid):          # (B, d_model) → (B, emb_dim)
        return F.normalize(self.proj(hmid), dim=-1)


# ===========================================================================
#  Stage 2 — GRU causal head
# ===========================================================================

class GRUHead(nn.Module):
    """
    Causal GRU that reads MEG embeddings in temporal order and outputs
    predicted next-word states in the LLM's hidden dimension.

    Forward  z: (B, T, meg_emb) → y: (B, T, d_model)

    The output y is then passed through the frozen lm_head to get q_t logits.
    h_0 is always zeros — hidden state resets at the start of every trial.
    """

    def __init__(self, meg_emb: int = MEG_EMB, gru_hidden: int = 256,
                 d_model: int = 960, dropout: float = 0.3):
        super().__init__()
        self.gru  = nn.GRU(meg_emb, gru_hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(gru_hidden, d_model)

    def forward(self, z):             # z: (B, T, meg_emb)
        h, _ = self.gru(z)           # (B, T, gru_hidden)
        return self.proj(self.drop(h))  # (B, T, d_model)


def load_lm_head(llm_name: str, device) -> nn.Linear:
    """
    Load only the lm_head Linear layer from a pretrained HF causal LM.
    The rest of the model is discarded after extraction.
    """
    from transformers import AutoModelForCausalLM
    print(f"Loading lm_head from {llm_name} ...")
    full_model = AutoModelForCausalLM.from_pretrained(llm_name)
    # deepcopy so the rest of the model can be GC'd
    lm_head = copy.deepcopy(full_model.lm_head)
    del full_model
    torch.cuda.empty_cache()
    for p in lm_head.parameters():
        p.requires_grad_(False)
    lm_head = lm_head.to(device)
    n = sum(p.numel() for p in lm_head.parameters())
    print(f"lm_head extracted  {n:,} params  frozen  → device={device}")
    return lm_head
