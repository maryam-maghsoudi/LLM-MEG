"""
meg_encoder.py
==============
MEG encoder for the LLM-guided decoder.

Replicates the MEGWordEncoderSmall architecture from the contrastive training
pipeline and provides a loader for the pretrained checkpoint (Option A: frozen).
The forward pass returns L2-normalized 128-d embeddings — the same output that
the contrastive decoder was trained with — which are then fed to the adapter.

Option B (train from scratch jointly with the adapter) is supported by simply
not calling load_pretrained() and leaving requires_grad=True.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_CHANNELS, WIN_SIZE, MEG_EMB_DIM, MEG_CKPT


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1, dropout: float = 0.3):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MEGEncoder(nn.Module):
    """
    Spatial-temporal conv encoder.  Mirrors MEGWordEncoderSmall from the
    contrastive pipeline so pretrained weights load without remapping.

    Input : (B, C, T)   — C=155 channels, T=100 samples
    Output: (B, emb_dim) — L2-normalized embedding
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        win_size: int = WIN_SIZE,
        emb_dim: int = MEG_EMB_DIM,
        dropout: float = 0.3,
    ):
        super().__init__()
        self._frozen = False
        self.spatial = nn.Sequential(
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
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, emb_dim),
        )
        self.emb_dim = emb_dim

    def freeze(self) -> None:
        """Freeze all parameters and lock BatchNorm in eval mode permanently."""
        for p in self.parameters():
            p.requires_grad_(False)
        self._frozen = True
        self.eval()

    def train(self, mode: bool = True) -> "MEGEncoder":
        # When frozen, outer model.train() must not flip BatchNorm to train mode —
        # that would replace stable running statistics with noisy batch statistics.
        if self._frozen:
            return super().train(False)
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial(x)
        x = self.temporal(x)
        x = self.pool(x).squeeze(-1)
        return F.normalize(self.proj(x), dim=-1)


def load_pretrained(
    freeze: bool = True,
    n_channels: int = N_CHANNELS,
    emb_dim: int = MEG_EMB_DIM,
    ckpt_path=MEG_CKPT,
) -> MEGEncoder:
    """
    Instantiate MEGEncoder and load the contrastively-trained checkpoint.

    Parameters
    ----------
    freeze     : if True (default, Option A), all parameters are frozen.
                 Set False for Option B (joint training with adapter).
    ckpt_path  : path to meg_encoder.pt from the contrastive pipeline.
    """
    encoder = MEGEncoder(n_channels=n_channels, emb_dim=emb_dim)
    state = torch.load(ckpt_path, map_location="cpu")
    encoder.load_state_dict(state)

    if freeze:
        encoder.freeze()

    n_params = sum(p.numel() for p in encoder.parameters())
    status = "frozen" if freeze else "trainable"
    # Returned on CPU — caller is responsible for .to(device) after loading.
    print(f"MEGEncoder loaded from {ckpt_path}  ({n_params:,} params, {status})")
    return encoder
