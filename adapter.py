"""
adapter.py
==========
Trainable MLP that projects MEG encoder embeddings into the LLM's token
embedding space.

Input : (B, meg_emb_dim)          — L2-normalized MEG embedding from encoder
Output: (B, n_soft, llm_d_model)  — soft-token vectors ready to be concatenated
                                     with real text embeddings before the LLM

This is the ONLY component with learnable weights in the default (Option A)
configuration.  The MEG encoder and LLM are both frozen.

Architecture (from Architecture.md):
  Linear(input_dim, hidden) → GELU → Linear(hidden, n_soft * d_model)
  reshape → (B, n_soft, d_model)
"""

import torch
import torch.nn as nn

from config import MEG_EMB_DIM, ADAPTER_HIDDEN, LLM_D_MODEL, N_SOFT_TOKENS


class Adapter(nn.Module):
    """
    Two-layer MLP: meg_emb_dim → hidden → n_soft * d_model.

    Parameters
    ----------
    input_dim  : MEG encoder output dimension (default 128)
    hidden_dim : bottleneck width (default 512)
    d_model    : LLM embedding dimension (default 768 for GPT-2)
    n_soft     : number of soft tokens produced per MEG window (default 1)
    """

    def __init__(
        self,
        input_dim: int = MEG_EMB_DIM,
        hidden_dim: int = ADAPTER_HIDDEN,
        d_model: int = LLM_D_MODEL,
        n_soft: int = N_SOFT_TOKENS,
    ):
        super().__init__()
        self.n_soft  = n_soft
        self.d_model = d_model

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_soft * d_model),
        )

    def forward(self, meg_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        meg_emb : (B, input_dim)

        Returns
        -------
        soft_tokens : (B, n_soft, d_model)
        """
        B = meg_emb.shape[0]
        out = self.net(meg_emb)                          # (B, n_soft * d_model)
        return out.view(B, self.n_soft, self.d_model)


def build_adapter(
    input_dim: int = MEG_EMB_DIM,
    hidden_dim: int = ADAPTER_HIDDEN,
    d_model: int = LLM_D_MODEL,
    n_soft: int = N_SOFT_TOKENS,
) -> Adapter:
    adapter = Adapter(input_dim, hidden_dim, d_model, n_soft)
    n_params = sum(p.numel() for p in adapter.parameters())
    print(
        f"Adapter: {input_dim} → {hidden_dim} → {n_soft}×{d_model}  "
        f"({n_params:,} trainable params)"
    )
    return adapter
