"""
model.py
========
Full LLM-guided MEG decoder.

Wires together:
  MEGEncoder  (frozen by default, Option A)
  Adapter     (trainable, the only learned component in Option A)
  LLM         (frozen; GPT-2 or similar via HuggingFace)

Two sequence designs are supported (selected by SEQUENCE_DESIGN in config):

  Design A — interleaved (recommended):
    [soft(w1)] [tok(w1)…] [soft(w2)] [tok(w2)…] …
    Loss computed only at text-token positions.  At each soft-token position
    the model is trained to predict the first text token of that word, giving
    a direct credit-assignment signal from brain evidence to word identity.

  Design B — upfront (baseline):
    [soft(w1)] [soft(w2)] … [soft(wN)] [tok(w1)…tok(wN)…]
    All MEG evidence presented first, then the LLM generates the full poem.
    Similar to image captioning

In both designs, labels = -100 at soft-token positions (ignored by CE loss),
and the HuggingFace internal shift (labels[:, 1:]) handles the standard
next-token prediction target.

Usage
-----
  from model import build_model
  model = build_model(device)

  # forward returns a HF CausalLMOutput; access .loss for backprop
  out = model(
      meg_windows    = batch["meg_windows"],     # (B, N, C, T)
      valid_mask     = batch["valid_mask"],       # (B, N)
      word_token_ids = batch["word_token_ids"],   # List[B × List[N × List[int]]]
  )
  out.loss.backward()
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from config import (
    LLM_NAME, LLM_D_MODEL, N_SOFT_TOKENS,
    MEG_EMB_DIM, ADAPTER_HIDDEN, N_CHANNELS,
    SEQUENCE_DESIGN,
)
from meg_encoder import MEGEncoder, load_pretrained
from adapter import Adapter, build_adapter


# =============================================================================
#  FROZEN LLM WRAPPER
# =============================================================================

class _FrozenLLM(nn.Module):
    """
    Thin wrapper around a HuggingFace causal LM that keeps the model in eval
    mode regardless of outer model.train() calls.  This prevents LLM dropout
    from randomly zeroing hidden states, which would inject noise into the
    gradient flowing back through inputs_embeds to the adapter.
    """

    def __init__(self, llm: nn.Module):
        super().__init__()
        self.llm = llm
        for p in self.llm.parameters():
            p.requires_grad_(False)
        self.llm.eval()

    def train(self, mode: bool = True) -> "_FrozenLLM":
        # Always stay in eval — same reasoning as MEGEncoder.train() override.
        return super().train(False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.llm.get_input_embeddings()

    def forward(self, **kwargs) -> CausalLMOutputWithCrossAttentions:
        return self.llm(**kwargs)


# =============================================================================
#  MAIN MODEL
# =============================================================================

class LLMDecoder(nn.Module):
    """
    Parameters
    ----------
    meg_encoder     : MEGEncoder instance (frozen via encoder.freeze(), Option A;
                      or trainable for Option B)
    adapter         : Adapter instance (always trainable)
    frozen_llm      : _FrozenLLM wrapping a HuggingFace causal LM
    sequence_design : "interleaved" (Design A) or "upfront" (Design B)
    """

    def __init__(
        self,
        meg_encoder:     MEGEncoder,
        adapter:         Adapter,
        frozen_llm:      _FrozenLLM,
        sequence_design: str = SEQUENCE_DESIGN,
    ):
        super().__init__()
        assert sequence_design in ("interleaved", "upfront"), (
            f"sequence_design must be 'interleaved' or 'upfront', got {sequence_design!r}"
        )
        self.meg_encoder     = meg_encoder
        self.adapter         = adapter
        self.frozen_llm      = frozen_llm
        self.sequence_design = sequence_design
        self.n_soft          = adapter.n_soft
        self.d_model         = adapter.d_model

    # ------------------------------------------------------------------
    #  MEG encoding
    # ------------------------------------------------------------------

    def _encode_meg(
        self,
        meg_windows: torch.Tensor,   # (B, N, C, T)
        valid_mask:  torch.Tensor,   # (B, N) bool — unused during encoding,
                                     # kept as arg for forward signature clarity
    ) -> torch.Tensor:               # (B, N, n_soft, d_model)
        B, N, C, T = meg_windows.shape
        flat    = meg_windows.view(B * N, C, T)
        meg_emb = self.meg_encoder(flat)             # (B*N, emb_dim)
        soft    = self.adapter(meg_emb)              # (B*N, n_soft, d_model)
        return soft.view(B, N, self.n_soft, self.d_model)

    # ------------------------------------------------------------------
    #  Sequence builders — one per design
    # ------------------------------------------------------------------

    def _build_interleaved(
        self,
        soft_tokens:    torch.Tensor,    # (N, n_soft, d_model)
        word_token_ids: List[List[int]], # N words × variable sub-tokens
    ):
        """
        Returns
        -------
        embeds : (seq_len, d_model)
        labels : (seq_len,) int64  — token IDs or -100
        """
        emb_table = self.frozen_llm.get_input_embeddings()
        device    = soft_tokens.device

        embed_parts: List[torch.Tensor] = []
        label_parts: List[torch.Tensor] = []

        for i, tok_ids in enumerate(word_token_ids):
            # Soft token(s) for word i
            soft = soft_tokens[i]                                   # (n_soft, d_model)
            embed_parts.append(soft)
            label_parts.append(
                torch.full((self.n_soft,), -100, dtype=torch.long, device=device)
            )

            # Text token embeddings for word i
            ids = torch.tensor(tok_ids, dtype=torch.long, device=device)
            embed_parts.append(emb_table(ids))                     # (n_toks, d_model)
            label_parts.append(ids)

        return torch.cat(embed_parts, dim=0), torch.cat(label_parts, dim=0)

    def _build_upfront(
        self,
        soft_tokens:    torch.Tensor,    # (N, n_soft, d_model)
        word_token_ids: List[List[int]],
    ):
        """
        Returns
        -------
        embeds : (seq_len, d_model)
        labels : (seq_len,) int64  — -100 for all soft-token positions
        """
        emb_table = self.frozen_llm.get_input_embeddings()
        device    = soft_tokens.device
        N         = soft_tokens.shape[0]

        # All soft tokens first
        all_soft   = soft_tokens.view(N * self.n_soft, self.d_model)
        soft_labels = torch.full(
            (N * self.n_soft,), -100, dtype=torch.long, device=device
        )

        # All text tokens concatenated in word order
        all_ids = [tid for tok_ids in word_token_ids for tid in tok_ids]
        ids     = torch.tensor(all_ids, dtype=torch.long, device=device)

        embeds = torch.cat([all_soft, emb_table(ids)], dim=0)
        labels = torch.cat([soft_labels, ids], dim=0)
        return embeds, labels

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        meg_windows:    torch.Tensor,       # (B, N, C, T)
        valid_mask:     torch.Tensor,       # (B, N) bool
        word_token_ids: List[List[List[int]]],  # B × N × variable
    ) -> CausalLMOutputWithCrossAttentions:
        """
        Returns a HuggingFace CausalLMOutput.  Access .loss for backprop
        and .logits (B, seq_len, vocab_size) for evaluation.
        """
        B = meg_windows.shape[0]
        soft_tokens = self._encode_meg(meg_windows, valid_mask)  # (B, N, n_soft, d_model)

        builder = (
            self._build_interleaved
            if self.sequence_design == "interleaved"
            else self._build_upfront
        )

        # Build per-trial sequences then pad to a common length
        all_embeds: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        for b in range(B):
            emb_b, lab_b = builder(soft_tokens[b], word_token_ids[b])
            all_embeds.append(emb_b)
            all_labels.append(lab_b)

        max_len = max(e.shape[0] for e in all_embeds)
        device  = meg_windows.device

        inputs_embeds  = torch.zeros(B, max_len, self.d_model, device=device)
        labels_padded  = torch.full((B, max_len), -100, dtype=torch.long, device=device)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)

        for b in range(B):
            L = all_embeds[b].shape[0]
            inputs_embeds[b, :L]  = all_embeds[b]
            labels_padded[b, :L]  = all_labels[b]
            attention_mask[b, :L] = 1

        return self.frozen_llm(
            inputs_embeds=inputs_embeds,
            labels=labels_padded,
            attention_mask=attention_mask,
        )

    # ------------------------------------------------------------------
    #  Convenience helpers
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """Returns only the parameters that should receive gradient updates."""
        params = list(self.adapter.parameters())
        if not self.meg_encoder._frozen:
            params += list(self.meg_encoder.parameters())
        return params

    def save_adapter(self, path: str) -> None:
        torch.save(self.adapter.state_dict(), path)

    def load_adapter(self, path: str, device: torch.device) -> None:
        self.adapter.load_state_dict(
            torch.load(path, map_location=device)
        )


# =============================================================================
#  FACTORY
# =============================================================================

def build_model(
    device:          torch.device,
    llm_name:        str  = LLM_NAME,
    freeze_encoder:  bool = True,
    sequence_design: str  = SEQUENCE_DESIGN,
) -> LLMDecoder:
    """
    Load all components and return a ready-to-train LLMDecoder.

    Components are loaded on CPU then moved to device together, which avoids
    double-allocating GPU memory during checkpoint loading.

    Parameters
    ----------
    device          : torch.device to place the full model on
    llm_name        : HuggingFace model ID (default "gpt2")
    freeze_encoder  : if True (Option A), MEG encoder weights are frozen;
                      set False for Option B joint training
    sequence_design : "interleaved" or "upfront"
    """
    print(f"\nBuilding LLMDecoder  (design={sequence_design!r}, device={device})")

    # 1. MEG encoder
    encoder = load_pretrained(freeze=freeze_encoder)      # returned on CPU

    # 2. Adapter
    adapter = build_adapter()

    # 3. Frozen LLM
    print(f"  Loading LLM: {llm_name} ...")
    llm        = AutoModelForCausalLM.from_pretrained(llm_name)
    frozen_llm = _FrozenLLM(llm)
    n_llm      = sum(p.numel() for p in frozen_llm.parameters())
    print(f"  LLM: {n_llm:,} params (frozen)")

    # 4. Assemble and move to device
    model = LLMDecoder(encoder, adapter, frozen_llm, sequence_design)
    model.to(device)

    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,} params total\n")

    return model
