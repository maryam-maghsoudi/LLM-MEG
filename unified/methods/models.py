"""
models.py — neural network modules shared across all three methods.

MEGEncoder          Spatial-temporal CNN (mirrors MEGWordEncoderSmall).
                    Input (B, C, T) → output (B, 128) L2-normalized.

BERTTextProjection  Maps frozen BERT hidden states → 128-d MEG embedding space.
                    Used by Method 1 (InfoNCE against BERT).

LLMTextProjection   Maps frozen LLM hidden states → 128-d MEG embedding space.
                    Used by Method 2 Stage 1 (InfoNCE against LLM).

GRUHead             Causal GRU for Method 2 Stage 2: reads z_t in order and
                    outputs y_t ∈ R^d_model for the frozen lm_head.

Adapter             Trainable MLP for Method 3: maps 128-d MEG embedding to
                    n_soft soft tokens in the LLM's embedding space.

load_lm_head        Extract frozen lm_head from a pretrained causal LM.
load_bert           Load frozen BERT and compute per-poem contextual embeddings.
"""

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
#  Defaults
# ---------------------------------------------------------------------------
N_CHANNELS = 155
WIN_SIZE   = 40      # [-100 ms, +300 ms] @ 100 Hz
MEG_EMB    = 128


# ===========================================================================
#  MEGEncoder  (shared across all three methods)
# ===========================================================================

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation=1, dropout=0.3):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation,
                      padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class MEGEncoder(nn.Module):
    """
    Spatial-temporal conv encoder (matches MEGWordEncoderSmall).
    Input : (B, C=155, T=40)
    Output: (B, 128)  L2-normalized
    """

    def __init__(self, n_channels=N_CHANNELS, win_size=WIN_SIZE,
                 emb_dim=MEG_EMB, dropout=0.3):
        super().__init__()
        self._frozen  = False
        self.emb_dim  = emb_dim
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

    def forward(self, x):          # x: (B, C, T)
        x = self.spatial(x)
        x = self.temporal(x)
        x = self.pool(x).squeeze(-1)
        return F.normalize(self.proj(x), dim=-1)


# ===========================================================================
#  Text projections  (one per method)
# ===========================================================================

class BERTTextProjection(nn.Module):
    """
    Method 1: maps BERT last-layer hidden states (768-d) → MEG embedding space.
    Architecture mirrors the text projection head from contrastive_word_meg.py.
    Output is L2-normalized to match MEGEncoder output.
    Trained alongside MEGEncoder during Method 1; discarded after training.
    """

    def __init__(self, bert_dim: int = 768, emb_dim: int = MEG_EMB,
                 dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, emb_dim),
        )

    def forward(self, h):          # (B, 768) → (B, 128)
        return F.normalize(self.proj(h), dim=-1)


class LLMTextProjection(nn.Module):
    """
    Method 2 Stage 1: maps LLM hidden states (d_model) → MEG embedding space.
    Output is L2-normalized. Discarded after Stage 1.
    """

    def __init__(self, d_model: int, emb_dim: int = MEG_EMB, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, emb_dim),
        )

    def forward(self, h):          # (B, d_model) → (B, 128)
        return F.normalize(self.proj(h), dim=-1)


# ===========================================================================
#  GRUHead  (Method 2 Stage 2)
# ===========================================================================

class GRUHead(nn.Module):
    """
    Causal GRU that reads MEG embeddings z_1..z_T in temporal order and
    outputs predicted next-word states in the LLM's hidden dimension.

    Forward  z: (B, T, meg_emb) → y: (B, T, d_model)
    The output is passed through the frozen lm_head to get q_t logits.
    h_0 is always zeros (state resets at the start of every trial).
    """

    def __init__(self, meg_emb: int = MEG_EMB, gru_hidden: int = 256,
                 d_model: int = 960, dropout: float = 0.3):
        super().__init__()
        self.gru  = nn.GRU(meg_emb, gru_hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(gru_hidden, d_model)

    def forward(self, z):          # (B, T, meg_emb)
        h, _ = self.gru(z)
        return self.proj(self.drop(h))   # (B, T, d_model)


# ===========================================================================
#  Adapter  (Method 3 — interleaved soft tokens)
# ===========================================================================

class Adapter(nn.Module):
    """
    Trainable MLP that maps a single MEG embedding (128-d) to n_soft soft
    tokens in the LLM's input embedding space (d_model-d each).

    Forward  z: (B, 128) → (B, n_soft, d_model)

    The soft tokens are injected into the LLM sequence immediately before the
    corresponding word's text tokens (interleaved design).
    """

    def __init__(self, meg_emb: int = MEG_EMB, n_soft: int = 1,
                 d_model: int = 768, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.n_soft  = n_soft
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(meg_emb, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_soft * d_model),
        )

    def forward(self, z):          # (B, 128) → (B, n_soft, d_model)
        B = z.shape[0]
        return self.mlp(z).view(B, self.n_soft, self.d_model)


# ===========================================================================
#  Loading helpers
# ===========================================================================

def load_lm_head(llm_name: str, device: torch.device) -> nn.Linear:
    """
    Load only the lm_head Linear layer from a pretrained HF causal LM.
    All other weights are discarded. The head is frozen.
    """
    from transformers import AutoModelForCausalLM
    print(f"Loading lm_head from {llm_name} ...")
    full = AutoModelForCausalLM.from_pretrained(llm_name)
    head = copy.deepcopy(full.lm_head)
    del full
    torch.cuda.empty_cache()
    for p in head.parameters():
        p.requires_grad_(False)
    head = head.to(device)
    n = sum(p.numel() for p in head.parameters())
    print(f"  lm_head  {n:,} params  frozen  device={device}")
    return head


def load_bert_hiddens(
    onset_dir: Path,
    bert_name: str = "bert-base-uncased",
    device:    str = "cpu",
    layer:     int = -1,          # which BERT layer to use (-1 = last)
) -> Dict[str, torch.Tensor]:
    """
    Run frozen BERT once per poem and return mean-pooled hidden states
    at the specified layer for every word position.

    Returns
    -------
    {poem: Tensor(N_words, 768)}  — one vector per word, same for all subjects/sessions
    """
    from transformers import AutoModel, AutoTokenizer
    import json

    print(f"Computing BERT hidden states (model={bert_name}, layer={layer}) ...")
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    model     = AutoModel.from_pretrained(bert_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hiddens: Dict[str, torch.Tensor] = {}

    for poem in ["poem1", "poem2"]:
        onsets = json.loads((onset_dir / f"{poem}_word_onsets.json").read_text())
        words  = [e["word"].strip().lower() for e in onsets]

        # Build subword alignment with leading space (mid-sentence BPE)
        all_ids: List[int] = []
        spans:   List[Tuple[int, int]] = []
        for word in words:
            ids = tokenizer.encode(" " + word, add_special_tokens=False)
            if not ids:
                ids = tokenizer.encode(word, add_special_tokens=False)
            if not ids:
                ids = [tokenizer.unk_token_id or 0]
            start = len(all_ids)
            all_ids.extend(ids)
            spans.append((start, len(all_ids)))

        # BERT expects [CLS] tokens [SEP]
        cls_id  = tokenizer.cls_token_id
        sep_id  = tokenizer.sep_token_id
        ids_t   = torch.tensor([cls_id] + all_ids + [sep_id],
                               dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(ids_t, output_hidden_states=True)

        # hidden_states: tuple of (1, T+2, 768), length n_layers+1
        hs = out.hidden_states[layer][0]   # (T+2, 768), +2 for CLS/SEP

        # Mean-pool over each word's subword span (offset +1 for CLS)
        pooled = []
        for s, e in spans:
            h = hs[s + 1 : e + 1].mean(dim=0)
            pooled.append(h.cpu())

        hiddens[poem] = torch.stack(pooled)   # (N_words, 768)
        print(f"  {poem}: {hiddens[poem].shape}")

    del model
    torch.cuda.empty_cache()
    return hiddens


def load_meg_encoder(ckpt_path: Optional[str], device: torch.device,
                     freeze: bool = True) -> MEGEncoder:
    """Load MEGEncoder from checkpoint, optionally freeze."""
    enc = MEGEncoder()
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu")
        # checkpoint may be a dict wrapping the state dict
        if "meg_encoder" in state:
            state = state["meg_encoder"]
        enc.load_state_dict(state)
        print(f"MEGEncoder loaded from {ckpt_path}")
    else:
        print("MEGEncoder initialised from random weights")
    if freeze:
        enc.freeze()
    enc = enc.to(device)
    n = sum(p.numel() for p in enc.parameters())
    status = "frozen" if freeze else "trainable"
    print(f"  MEGEncoder  {n:,} params  {status}")
    return enc
