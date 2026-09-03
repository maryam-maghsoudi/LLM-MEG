"""
new_models.py — neural network modules for the continuous-encoder MEG decoder

MEGEncoder            Causal, continuous spatial-temporal conv encoder (§4).
                      Replaces the old fixed-[-100,+300]ms-window design:
                      input is a whole trial, output is the full dense
                      (B, T_out, D) sequence — NO pooling inside the encoder.
SharedSpatialConv     Shared (not per-subject) 1x1 conv across MEG channels.
AudioProjectionHead   Dense, per-frame: applied directly to the encoder's
                      full sequence output. Projects to JOINT_DIM=128.
WordProjectionHead    Pooled, per-word: applied to per-word vectors produced
                      by the (separate, not-yet-written) pooling module.
                      Same architecture as AudioProjectionHead, different
                      input shape convention.
GRUHead               Causal GRU trunk + one small linear head per
                      injection-depth sweep layer L. Generalizes the old
                      single-final-layer Stage 2 design into the depth-sweep
                      architecture — shared trunk keeps trainable parameters
                      from multiplying by len(sweep_layers).
load_meg_encoder      Instantiate + optionally load pretrained MEGEncoder.
load_lm_head          Extract only lm_head from a pretrained HF causal LM —
                      sufficient for the L=final sweep point only.
load_frozen_gpt2      Load + freeze the FULL GPT2Model — needed for every
                      sweep point L < final (continue_forward_from_layer in
                      teacher_cache.py needs blocks[L:], ln_f, wte.weight).


"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
#  Defaults
# ---------------------------------------------------------------------------
N_CHANNELS = 155
JOINT_DIM  = 128  
TOTAL_STRIDE = 2


try:
    from .teacher_cache import GPT2_SWEEP_LAYERS
except ImportError:
    GPT2_SWEEP_LAYERS = [0, 4, 8, 12]  # same as teacher_cache.py


# ===========================================================================
#  MEGEncoder — causal, continuous, no internal pooling
# ===========================================================================

class _ConvBlock(nn.Module):
    """
    One causal (left-only-padded) temporal conv block. Padding
    (kernel-1)*dilation is applied ONLY on the left, so output position t
    depends only on input positions <= t, never future ones. This is the
    key difference from the old block: nn.Conv1d(padding=...) pads
    symmetrically, which lets output t see (kernel-1)*dilation//2 samples
    of FUTURE input — silently non-causal.
    """

    def __init__(self, in_ch, out_ch, kernel, dilation=1, stride=1, dropout=0.3):
        super().__init__()
        self.left_pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                               dilation=dilation, padding=0, bias=False)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.pad(x, (self.left_pad, 0))   # left-only pad along time
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return self.drop(x)


class SharedSpatialConv(nn.Module):
    """
    Shared (not per-subject) 1x1 conv across MEG channels — mixes channels
    at each timestep independently. Kernel size 1 has no temporal
    receptive field, so there's nothing to make causal here.

    Per-subject spatial attention is deferred (§4): mechanically
    incompatible with strict LOSO without few-shot calibration per
    held-out subject. Revisit later.
    """

    def __init__(self, n_channels, out_ch=32):
        super().__init__()
        self.conv = nn.Conv1d(n_channels, out_ch, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act  = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MEGEncoder(nn.Module):
    """
    Continuous, causal spatial-temporal conv encoder. Runs ONCE over the
    whole trial and does NOT pool — word-level extraction happens
    downstream, via a separate pooling module operating on this output
    plus dataset.py's onset_samples / offset_samples.

    Input : (B, C, T)      C=155, T = full trial length (variable per trial)
    Output: (B, T_out, D)  T_out ~= T/2 (one stride-2 block below), D=128.
             NOT L2-normalized here — normalization now happens per-head
             (AudioProjectionHead / WordProjectionHead) instead, since the
             two heads serve different losses and shouldn't be forced to
             share a normalization decision made before either target is
             pooled/resampled.

    Stride budget: total stride = 2 across the whole stack (one stride-2
    block, landing 100Hz -> ~50Hz, matching wav2vec2's native frame rate),
    with receptive-field growth coming from dilation (1, 2, 4, 8) instead
    of further downsampling..


    """

    def __init__(self, n_channels=N_CHANNELS, backbone_dim=128, dropout=0.3):
        super().__init__()
        self._frozen = False
        self.backbone_dim = backbone_dim

        self.spatial  = SharedSpatialConv(n_channels, out_ch=32)
        self.temporal = nn.Sequential(
            _ConvBlock(32,  64,           kernel=7, dilation=1, stride=2, dropout=dropout),  # 100Hz -> ~50Hz
            _ConvBlock(64,  128,          kernel=5, dilation=2, stride=1, dropout=dropout),
            _ConvBlock(128, 128,          kernel=3, dilation=4, stride=1, dropout=dropout),
            _ConvBlock(128, backbone_dim, kernel=3, dilation=8, stride=1, dropout=dropout),
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
        # x: (B, C, T)
        x = self.spatial(x)        # (B, 32, T)
        x = self.temporal(x)       # (B, backbone_dim, T_out)
        return x.transpose(1, 2)   # (B, T_out, backbone_dim) — (B, T, D)
                                    # convention matching dataset.py / the
                                    # pooling module.


def load_meg_encoder(ckpt_path=None, freeze=True) -> MEGEncoder:
    """
    Instantiate MEGEncoder, optionally loading pretrained contrastive weights.
    ckpt_path=None -> train from scratch.
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
#  Stage 1 — MEG-side projection heads (audio + word)
# ===========================================================================

class _ProjectionHead(nn.Module):
    """
    Shared implementation for AudioProjectionHead and WordProjectionHead —
    architecturally identical (Linear -> GELU -> Dropout -> Linear,
    L2-normalized output). They differ only in WHEN/WHAT they're applied
    to: AudioProjectionHead runs densely over every frame of the encoder's
    output; WordProjectionHead runs on pooled per-word vectors from the
    pooling module. nn.Linear broadcasts over leading dimensions
    automatically, so one forward() covers both (B, T, D) dense input and
    (B, N_words, D) pooled input.
    """

    def __init__(self, in_dim, joint_dim=JOINT_DIM, hidden=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, joint_dim),
        )

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


class AudioProjectionHead(_ProjectionHead):
    """Dense, per-frame: applied to the encoder's full (B, T, D) output directly."""
    pass


class WordProjectionHead(_ProjectionHead):
    """Pooled, per-word: applied to (B, N_words, D) vectors from the pooling module."""
    pass


# ===========================================================================
#  Stage 2 — causal GRU trunk + per-sweep-layer heads
# ===========================================================================

class GRUHead(nn.Module):
    """
    Causal GRU trunk + one small linear head per injection-depth sweep
    layer L. Generalizes the old single-final-layer design into the
    depth-sweep architecture we designed the KL-distillation experiment
    around: ONE shared GRU trunk, multiple small Linear(gru_hidden,
    d_model) heads branching off it — not a separate GRU per L — to keep
    trainable parameters from multiplying by len(sweep_layers) given how
    little data there is (~140 word instances across both poems), and to
    keep training budget matched across every L in the comparison.

    Forward  z: (B, T, joint_dim) -> {L: y_L of shape (B, T, d_model)
                                       for L in sweep_layers}

    y_L is the synthetic hidden state destined for
    continue_forward_from_layer() (teacher_cache.py) at layer L — this
    module only produces the MEG-side mapping; no GPT-2 computation
    happens here.

    h_0 is always zeros — hidden state resets at the start of every trial.
    """

    def __init__(self, joint_dim=JOINT_DIM, gru_hidden=256, d_model=768,
                 sweep_layers=None, dropout=0.3):
        super().__init__()
        if sweep_layers is None:
            sweep_layers = GPT2_SWEEP_LAYERS
        self.sweep_layers = sweep_layers

        self.gru  = nn.GRU(joint_dim, gru_hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({
            str(L): nn.Linear(gru_hidden, d_model) for L in sweep_layers
        })

    def forward(self, z):   # z: (B, T, joint_dim)
        h, _ = self.gru(z)             # (B, T, gru_hidden)
        h = self.drop(h)
        return {L: self.heads[str(L)](h) for L in self.sweep_layers}


def load_lm_head(llm_name: str, device) -> nn.Linear:
    """
    Load only the lm_head Linear layer from a pretrained HF causal LM.
    The rest of the model is discarded after extraction.

    Sufficient ONLY for the old L=final sweep point (skip straight to
    lm_head, no frozen blocks touched). Every other sweep point needs
    load_frozen_gpt2() instead.
    """
    import copy
    from transformers import AutoModelForCausalLM
    print(f"Loading lm_head from {llm_name} ...")
    full_model = AutoModelForCausalLM.from_pretrained(llm_name)
    lm_head = copy.deepcopy(full_model.lm_head)   # deepcopy 
    del full_model
    torch.cuda.empty_cache()
    for p in lm_head.parameters():
        p.requires_grad_(False)
    lm_head = lm_head.to(device)
    n = sum(p.numel() for p in lm_head.parameters())
    print(f"lm_head extracted  {n:,} params  frozen  -> device={device}")
    return lm_head


def load_frozen_gpt2(llm_name: str = "gpt2", device="cpu"):
    """
    Load and freeze the FULL GPT2Model (not just lm_head) — needed for
    every sweep layer L < final, since continue_forward_from_layer()
    (teacher_cache.py) needs blocks[L:], ln_f, and wte.weight all
    available at train/eval time.
    """
    from transformers import GPT2Model
    print(f"Loading full frozen GPT-2 from {llm_name} ...")
    lm = GPT2Model.from_pretrained(llm_name).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    lm = lm.to(device)
    n = sum(p.numel() for p in lm.parameters())
    print(f"GPT2Model  ({llm_name})  {n:,} params  frozen  -> device={device}")
    return lm
