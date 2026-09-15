"""
pooling.py — word-level attention-pooling module .

Converts MEGEncoder's dense (B, T_out, D) continuous output into one
backbone_dim-d vector PER WORD, using a WIDE window of encoder frames
around each word's onset and a LEARNED (single, global) query that
attends over that window — not a hard slice the way the old fixed
[-100,+300]ms design worked.

Why attention, not a hard slice, and why WIDE:
  - Onset jitter (§6): training perturbs listening onsets by +-50-150ms so
    this module learns tolerance to timing error. That tolerance is only
    possible if the window is wide enough to still contain the true word
    span after perturbation, and soft (learned) weighting lets the model
    downweight frames it's unsure about rather than being forced to use
    exactly what a fixed slice happened to contain.
  - The imagined condition's onset estimates (§9) are inherently noisier
    (duration-scaled priors / entrainment-phase extraction) than the
    forced-aligned listening onsets — this must already be robust to
    timing error before the imagined branch can be trusted downstream.

WINDOW WIDTH IS A TUNABLE ASSUMPTION, NOT A SPEC'D VALUE: the design spec
only says "wide window" qualitatively. Defaults below (200ms pre / 600ms
post, asymmetric — same ~1:3 ratio as the old 100/300ms window, doubled)
are a reasonable starting point, not something to trust without sweeping.

PADDING-AWARENESS: dataset.py's collate_continuous_trials right-pads
variable-length trials to a shared T_max and returns trial_mask (raw-
sample resolution, True = real). This module converts trial_mask to a
per-trial REAL encoder-frame count and uses THAT (not the batch's global
T_out) as each trial's pooling boundary — using the global T_out alone
would silently let a short trial's pooling window read PADDING zeros as
if they were real signal.
"""

import torch
import torch.nn as nn

try:
    from .new_dataset import SFREQ_DS
except ImportError:
    SFREQ_DS = 100.0  # Sampling frequency

try:
    from .new_models import TOTAL_STRIDE
except ImportError:
    TOTAL_STRIDE = 2  #  MEGEncoder


def raw_length_to_encoder_frames(raw_length: torch.Tensor, total_stride: int) -> torch.Tensor:
    """
    T_out = floor((T_raw - 1) / total_stride) + 1 for T_raw > 0 — matches
    MEGEncoder's SPECIFIC stride-2 block (kernel=7, dilation=1, causal
    left-pad=6). Architecture-specific.


    """
    raw_length = raw_length.clamp(min=0)
    floor_term = torch.div(raw_length - 1, total_stride, rounding_mode="floor") + 1
    return torch.where(raw_length > 0, floor_term, torch.zeros_like(raw_length))


def exact_slice_pooling(z: torch.Tensor, onset_samples: torch.Tensor, offset_samples: torch.Tensor,
                         total_stride: int = TOTAL_STRIDE):
    """
    z              : (B, T_out, D) — MEGEncoder's dense output
    onset_samples  : (B, N) long, RAW sample onset per word; -1 = invalid
    offset_samples : (B, N) long, RAW sample offset per word; -1 = invalid

    Returns pooled (B, N, D), valid (B, N) bool.

    """
    B, T_out, D = z.shape
    N = onset_samples.shape[1]
    device = z.device

    onset_frame  = torch.div(onset_samples,  total_stride, rounding_mode="floor")
    offset_frame = torch.div(offset_samples, total_stride, rounding_mode="floor")
    has_span = (onset_samples >= 0) & (offset_samples >= 0) & (offset_frame > onset_frame)

    pooled = torch.zeros(B, N, D, device=device)
    valid  = torch.zeros(B, N, dtype=torch.bool, device=device)

    for b in range(B):
        for i in range(N):
            if not has_span[b, i]:
                continue
            s = int(onset_frame[b, i].item())
            e = min(int(offset_frame[b, i].item()), T_out)   # defensive only, see docstring — should never trigger
            if e <= s:
                continue
            pooled[b, i] = z[b, s:e].mean(dim=0)
            valid[b, i] = True

    return pooled, valid



class WordAttentionPooling(nn.Module):
    """
    the wide-window, learned version. 

    forward() combines TWO independent validity signals into pool_valid:
      1. onset_samples == -1 (dataset.py's own "word alignment failed" sentinel)
      2. the pooling window found zero real, in-bounds encoder frames
         (can happen even with a valid onset, e.g. right at a trial's edge)
    Callers should AND this with dataset.py's own valid_mask before use —
    that one is about whether the word's alignment existed at all; this
    one is about whether pooling actually had anything real to pool over.
    They are different failure modes.
    """

    def __init__(
        self,
        backbone_dim: int,
        window_pre_ms: float = 200.0,
        window_post_ms: float = 600.0,
        sfreq_raw: float = SFREQ_DS,
        total_stride: int = TOTAL_STRIDE,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone_dim = backbone_dim
        self.total_stride = total_stride
        self.sfreq_raw    = sfreq_raw
        sfreq_enc         = sfreq_raw / total_stride

        self.window_pre_frames  = int(round(window_pre_ms  / 1000.0 * sfreq_enc))
        self.window_post_frames = int(round(window_post_ms / 1000.0 * sfreq_enc))


        self.query   = nn.Parameter(torch.randn(backbone_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def _raw_length_to_encoder_frames(self, raw_length: torch.Tensor) -> torch.Tensor:
        return raw_length_to_encoder_frames(raw_length, self.total_stride)

    def forward(
        self,
        z: torch.Tensor,               # (B, T_out, D) — MEGEncoder's dense output
        onset_samples: torch.Tensor,   # (B, N) long, RAW sample indices (100Hz); -1 = invalid
        trial_mask: torch.Tensor,      # (B, T_raw) bool, RAW-resolution padding mask (True = real)
        jitter_ms: "tuple[float, float] | None" = None,
    ):
        """
        jitter_ms : optional (low, high) MAGNITUDE range in ms — e.g.
                    (50, 150) per §6. If given AND self.training is True,
                    each onset is shifted by a random +-magnitude in that
                    range before pooling. Ignored during eval() —
                    jitter is a training-time augmentation only, same
                    train/eval convention as Dropout. The -1 sentinel is
                    never jittered.

        Returns
        -------
        pooled     : (B, N, D)
        pool_valid : (B, N) bool .
        """
        device = z.device
        B, T_out, D = z.shape
        N = onset_samples.shape[1]

        if jitter_ms is not None and self.training:
            low_samp  = int(round(jitter_ms[0] / 1000.0 * self.sfreq_raw))
            high_samp = int(round(jitter_ms[1] / 1000.0 * self.sfreq_raw))
            magnitude = torch.randint(low_samp, high_samp + 1, onset_samples.shape, device=device)
            sign      = torch.randint(0, 2, onset_samples.shape, device=device) * 2 - 1
            jittered  = onset_samples + magnitude * sign
            onset_samples = torch.where(onset_samples >= 0, jittered, onset_samples)

        real_frames = self._raw_length_to_encoder_frames(trial_mask.sum(dim=1))   # (B,)

        has_onset = onset_samples >= 0                                            # (B, N)
        centers   = torch.div(onset_samples, self.total_stride, rounding_mode="floor")  # (B, N)

        offsets = torch.arange(-self.window_pre_frames, self.window_post_frames + 1, device=device)  # (W,)
        idx = centers.unsqueeze(-1) + offsets.view(1, 1, -1)                      # (B, N, W)

        in_bounds = (
            (idx >= 0)
            & (idx < real_frames.view(B, 1, 1))    # excludes padding AND genuine trial-end overrun
            & has_onset.unsqueeze(-1)
        )
        idx_clamped = idx.clamp(0, T_out - 1)       # safe for gather; masked out below regardless

        idx_expand = idx_clamped.unsqueeze(-1).expand(-1, -1, -1, D)   # (B, N, W, D)
        z_expand   = z.unsqueeze(1).expand(-1, N, -1, -1)               # (B, N, T_out, D), a view — no copy
        window     = torch.gather(z_expand, 2, idx_expand)              # (B, N, W, D)

        attn_logits = (window * self.query.view(1, 1, 1, D)).sum(-1) / (D ** 0.5)   # (B, N, W)
        attn_logits = attn_logits.masked_fill(~in_bounds, float("-inf"))

        pool_valid = in_bounds.any(dim=-1)   # (B, N) 
        attn_logits = torch.where(pool_valid.unsqueeze(-1), attn_logits, torch.zeros_like(attn_logits))

        attn   = self.dropout(torch.softmax(attn_logits, dim=-1))   # (B, N, W)
        pooled = (attn.unsqueeze(-1) * window).sum(dim=2)            # (B, N, D)
        pooled = pooled * pool_valid.unsqueeze(-1)

        return pooled, pool_valid


def pool_words(mode, z, onset_samples, offset_samples=None, trial_mask=None,
                attention_module=None, jitter_ms=None):
    """
    mode="exact" : Setting A (exact_slice_pooling) — needs offset_samples
    mode="wide"  : Setting B (WordAttentionPooling) — needs trial_mask, already-constructed attention_module instance
    """
    if mode == "exact":
        assert offset_samples is not None, "exact mode needs offset_samples"
        return exact_slice_pooling(z, onset_samples, offset_samples)
    elif mode == "wide":
        assert attention_module is not None, "wide mode needs a WordAttentionPooling instance"
        assert trial_mask is not None, "wide mode needs trial_mask"
        return attention_module(z, onset_samples, trial_mask, jitter_ms=jitter_ms)
    else:
        raise ValueError(f"unknown pooling mode: {mode!r}")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("=== pooling.py sanity check ===\n")

    # ------------------------------------------------------------------
    # 1. exact_slice_pooling — correctness against a
    #    hand-computed average, plus the direct answer to "do we need
    #    the learned vector in both settings?": compare trainable
    # ------------------------------------------------------------------
    print("\n=== exact_slice_pooling (Setting A) sanity check ===")
    z_ex = torch.randn(1, 20, 4)
    onset_ex  = torch.tensor([[6, -1]])
    offset_ex = torch.tensor([[14, -1]])
    pooled_ex, valid_ex = exact_slice_pooling(z_ex, onset_ex, offset_ex, total_stride=2)

    # by hand: onset_frame = 6//2 = 3, offset_frame = 14//2 = 7 -> average frames 3..6 (slice 3:7)
    expected_ex = z_ex[0, 3:7].mean(dim=0)
    assert torch.allclose(pooled_ex[0, 0], expected_ex, atol=1e-6), "doesn't match a hand-computed average"
    assert valid_ex.tolist() == [[True, False]], "the -1 sentinel word must come back invalid"
    assert torch.all(pooled_ex[0, 1] == 0), "invalid word must pool to exactly zero"
    print("[OK] exact_slice_pooling matches a hand-computed average; sentinel word correctly invalid")

    n_params_wide = sum(p.numel() for p in WordAttentionPooling(backbone_dim=4).parameters())
    print(f"[OK] trainable parameters — Setting A: 0 (plain function, nothing to learn)  "
          f"Setting B: {n_params_wide} (the query vector — one learned number per feature dimension)")


