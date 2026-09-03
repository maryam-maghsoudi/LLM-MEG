"""
pooling.py — word-level attention-pooling module (§4/§10, item 1).

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

REQUIRED COMPANION EDIT (not made here): new_models.py does not currently
export a TOTAL_STRIDE constant — add `TOTAL_STRIDE = 2` next to JOINT_DIM
there so the try/except import below actually succeeds instead of
silently falling back to a hardcoded duplicate. Same risk pattern as
POEM_LINES / GPT2_SWEEP_LAYERS duplication elsewhere in this project.

RAW-SAMPLE -> ENCODER-FRAME CONVERSION IS ARCHITECTURE-SPECIFIC: the
formula in _raw_length_to_encoder_frames only matches MEGEncoder's
SPECIFIC stride-2 block (kernel=7, dilation=1, causal left-pad=6). If that
block's kernel/dilation/stride ever changes, this formula must be
recomputed to match — ideally this becomes a method on MEGEncoder itself
(single source of truth) rather than living here.

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
    SFREQ_DS = 100.0  # keep in sync with new_dataset.py

try:
    from .new_models import TOTAL_STRIDE
except ImportError:
    TOTAL_STRIDE = 2  # keep in sync with new_models.py's MEGEncoder stride budget


def raw_length_to_encoder_frames(raw_length: torch.Tensor, total_stride: int) -> torch.Tensor:
    """
    T_out = floor((T_raw - 1) / total_stride) + 1 for T_raw > 0 — matches
    MEGEncoder's SPECIFIC stride-2 block (kernel=7, dilation=1, causal
    left-pad=6). Architecture-specific — see module docstring above.

    Module-level (not just a WordAttentionPooling method) so OTHER callers
    — train.py, aligning audio_target to the same real-encoder-frame
    boundary — reuse this exact formula instead of re-deriving it a third
    time (see new_models.py's TOTAL_STRIDE docstring for the same concern).
    """
    raw_length = raw_length.clamp(min=0)
    floor_term = torch.div(raw_length - 1, total_stride, rounding_mode="floor") + 1
    return torch.where(raw_length > 0, floor_term, torch.zeros_like(raw_length))


def exact_slice_pooling(z: torch.Tensor, onset_samples: torch.Tensor, offset_samples: torch.Tensor,
                         total_stride: int = TOTAL_STRIDE):
    """
    SETTING A — the simple version. Assumes onset AND offset are exactly
    right, and just averages every encoder frame between them, equally.
    No learned parameters, no window, no jitter — nothing to tune.

    Use this as a clean reference point against WordAttentionPooling
    (SETTING B, below): if the answer barely changes between the two, the
    extra machinery in Setting B isn't buying much. If it changes a lot,
    that's a real measurement of what the wide window + learned weighting
    + jitter are actually contributing.

    z              : (B, T_out, D) — MEGEncoder's dense output
    onset_samples  : (B, N) long, RAW sample onset per word; -1 = invalid
    offset_samples : (B, N) long, RAW sample offset per word; -1 = invalid

    Returns pooled (B, N, D), valid (B, N) bool.

    No padding-awareness needed here (unlike Setting B): dataset.py
    already guarantees onset/offset fall inside that trial's OWN real
    length before any batching/padding happens, and this function never
    looks past onset/offset themselves — so there's nothing left that
    could reach into padded (fake) frames. Setting B's padding risk comes
    specifically from adding a margin BEYOND the validated onset; this
    function adds no margin at all.
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
    SETTING B — the wide-window, learned version. See module docstring
    for design rationale, and exact_slice_pooling (Setting A, above) for
    the simple alternative this is being compared against.

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

        # Single, global learned query — deliberately NOT multi-head and
        # NOT preceded by key/value projections: with ~140 word instances
        # total, extra attention-pooling parameters are capacity the data
        # can't support fitting reliably. Query attends directly to raw
        # encoder features; pooled output is a weighted sum of those same
        # raw features (no separate value projection). Easy to add
        # key/value projections later if this empirically underfits —
        # deliberately left out for now, not an oversight.
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
        pool_valid : (B, N) bool — see class docstring.
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
        # An all-invalid word produces an all -inf logit row -> NaN
        # softmax. Substitute a neutral row there so gradients for OTHER,
        # valid words in the same batch stay clean; the pooled output for
        # an invalid word is zeroed explicitly below regardless.
        attn_logits = torch.where(pool_valid.unsqueeze(-1), attn_logits, torch.zeros_like(attn_logits))

        attn   = self.dropout(torch.softmax(attn_logits, dim=-1))   # (B, N, W)
        pooled = (attn.unsqueeze(-1) * window).sum(dim=2)            # (B, N, D)
        pooled = pooled * pool_valid.unsqueeze(-1)

        return pooled, pool_valid


def pool_words(mode, z, onset_samples, offset_samples=None, trial_mask=None,
                attention_module=None, jitter_ms=None):
    """
    One entry point for both settings, so callers pick a mode with one
    flag instead of branching everywhere.

    mode="exact" : Setting A (exact_slice_pooling) — needs offset_samples
    mode="wide"  : Setting B (WordAttentionPooling) — needs trial_mask
                   and an already-constructed attention_module instance
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
    # 1. _raw_length_to_encoder_frames — hand-derived values, per the
    #    module docstring's formula (architecture-specific to MEGEncoder's
    #    stride-2 block: kernel=7, dilation=1).
    # ------------------------------------------------------------------
    pool = WordAttentionPooling(backbone_dim=8, sfreq_raw=100.0, total_stride=2, dropout=0.0)
    test_lengths = torch.tensor([0, 1, 2, 3, 7, 63, 100])
    expected     = torch.tensor([0, 1, 1, 2, 4, 32, 50])
    got = pool._raw_length_to_encoder_frames(test_lengths)
    assert torch.equal(got, expected), f"length formula mismatch: got {got.tolist()}, expected {expected.tolist()}"
    print(f"[OK] _raw_length_to_encoder_frames  {test_lengths.tolist()} -> {got.tolist()}")

    # ------------------------------------------------------------------
    # 2. Parameter count — should be exactly backbone_dim (just the query
    #    vector; no key/value projections, see class docstring).
    # ------------------------------------------------------------------
    n_params = sum(p.numel() for p in pool.parameters())
    assert n_params == pool.backbone_dim, (
        f"expected exactly backbone_dim={pool.backbone_dim} trainable params, got {n_params}"
    )
    print(f"[OK] parameter count = {n_params} (== backbone_dim, as intended)")

    # ------------------------------------------------------------------
    # 3. Shapes, sentinel handling, and — the important one — padding
    #    leakage: trial 1 is shorter than trial 0 within the same batch,
    #    its padded region is set to an extreme value (1000.0), and a
    #    word's window near trial 1's real/padded boundary is deliberately
    #    wide enough to nominally overrun into it. The pooled output must
    #    NOT show any trace of the 1000.0 padding.
    # ------------------------------------------------------------------
    print("\n=== padding-awareness test ===")
    D = 8
    pool2 = WordAttentionPooling(backbone_dim=D, window_pre_ms=200, window_post_ms=600,
                                  sfreq_raw=100.0, total_stride=2, dropout=0.0)
    pool2.eval()  # deterministic: no dropout, no jitter

    B, T_raw_max, T_out = 2, 60, 30   # T_out = floor((60-1)/2)+1 = 30
    z = torch.randn(B, T_out, D) * 0.1
    z[1, 20:, :] = 1000.0             # trial 1's PADDED region (frames 20..29) — must never be pooled

    trial_mask = torch.zeros(B, T_raw_max, dtype=torch.bool)
    trial_mask[0, :60] = True   # trial 0: fully real
    trial_mask[1, :40] = True   # trial 1: real only up to raw sample 40 (== encoder frame 20)

    onset_samples = torch.tensor([
        [10, -1],   # trial 0: one valid word, one sentinel-invalid word
        [35,  5],   # trial 1: word whose wide window nominally overruns into padding; one normal word
    ])

    pooled, pool_valid = pool2(z, onset_samples, trial_mask)
    assert pooled.shape == (B, 2, D)
    assert pool_valid.tolist() == [[True, False], [True, True]], f"unexpected pool_valid: {pool_valid.tolist()}"
    assert torch.all(pooled[0, 1] == 0), "sentinel-invalid (-1) word must pool to exactly zero"

    leaked = pooled[1, 0].abs().max().item()
    assert leaked < 5.0, f"padding leaked into pooled output! max |value| = {leaked:.3f} (padding was 1000.0)"
    print(f"[OK] shapes correct, sentinel word is exactly zero, "
          f"near-boundary word max |pooled value| = {leaked:.3f} (no leakage from 1000.0 padding)")

    # ------------------------------------------------------------------
    # 4. Gradients flow cleanly to both z and the learned query.
    # ------------------------------------------------------------------
    print("\n=== gradient flow test ===")
    z2 = (torch.randn(B, T_out, D) * 0.1).requires_grad_(True)
    pool2.train()
    pooled2, _ = pool2(z2, onset_samples, trial_mask, jitter_ms=None)
    pooled2.sum().backward()
    assert z2.grad is not None and torch.isfinite(z2.grad).all(), "gradient did not flow cleanly back to z"
    assert pool2.query.grad is not None and torch.isfinite(pool2.query.grad).all()
    print("[OK] gradients flow to both z and the learned query, no NaN/Inf")

    # ------------------------------------------------------------------
    # 5. Onset jitter: doesn't mutate caller's tensor in place, and is
    #    correctly disabled in eval() mode (deterministic output).
    # ------------------------------------------------------------------
    print("\n=== onset jitter test ===")
    pool3 = WordAttentionPooling(backbone_dim=D, sfreq_raw=100.0, total_stride=2)
    onset_before = onset_samples.clone()
    pool3.train()
    pool3(z, onset_samples, trial_mask, jitter_ms=(50, 150))
    assert torch.equal(onset_samples, onset_before), "jitter must not mutate the caller's onset_samples in place"

    pool3.eval()
    out_a, _ = pool3(z, onset_samples, trial_mask, jitter_ms=(50, 150))
    out_b, _ = pool3(z, onset_samples, trial_mask, jitter_ms=(50, 150))
    assert torch.equal(out_a, out_b), "jitter_ms must be ignored in eval() mode — output should be deterministic"
    print("[OK] jitter doesn't mutate input in place; eval() mode ignores jitter_ms and is deterministic")

    # ------------------------------------------------------------------
    # 6. exact_slice_pooling (Setting A) — correctness against a
    #    hand-computed average, plus the direct answer to "do we need
    #    the learned vector in both settings?": compare trainable
    #    parameter counts between Setting A and Setting B directly.
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

    # ------------------------------------------------------------------
    # 7. pool_words dispatcher — both modes reachable from one call.
    # ------------------------------------------------------------------
    print("\n=== pool_words dispatcher ===")
    pooled_a, valid_a = pool_words("exact", z_ex, onset_ex, offset_samples=offset_ex)
    assert torch.equal(pooled_a, pooled_ex) and torch.equal(valid_a, valid_ex)

    fake_attn = WordAttentionPooling(backbone_dim=4)
    fake_attn.eval()
    fake_trial_mask = torch.ones(1, 20, dtype=torch.bool)
    pooled_b, valid_b = pool_words("wide", z_ex, onset_ex, trial_mask=fake_trial_mask, attention_module=fake_attn)
    assert pooled_b.shape == (1, 2, 4)
    print("[OK] pool_words correctly dispatches to both modes")

    print("\n=== ALL CHECKS PASSED ===")
