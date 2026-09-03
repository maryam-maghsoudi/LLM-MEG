"""
losses.py — Stage 1 contrastive losses (§6).

multi_positive_contrastive_loss  General, reusable SupCon-style loss:
                                  positives are determined by LABEL equality,
                                  not index equality ("clean diagonal"). Zero
                                  learnable parameters.
llm_contrastive_loss              Thin wrapper: builds (poem_id, word_pos)
                                  labels automatically, calls the function
                                  above. This is the piece §6 explicitly
                                  requires ("flatten (batch, word) into one
                                  pool, treat every same-(poem, word_pos)
                                  pair across the batch as a positive").
audio_contrastive_loss            Simpler diagonal-only InfoNCE for the
                                  dense per-frame audio target. See the
                                  caveat in its docstring — the SAME
                                  duplicate-target concern applies here in
                                  principle, but the design spec only
                                  explicitly calls for fixing it at the
                                  word level.
stage1_anneal_weights /
stage1_loss                       Linear crossfade of alpha (audio) / beta
                                  (LLM) weights over ~15 epochs (§6).

WHY THE FIX MATTERS (concretely, not just in the abstract): h_mid targets
in teacher_cache.py depend ONLY on (poem, word_pos) — not on subject or
session. So two different trials' occurrences of the same word carry
IDENTICAL target vectors. Under plain diagonal InfoNCE, if both land in
the same batch, one gets treated as the true positive for its own anchor
while the OTHER gets treated as a hard NEGATIVE for that same anchor —
demanding the anchor be simultaneously similar and dissimilar to a vector
that is numerically identical either way. Label-based positives (this
file) instead average over every batch entry sharing the same
(poem, word_pos), so a duplicate is correctly reinforcing, never
contradictory.

NOT SPEC'D, ADDED AS A REASONABLE DEFAULT — flagged so it isn't silently
assumed: symmetric=True (also computing the reverse, key-as-anchor
direction and averaging, CLIP-style). The design spec doesn't mention this
either way; it's cheap, standard, and toggleable off if unwanted. Also not
spec'd: the anneal curve shape (linear here) — "crossfading over ~15
epochs" doesn't specify linear vs. cosine vs. anything else; linear is the
simplest faithful reading.
"""

import torch
import torch.nn.functional as F


def multi_positive_contrastive_loss(
    anchors: torch.Tensor,      # (B, N, D)
    keys: torch.Tensor,         # (B, N, D)
    labels: torch.Tensor,       # (B, N) long — entries sharing a label are mutual positives
    valid_mask: "torch.Tensor | None" = None,   # (B, N) bool
    temperature: float = 0.1,
    symmetric: bool = True,
) -> torch.Tensor:
    """
    See module docstring for the exact problem this solves. anchors[i] and
    keys[i] should be the "natural" pairing (e.g. one MEG embedding and its
    own occurrence's LLM target) — labels then determine ALL positives,
    including but not limited to that natural pairing.

    Both anchors and keys are re-normalized internally (L2, cosine
    similarity), regardless of whether the caller already normalized them
    — safe/idempotent if they did (e.g. WordProjectionHead's output),
    necessary if they didn't (teacher_cache.py's h_mid is NOT guaranteed
    unit-norm — the JL random projection shrinks norms, doesn't preserve
    them).

    Returns a scalar. Returns 0.0 (no gradient contribution) if valid_mask
    leaves nothing valid in this batch — callers should still guard against
    calling this on an empty batch upstream if that's a possible state.
    """
    def _directional(a, k, lab):
        B, N, D = a.shape
        a = a.reshape(B * N, D)
        k = k.reshape(B * N, D)
        lab = lab.reshape(B * N)

        if valid_mask is not None:
            m = valid_mask.reshape(B * N)
            a, k, lab = a[m], k[m], lab[m]

        M = a.shape[0]
        if M == 0:
            return a.new_zeros(())

        a = F.normalize(a, dim=-1)
        k = F.normalize(k, dim=-1)

        logits   = a @ k.T / temperature                              # (M, M)
        log_prob = F.log_softmax(logits, dim=1)                        # (M, M)
        pos_mask = (lab.unsqueeze(0) == lab.unsqueeze(1)).float()      # (M, M)
        pos_count = pos_mask.sum(dim=1).clamp(min=1.0)

        per_anchor = -(log_prob * pos_mask).sum(dim=1) / pos_count
        return per_anchor.mean()

    loss = _directional(anchors, keys, labels)
    if symmetric:
        loss = 0.5 * (loss + _directional(keys, anchors, labels))
    return loss


def llm_contrastive_loss(
    z_word: torch.Tensor,        # (B, N, D) — MEG side, e.g. WordProjectionHead output
    h_mid_target: torch.Tensor,  # (B, N, D) — LLM side, e.g. teacher_cache.py's h_mid
    valid_mask: "torch.Tensor | None",
    poem_ids: torch.Tensor,      # (B, N) long — small integer per poem (e.g. 0/1)
    word_pos: torch.Tensor,      # (B, N) long — word's position within its poem
    max_word_pos: int = 100,     # must exceed the largest real word_pos (poem2 goes to 60)
    temperature: float = 0.1,
    symmetric: bool = True,
) -> torch.Tensor:
    """
    Thin, spec-specific wrapper: builds the (poem, word_pos) label directly
    from the two id tensors so callers don't construct it by hand.
    max_word_pos is an ENCODING detail only (collision-free packing of two
    ids into one integer), not a modeling choice — default of 100 is safely
    above both poems' real word counts.
    """
    labels = poem_ids * max_word_pos + word_pos
    return multi_positive_contrastive_loss(z_word, h_mid_target, labels, valid_mask, temperature, symmetric)


def audio_contrastive_loss(
    z_dense: torch.Tensor,        # (B, T, D) — encoder's dense per-frame output
    audio_target: torch.Tensor,   # (B, T, D) — resampled wav2vec2 target, same T
    frame_valid_mask: "torch.Tensor | None" = None,   # (B, T) bool
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Standard diagonal-positive, symmetric InfoNCE between dense encoder
    frames and the resampled audio target at matching real-time positions.

    CAVEAT, not fixed here: audio_target depends only on (poem, time), not
    subject — so in principle the same duplicate-target problem
    multi_positive_contrastive_loss solves for words also applies here
    (two subjects' frames at the same instant of the same poem share a
    target). Implemented as plain diagonal InfoNCE because that's what the
    design spec explicitly scopes the multi-positive fix to (word-level
    only, §6) — revisit if audio-side batches routinely contain multiple
    trials of the same poem at overlapping frame ranges.
    """
    B, T, D = z_dense.shape
    z = z_dense.reshape(B * T, D)
    a = audio_target.reshape(B * T, D)

    if frame_valid_mask is not None:
        m = frame_valid_mask.reshape(B * T)
        z, a = z[m], a[m]

    if z.shape[0] == 0:
        return z.new_zeros(())

    z = F.normalize(z, dim=-1)
    a = F.normalize(a, dim=-1)

    logits  = z @ a.T / temperature
    targets = torch.arange(z.shape[0], device=z.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def stage1_anneal_weights(epoch: float, total_anneal_epochs: float = 15.0,
                           alpha_start: float = 0.8, alpha_end: float = 0.2):
    """
    Linear crossfade: alpha (audio weight) 0.8 -> 0.2, beta (LLM
    weight) 0.2 -> 0.8, over total_anneal_epochs, held fixed at the end
    values afterward.
    """
    t = min(max(epoch / total_anneal_epochs, 0.0), 1.0)
    alpha = alpha_start + t * (alpha_end - alpha_start)
    beta  = 1.0 - alpha
    return alpha, beta


def stage1_loss(audio_loss: torch.Tensor, llm_loss: torch.Tensor, epoch: float,
                 total_anneal_epochs: float = 15.0, alpha_start: float = 0.8, alpha_end: float = 0.2):
    """Returns (combined_loss, alpha, beta) — alpha/beta returned for logging."""
    alpha, beta = stage1_anneal_weights(epoch, total_anneal_epochs, alpha_start, alpha_end)
    return alpha * audio_loss + beta * llm_loss, alpha, beta


if __name__ == "__main__":
    torch.manual_seed(0)
    print("=== losses.py sanity check ===\n")

    # ------------------------------------------------------------------
    # 1. multi_positive_contrastive_loss vs. an independently computed
    #    reference, for a small, non-degenerate example.
    # ------------------------------------------------------------------
    D, M = 3, 4
    anchors = F.normalize(torch.randn(M, D), dim=-1)
    keys    = F.normalize(torch.randn(M, D), dim=-1)
    labels  = torch.tensor([0, 1, 0, 1])   # 0&2 share a label, 1&3 share a label
    temperature = 0.5

    logits    = anchors @ keys.T / temperature
    log_prob  = F.log_softmax(logits, dim=1)
    pos_mask  = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    expected  = (-(log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)).mean()

    got = multi_positive_contrastive_loss(
        anchors.unsqueeze(0), keys.unsqueeze(0), labels.unsqueeze(0),
        valid_mask=None, temperature=temperature, symmetric=False,
    )
    assert torch.allclose(got, expected, atol=1e-5), (
        f"mismatch vs. reference: got {got.item():.6f}, expected {expected.item():.6f}"
    )
    print(f"[OK] multi_positive_contrastive_loss matches independent reference ({got.item():.4f})")

    # ------------------------------------------------------------------
    # 2. The actual property being fixed: same-label, different-index
    #    pairs are positives, not false negatives.
    # ------------------------------------------------------------------
    assert pos_mask[0, 2] == 1 and pos_mask[2, 0] == 1, "same-label off-diagonal pairs must be positive"
    assert pos_mask[0, 1] == 0 and pos_mask[0, 3] == 0, "different-label pairs must be negative"
    print("[OK] same-(poem,word_pos) pairs across different batch indices are positives, not false negatives")

    # ------------------------------------------------------------------
    # 3. valid_mask excludes entries entirely — perturbing a masked-out
    #    entry must not move the loss at all.
    # ------------------------------------------------------------------
    valid_mask = torch.tensor([[True, True, True, False]])
    a2, k2 = anchors.clone().unsqueeze(0), keys.clone().unsqueeze(0)
    loss_before = multi_positive_contrastive_loss(a2, k2, labels.unsqueeze(0), valid_mask, temperature, symmetric=False)
    a2[:, 3], k2[:, 3] = torch.randn(D), torch.randn(D)
    loss_after = multi_positive_contrastive_loss(a2, k2, labels.unsqueeze(0), valid_mask, temperature, symmetric=False)
    assert torch.allclose(loss_before, loss_after, atol=1e-6), "perturbing a masked-invalid entry changed the loss"
    print("[OK] valid_mask correctly excludes invalid entries from the loss entirely")

    # ------------------------------------------------------------------
    # 4. symmetric=True equals the average of both directions.
    # ------------------------------------------------------------------
    loss_fwd = multi_positive_contrastive_loss(anchors.unsqueeze(0), keys.unsqueeze(0), labels.unsqueeze(0), None, temperature, symmetric=False)
    loss_rev = multi_positive_contrastive_loss(keys.unsqueeze(0), anchors.unsqueeze(0), labels.unsqueeze(0), None, temperature, symmetric=False)
    loss_sym = multi_positive_contrastive_loss(anchors.unsqueeze(0), keys.unsqueeze(0), labels.unsqueeze(0), None, temperature, symmetric=True)
    assert torch.allclose(loss_sym, 0.5 * (loss_fwd + loss_rev), atol=1e-5)
    print(f"[OK] symmetric=True == average of both directions ({loss_sym.item():.4f})")

    # ------------------------------------------------------------------
    # 5. llm_contrastive_loss — poem_id/word_pos label construction runs
    #    end-to-end.
    # ------------------------------------------------------------------
    poem_ids = torch.tensor([[0, 0, 1, 1]])
    word_pos = torch.tensor([[5, 5, 5, 3]])
    llm_loss = llm_contrastive_loss(anchors.unsqueeze(0), keys.unsqueeze(0), None, poem_ids, word_pos, temperature=temperature)
    assert torch.isfinite(llm_loss)
    print(f"[OK] llm_contrastive_loss runs end-to-end ({llm_loss.item():.4f})")

    # ------------------------------------------------------------------
    # 6. audio_contrastive_loss — perfect alignment near-zero, random pairs
    #    much higher.
    # ------------------------------------------------------------------
    Bc, T, Da = 1, 50, 16
    z_perfect = F.normalize(torch.randn(Bc, T, Da), dim=-1)
    loss_perfect = audio_contrastive_loss(z_perfect, z_perfect.clone(), temperature=0.1)
    z_random = F.normalize(torch.randn(Bc, T, Da), dim=-1)
    a_random = F.normalize(torch.randn(Bc, T, Da), dim=-1)
    loss_random = audio_contrastive_loss(z_random, a_random, temperature=0.1)
    assert loss_perfect.item() < 0.1, f"perfect alignment should be near-zero, got {loss_perfect.item()}"
    assert loss_random.item() > loss_perfect.item() + 1.0, "unrelated pairs should be much higher"
    print(f"[OK] audio_contrastive_loss: perfect={loss_perfect.item():.4f}, random={loss_random.item():.4f}")

    # ------------------------------------------------------------------
    # 7. stage1_anneal_weights — endpoints, midpoint, and holding past the
    #    anneal window.
    # ------------------------------------------------------------------
    a0, b0   = stage1_anneal_weights(0, total_anneal_epochs=15)
    a15, b15 = stage1_anneal_weights(15, total_anneal_epochs=15)
    a_mid, _ = stage1_anneal_weights(7.5, total_anneal_epochs=15)
    a30, _   = stage1_anneal_weights(30, total_anneal_epochs=15)
    assert abs(a0 - 0.8) < 1e-6 and abs(b0 - 0.2) < 1e-6
    assert abs(a15 - 0.2) < 1e-6 and abs(b15 - 0.8) < 1e-6
    assert abs(a_mid - 0.5) < 1e-6
    assert abs(a30 - 0.2) < 1e-6, "weights must hold at the end value past the anneal window"
    print(f"[OK] anneal schedule: epoch0=({a0:.2f},{b0:.2f})  epoch7.5=({a_mid:.2f})  "
          f"epoch15=({a15:.2f},{b15:.2f})  epoch30(past window)=({a30:.2f})")

    # ------------------------------------------------------------------
    # 8. Gradients flow cleanly through the combined stage1_loss.
    # ------------------------------------------------------------------
    z_audio  = (torch.randn(1, 20, 16) * 0.1).requires_grad_(True)
    a_target = torch.randn(1, 20, 16)
    z_word   = (torch.randn(1, 4, 16) * 0.1).requires_grad_(True)
    h_target = torch.randn(1, 4, 16)
    p_ids = torch.tensor([[0, 0, 1, 1]])
    w_pos = torch.tensor([[0, 1, 0, 1]])

    aud_loss = audio_contrastive_loss(z_audio, a_target)
    llmw_loss = llm_contrastive_loss(z_word, h_target, None, p_ids, w_pos)
    combined, alpha, beta = stage1_loss(aud_loss, llmw_loss, epoch=3)
    combined.backward()
    assert z_audio.grad is not None and torch.isfinite(z_audio.grad).all()
    assert z_word.grad is not None and torch.isfinite(z_word.grad).all()
    print(f"[OK] stage1_loss gradients flow cleanly to both z_audio and z_word (alpha={alpha:.3f}, beta={beta:.3f})")

    print("\n=== ALL CHECKS PASSED ===")
