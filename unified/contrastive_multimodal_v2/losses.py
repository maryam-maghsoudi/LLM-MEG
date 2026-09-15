"""
losses.py — Stage 1 contrastive losses.

multi_positive_contrastive_loss  SupCon-style loss:
                                  positives are determined by LABEL equality,
                                  not index equality ("clean diagonal"). 
                                  
llm_contrastive_loss               builds (poem_id, word_pos)
                                  labels automatically,  treat every same-(poem, word_pos)
                                  pair across the batch as a positive").
audio_contrastive_loss            Simpler diagonal-only InfoNCE for the
                                  dense per-frame audio target.
stage1_anneal_weights /
stage1_loss                       Linear crossfade of alpha (audio) / beta
                                  (LLM) weights over ~15 epochs (§6).


"""

import torch
import torch.nn.functional as F

def multi_positive_contrastive_loss(
    anchors: torch.Tensor, keys: torch.Tensor, labels: torch.Tensor,
    valid_mask: "torch.Tensor | None" = None, temperature: float = 0.1,
    symmetric: bool = True, weights: "torch.Tensor | None" = None,   # NEW
) -> torch.Tensor:
  
    def _directional(a, k, lab):
        B, N, D = a.shape
        a = a.reshape(B * N, D)
        k = k.reshape(B * N, D)
        lab = lab.reshape(B * N)
        w = weights.reshape(B * N) if weights is not None else None

        if valid_mask is not None:
            m = valid_mask.reshape(B * N)
            a, k, lab = a[m], k[m], lab[m]
            if w is not None:
                w = w[m]

        M = a.shape[0]
        if M == 0:
            return a.new_zeros(())

        a = F.normalize(a, dim=-1)
        k = F.normalize(k, dim=-1)

        logits   = a @ k.T / temperature
        log_prob = F.log_softmax(logits, dim=1)
        pos_mask = (lab.unsqueeze(0) == lab.unsqueeze(1)).float()
        pos_count = pos_mask.sum(dim=1).clamp(min=1.0)

        per_anchor = -(log_prob * pos_mask).sum(dim=1) / pos_count
        if w is None:
            return per_anchor.mean()
        return (per_anchor * w).sum() / w.sum().clamp(min=1e-8)

    loss = _directional(anchors, keys, labels)
    if symmetric:
        loss = 0.5 * (loss + _directional(keys, anchors, labels))
    return loss


def llm_contrastive_loss(
    z_word: torch.Tensor, h_mid_target: torch.Tensor, valid_mask: "torch.Tensor | None",
    poem_ids: torch.Tensor, word_pos: torch.Tensor, max_word_pos: int = 100,
    temperature: float = 0.1, symmetric: bool = True,
    weights: "torch.Tensor | None" = None,   # NEW — forwarded, nothing else changes
) -> torch.Tensor:
    labels = poem_ids * max_word_pos + word_pos
    return multi_positive_contrastive_loss(z_word, h_mid_target, labels, valid_mask,
                                            temperature, symmetric, weights=weights)

def audio_contrastive_loss(
    z_dense: torch.Tensor,        # (B, T, D) — encoder's dense per-frame output
    audio_target: torch.Tensor,   # (B, T, D) — resampled wav2vec2 target, same T
    frame_valid_mask: "torch.Tensor | None" = None,   # (B, T) bool
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Standard diagonal-positive, symmetric InfoNCE between dense encoder
    frames and the resampled audio target at matching real-time positions.


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


# def stage1_anneal_weights(epoch: float, total_anneal_epochs: float = 15.0,
#                            alpha_start: float = 0.8, alpha_end: float = 0.2):
#     """
#     Linear crossfade: alpha (audio weight) 0.8 -> 0.2, beta (LLM
#     weight) 0.2 -> 0.8, over total_anneal_epochs, held fixed at the end
#     values afterward.
#     """
#     t = min(max(epoch / total_anneal_epochs, 0.0), 1.0)
#     alpha = alpha_start + t * (alpha_end - alpha_start)
#     beta  = 1.0 - alpha
#     return alpha, beta


def stage1_anneal_weights(epoch, total_anneal_epochs=15.0, alpha_start=0.8, alpha_end=0.2, shape="linear"):
    t = min(max(epoch / total_anneal_epochs, 0.0), 1.0)
    if shape == "cosine":
        t = 0.5 * (1 - torch.cos(torch.tensor(t * 3.14159265)).item())
    alpha = alpha_start + t * (alpha_end - alpha_start)
    return alpha, 1.0 - alpha


def stage1_loss(audio_loss: torch.Tensor, llm_loss: torch.Tensor, epoch: float,
                 total_anneal_epochs: float = 15.0, alpha_start: float = 0.8, alpha_end: float = 0.2):
    """Returns (combined_loss, alpha, beta) — alpha/beta returned for logging."""
    alpha, beta = stage1_anneal_weights(epoch, total_anneal_epochs, alpha_start, alpha_end)
    return alpha * audio_loss + beta * llm_loss, alpha, beta






