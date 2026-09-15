"""
train_stage2.py — Stage 2 (KL next-word distillation) is not yet written.

This module currently provides ONLY load_stage1_checkpoint, the loader that
eval_stage1.py (and future Stage 2 code) uses to restore a frozen, converged
Stage 1 encoder + word head + pooling module from a checkpoint saved by
train.py. It matches the checkpoint schema train.py writes:

    ckpt = {
        "encoder":         MEGEncoder state_dict,
        "word_head":       WordProjectionHead state_dict,
        "pooling":         WordAttentionPooling state_dict (only if pooling_mode="wide"),
        "pooling_mode":    "exact" | "wide",
        "heldout_subject": str,
        "epoch":           int,
        "val_loss":        float,
    }

Returns 5 values (the signature eval_stage1.py unpacks):
    encoder, word_head, pooling_module, pooling_mode, ckpt_subject
"""
import torch

from new_models import MEGEncoder, WordProjectionHead
from pooling import WordAttentionPooling


def load_stage1_checkpoint(ckpt_path: str, device):
    """
    Load encoder, word_head, and pooling_module from a Stage 1 checkpoint.

    The encoder is frozen (eval mode, no grad). For pooling_mode="exact" no
    learned pooling parameters exist, so a fresh WordAttentionPooling is
    returned but never used by pool_words("exact", ...) — kept only so the
    return signature is uniform across both pooling modes.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pooling_mode = ckpt.get("pooling_mode", "exact")
    ckpt_subject = ckpt.get("heldout_subject", None)

    encoder = MEGEncoder().to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.freeze()

    word_head = WordProjectionHead(encoder.backbone_dim).to(device)
    word_head.load_state_dict(ckpt["word_head"])
    word_head.eval()

    pooling_module = WordAttentionPooling(encoder.backbone_dim).to(device)
    if "pooling" in ckpt:
        pooling_module.load_state_dict(ckpt["pooling"])
    pooling_module.eval()

    val_loss = ckpt.get("val_loss", float("nan"))
    print(f"Loaded Stage 1 checkpoint: {ckpt_path}  "
          f"(epoch={ckpt.get('epoch', '?')}, val_loss={val_loss:.4f}, "
          f"pooling_mode={pooling_mode}, heldout={ckpt_subject})")
    return encoder, word_head, pooling_module, pooling_mode, ckpt_subject
