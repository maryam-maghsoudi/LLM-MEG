"""
evaluate.py — Step 10
======================
Load trained Stage 1 + Stage 2 checkpoints and evaluate on the held-out test set.

Stage 1 metrics  (ranking against all cached hmid_t vectors for both poems):
    R@1, R@5, R@10, MRR, median rank

Stage 2 metrics  (per-trial sequence evaluation):
    mean KL(p_t || q_t)         — distillation quality
    next-word top-1 agreement   — fraction where argmax(q_t) == argmax(p_t)
    next-word true accuracy     — fraction where argmax(q_t) == true next word

Usage
-----
python evaluate.py --heldout sub-01
python evaluate.py --heldout sub-01 --llm_name HuggingFaceTB/SmolLM2-360M
python evaluate.py --heldout sub-01 --skip_stage2   # Stage 1 only
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(1, str(_HERE.parent))

from dataset import (
    MEGWordDatasetLLM,
    MEGSequenceDataset,
    collate_sequences,
    load_poem_cache,
    load_vocab_info,
    make_loso_splits,
    model_tag,
    POEM_KEYS,
)
from models import MEGEncoder, LLMTextProjection, GRUHead, load_lm_head
from train import MODEL_CONFIGS, kl_loss, shuffle_time_per_trial, MEG_EMB, GRU_HIDDEN


# ===========================================================================
#  Stage 1 — ranking evaluation
# ===========================================================================

def evaluate_ranking(
    meg_enc:   MEGEncoder,
    text_proj: LLMTextProjection,
    test_loader: DataLoader,
    device,
    verbose: bool = True,
) -> dict:
    """
    For every test MEG window, rank it against ALL test hmid_t vectors.
    Metrics: R@1, R@5, R@10, MRR, median_rank.

    Note: the gallery here is all test occurrences (not the full train set).
    A larger gallery = harder task.
    """
    meg_enc.eval()
    text_proj.eval()

    all_z = []
    all_t = []
    with torch.no_grad():
        for x, h in test_loader:
            all_z.append(meg_enc(x.to(device)).cpu())
            all_t.append(text_proj(h.to(device)).cpu())

    Z = torch.cat(all_z, dim=0)   # (N, emb)
    T = torch.cat(all_t, dim=0)   # (N, emb)
    N = Z.shape[0]

    # Cosine similarity matrix (already L2-normalized outputs)
    sim = Z @ T.T                  # (N, N)
    # Diagonal = matching pair; sort each row descending
    ranks = []
    for i in range(N):
        row = sim[i].clone()
        row[i] = -1e9              # exclude self (same occurrence)
        order = torch.argsort(row, descending=True)
        # rank of the diagonal = ground-truth match (label i)
        # For occurrence-level eval: correct match is position i itself.
        # But we set sim[i,i]=-inf, so the "next best" for the same word type
        # might rank first. We report standard retrieval R@k.
        # Restore diagonal for true retrieval rank:
        row2 = sim[i].clone()
        rank = (torch.argsort(row2, descending=True) == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    results = {
        "R@1":         float((ranks <= 1).mean()),
        "R@5":         float((ranks <= 5).mean()),
        "R@10":        float((ranks <= 10).mean()),
        "MRR":         float((1.0 / ranks).mean()),
        "median_rank": float(np.median(ranks)),
        "n_test":      int(N),
    }

    if verbose:
        print("\n── Stage 1 Ranking (test set) ──────────────────────────")
        for k, v in results.items():
            print(f"  {k:15s}: {v:.4f}" if isinstance(v, float) else f"  {k:15s}: {v}")
        print()

    return results


# ===========================================================================
#  Stage 2 — KL and next-word accuracy
# ===========================================================================

def evaluate_stage2(
    meg_enc:  MEGEncoder,
    gru_head: GRUHead,
    lm_head:  torch.nn.Module,
    r_ids:    torch.Tensor,
    test_loader: DataLoader,
    device,
    verbose: bool = True,
) -> dict:
    """
    Per-position KL + accuracy on test trials.

    next_word_agreement : argmax(q_t) == argmax(p_t)   [mimicry of teacher]
    next_word_accuracy  : argmax(q_t) == true word id  [actual decoding]
    """
    meg_enc.eval()
    gru_head.eval()

    total_kl     = 0.0
    total_agree  = 0
    total_acc    = 0
    total_valid  = 0

    with torch.no_grad():
        for batch in test_loader:
            meg_seq    = batch["meg_windows"].to(device)      # (B, N, C, T)
            p_t_r      = batch["p_t_restricted"].to(device)   # (B, N, R)
            valid_mask = batch["valid_mask"].to(device)        # (B, N)

            B, N, C, T = meg_seq.shape
            z  = meg_enc(meg_seq.view(B * N, C, T)).view(B, N, -1)
            y  = gru_head(z)                                  # (B, N, d_model)
            q  = lm_head(y)[:, :, r_ids]                     # (B, N, R)

            # KL for positions 0..N-2; mask on t+1 existence (not t)
            kl = kl_loss(q, p_t_r, valid_mask)
            mask_pos = valid_mask[:, 1:]                      # (B, N-1): word t+1 exists?
            n_valid  = mask_pos.sum().item()
            total_kl    += kl.item() * n_valid
            total_valid += n_valid

            # Top-1 agreement (q vs p) at same valid positions
            q_top1 = q[:, :-1, :].argmax(-1)       # (B, N-1)
            p_top1 = p_t_r[:, :-1, :].argmax(-1)   # (B, N-1)
            agree  = ((q_top1 == p_top1) & mask_pos).sum().item()
            total_agree += agree

    mean_kl   = total_kl   / max(total_valid, 1)
    agreement = total_agree / max(total_valid, 1)

    results = {
        "mean_kl":               float(mean_kl),
        "next_word_agreement":   float(agreement),
        "n_valid_positions":     int(total_valid),
    }

    if verbose:
        print("\n── Stage 2 (test set) ───────────────────────────────────")
        for k, v in results.items():
            print(f"  {k:25s}: {v:.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")
        print()

    return results


# ===========================================================================
#  Stage 2 — eval-only control ablations (no retrain)
# ===========================================================================

@torch.no_grad()
def evaluate_stage2_control(
    meg_enc:  MEGEncoder,
    gru_head: GRUHead,
    lm_head:  torch.nn.Module,
    r_ids:    torch.Tensor,
    test_loader: DataLoader,
    device,
    mode: str = "zero",
) -> dict:
    """
    Eval-only ablation using the already-trained stage2_best.pt weights.
    Tests whether the trained GRU actually relies on real z_t.

    mode='zero'    — replace z_t with zeros
    mode='shuffle' — randomly permute trials in each batch (simple across-batch
                     shuffle; fine for eval since we just want a signal check)

    If the trained model's KL / agreement barely changes under either mode,
    the GRU was never leaning on MEG content in the first place.
    """
    meg_enc.eval()
    gru_head.eval()

    total_kl    = 0.0
    total_agree = 0
    total_valid = 0

    for batch in test_loader:
        meg_seq    = batch["meg_windows"].to(device)
        p_t_r      = batch["p_t_restricted"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        B, N, C, T = meg_seq.shape

        z = meg_enc(meg_seq.view(B * N, C, T)).view(B, N, -1)

        if mode == "zero":
            z = torch.zeros_like(z)
        elif mode == "shuffle":
            perm = torch.randperm(B, device=device)
            z    = z[perm]
        elif mode == "shuffle_time":
            z = shuffle_time_per_trial(z, p_t_r)

        y  = gru_head(z)
        q  = lm_head(y)[:, :, r_ids]

        kl       = kl_loss(q, p_t_r, valid_mask)
        mask_pos = valid_mask[:, 1:]
        n_valid  = mask_pos.sum().item()
        total_kl    += kl.item() * n_valid
        total_valid += n_valid

        q_top1 = q[:, :-1, :].argmax(-1)
        p_top1 = p_t_r[:, :-1, :].argmax(-1)
        total_agree += ((q_top1 == p_top1) & mask_pos).sum().item()

    mean_kl   = total_kl   / max(total_valid, 1)
    agreement = total_agree / max(total_valid, 1)
    return {
        "mean_kl":             float(mean_kl),
        "next_word_agreement": float(agreement),
        "n_valid_positions":   int(total_valid),
    }


# ===========================================================================
#  Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained two-stage MEG decoder")
    p.add_argument("--heldout",    required=True)
    p.add_argument("--llm_name",   default="HuggingFaceTB/SmolLM2-360M",
                   choices=list(MODEL_CONFIGS.keys()))
    p.add_argument("--hmid_layer", type=int, default=None)
    p.add_argument("--gru_hidden", type=int, default=GRU_HIDDEN)
    p.add_argument("--skip_stage2", action="store_true")
    p.add_argument("--out_root",   default=str(_HERE / "out"))
    p.add_argument("--ckpt_dir",   default=None,
                   help="Override checkpoint directory (default: out_root/model_tag/heldout). "
                        "Useful for evaluating control runs whose directory names differ from heldout.")
    p.add_argument("--device",     default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    cfg        = MODEL_CONFIGS[args.llm_name]
    d_model    = cfg["d_model"]
    hmid_layer = args.hmid_layer if args.hmid_layer is not None else cfg["hmid_layer"]

    tag     = model_tag(args.llm_name)
    out_dir = Path(args.ckpt_dir) if args.ckpt_dir else Path(args.out_root) / tag / args.heldout
    ckpt_s1 = out_dir / "stage1_best.pt"
    ckpt_s2 = out_dir / "stage2_best.pt"

    if not ckpt_s1.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {ckpt_s1}\n"
                                f"Run train.py --heldout {args.heldout} first.")

    print(f"\nEvaluating  heldout={args.heldout}  llm={args.llm_name}")
    print(f"  Stage 1 ckpt : {ckpt_s1}")
    print(f"  Stage 2 ckpt : {ckpt_s2}  (skip={args.skip_stage2})")
    print(f"  device       : {device}")

    # ── vocab info ───────────────────────────────────────────────────────
    vocab_info = load_vocab_info(args.llm_name)
    r_ids      = torch.tensor(vocab_info["restricted_first_token_ids"],
                              dtype=torch.long, device=device)

    # ── LOSO split (test only) ───────────────────────────────────────────
    splits   = make_loso_splits(args.heldout)
    test_cfg = splits["test"]

    # ── Stage 1 evaluation ───────────────────────────────────────────────
    s1_state = torch.load(ckpt_s1, map_location="cpu")

    meg_enc   = MEGEncoder().to(device)
    text_proj = LLMTextProjection(d_model=d_model).to(device)
    meg_enc.load_state_dict(s1_state["meg_encoder"])
    text_proj.load_state_dict(s1_state["text_proj"])

    print(f"\nBuilding Stage 1 test dataset ...")
    ds_test_s1 = MEGWordDatasetLLM(
        **test_cfg, llm_name=args.llm_name, hmid_layer=hmid_layer, augment=False
    )
    dl_test_s1 = DataLoader(ds_test_s1, batch_size=128, shuffle=False, num_workers=2)

    s1_results = evaluate_ranking(meg_enc, text_proj, dl_test_s1, device)
    (out_dir / "eval_stage1.json").write_text(json.dumps(s1_results, indent=2))
    print(f"Saved → {out_dir / 'eval_stage1.json'}")

    if args.skip_stage2 or not ckpt_s2.exists():
        if not args.skip_stage2:
            print(f"\nNote: Stage 2 checkpoint not found at {ckpt_s2} — skipping.")
        return

    # ── Stage 2 evaluation ───────────────────────────────────────────────
    s2_state = torch.load(ckpt_s2, map_location="cpu")

    meg_enc.freeze()
    gru_head = GRUHead(meg_emb=MEG_EMB, gru_hidden=args.gru_hidden,
                       d_model=d_model).to(device)
    gru_head.load_state_dict(s2_state["gru_head"])

    lm_head = load_lm_head(args.llm_name, device=device)

    print(f"\nBuilding Stage 2 test dataset ...")
    ds_test_s2 = MEGSequenceDataset(**test_cfg, llm_name=args.llm_name)
    dl_test_s2 = DataLoader(ds_test_s2, batch_size=4, shuffle=False,
                            collate_fn=collate_sequences, num_workers=2)

    s2_results = evaluate_stage2(meg_enc, gru_head, lm_head, r_ids,
                                 dl_test_s2, device)
    (out_dir / "eval_stage2.json").write_text(json.dumps(s2_results, indent=2))
    print(f"Saved → {out_dir / 'eval_stage2.json'}")

    # ── Eval-only controls (Experiment #1 — no retrain) ──────────────────
    print("\n── Stage 2 Controls (eval-only swap, existing weights) ──────")
    ctrl_results = {"real": s2_results}
    for mode in ("zero", "shuffle", "shuffle_time"):
        ctrl_results[mode] = evaluate_stage2_control(
            meg_enc, gru_head, lm_head, r_ids, dl_test_s2, device, mode=mode
        )

    print(f"  {'mode':8s}  {'KL':>8}  {'agreement':>10}")
    print(f"  {'-'*34}")
    for mode, r in ctrl_results.items():
        marker = "  ← trained model" if mode == "real" else ""
        print(f"  {mode:8s}  {r['mean_kl']:8.4f}  {r['next_word_agreement']:10.4f}{marker}")

    (out_dir / "eval_stage2_controls.json").write_text(
        json.dumps(ctrl_results, indent=2)
    )
    print(f"Saved → {out_dir / 'eval_stage2_controls.json'}")


if __name__ == "__main__":
    main()
