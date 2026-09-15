"""
train.py — Stage 1 (contrastive alignment) training loop.

  --freq_weighted_loss    Reweights the contrastive loss's final average
                           by inverse word-type frequency (losses.py).
                           Does NOT change which pairs are positives/
                           negatives, only how much each anchor's loss
                           term counts toward the batch average.

  --oversample_rare_types Resamples (with replacement) at the flattened
                           WORD-OCCURRENCE level, weighted inverse to
                           word-type frequency, before the loss is
                           computed. Deliberately NOT trial-level: every
                           trial replays the same fixed poem, so
                           "clatter" appears exactly once and "the"
                           appears however many times it appears in the
                           poem text, in EVERY trial regardless of which
                           trials get loaded — the only way to change
                           that ratio is to resample individual word rows
                           after pooling has already flattened them out
                           of their trial structure.


"""

import argparse
import os
from collections import Counter

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from new_dataset import MEGContinuousTrialDataset, collate_continuous_trials, _load_onsets
from new_models import MEGEncoder, AudioProjectionHead, WordProjectionHead, JOINT_DIM, TOTAL_STRIDE
from pooling import WordAttentionPooling, pool_words, raw_length_to_encoder_frames
from losses import audio_contrastive_loss, llm_contrastive_loss, stage1_anneal_weights
from splits import make_loso_splits

POEM_TO_ID = {"poem1": 0, "poem2": 1}
ID_TO_POEM = {v: k for k, v in POEM_TO_ID.items()}


# ===========================================================================
#  Ablation 
# ===========================================================================

def get_stage1_weights(mode: str, epoch: float, total_anneal_epochs: float = 15.0,
                        hard_stage_split_epoch: float = 15.0):
    if mode == "joint_annealed":
        return stage1_anneal_weights(epoch, total_anneal_epochs, alpha_start=0.8, alpha_end=0.2, shape='cosine')
    elif mode == "llm_only":
        return 0.0, 1.0
    elif mode == "audio_only":
        return 1.0, 0.0
    elif mode == "joint_fixed":
        return 0.5, 0.5
    elif mode == "hard_staged":
        return (1.0, 0.0) if epoch < hard_stage_split_epoch else (0.0, 1.0)
    else:
        raise ValueError(f"unknown stage1 ablation mode: {mode!r}")


# ===========================================================================
#  Batch 
# ===========================================================================

def gather_word_targets(batch, teacher_cache, target_key, target_dim, poem_to_id=POEM_TO_ID):
    B, N = batch["onset_samples"].shape
    target     = torch.zeros(B, N, target_dim)
    poem_ids   = torch.zeros(B, N, dtype=torch.long)
    word_pos_t = torch.zeros(B, N, dtype=torch.long)

    for b in range(B):
        poem   = batch["poem"][b]
        poses  = batch["word_poses"][b]
        source = teacher_cache[poem][target_key]
        pid    = poem_to_id[poem]
        for i, pos in enumerate(poses):
            if i >= N:
                break
            target[b, i]     = source[pos]
            poem_ids[b, i]   = pid
            word_pos_t[b, i] = pos

    return target, poem_ids, word_pos_t


def build_word_targets(batch, teacher_cache, poem_to_id=POEM_TO_ID, joint_dim=JOINT_DIM):
    return gather_word_targets(batch, teacher_cache, "h_mid", joint_dim, poem_to_id)


def build_audio_targets(batch, teacher_cache, T_out, total_stride=TOTAL_STRIDE):
    B = batch["meg_trial"].shape[0]
    real_frames = raw_length_to_encoder_frames(batch["trial_mask"].sum(dim=1), total_stride)

    audio_target     = torch.zeros(B, T_out, JOINT_DIM)
    frame_valid_mask = torch.zeros(B, T_out, dtype=torch.bool)

    for b in range(B):
        poem = batch["poem"][b]
        a = teacher_cache[poem]["audio_target"]
        n = min(a.shape[0], int(real_frames[b].item()), T_out)
        audio_target[b, :n]     = a[:n]
        frame_valid_mask[b, :n] = True

    return audio_target, frame_valid_mask


# ===========================================================================
#  Word-type frequency weighting / oversampling 
# ===========================================================================

def build_word_weight_cache(alpha: float = 0.5, poem_to_id=POEM_TO_ID):
    """
    One weight per word OCCURRENCE, ordered identically to h_mid's rows
    (same _load_onsets call/order eval_stage1.py's build_candidate_bank
    uses) -- gathered via the SAME gather_word_targets() machinery as
    h_mid, guaranteeing identical (poem, word_pos) alignment.
    Inverse to word-TYPE frequency (repeated word STRINGS), not
    occurrence-label frequency (already ~balanced by construction, §6).
    Normalized to mean 1 so total loss scale stays comparable across
    weighted/unweighted runs.
    """
    type_counts = Counter()
    per_poem_words = {}
    for poem in ("poem1", "poem2"):
        words = [e["word"].strip().lower() for e in _load_onsets(poem)]
        per_poem_words[poem] = words
        type_counts.update(words)

    weight_cache, all_weights = {}, []
    for poem, words in per_poem_words.items():
        w = torch.tensor([type_counts[t] for t in words], dtype=torch.float32).pow(-alpha)
        weight_cache[poem] = {"word_weight": w.unsqueeze(-1)}   # (n_words, 1)
        all_weights.append(w)

    mean_w = torch.cat(all_weights).mean()
    for poem in weight_cache:
        weight_cache[poem]["word_weight"] /= mean_w
    return weight_cache


def build_word_weights(batch, weight_cache, poem_to_id=POEM_TO_ID):
    """Per-batch (B, N) frequency weight, reusing gather_word_targets verbatim."""
    weights, _, _ = gather_word_targets(batch, weight_cache, "word_weight", target_dim=1, poem_to_id=poem_to_id)
    return weights.squeeze(-1)


def _flatten_valid_words(z_word, h_mid_target, poem_ids, word_pos_t, combined_valid):
    """
    Flattens the (B, N, ...) grid down to (M, ...) for just the valid
    word occurrences -- oversampling needs this flat form, since
    duplicating whole TRIALS can never rebalance word-type ratios (every
    trial replays the same fixed poem text).
    """
    m = combined_valid.reshape(-1)
    B, N, D = z_word.shape
    z_flat    = z_word.reshape(B * N, D)[m]
    h_flat    = h_mid_target.reshape(B * N, D)[m]
    poem_flat = poem_ids.reshape(-1)[m]
    pos_flat  = word_pos_t.reshape(-1)[m]
    return z_flat, h_flat, poem_flat, pos_flat


def resample_word_occurrences(z_flat, h_flat, poem_flat, pos_flat, word_weight_flat,
                               target_size=None, generator=None):
    """
    Resamples WITH REPLACEMENT, weighted inverse to word-type frequency.
    target_size defaults to the original valid-word count so step size /
    gradient noise stays roughly comparable to an unweighted run.
    """
    if target_size is None:
        target_size = z_flat.shape[0]
    probs = word_weight_flat / word_weight_flat.sum()
    idx = torch.multinomial(probs.cpu(), target_size, replacement=True, generator=generator).to(probs.device)
    return z_flat[idx], h_flat[idx], poem_flat[idx], pos_flat[idx]


def _weights_from_flat_positions(poem_flat, pos_flat, weight_cache, poem_to_id=POEM_TO_ID):
    """Reconstructs per-row frequency weight from (poem_id, word_pos) after resampling --
    used only when BOTH --oversample_rare_types and --freq_weighted_loss are set."""
    out = torch.zeros(pos_flat.shape[0], dtype=torch.float32, device=pos_flat.device)
    for pid in poem_flat.unique().tolist():
        poem = ID_TO_POEM[pid]
        w = weight_cache[poem]["word_weight"].squeeze(-1).to(pos_flat.device)
        mask = poem_flat == pid
        out[mask] = w[pos_flat[mask]]
    return out


# ===========================================================================
#  Shared forward pass — used by real training, evaluation
# ===========================================================================

def compute_stage1_losses(batch, teacher_cache, encoder, audio_head, word_head, pooling_module,
                           poem_to_id=POEM_TO_ID, jitter_ms=None, pooling_mode="wide",
                           audio_temperature=0.1, llm_temperature=0.1,
                           weight_cache=None, freq_weighted_loss=False,
                           oversample_rare_types=False, rebalance_generator=None):
    z_dense = encoder(batch["meg_trial"])
    z_audio = audio_head(z_dense)

    pooled, pool_valid = pool_words(
        pooling_mode, z_dense, batch["onset_samples"],
        offset_samples=batch["offset_samples"], trial_mask=batch["trial_mask"],
        attention_module=pooling_module, jitter_ms=jitter_ms,
    )
    z_word = word_head(pooled)
    combined_valid = pool_valid & batch["valid_mask"]

    device = z_dense.device
    audio_target, frame_valid_mask = build_audio_targets(batch, teacher_cache, z_dense.shape[1])
    h_mid_target, poem_ids, word_pos_t = build_word_targets(batch, teacher_cache, poem_to_id)

    audio_loss = audio_contrastive_loss(
        z_audio, audio_target.to(device), frame_valid_mask.to(device), temperature=audio_temperature
    )

    if oversample_rare_types:
        assert weight_cache is not None, "--oversample_rare_types requires a weight_cache (build via build_word_weight_cache)"
        word_weight = build_word_weights(batch, weight_cache, poem_to_id).to(device)
        z_flat, h_flat, poem_flat, pos_flat = _flatten_valid_words(
            z_word, h_mid_target.to(device), poem_ids.to(device), word_pos_t.to(device), combined_valid
        )
        if z_flat.shape[0] == 0:
            return audio_loss, audio_loss.new_zeros(())

        w_flat = word_weight.reshape(-1)[combined_valid.reshape(-1)]
        z_rs, h_rs, poem_rs, pos_rs = resample_word_occurrences(
            z_flat, h_flat, poem_flat, pos_flat, w_flat, generator=rebalance_generator
        )

        loss_weights = None
        if freq_weighted_loss:
            # BOTH flags set: reweight the already-rebalanced batch too --
            # compounds the two mitigations deliberately. Test each alone
            # before trusting this combination (see module docstring).
            w_rs = _weights_from_flat_positions(poem_rs, pos_rs, weight_cache, poem_to_id)
            loss_weights = w_rs.unsqueeze(0)

        llm_loss = llm_contrastive_loss(
            z_rs.unsqueeze(0), h_rs.unsqueeze(0), None,
            poem_rs.unsqueeze(0), pos_rs.unsqueeze(0),
            temperature=llm_temperature, weights=loss_weights,
        )
    else:
        word_weights = None
        if freq_weighted_loss:
            assert weight_cache is not None, "--freq_weighted_loss requires a weight_cache (build via build_word_weight_cache)"
            word_weights = build_word_weights(batch, weight_cache, poem_to_id).to(device)

        llm_loss = llm_contrastive_loss(
            z_word, h_mid_target.to(device), combined_valid, poem_ids.to(device), word_pos_t.to(device),
            temperature=llm_temperature, weights=word_weights,
        )
    ## Adding MSE loss for absolute alignment 
#     mse_weight = 0.05
#     mse_loss = z_word.new_zeros(())
#     if mse_weight > 0:
#         diff = (z_word - h_mid_target.to(device)) ** 2
#         per_word_mse = diff.mean(dim=-1)                                    # (B, N) — mean over 768 dims, not sum
#         mse_loss = (per_word_mse * combined_valid).sum() / combined_valid.sum().clamp(min=1)
# #         mse_loss = (diff.sum(-1) * combined_valid).sum() / combined_valid.sum().clamp(min=1)
#     llm_loss = llm_loss + mse_weight*mse_loss
    return audio_loss, llm_loss

@torch.no_grad()
def check_scale_and_direction(word_head, encoder, pooling_module, loader, teacher_cache, pooling_mode, device,
                               poem_to_id=POEM_TO_ID):
    """
    Diagnostic (not used in training) -- checks whether z_word's RAW norm
    (magnitude) and direction actually match the true h_mid target, now
    that WordProjectionHead no longer internally normalizes. The
    contrastive loss only ever supervises DIRECTION (both sides get
    L2-normalized inside multi_positive_contrastive_loss regardless); an
    optional MSE term is what's meant to additionally supervise magnitude
    -- this is how to check whether it's actually doing that, separately
    from whether it's damaging the direction the contrastive loss already
    established.
    """
    encoder.eval(); word_head.eval(); pooling_module.eval()
    norm_ratios, cos_sims = [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        z_dense = encoder(batch["meg_trial"])
        pooled, pool_valid = pool_words(
            pooling_mode, z_dense, batch["onset_samples"],
            offset_samples=batch["offset_samples"], trial_mask=batch["trial_mask"],
            attention_module=pooling_module,
        )
        z_word = word_head(pooled)
        h_mid_target, _, _ = build_word_targets(batch, teacher_cache, poem_to_id)
        h_mid_target = h_mid_target.to(device)
        combined_valid = pool_valid & batch["valid_mask"]

        r = (z_word.norm(dim=-1) / h_mid_target.norm(dim=-1).clamp(min=1e-8))[combined_valid]
        c = F.cosine_similarity(z_word, h_mid_target, dim=-1)[combined_valid]
        norm_ratios.append(r.cpu()); cos_sims.append(c.cpu())

    norm_ratios, cos_sims = torch.cat(norm_ratios), torch.cat(cos_sims)
    return {
        "norm_ratio_mean": norm_ratios.mean().item(), "norm_ratio_std": norm_ratios.std().item(),
        "norm_ratio_min": norm_ratios.min().item(), "norm_ratio_max": norm_ratios.max().item(),
        "cos_sim_mean": cos_sims.mean().item(), "cos_sim_std": cos_sims.std().item(),
        "cos_sim_min": cos_sims.min().item(), "cos_sim_max": cos_sims.max().item(),
    }

def _move_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ===========================================================================
#  Train / eval epochs
# ===========================================================================

def train_one_epoch(loader, teacher_cache, encoder, audio_head, word_head, pooling_module,
                     optimizer, epoch, anneal_mode, device,
                     jitter_ms=(50.0, 150.0), total_anneal_epochs=15.0, pooling_mode="wide",
                     weight_cache=None, freq_weighted_loss=False, oversample_rare_types=False,
                     rebalance_generator=None):
    encoder.train(); audio_head.train(); word_head.train(); pooling_module.train()
    alpha, beta = get_stage1_weights(anneal_mode, epoch, total_anneal_epochs)

    running_loss = running_audio = running_llm = 0.0
    n_batches = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad()

        audio_loss, llm_loss = compute_stage1_losses(
            batch, teacher_cache, encoder, audio_head, word_head, pooling_module,
            jitter_ms=jitter_ms, pooling_mode=pooling_mode,
            weight_cache=weight_cache, freq_weighted_loss=freq_weighted_loss,
            oversample_rare_types=oversample_rare_types, rebalance_generator=rebalance_generator,
        )
        loss = alpha * audio_loss + beta * llm_loss
        loss.backward()
        optimizer.step()

        running_loss  += loss.item()
        running_audio += audio_loss.item()
        running_llm   += llm_loss.item()
        n_batches += 1

    n_batches = max(n_batches, 1)
    return running_loss / n_batches, running_audio / n_batches, running_llm / n_batches, alpha, beta


@torch.no_grad()
def evaluate(loader, teacher_cache, encoder, audio_head, word_head, pooling_module,
             epoch, anneal_mode, device, total_anneal_epochs=15.0, pooling_mode="wide",
             weight_cache=None, freq_weighted_loss=False, oversample_rare_types=False):
    # NOTE: freq_weighted_loss/oversample_rare_types are passed through here for
    # consistency, but val_loss is a MODEL-SELECTION signal, not a training
    # target — deliberately NOT reseeding rebalance_generator per call, so
    # oversampling here uses fresh randomness each time (unlike training,
    # this isn't a step you'd want bit-reproducible).
    encoder.eval(); audio_head.eval(); word_head.eval(); pooling_module.eval()
    alpha, beta = get_stage1_weights(anneal_mode, epoch, total_anneal_epochs)

    running_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        audio_loss, llm_loss = compute_stage1_losses(
            batch, teacher_cache, encoder, audio_head, word_head, pooling_module,
            jitter_ms=None, pooling_mode=pooling_mode,
            weight_cache=weight_cache, freq_weighted_loss=freq_weighted_loss,
            oversample_rare_types=oversample_rare_types,
        )
        running_loss += (alpha * audio_loss + beta * llm_loss).item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


# ===========================================================================
#  Main
# ===========================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rebalance_generator = torch.Generator(device="cpu").manual_seed(args.seed) if args.oversample_rare_types else None

    print(f"=== Stage 1 training  heldout={args.heldout_subject}  mode={args.anneal_mode}  "
          f"freq_weighted={args.freq_weighted_loss}  oversample={args.oversample_rare_types}  device={device} ===")

    splits = make_loso_splits(args.heldout_subject)

    train_ds = MEGContinuousTrialDataset(splits["train"]["trials"], word_filter=splits["train"]["word_filter"],
                                          meg_base=args.meg_base)
    val_ds   = MEGContinuousTrialDataset(splits["val"]["trials"],   word_filter=splits["val"]["word_filter"],
                                          meg_base=args.meg_base)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_continuous_trials)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                               collate_fn=collate_continuous_trials)

    teacher_cache = torch.load(args.teacher_cache_path, weights_only=False)
    if args.gpt2_mid_layer is not None:
        for poem in ("poem1", "poem2"):
            teacher_cache[poem]["h_mid"] = teacher_cache[poem]["hidden_states_word_aligned"][args.gpt2_mid_layer]
        print(f"  [layer override] using GPT-2 layer {args.gpt2_mid_layer} as h_mid target")

    if args.audio_layer is not None:
        from teacher_cache import resample_time
        for poem in ("poem1", "poem2"):
            native = teacher_cache[poem]["audio_hidden_states_full"][args.audio_layer]
            teacher_cache[poem]["audio_target"] = resample_time(native, src_hz=49.95, tgt_hz=50.0)
        print(f"  [layer override] using wav2vec2 layer {args.audio_layer} as audio_target")

    weight_cache = None
    if args.freq_weighted_loss or args.oversample_rare_types:
        weight_cache = build_word_weight_cache(args.rebalance_alpha)
        print(f"Built word-type frequency weight cache (alpha={args.rebalance_alpha})")

    encoder         = MEGEncoder().to(device)
    audio_head      = AudioProjectionHead(encoder.backbone_dim).to(device)
    word_head       = WordProjectionHead(encoder.backbone_dim).to(device)
    pooling_module  = WordAttentionPooling(encoder.backbone_dim).to(device)

    params = list(encoder.parameters()) + list(audio_head.parameters()) + list(word_head.parameters())
    if args.pooling_mode == "wide":
        params += list(pooling_module.parameters())
    print(f"Trainable parameters: {sum(p.numel() for p in params):,}  pooling_mode={args.pooling_mode}  "
          f"(train trials={len(train_ds)}, val trials={len(val_ds)})")

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    patience_left = args.patience
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_audio, train_llm, alpha, beta = train_one_epoch(
            train_loader, teacher_cache, encoder, audio_head, word_head, pooling_module,
            optimizer, epoch, args.anneal_mode, device,
            jitter_ms=(args.jitter_low_ms, args.jitter_high_ms), total_anneal_epochs=args.anneal_epochs,
            pooling_mode=args.pooling_mode, weight_cache=weight_cache,
            freq_weighted_loss=args.freq_weighted_loss, oversample_rare_types=args.oversample_rare_types,
            rebalance_generator=rebalance_generator,
        )
        val_loss = evaluate(
            val_loader, teacher_cache, encoder, audio_head, word_head, pooling_module,
            epoch, args.anneal_mode, device, total_anneal_epochs=args.anneal_epochs,
            pooling_mode=args.pooling_mode, weight_cache=weight_cache,
            freq_weighted_loss=args.freq_weighted_loss, oversample_rare_types=args.oversample_rare_types,
        )

        print(f"epoch {epoch:3d}  alpha={alpha:.2f} beta={beta:.2f}  "
              f"train_loss={train_loss:.4f} (audio={train_audio:.4f} llm={train_llm:.4f})  "
              f"val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val, patience_left = val_loss, args.patience
            ckpt = {
                "encoder": encoder.state_dict(), "audio_head": audio_head.state_dict(),
                "word_head": word_head.state_dict(), "pooling": pooling_module.state_dict(),
                "epoch": epoch, "val_loss": val_loss,
                "heldout_subject": args.heldout_subject, "anneal_mode": args.anneal_mode,
                "pooling_mode": args.pooling_mode,
                "freq_weighted_loss": args.freq_weighted_loss,        # NEW — record on checkpoint
                "oversample_rare_types": args.oversample_rare_types,  # NEW — so eval/downstream can tell variants apart
                "rebalance_alpha": args.rebalance_alpha,
                "gpt2_mid_layer": args.gpt2_mid_layer, "audio_layer": args.audio_layer,
            }
            tag = ""
            if args.freq_weighted_loss:
                tag += "_freqw"
            if args.oversample_rare_types:
                tag += "_oversample"
            path = os.path.join(
                args.out_dir, f"stage1_best_{args.heldout_subject}_{args.anneal_mode}_{args.pooling_mode}{tag}.pt"
            )
            torch.save(ckpt, path)
            print(f"  [saved new best checkpoint -> {path}]")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stopping at epoch {epoch}: no val improvement for {args.patience} epochs.")
                break

    print(f"\nDone. Best val_loss={best_val:.4f}")

def build_arg_parser():
    p = argparse.ArgumentParser(description="Stage 1 (contrastive) MEG encoder training.")
    p.add_argument("--heldout_subject", type=str, default="sub-01")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--anneal_mode", type=str, default="joint_annealed",
                    choices=["joint_annealed", "llm_only", "joint_fixed", "hard_staged"])
    p.add_argument("--pooling_mode", type=str, default="wide", choices=["wide", "exact"])
    p.add_argument("--anneal_epochs", type=float, default=15.0)
    p.add_argument("--jitter_low_ms", type=float, default=50.0)
    p.add_argument("--jitter_high_ms", type=float, default=150.0)
    p.add_argument("--teacher_cache_path", type=str, default="teacher_cache.pt")
    p.add_argument("--meg_base", type=str, default="/fs/nexus-projects/brain_project/maryam_meg_dataset/icaed_Sai")
    p.add_argument("--out_dir", type=str, default="./checkpoints")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--freq_weighted_loss", action="store_true",
                    help="Reweight the contrastive loss's final average by inverse word-type frequency.")
    p.add_argument("--oversample_rare_types", action="store_true",
                    help="Resample word occurrences (with replacement) weighted inverse to word-type frequency, "
                         "before the loss is computed. Combine with --freq_weighted_loss deliberately, not by accident.")
    p.add_argument("--rebalance_alpha", type=float, default=0.5,
                    help="Weight = freq^(-alpha). 0=uniform/no effect, 1=fully inverse-frequency.")
    p.add_argument("--gpt2_mid_layer", type=int, default=None,
                    help="Override which cached GPT-2 sweep layer to use as h_mid.")
    p.add_argument("--audio_layer", type=int, default=None,
                    help="Override which cached wav2vec2 sweep layer to use as audio_target.")
    
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
