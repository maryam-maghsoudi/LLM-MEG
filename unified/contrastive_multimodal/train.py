"""
train.py — Stage 1 (contrastive alignment) training loop.

Orchestrates: splits.py -> new_dataset.py -> new_models.py (encoder + both
projection heads) -> pooling.py -> losses.py, wired exactly as designed
across the earlier pieces. This is STAGE 1 ONLY (encoder pretraining) —
Stage 2 (GRUHead, the injection-depth sweep) trains against this stage's
FROZEN, already-converged output and needs its own separate training
script once this one has produced a checkpoint.

DATASET SIZE IS THE GOVERNING CONSTRAINT THROUGHOUT THIS FILE, not an
afterthought — concretely:
  - batch_size defaults to 4: LOSO's train split is ~192 trials total
    (12 subjects x 2 poems x 8 sessions), so even a "small" batch size by
    ordinary standards is a meaningful fraction of an epoch.
  - Early stopping (--patience) is ON by default and enforced, not just
    available — with ~140 word instances underlying the LLM contrastive
    term, chasing train loss down past the point of val improvement is a
    memorization risk, not progress.
  - No multi-worker DataLoader, no streaming/lazy loading: total data is
    small enough that MEGContinuousTrialDataset already loads everything
    into memory eagerly at construction time (see new_dataset.py) — adding
    that infrastructure here would be solving a problem this project
    doesn't have.
  - The 4-way ablation dispatcher (get_stage1_weights) exists because §6
    explicitly calls for comparing joint-annealed against 3 alternatives
    given how little data there is to justify the more complex annealed
    design over a simpler fixed-weight or hard-staged one — this makes
    that comparison one flag away (--anneal_mode) instead of 4 near-
    duplicate scripts.

IMPORTS ASSUME CO-LOCATION: this file uses plain (not relative) imports —
teacher_cache.py, new_dataset.py, new_models.py, pooling.py, losses.py,
splits.py are assumed to sit in the same directory and this script is run
directly from there (matching how teacher_cache.py has been run throughout
this project: `python teacher_cache.py`, not `python -m package.teacher_cache`).

RUN MODES:
  python train.py --dry_run                     # synthetic smoke test, no files needed
  python train.py --heldout_subject sub-01 ...   # real training
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from new_dataset import MEGContinuousTrialDataset, collate_continuous_trials
from new_models import MEGEncoder, AudioProjectionHead, WordProjectionHead, JOINT_DIM, TOTAL_STRIDE
from pooling import WordAttentionPooling, pool_words, raw_length_to_encoder_frames
from losses import audio_contrastive_loss, llm_contrastive_loss, stage1_anneal_weights
from splits import make_loso_splits

POEM_TO_ID = {"poem1": 0, "poem2": 1}


# ===========================================================================
#  Ablation dispatcher (§6: joint-annealed vs 3 alternatives, 4 rows)
# ===========================================================================

def get_stage1_weights(mode: str, epoch: float, total_anneal_epochs: float = 15.0,
                        hard_stage_split_epoch: float = 15.0):
    """
    Returns (alpha, beta) for one of the 4 rows named in §6:
      "joint_annealed" (primary) — stage1_anneal_weights, 0.8 -> 0.2 crossfade
      "llm_only"       (alpha=0) — audio term ablated out entirely
      "joint_fixed"    (no anneal) — constant 0.5/0.5 every epoch
      "hard_staged"    (the ORIGINAL proposal this project moved away from)
                        — pure audio for hard_stage_split_epoch epochs, then
                        an ABRUPT switch to pure LLM. Kept runnable so the
                        "why we moved to joint-annealed" comparison in §6
                        is an actual result, not just an assertion.
    """
    if mode == "joint_annealed":
        return stage1_anneal_weights(epoch, total_anneal_epochs, alpha_start=0.8, alpha_end=0.2)
    elif mode == "llm_only":
        return 0.0, 1.0
    elif mode == "joint_fixed":
        return 0.5, 0.5
    elif mode == "hard_staged":
        return (1.0, 0.0) if epoch < hard_stage_split_epoch else (0.0, 1.0)
    else:
        raise ValueError(f"unknown stage1 ablation mode: {mode!r}")


# ===========================================================================
#  Batch -> teacher targets (glue between new_dataset.py and teacher_cache.py)
# ===========================================================================

def gather_word_targets(batch, teacher_cache, target_key, target_dim, poem_to_id=POEM_TO_ID):
    """
    General per-word target gatherer: pulls teacher_cache[poem][target_key][word_pos]
    for every word in this batch, matching (poem, word_pos). Generalized
    (originally just h_mid-specific) so train_stage2.py can reuse this
    exact loop for "hf_full" (768-d) targets instead of duplicating it —
    same reasoning as every other "keep one source of truth" fix already
    made across this project (POEM_LINES, GPT2_SWEEP_LAYERS, TOTAL_STRIDE).

    A plain Python loop over (B, N) — deliberately not vectorized: B and N
    are both small (dataset size again), and this is far more readable/
    less bug-prone than a vectorized gather for a loop this cheap.

    Returns target (B,N,target_dim), poem_ids (B,N) long, word_pos (B,N) long.
    """
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
    """Thin wrapper: h_mid targets specifically, for Stage 1's LLM contrastive loss."""
    return gather_word_targets(batch, teacher_cache, "h_mid", joint_dim, poem_to_id)



def build_audio_targets(batch, teacher_cache, T_out, total_stride=TOTAL_STRIDE):
    """
    Aligns each trial's poem-level, fixed audio_target (teacher_cache.py —
    depends only on (poem, time), not subject) to THIS batch's actual
    encoder output length. Truncates to
    min(cached target length, this trial's REAL encoder-frame count,
    batch T_out) and marks anything beyond as invalid — handles minor
    length mismatches between the cached target and a specific trial's
    real duration without assuming they're always exactly equal.

    Uses raw_length_to_encoder_frames (pooling.py) — the SAME formula
    pooling.py itself uses to find each trial's real (non-padded) encoder
    frame count, so audio-side and word-side alignment can never silently
    disagree about where a trial's real content ends.
    """
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
#  Shared forward pass — used by real training, evaluation, AND the smoke
#  test below. One code path, not three, so the dry run actually tests the
#  logic real training uses rather than a parallel reimplementation of it.
# ===========================================================================

def compute_stage1_losses(batch, teacher_cache, encoder, audio_head, word_head, pooling_module,
                           poem_to_id=POEM_TO_ID, jitter_ms=None, pooling_mode="wide",
                           audio_temperature=0.1, llm_temperature=0.1):
    z_dense = encoder(batch["meg_trial"])                 # (B, T_out, D)
    z_audio = audio_head(z_dense)                          # (B, T_out, JOINT_DIM)

    pooled, pool_valid = pool_words(
        pooling_mode, z_dense, batch["onset_samples"],
        offset_samples=batch["offset_samples"],       # only used by mode="exact"
        trial_mask=batch["trial_mask"],                # only used by mode="wide"
        attention_module=pooling_module,                # only used by mode="wide"
        jitter_ms=jitter_ms,                             # only used by mode="wide" (ignored for "exact")
    )
    z_word = word_head(pooled)                              # (B, N, JOINT_DIM)
    combined_valid = pool_valid & batch["valid_mask"]        # pooling-window validity AND alignment validity

    device = z_dense.device
    audio_target, frame_valid_mask = build_audio_targets(batch, teacher_cache, z_dense.shape[1])
    h_mid_target, poem_ids, word_pos_t = build_word_targets(batch, teacher_cache, poem_to_id)

    audio_loss = audio_contrastive_loss(
        z_audio, audio_target.to(device), frame_valid_mask.to(device), temperature=audio_temperature
    )
    llm_loss = llm_contrastive_loss(
        z_word, h_mid_target.to(device), combined_valid, poem_ids.to(device), word_pos_t.to(device),
        temperature=llm_temperature,
    )
    return audio_loss, llm_loss


def _move_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ===========================================================================
#  Train / eval epochs
# ===========================================================================

def train_one_epoch(loader, teacher_cache, encoder, audio_head, word_head, pooling_module,
                     optimizer, epoch, anneal_mode, device,
                     jitter_ms=(50.0, 150.0), total_anneal_epochs=15.0, pooling_mode="wide"):
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
             epoch, anneal_mode, device, total_anneal_epochs=15.0, pooling_mode="wide"):
    encoder.eval(); audio_head.eval(); word_head.eval(); pooling_module.eval()
    alpha, beta = get_stage1_weights(anneal_mode, epoch, total_anneal_epochs)

    running_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        audio_loss, llm_loss = compute_stage1_losses(
            batch, teacher_cache, encoder, audio_head, word_head, pooling_module,
            jitter_ms=None, pooling_mode=pooling_mode,
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

    print(f"=== Stage 1 training  heldout={args.heldout_subject}  mode={args.anneal_mode}  device={device} ===")

    splits = make_loso_splits(args.heldout_subject)

    train_ds = MEGContinuousTrialDataset(splits["train"]["trials"], word_filter=splits["train"]["word_filter"],
                                          meg_base=args.meg_base)
    val_ds   = MEGContinuousTrialDataset(splits["val"]["trials"],   word_filter=splits["val"]["word_filter"],
                                          meg_base=args.meg_base)
    # MEGContinuousTrialDataset.save_items() / .from_cache() are available
    # for reuse without re-reading .fif files on repeat runs — not wired up
    # here to keep this script's primary path simple; add if .fif loading
    # becomes a real bottleneck.

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_continuous_trials)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                               collate_fn=collate_continuous_trials)

    teacher_cache = torch.load(args.teacher_cache_path, weights_only=False)

    encoder         = MEGEncoder().to(device)
    audio_head      = AudioProjectionHead(encoder.backbone_dim).to(device)
    word_head       = WordProjectionHead(encoder.backbone_dim).to(device)
    pooling_module  = WordAttentionPooling(encoder.backbone_dim).to(device)

    params = list(encoder.parameters()) + list(audio_head.parameters()) + list(word_head.parameters())
    if args.pooling_mode == "wide":
        params += list(pooling_module.parameters())
    # mode="exact" never calls pooling_module.forward() at all (see
    # compute_stage1_losses -> pool_words), so its query would sit at its
    # random init forever if included here — left out entirely rather
    # than added-but-never-updated, which would be confusing to read later.
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
            pooling_mode=args.pooling_mode,
        )
        val_loss = evaluate(
            val_loader, teacher_cache, encoder, audio_head, word_head, pooling_module,
            epoch, args.anneal_mode, device, total_anneal_epochs=args.anneal_epochs,
            pooling_mode=args.pooling_mode,
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
            }
            path = os.path.join(
                args.out_dir, f"stage1_best_{args.heldout_subject}_{args.anneal_mode}_{args.pooling_mode}.pt"
            )
            torch.save(ckpt, path)
            print(f"  [saved new best checkpoint -> {path}]")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stopping at epoch {epoch}: no val improvement for {args.patience} epochs "
                      f"— with this little data, stopping promptly matters more than chasing train loss.")
                break

    print(f"\nDone. Best val_loss={best_val:.4f}")


# ===========================================================================
#  --dry_run: synthetic smoke test, zero external files required.
#  Uses compute_stage1_losses / build_word_targets / build_audio_targets /
#  get_stage1_weights directly — the SAME functions real training calls —
#  so this actually tests the orchestration logic, not a separate stub.
# ===========================================================================

def _make_fake_batch(device):
    torch.manual_seed(0)
    B = 3
    T_raws = [120, 100, 140]
    T_raw_max = max(T_raws)
    N_words_list = [5, 4, 5]
    N_max = max(N_words_list)
    poem_list = ["poem1", "poem1", "poem2"]

    from new_dataset import N_CHANNELS as C

    meg_trial   = torch.zeros(B, C, T_raw_max)
    trial_mask  = torch.zeros(B, T_raw_max, dtype=torch.bool)
    onset_samples  = torch.full((B, N_max), -1, dtype=torch.long)
    offset_samples = torch.full((B, N_max), -1, dtype=torch.long)
    valid_mask  = torch.zeros(B, N_max, dtype=torch.bool)
    word_poses  = []

    for b in range(B):
        T = T_raws[b]
        meg_trial[b, :, :T] = torch.randn(C, T) * 0.1
        trial_mask[b, :T] = True
        n = N_words_list[b]
        poses = list(range(n))
        word_poses.append(poses)
        for i, pos in enumerate(poses):
            onset = int((i + 0.5) / n * T * 0.8)
            offset = onset + 5
            if offset < T:
                onset_samples[b, i], offset_samples[b, i] = onset, offset
                valid_mask[b, i] = True

    batch = {
        "meg_trial": meg_trial.to(device), "trial_mask": trial_mask.to(device),
        "onset_samples": onset_samples.to(device), "offset_samples": offset_samples.to(device),
        "valid_mask": valid_mask.to(device),
        "word_texts": [["w"] * n for n in N_words_list], "word_poses": word_poses,
        "poem": poem_list, "subject": ["sub-fake"] * B, "session": [0] * B,
    }
    fake_teacher_cache = {
        "poem1": {"h_mid": torch.randn(10, JOINT_DIM), "audio_target": torch.randn(80, JOINT_DIM)},
        "poem2": {"h_mid": torch.randn(10, JOINT_DIM), "audio_target": torch.randn(80, JOINT_DIM)},
    }
    return batch, fake_teacher_cache


def run_dry_run(device):
    print("=== train.py DRY RUN (synthetic data, no real files needed) ===\n")
    batch, teacher_cache = _make_fake_batch(device)

    encoder        = MEGEncoder().to(device)
    audio_head     = AudioProjectionHead(encoder.backbone_dim).to(device)
    word_head      = WordProjectionHead(encoder.backbone_dim).to(device)
    pooling_module = WordAttentionPooling(encoder.backbone_dim).to(device)

    params = (list(encoder.parameters()) + list(audio_head.parameters())
              + list(word_head.parameters()) + list(pooling_module.parameters()))
    opt = torch.optim.AdamW(params, lr=1e-3)
    probe_before = next(encoder.parameters()).clone()

    for mode in ["joint_annealed", "llm_only", "joint_fixed", "hard_staged"]:
        encoder.train(); audio_head.train(); word_head.train(); pooling_module.train()
        opt.zero_grad()

        audio_loss, llm_loss = compute_stage1_losses(
            batch, teacher_cache, encoder, audio_head, word_head, pooling_module, jitter_ms=(50.0, 150.0)
        )
        alpha, beta = get_stage1_weights(mode, epoch=3)
        loss = alpha * audio_loss + beta * llm_loss

        assert torch.isfinite(loss), f"[{mode}] loss is not finite: {loss}"
        loss.backward()
        for p in params:
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"[{mode}] non-finite gradient found"
        opt.step()
        print(f"[OK] mode={mode:15s} alpha={alpha:.2f} beta={beta:.2f}  "
              f"audio_loss={audio_loss.item():.4f}  llm_loss={llm_loss.item():.4f}  total={loss.item():.4f}")

    probe_after = next(encoder.parameters())
    assert not torch.allclose(probe_before, probe_after), (
        "encoder parameters did not change after optimizer steps — something is detached from the graph"
    )
    print("\n[OK] encoder parameters changed after training steps — gradient wiring confirmed end-to-end")

    print("\n=== pooling_mode: wide vs exact ===")
    for mode in ["wide", "exact"]:
        encoder.train(); audio_head.train(); word_head.train(); pooling_module.train()
        opt.zero_grad()

        audio_loss, llm_loss = compute_stage1_losses(
            batch, teacher_cache, encoder, audio_head, word_head, pooling_module,
            jitter_ms=(50.0, 150.0), pooling_mode=mode,
        )
        loss = audio_loss + llm_loss
        assert torch.isfinite(loss), f"[pooling_mode={mode}] loss is not finite: {loss}"
        loss.backward()

        query_touched = pooling_module.query.grad is not None and pooling_module.query.grad.abs().sum().item() > 0
        if mode == "wide":
            assert query_touched, "wide mode must send a gradient to the learned query"
        else:
            assert not query_touched, (
                "exact mode must NEVER touch the learned query — it doesn't call WordAttentionPooling at all"
            )
        print(f"[OK] pooling_mode={mode:6s}  loss={loss.item():.4f}  query received gradient: {query_touched}")

    print("[OK] confirms empirically: the learned query is used in 'wide' mode only, exactly as designed")

    # get_stage1_weights dispatch correctness, independent of the forward pass above
    a, b = get_stage1_weights("llm_only", epoch=0)
    assert a == 0.0 and b == 1.0
    a, b = get_stage1_weights("joint_fixed", epoch=100)
    assert a == 0.5 and b == 0.5
    a, b = get_stage1_weights("hard_staged", epoch=5, hard_stage_split_epoch=15)
    assert (a, b) == (1.0, 0.0)
    a, b = get_stage1_weights("hard_staged", epoch=20, hard_stage_split_epoch=15)
    assert (a, b) == (0.0, 1.0)
    print("[OK] get_stage1_weights: all 4 ablation modes dispatch correctly")

    print("\n=== DRY RUN PASSED ===")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Stage 1 (contrastive) MEG encoder training.")
    p.add_argument("--heldout_subject", type=str, default="sub-01")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4,
                    help="Kept small deliberately — LOSO train split is only ~192 trials total.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10,
                    help="Early-stopping patience (epochs). With ~140 word instances total, stopping "
                         "promptly matters more than in a large-data setting.")
    p.add_argument("--anneal_mode", type=str, default="joint_annealed",
                    choices=["joint_annealed", "llm_only", "joint_fixed", "hard_staged"])
    p.add_argument("--pooling_mode", type=str, default="wide", choices=["wide", "exact"],
                    help="wide = WordAttentionPooling (learned query, jitter-robust). "
                         "exact = plain onset-to-offset averaging, no learned params, no jitter.")
    p.add_argument("--anneal_epochs", type=float, default=15.0)
    p.add_argument("--jitter_low_ms", type=float, default=50.0)
    p.add_argument("--jitter_high_ms", type=float, default=150.0)
    p.add_argument("--teacher_cache_path", type=str, default="teacher_cache.pt")
    p.add_argument("--meg_base", type=str, default="/fs/nexus-projects/brain_project/maryam_meg_dataset/icaed_Sai")
    p.add_argument("--out_dir", type=str, default="./checkpoints")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry_run", action="store_true",
                    help="Run the synthetic end-to-end smoke test instead of real training. "
                         "No .fif files, teacher_cache.pt, or GPU required.")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dry_run:
        run_dry_run(_device)
    else:
        main(args)
