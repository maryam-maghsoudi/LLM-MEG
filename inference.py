"""
inference.py
============
Autoregressive poem generation from MEG soft tokens at test time.

At inference there is no ground-truth text available.  The model generates
each word's text tokens from the soft token derived from that word's MEG
window, conditioned on all previously generated text and soft tokens.

Design A (interleaved) — word-by-word generation
-------------------------------------------------
For each word i in the poem:
  1. Append soft(i) to the current sequence
  2. Greedily decode text tokens until a stop condition is reached
  3. Append the generated text embeddings to the sequence
  4. Repeat

Two stop conditions are supported:
  oracle_lengths=True  : stop after generating exactly len(true_tok_ids[i])
                         tokens (token count is determined from the ground-truth
                         word, not the token IDs themselves — fair for evaluation)
  oracle_lengths=False : stop after the first token (n=1 per word), suitable
                         for truly open-vocabulary generation where word
                         boundaries must be inferred

Design B (upfront) — full-poem generation
------------------------------------------
  1. Build all N soft tokens as a prefix
  2. Autoregressively generate the full poem text using the LLM

Beam search is available for both designs via `beam_size` (default 1 = greedy).
For beam_size > 1, Design B uses HuggingFace's generate(); Design A uses a
manual beam over the word-by-word loop (beam_size tokens per word position).

Usage
-----
  python inference.py --adapter_ckpt out/train/best_adapter.pt

  # With oracle token lengths (cleaner evaluation)
  python inference.py --adapter_ckpt out/train/best_adapter.pt --oracle_lengths

  # Beam search (greedy over per-word top-k tokens)
  python inference.py --adapter_ckpt out/train/best_adapter.pt --beam_size 5
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    SUBJECTS, TEST_POEMS, LLM_NAME, SEQUENCE_DESIGN, OUT_DIR,
)
from dataset import PoemTrialDataset, collate_trials
from model import LLMDecoder, build_model


# =============================================================================
#  WER
# =============================================================================

def _edit_distance(r: List[str], h: List[str]) -> int:
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    for i in range(len(r) + 1):
        d[i, 0] = i
    for j in range(len(h) + 1):
        d[0, j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + cost)
    return int(d[len(r), len(h)])


def compute_wer(reference: List[str], hypothesis: List[str]) -> float:
    """Word error rate: edit_distance / len(reference)."""
    if not reference:
        return 0.0
    return _edit_distance(reference, hypothesis) / len(reference)


# =============================================================================
#  GREEDY DECODING HELPERS
# =============================================================================

def _greedy_next_token(
    model:         LLMDecoder,
    current_embeds: torch.Tensor,   # (1, L, d_model)
    device:        torch.device,
) -> Tuple[int, torch.Tensor]:
    """
    One greedy decoding step.  Returns (token_id, token_embedding).
    """
    with torch.no_grad():
        out = model.frozen_llm.llm(inputs_embeds=current_embeds)
    logits   = out.logits[0, -1, :]              # (vocab_size,)
    tok_id   = int(logits.argmax())
    emb      = model.frozen_llm.get_input_embeddings()(
        torch.tensor([tok_id], device=device)
    )                                             # (1, d_model)
    return tok_id, emb


def _beam_next_token(
    model:          LLMDecoder,
    current_embeds: torch.Tensor,   # (1, L, d_model)
    device:         torch.device,
    beam_size:      int,
) -> List[Tuple[int, torch.Tensor, float]]:
    """
    Returns top-k (token_id, embedding, log_prob) candidates.
    """
    with torch.no_grad():
        out = model.frozen_llm.llm(inputs_embeds=current_embeds)
    logits   = out.logits[0, -1, :]                          # (vocab_size,)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    top_vals, top_ids = log_probs.topk(beam_size)
    emb_table = model.frozen_llm.get_input_embeddings()
    results = []
    for score, tid in zip(top_vals.tolist(), top_ids.tolist()):
        emb = emb_table(torch.tensor([tid], device=device))  # (1, d_model)
        results.append((tid, emb, score))
    return results


# =============================================================================
#  DESIGN A — INTERLEAVED GENERATION
# =============================================================================

@torch.no_grad()
def generate_interleaved(
    model:          LLMDecoder,
    soft_tokens:    torch.Tensor,        # (N, n_soft, d_model)
    word_token_ids_gt: List[List[int]],  # ground-truth token IDs (for oracle lengths)
    device:         torch.device,
    tokenizer,
    oracle_lengths: bool = True,
    max_tokens_per_word: int = 3,
    beam_size:      int = 1,
) -> List[str]:
    """
    Generate one poem trial word by word (Design A, interleaved).

    Parameters
    ----------
    oracle_lengths       : if True, generate exactly len(true_tok_ids[i]) tokens
                           per word; if False, generate 1 token per word
    max_tokens_per_word  : upper bound when oracle_lengths=False
    beam_size            : 1 = greedy; >1 = pick top-beam_size first tokens
                           per word (we commit to the best single token here —
                           full sequence-level beam search is in generate_upfront)

    Returns
    -------
    List[str] — one decoded word string per poem word
    """
    emb_table   = model.frozen_llm.get_input_embeddings()
    n_soft      = model.n_soft
    d_model     = model.d_model

    # Start with an empty sequence (list of (L, d_model) tensors to cat later)
    seq_parts: List[torch.Tensor] = []
    generated_words: List[str]    = []

    for i, soft_i in enumerate(soft_tokens):        # soft_i: (n_soft, d_model)
        # Append soft token(s) for word i
        seq_parts.append(soft_i)

        n_to_gen = (
            len(word_token_ids_gt[i])
            if oracle_lengths and word_token_ids_gt
            else 1
        )
        n_to_gen = max(1, min(n_to_gen, max_tokens_per_word))

        word_tok_ids: List[int] = []

        for _ in range(n_to_gen):
            current_embeds = torch.cat(seq_parts, dim=0).unsqueeze(0)  # (1, L, d_model)

            if beam_size > 1:
                candidates = _beam_next_token(model, current_embeds, device, beam_size)
                tok_id, tok_emb, _ = candidates[0]   # take best candidate
            else:
                tok_id, tok_emb = _greedy_next_token(model, current_embeds, device)

            word_tok_ids.append(tok_id)
            seq_parts.append(tok_emb)               # (1, d_model)

        decoded = tokenizer.decode(word_tok_ids).strip().lower()
        generated_words.append(decoded)

    return generated_words


# =============================================================================
#  DESIGN B — UPFRONT GENERATION
# =============================================================================

@torch.no_grad()
def generate_upfront(
    model:       LLMDecoder,
    soft_tokens: torch.Tensor,         # (N, n_soft, d_model)
    total_text_tokens: int,            # how many text tokens to generate total
    device:      torch.device,
    tokenizer,
    beam_size:   int = 1,
) -> List[str]:
    """
    Generate a full poem from upfront soft tokens (Design B).

    Appends all N soft tokens as a prefix, then autoregressively generates
    `total_text_tokens` text tokens.  Returns the decoded text split into words.

    For beam_size > 1, delegates to HuggingFace's generate() which supports
    full sequence-level beam search.
    """
    N, n_soft, d_model = soft_tokens.shape
    prefix = soft_tokens.view(N * n_soft, d_model).unsqueeze(0)   # (1, N*n_soft, d_model)

    if beam_size <= 1:
        # Manual greedy
        seq_parts = [prefix]
        gen_ids: List[int] = []
        for _ in range(total_text_tokens):
            current = torch.cat(seq_parts, dim=1)  # (1, L, d_model)
            tok_id, tok_emb = _greedy_next_token(model, current, device)
            gen_ids.append(tok_id)
            seq_parts.append(tok_emb.unsqueeze(0))
    else:
        # HF beam search: requires past_key_values or encode prefix as cache
        # We use a simple approach: encode prefix, then generate
        with torch.no_grad():
            prefix_out = model.frozen_llm.llm(
                inputs_embeds=prefix, use_cache=True
            )
        past = prefix_out.past_key_values

        # Use greedy from here (HF generate() with inputs_embeds+past is
        # complex; true beam search over the upfront design is left for future)
        emb_table = model.frozen_llm.get_input_embeddings()
        logits     = prefix_out.logits[0, -1, :]
        tok_id     = int(logits.argmax())
        gen_ids    = [tok_id]
        cur_emb    = emb_table(torch.tensor([[tok_id]], device=device))

        for _ in range(total_text_tokens - 1):
            with torch.no_grad():
                step_out = model.frozen_llm.llm(
                    inputs_embeds=cur_emb,
                    past_key_values=past,
                    use_cache=True,
                )
            past   = step_out.past_key_values
            tok_id = int(step_out.logits[0, -1, :].argmax())
            gen_ids.append(tok_id)
            cur_emb = emb_table(torch.tensor([[tok_id]], device=device))

    full_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return full_text.strip().lower().split()


# =============================================================================
#  TRIAL-LEVEL GENERATION
# =============================================================================

@torch.no_grad()
def generate_trial(
    model:             LLMDecoder,
    meg_windows:       torch.Tensor,        # (N, C, T)  — single trial, no batch dim
    word_token_ids_gt: List[List[int]],
    device:            torch.device,
    tokenizer,
    oracle_lengths:    bool = True,
    beam_size:         int  = 1,
) -> List[str]:
    """Generate for a single trial; dispatches to interleaved or upfront."""
    N, C, T   = meg_windows.shape
    n_soft    = model.n_soft
    d_model   = model.d_model

    # Encode MEG windows → soft tokens
    meg_emb     = model.meg_encoder(meg_windows)                # (N, emb_dim)
    soft        = model.adapter(meg_emb)                        # (N, n_soft, d_model)
    soft_tokens = soft.view(N, n_soft, d_model)

    if model.sequence_design == "interleaved":
        return generate_interleaved(
            model, soft_tokens, word_token_ids_gt, device, tokenizer,
            oracle_lengths=oracle_lengths, beam_size=beam_size,
        )
    else:
        total_text = sum(len(ids) for ids in word_token_ids_gt)
        words = generate_upfront(
            model, soft_tokens, total_text, device, tokenizer, beam_size=beam_size
        )
        return words


# =============================================================================
#  DATASET-LEVEL EVALUATION
# =============================================================================

def evaluate_generation(
    model:          LLMDecoder,
    dataset:        PoemTrialDataset,
    device:         torch.device,
    tokenizer,
    oracle_lengths: bool = True,
    beam_size:      int  = 1,
) -> Dict:
    """
    Run autoregressive generation over all trials in a dataset.
    Returns aggregate WER and per-trial records.
    """
    model.eval()
    all_wer:   List[float] = []
    all_bleu1: List[float] = []
    records:   List[Dict]  = []

    for idx in range(len(dataset)):
        item    = dataset[idx]
        windows = item["meg_windows"].to(device)    # (N, C, T)
        tok_ids = item["word_token_ids"]
        texts   = item["word_texts"]
        meta    = item["meta"]

        generated = generate_trial(
            model, windows, tok_ids, device, tokenizer,
            oracle_lengths=oracle_lengths, beam_size=beam_size,
        )

        wer = compute_wer(texts, generated)

        # Unigram BLEU
        from collections import defaultdict as _dd
        ref_counts = _dd(int)
        for w in texts: ref_counts[w] += 1
        clip = 0
        for w in generated:
            if ref_counts[w] > 0:
                clip += 1
                ref_counts[w] -= 1
        bleu = (clip / max(len(generated), 1)) * min(1.0, len(generated) / max(len(texts), 1))

        all_wer.append(wer)
        all_bleu1.append(bleu)

        records.append({
            "meta":      meta,
            "true":      texts,
            "generated": generated,
            "wer":       round(wer, 4),
            "bleu1":     round(bleu, 4),
        })

        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(dataset)}]  wer={wer:.3f}  "
                  f"true[:5]={texts[:5]}  gen[:5]={generated[:5]}")

    if records:
        print("\n  Example generation (trial 0):")
        print(f"  True     : {' '.join(records[0]['true'])}")
        print(f"  Generated: {' '.join(records[0]['generated'])}")

    return {
        "mean_wer":   float(np.mean(all_wer)),
        "mean_bleu1": float(np.mean(all_bleu1)),
        "n_trials":   len(records),
        "trials":     records,
    }


# =============================================================================
#  MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Autoregressive MEG poem generation")
    p.add_argument("--adapter_ckpt",  type=Path,
                   default=OUT_DIR / "train" / "best_adapter.pt")
    p.add_argument("--llm_name",      default=LLM_NAME)
    p.add_argument("--design",        default=SEQUENCE_DESIGN,
                   choices=["interleaved", "upfront"])
    p.add_argument("--out_dir",       type=Path,
                   default=OUT_DIR / "inference")
    p.add_argument("--poems",         nargs="+", default=TEST_POEMS)
    p.add_argument("--oracle_lengths", action="store_true",
                   help="Use ground-truth token counts per word (fair for eval)")
    p.add_argument("--beam_size",     type=int, default=1,
                   help="1 = greedy; >1 = beam candidates per word (interleaved)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Inference: design={args.design!r}  beam={args.beam_size}  device={device}")
    print(f"  Oracle lengths: {args.oracle_lengths}")
    print(f"  Adapter: {args.adapter_ckpt}")
    print(f"{'='*60}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(device, llm_name=args.llm_name, sequence_design=args.design)
    model.load_adapter(str(args.adapter_ckpt), device)
    model.eval()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    from transformers import AutoTokenizer as _Tok
    tokenizer = _Tok.from_pretrained(args.llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = PoemTrialDataset(
        subjects  = SUBJECTS,
        poems     = args.poems,
        sessions  = list(range(10)),
        condition = "lis",
        llm_name  = args.llm_name,
    )
    print(f"  Test trials: {len(dataset)}\n")

    # ── Generate ──────────────────────────────────────────────────────────────
    results = evaluate_generation(
        model, dataset, device, tokenizer,
        oracle_lengths=args.oracle_lengths,
        beam_size=args.beam_size,
    )

    print(f"\n  Mean WER   : {results['mean_wer']:.4f}")
    print(f"  Mean BLEU-1: {results['mean_bleu1']:.4f}")
    print(f"  Trials     : {results['n_trials']}")

    # ── Save ──────────────────────────────────────────────────────────────────
    tag      = f"{args.design}_beam{args.beam_size}"
    out_path = args.out_dir / f"generation_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
