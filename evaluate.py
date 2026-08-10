"""
evaluate.py
===========
Teacher-forced evaluation and ablation analysis for the trained LLM decoder.

For each word in a trial the model sees the ground-truth preceding context
(previous soft tokens + previous text tokens).  We extract the logit at each
soft-token position: that logit is the model's prediction for the first text
token of that word conditioned on its MEG evidence.

Metrics
-------
  exact_match      : fraction of words where argmax at soft-token position ==
                     first token of the true word
  bert_sim         : mean cosine similarity (BERT embeddings) of predicted
                     vs. true word strings
  bleu1            : corpus-level unigram BLEU over the full poem sequence
  restricted_R@k   : rank of true word's first-token probability among the
                     76-word closed vocabulary (for comparison to contrastive
                     decoder R@k from the main paper)
  restricted_MRR   : mean reciprocal rank over the 76-word vocabulary

Ablations (run all before trusting any positive result)
---------
  shuffle          : MEG windows randomly permuted across word positions before
                     encoding — tests whether soft tokens carry word-specific info
  random_soft      : soft tokens replaced with Gaussian noise at test time —
                     tests whether the LLM uses soft tokens at all at inference
  no_soft          : LLM runs on text tokens only, no soft tokens — measures
                     what the language prior alone achieves on these specific poems

Usage
-----
  # Full evaluation + all ablations
  python evaluate.py --adapter_ckpt out/train/best_adapter.pt

  # Specific ablation only
  python evaluate.py --adapter_ckpt out/train/best_adapter.pt --ablation shuffle
"""

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from config import (
    SUBJECTS, TEST_POEMS, TRAIN_POEMS, TRAIN_SESSIONS, VAL_SESSIONS,
    POEM_KEYS, LLM_NAME, SEQUENCE_DESIGN, N_SOFT_TOKENS, LLM_D_MODEL,
    OUT_DIR,
)
from dataset import PoemTrialDataset, collate_trials
from model import LLMDecoder, build_model

warnings.filterwarnings("ignore")


# =============================================================================
#  SEQUENCE POSITION HELPERS
# =============================================================================

def _soft_positions_interleaved(
    word_token_ids: List[List[int]], n_soft: int
) -> List[int]:
    """
    Returns the sequence position of the soft token for each word in an
    interleaved sequence.  The logit at this position predicts the first
    text token of that word.
    """
    positions = []
    pos = 0
    for tok_ids in word_token_ids:
        positions.append(pos)
        pos += n_soft + len(tok_ids)
    return positions


def _soft_positions_upfront(
    word_token_ids: List[List[int]], n_soft: int
) -> List[int]:
    """
    Returns the position whose logit predicts the first text token of each
    word in an upfront sequence: [soft*N] [text_w0] [text_w1] ...

    For word 0: last soft-token position (N*n_soft - 1)
    For word i: last position before word i's text block
    """
    N = len(word_token_ids)
    positions = []
    cursor = N * n_soft   # first text token of word 0
    for i, tok_ids in enumerate(word_token_ids):
        positions.append(cursor - 1)
        cursor += len(tok_ids)
    return positions


def _get_soft_positions(
    word_token_ids: List[List[int]], n_soft: int, design: str
) -> List[int]:
    if design == "interleaved":
        return _soft_positions_interleaved(word_token_ids, n_soft)
    return _soft_positions_upfront(word_token_ids, n_soft)


# =============================================================================
#  PER-WORD PREDICTION EXTRACTION
# =============================================================================

def extract_word_predictions(
    logits_b:       torch.Tensor,      # (seq_len, vocab_size)
    word_token_ids: List[List[int]],
    design:         str,
    n_soft:         int,
) -> Tuple[List[int], List[torch.Tensor]]:
    """
    Extract per-word predictions from a single trial's logits.

    Returns
    -------
    pred_first_tokens : List[int]  — argmax predicted first token per word
    word_logits       : List[Tensor]  — (vocab_size,) logit vector per word,
                        used for restricted rank computation
    """
    positions      = _get_soft_positions(word_token_ids, n_soft, design)
    pred_tokens    = []
    word_logit_list = []
    for pos in positions:
        logit_vec = logits_b[pos]                  # (vocab_size,)
        pred_tokens.append(int(logit_vec.argmax()))
        word_logit_list.append(logit_vec)
    return pred_tokens, word_logit_list


# =============================================================================
#  METRICS
# =============================================================================

def exact_match(
    pred_first_tokens: List[int],
    word_token_ids:    List[List[int]],
) -> float:
    """Fraction of words where predicted first token == true first token."""
    correct = sum(
        p == ids[0]
        for p, ids in zip(pred_first_tokens, word_token_ids)
        if ids
    )
    total = sum(1 for ids in word_token_ids if ids)
    return correct / max(total, 1)


def bleu1(predicted_words: List[str], true_words: List[str]) -> float:
    """Unigram BLEU (corpus-level) between predicted and reference word lists."""
    ref_counts: Dict[str, int] = defaultdict(int)
    for w in true_words:
        ref_counts[w] += 1

    clip_count = 0
    for w in predicted_words:
        if ref_counts[w] > 0:
            clip_count += 1
            ref_counts[w] -= 1

    precision = clip_count / max(len(predicted_words), 1)
    # Brevity penalty
    bp = min(1.0, len(predicted_words) / max(len(true_words), 1))
    return bp * precision


def restricted_rank(
    word_logit_list:  List[torch.Tensor],   # per-word (vocab_size,) logits
    word_token_ids:   List[List[int]],       # true token IDs per word
    vocab_words:      List[str],             # 76-word closed vocabulary
    tokenizer,
) -> Dict[str, float]:
    """
    For each word position, rank the true word's first-token probability among
    all 76 vocabulary words' first tokens.

    This allows apples-to-apples comparison with the contrastive decoder's R@k.
    Note: multiple vocab words may share the same first token (BPE ambiguity);
    we rank by unique first-token probabilities, so effective vocabulary may be
    smaller than 76 in some positions.
    """
    # Pre-compute first token IDs for all 76 vocab words (done once)
    vocab_first_tokens = []
    for w in vocab_words:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        vocab_first_tokens.append(ids[0] if ids else tokenizer.unk_token_id)

    ranks = []
    for logit_vec, true_ids in zip(word_logit_list, word_token_ids):
        if not true_ids:
            continue
        true_first = true_ids[0]
        probs = torch.softmax(logit_vec.float(), dim=-1)

        # Score of the true word
        true_score = probs[true_first].item()

        # Rank among 76 vocab first-token scores (1 = best)
        rank = 1 + sum(
            1 for ft in vocab_first_tokens
            if probs[ft].item() > true_score
        )
        ranks.append(rank)

    if not ranks:
        return {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "MRR": 0.0,
                "median_rank": 0, "n": 0}

    ranks_arr = np.array(ranks)
    return {
        "R@1":         float((ranks_arr <= 1).mean()),
        "R@5":         float((ranks_arr <= 5).mean()),
        "R@10":        float((ranks_arr <= 10).mean()),
        "MRR":         float((1.0 / ranks_arr).mean()),
        "median_rank": int(np.median(ranks_arr)),
        "n":           len(ranks_arr),
    }


# =============================================================================
#  BERT SIMILARITY
# =============================================================================

class _BERTSimilarity:
    """Cache BERT embeddings for the 76 vocab words and compute cosine sim."""

    def __init__(self, device: torch.device, model_name: str = "bert-base-uncased"):
        print(f"  Loading BERT ({model_name}) for similarity metric...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device    = device
        self._cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def embed(self, word: str) -> torch.Tensor:
        if word not in self._cache:
            enc = self.tokenizer(word, return_tensors="pt").to(self.device)
            out = self.model(**enc).last_hidden_state          # (1, L, 768)
            emb = out[0].mean(dim=0)                           # (768,)
            self._cache[word] = torch.nn.functional.normalize(emb, dim=0)
        return self._cache[word]

    def similarity(self, pred_word: str, true_word: str) -> float:
        ep = self.embed(pred_word)
        et = self.embed(true_word)
        return float((ep * et).sum())


# =============================================================================
#  CORE EVALUATION PASS
# =============================================================================

@torch.no_grad()
def run_eval_pass(
    model:           LLMDecoder,
    loader:          DataLoader,
    device:          torch.device,
    tokenizer,
    vocab_words:     List[str],
    bert_sim:        Optional[_BERTSimilarity] = None,
    shuffle_meg:     bool = False,
    random_soft:     bool = False,
    no_soft:         bool = False,
) -> Dict:
    """
    Teacher-forced evaluation pass.

    Parameters
    ----------
    shuffle_meg  : permute MEG windows randomly across word positions per trial
    random_soft  : replace adapter output with Gaussian noise (same shape)
    no_soft      : feed text tokens only to the LLM (no soft tokens at all)
    """
    model.eval()
    n_soft  = model.n_soft
    d_model = model.d_model
    design  = model.sequence_design
    emb_table = model.frozen_llm.get_input_embeddings()

    all_em:          List[float] = []
    all_bert:        List[float] = []
    all_bleu:        List[float] = []
    all_word_logits: List[torch.Tensor] = []
    all_true_ids:    List[List[int]] = []
    total_loss       = 0.0
    n_batches        = 0

    for batch in loader:
        meg_windows    = batch["meg_windows"].to(device)   # (B, N, C, T): N is number of words in test poem
        valid_mask     = batch["valid_mask"].to(device)
        word_token_ids = batch["word_token_ids"]

        B, N, C, T = meg_windows.shape

        # ── Encode MEG → soft tokens ────────────────────────────────────────
        if no_soft:
            soft_tokens = None
        else:
            flat    = meg_windows.view(B * N, C, T)
            meg_emb = model.meg_encoder(flat)              # (B*N, emb_dim)

            if shuffle_meg:
                # Re-pair MEG embeddings to random word positions within each trial
                meg_emb = meg_emb.view(B, N, -1)
                for b in range(B):
                    perm = torch.randperm(N, device=device)
                    meg_emb[b] = meg_emb[b][perm]
                meg_emb = meg_emb.view(B * N, -1)

            soft = model.adapter(meg_emb)                  # (B*N, n_soft, d_model)

            if random_soft:
                soft = torch.randn_like(soft)

            soft_tokens = soft.view(B, N, n_soft, d_model)

        # ── Build sequences and collect logits ──────────────────────────────
        all_embeds: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        for b in range(B):
            tok_ids_b = word_token_ids[b]

            if no_soft:
                # Text-only baseline: just feed all text tokens, no soft tokens
                all_text = [tid for tok_ids in tok_ids_b for tid in tok_ids]
                ids_t    = torch.tensor(all_text, dtype=torch.long, device=device)
                emb_b    = emb_table(ids_t)
                lab_b    = ids_t
            elif design == "interleaved":
                emb_b, lab_b = model._build_interleaved(soft_tokens[b], tok_ids_b)
            else:
                emb_b, lab_b = model._build_upfront(soft_tokens[b], tok_ids_b)

            all_embeds.append(emb_b)
            all_labels.append(lab_b)

        max_len        = max(e.shape[0] for e in all_embeds)
        inputs_embeds  = torch.zeros(B, max_len, d_model, device=device)
        labels_padded  = torch.full((B, max_len), -100, dtype=torch.long, device=device)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for b in range(B):
            L = all_embeds[b].shape[0]
            inputs_embeds[b, :L]  = all_embeds[b]
            labels_padded[b, :L]  = all_labels[b]
            attention_mask[b, :L] = 1

        out = model.frozen_llm(
            inputs_embeds=inputs_embeds,
            labels=labels_padded,
            attention_mask=attention_mask,
        )
        total_loss += out.loss.item()
        n_batches  += 1

        logits = out.logits   # (B, seq_len, vocab_size)

        # ── Per-word metrics ─────────────────────────────────────────────────
        for b in range(B):
            tok_ids_b = word_token_ids[b]
            logits_b  = logits[b]   # (seq_len, vocab_size)

            if no_soft:
                # In text-only mode, position j predicts token j+1.
                # Word i's first token is at cumulative text position;
                # the logit predicting it is at the position just before.
                pred_tokens = []
                word_logit_list = []
                cursor = 0
                for tok_ids in tok_ids_b:
                    pos = max(cursor - 1, 0)   # logit before word i's first token
                    pred_tokens.append(int(logits_b[pos].argmax()))
                    word_logit_list.append(logits_b[pos])
                    cursor += len(tok_ids)
            else:
                pred_tokens, word_logit_list = extract_word_predictions(
                    logits_b, tok_ids_b, design, n_soft
                )

            # Decode predicted first tokens to word strings
            pred_words = [
                tokenizer.decode([pt]).strip().lower()
                for pt in pred_tokens
            ]
            true_words = [
                tokenizer.decode(ids).strip().lower()
                for ids in tok_ids_b
            ]

            all_em.append(exact_match(pred_tokens, tok_ids_b))
            all_bleu.append(bleu1(pred_words, true_words))

            if bert_sim is not None:
                sims = [
                    bert_sim.similarity(pw, tw)
                    for pw, tw in zip(pred_words, true_words)
                    if pw and tw
                ]
                if sims:
                    all_bert.append(float(np.mean(sims)))

            all_word_logits.extend(word_logit_list)
            all_true_ids.extend(tok_ids_b)

    # ── Aggregate ────────────────────────────────────────────────────────────
    rrank = restricted_rank(all_word_logits, all_true_ids, vocab_words, tokenizer)

    return {
        "loss":          total_loss / max(n_batches, 1),
        "exact_match":   float(np.mean(all_em)) if all_em else 0.0,
        "bleu1":         float(np.mean(all_bleu)) if all_bleu else 0.0,
        "bert_sim":      float(np.mean(all_bert)) if all_bert else None,
        **{f"restricted_{k}": v for k, v in rrank.items()},
    }


# =============================================================================
#  NO-SOFT BASELINE (text-only LM forward pass)
# =============================================================================

def run_no_soft_baseline(
    model: LLMDecoder, loader: DataLoader,
    device: torch.device, tokenizer, vocab_words: List[str],
    bert_sim: Optional[_BERTSimilarity] = None,
) -> Dict:
    return run_eval_pass(
        model, loader, device, tokenizer, vocab_words, bert_sim,
        no_soft=True,
    )


# =============================================================================
#  MAIN
# =============================================================================

def _load_model_for_eval(
    adapter_ckpt: Path, device: torch.device, llm_name: str, design: str
) -> LLMDecoder:
    model = build_model(device, llm_name=llm_name, sequence_design=design)
    model.load_adapter(str(adapter_ckpt), device)
    print(f"  Adapter loaded from {adapter_ckpt}")
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate LLM-guided MEG decoder")
    p.add_argument("--adapter_ckpt", type=Path,
                   default=OUT_DIR / "train" / "best_adapter.pt")
    p.add_argument("--llm_name",     default=LLM_NAME)
    p.add_argument("--design",       default=SEQUENCE_DESIGN,
                   choices=["interleaved", "upfront"])
    p.add_argument("--out_dir",      type=Path,
                   default=OUT_DIR / "eval")
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--ablation",
                   choices=["all", "shuffle", "random_soft", "no_soft"],
                   default="all",
                   help="Which ablations to run (default: all)")
    p.add_argument("--no_bert",      action="store_true",
                   help="Skip BERT similarity (faster, no BERT download)")
    p.add_argument("--poems",        nargs="+", default=TEST_POEMS,
                   help="Poems to evaluate on (default: TEST_POEMS from config)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Evaluation: design={args.design!r}  device={device}")
    print(f"  Adapter : {args.adapter_ckpt}")
    print(f"  Poems   : {args.poems}")
    print(f"{'='*60}\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    test_ds = PoemTrialDataset(
        subjects  = SUBJECTS,
        poems     = args.poems,
        sessions  = list(range(10)),
        condition = "lis",
        llm_name  = args.llm_name,
    )
    loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_trials,
    )

    # Build vocab from ALL poems so restricted_rank uses the full 76-word
    # candidate set — same as the contrastive decoder, making R@k comparable.
    full_vocab_ds = PoemTrialDataset(
        subjects  = SUBJECTS[:1],   # one subject is enough to get all word types
        poems     = POEM_KEYS,
        sessions  = [0],
        condition = "lis",
        llm_name  = args.llm_name,
    )
    vocab_words = full_vocab_ds.vocab
    print(f"  Test trials: {len(test_ds)}  Vocab size (all poems): {len(vocab_words)}")

    # ── Tokenizer (for decoding predictions) ─────────────────────────────────
    from transformers import AutoTokenizer as _Tok
    tokenizer = _Tok.from_pretrained(args.llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Optional BERT similarity ──────────────────────────────────────────────
    bert_sim = None if args.no_bert else _BERTSimilarity(device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = _load_model_for_eval(
        args.adapter_ckpt, device, args.llm_name, args.design
    )

    results = {}

    # ── Main evaluation ───────────────────────────────────────────────────────
    print("\n[1/4] Standard evaluation (real MEG)...")
    results["standard"] = run_eval_pass(
        model, loader, device, tokenizer, vocab_words, bert_sim
    )
    _print_metrics("standard", results["standard"])

    # ── Ablations ─────────────────────────────────────────────────────────────
    run_all = args.ablation == "all"

    if run_all or args.ablation == "shuffle":
        print("\n[2/4] Shuffle ablation (MEG re-paired to random words)...")
        results["shuffle"] = run_eval_pass(
            model, loader, device, tokenizer, vocab_words, bert_sim,
            shuffle_meg=True,
        )
        _print_metrics("shuffle", results["shuffle"])

    if run_all or args.ablation == "random_soft":
        print("\n[3/4] Random soft-token ablation (Gaussian noise at test time)...")
        results["random_soft"] = run_eval_pass(
            model, loader, device, tokenizer, vocab_words, bert_sim,
            random_soft=True,
        )
        _print_metrics("random_soft", results["random_soft"])

    if run_all or args.ablation == "no_soft":
        print("\n[4/4] No-soft-token baseline (LLM language prior only)...")
        results["no_soft"] = run_no_soft_baseline(
            model, loader, device, tokenizer, vocab_words, bert_sim
        )
        _print_metrics("no_soft", results["no_soft"])

    # ── Save ─────────────────────────────────────────────────────────────────
    poem_tag = "_".join(sorted(args.poems))
    out_path = args.out_dir / f"results_{args.design}_{poem_tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


def _print_metrics(tag: str, m: Dict) -> None:
    bert = f"{m['bert_sim']:.3f}" if m.get("bert_sim") is not None else "N/A"
    print(
        f"  [{tag}]  loss={m['loss']:.3f}  "
        f"exact_match={m['exact_match']:.3f}  "
        f"bleu1={m['bleu1']:.3f}  "
        f"bert_sim={bert}  "
        f"R@1={m.get('restricted_R@1', 0):.3f}  "
        f"R@5={m.get('restricted_R@5', 0):.3f}  "
        f"MRR={m.get('restricted_MRR', 0):.3f}"
    )


if __name__ == "__main__":
    main()
