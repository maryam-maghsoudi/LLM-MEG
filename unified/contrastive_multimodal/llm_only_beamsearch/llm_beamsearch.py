"""
llm_beamsearch.py — pure LLM beam search over the closed poem vocabulary.

This is the MEG-free counterpart of ../fusion_beamsearch.py.

In fusion_beamsearch.py, even at alpha=1 the LLM is still confined to the
top-k words MEG nominated at each position — so "LLM only" there is really
"LLM re-ranking within MEG's shortlist". Here there is NO MEG at all: at every
position the LLM scores the ENTIRE poem vocabulary (|V| word types) conditioned
on the beam's own predicted history, and standard beam search keeps the top-B
partial sequences.

Determinism
-----------
There is no MEG input and no sampling (the model is in eval mode and we take
exact top-B). The result therefore depends ONLY on:
    (poem vocabulary, poem length N, beam width, LLM, normalization)
It does NOT depend on subject or session. Consequently there is exactly ONE
result per poem — poem1 and poem2 — shared across all subjects.

Generation
----------
Each beam starts from BOS (matching fusion_beamsearch.py). We generate exactly
N positions, where N = number of word occurrences in the poem, then compare the
best beam's sequence against the poem's ground-truth ordered word sequence
(BLEU-1 and word accuracy over all positions).

Usage (run from inside contrastive_multimodal/llm_only_beamsearch/):
    python llm_beamsearch.py
    python llm_beamsearch.py --beam_width 5 --llm_name gpt2
    python llm_beamsearch.py --normalization row_zscore
    python llm_beamsearch.py --no_repeat_ngram 2      # block immediate repetition
    python llm_beamsearch.py --poems poem1            # single poem only
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

# fusion_beamsearch.py and new_dataset.py live one level up.
PARENT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PARENT)

import fusion_beamsearch as fb                       # noqa: E402
from new_dataset import _load_onsets                 # noqa: E402


# ===========================================================================
#  Pure LLM beam search over the full poem vocabulary
# ===========================================================================

@torch.no_grad()
def llm_only_beam_search(
    vocab:           List[str],
    n_positions:     int,
    tokenizer,
    model,
    device:          torch.device,
    beam_width:      int = 5,
    normalization:   str = "logsoftmax",
    no_repeat_ngram: int = 0,
) -> Dict:
    """
    Beam search using the LLM alone over the closed vocabulary `vocab`.

    At each of `n_positions` steps every beam is scored against ALL |V| words
    (never a MEG-defined subset). Returns the best sequence and its cumulative
    score. Deterministic for fixed inputs.
    """
    V = len(vocab)
    cand_list = list(range(V))          # full vocab at every position

    # Pre-tokenize each vocab word (full subword IDs, leading space like GPT-2).
    word_to_tids: Dict[str, List[int]] = {}
    for w in vocab:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        word_to_tids[w] = ids if ids else [tokenizer.unk_token_id or 0]

    bos: List[int] = ([tokenizer.bos_token_id]
                      if tokenizer.bos_token_id is not None else [])
    beams = [{"history": [], "token_ids": list(bos), "cum_score": 0.0}]

    for t in range(n_positions):
        # ── LLM scores per beam over the FULL vocab ───────────────────────────
        llm_norm = torch.zeros(len(beams), V, device=device)

        for bi, beam in enumerate(beams):
            h = beam["token_ids"]
            if not h:
                continue   # empty context (no BOS) → leave scores at zero

            hist_ids  = torch.tensor([h], dtype=torch.long, device=device)
            hist_out  = model(hist_ids, use_cache=True)
            past_kv   = hist_out.past_key_values
            last_logp = F.log_softmax(hist_out.logits[0, -1, :], dim=-1)

            # Batch first-token log-prob for every vocab word at once.
            first_tids = torch.tensor(
                [word_to_tids[vocab[wi]][0] for wi in cand_list],
                dtype=torch.long, device=device,
            )
            raw_list: List[float] = last_logp[first_tids].tolist()

            # Multi-token words: extend the KV cache token-by-token.
            for j, wi in enumerate(cand_list):
                tids = word_to_tids[vocab[wi]]
                if len(tids) == 1:
                    continue
                curr_kv = past_kv
                extra   = 0.0
                for step in range(len(tids) - 1):
                    inp     = torch.tensor([[tids[step]]],
                                           dtype=torch.long, device=device)
                    out     = model(inp, past_key_values=curr_kv, use_cache=True)
                    curr_kv = out.past_key_values
                    extra  += (F.log_softmax(out.logits[0, -1, :], dim=-1)
                                [tids[step + 1]].item())
                raw_list[j] += extra

            raw = torch.tensor(raw_list, device=device)
            llm_norm[bi] = fb._normalize_row(raw, normalization)

        # ── Expand beams ──────────────────────────────────────────────────────
        candidates: List[Dict] = []
        for bi, beam in enumerate(beams):
            blocked: List[Dict] = []
            for j, wi in enumerate(cand_list):
                word  = vocab[wi]
                score = llm_norm[bi, j].item()          # alpha = 1: LLM only
                entry = {
                    "history":   beam["history"]   + [word],
                    "token_ids": beam["token_ids"] + word_to_tids[word],
                    "cum_score": beam["cum_score"] + score,
                }
                if (no_repeat_ngram > 0
                        and fb._would_repeat_ngram(beam["history"], word,
                                                   no_repeat_ngram)):
                    blocked.append(entry)
                else:
                    candidates.append(entry)
            # Never stall: if every candidate for this beam was blocked, admit
            # the blocked set so search always produces B beams.
            if not any(c["history"][:-1] == beam["history"] for c in candidates):
                candidates.extend(blocked)

        candidates.sort(key=lambda c: c["cum_score"], reverse=True)
        beams = candidates[:beam_width]

    best = beams[0]
    return {"pred_sequence": best["history"], "cum_score": float(best["cum_score"])}


# ===========================================================================
#  Main
# ===========================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  llm={args.llm_name}  norm={args.normalization}  "
          f"beam_width={args.beam_width}  no_repeat_ngram={args.no_repeat_ngram}")

    tokenizer, model = fb.load_fusion_llm(args.llm_name, device)

    results: Dict[str, Dict] = {}
    for poem in args.poems:
        onsets     = _load_onsets(poem)
        word_texts = [e["word"].strip().lower() for e in onsets]  # ground truth (ordered)
        vocab      = sorted(set(word_texts))
        N          = len(word_texts)
        print(f"\n=== {poem}: {N} positions, {len(vocab)} unique words ===")

        out = llm_only_beam_search(
            vocab, N, tokenizer, model, device,
            beam_width=args.beam_width,
            normalization=args.normalization,
            no_repeat_ngram=args.no_repeat_ngram,
        )

        # All positions are valid (no MEG gating).
        valid_mask = [True] * N
        metrics = fb.eval_sequence(out["pred_sequence"], word_texts, valid_mask)

        print(f"  BLEU-1={metrics['bleu1']*100:.2f}%  "
              f"word_acc={metrics['word_acc']*100:.2f}%  "
              f"cum_score={out['cum_score']:.2f}")
        print(f"  pred : {' '.join(out['pred_sequence'][:20])} ...")
        print(f"  truth: {' '.join(word_texts[:20])} ...")

        results[poem] = {
            "pred_sequence": out["pred_sequence"],
            "word_texts":    word_texts,
            "cum_score":     out["cum_score"],
            "bleu1":         metrics["bleu1"],
            "word_acc":      metrics["word_acc"],
            "n_positions":   N,
            "vocab_size":    len(vocab),
        }

    os.makedirs(args.out_dir, exist_ok=True)
    llm_tag  = args.llm_name.replace("/", "_")
    out_path = os.path.join(
        args.out_dir,
        f"llm_beamsearch_B{args.beam_width}_{llm_tag}_{args.normalization}.json",
    )
    output = {
        "llm_name":        args.llm_name,
        "normalization":   args.normalization,
        "beam_width":      args.beam_width,
        "no_repeat_ngram": args.no_repeat_ngram,
        "subject_independent": True,   # deterministic; no MEG, shared across subjects
        "results":         results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Pure LLM beam search over the closed poem vocabulary (no MEG)."
    )
    p.add_argument("--llm_name", type=str, default="gpt2")
    p.add_argument("--normalization", type=str, default="logsoftmax",
                   choices=["logsoftmax", "row_zscore"],
                   help="Per-position normalization over the vocab. logsoftmax "
                        "gives standard log-prob beam search.")
    p.add_argument("--beam_width", type=int, default=5,
                   help="Number of beams to maintain at each step.")
    p.add_argument("--no_repeat_ngram", type=int, default=0,
                   help="Block repeated n-grams of this length (0=disabled).")
    p.add_argument("--poems", nargs="+", default=["poem1", "poem2"],
                   choices=["poem1", "poem2"],
                   help="Which poem(s) to decode. One result per poem.")
    p.add_argument("--out_dir", type=str, default="results")
    args = p.parse_args()

    main(args)
