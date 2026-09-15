"""
fusion_beamsearch_null.py — RANDOM-candidate null baseline for fusion_beamsearch.py.

This is a controlled null for MEG-guided beam-search fusion.  It is IDENTICAL to
fusion_beamsearch.py in every respect (LLM scoring, normalization, fusion rule,
beam pruning, evaluation, output format) EXCEPT the candidate-selection step:

    fusion_beamsearch.py :  at a valid position the k candidates are MEG's top-k
                            word types (by cosine similarity).
    THIS SCRIPT          :  at a valid position the k candidates are a RANDOM
                            set of k word types.  Their MEG scores are still the
                            real cosine values (just for that random set), and
                            they are normalized and fused with the LLM exactly
                            as in the real script.

If MEG is genuinely informative, the real top-k script should beat this null.
If the fusion gains survive under random candidates, they come from the LLM /
vocabulary prior, not from MEG.

Parity note
-----------
In the real script the top-k set at position t depends only on the MEG row (not
on alpha or on beam history), so all 35 alpha beam searches see the SAME
candidate set at each position.  To keep the null a fair mirror, the random
k-subset is drawn ONCE per (trial, position) with a seeded RNG and reused across
every alpha.  This removes any extra variance the real run does not have.  Set
--seed to reproduce or to average several null draws.

Usage (run from inside contrastive_multimodal/):
    python fusion_beamsearch_null.py --heldout_subject sub-01
    python fusion_beamsearch_null.py --heldout_subject sub-01 \\
        --normalization row_zscore --beam_width 5 --top_k 5 --seed 0
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

# run from inside contrastive_multimodal/
sys.path.insert(0, os.path.dirname(__file__))

import fusion_beamsearch as fb                                   # noqa: E402
from fusion_beamsearch import (                                  # noqa: E402
    ALPHA_GRID,
    _normalize_row,
    _would_repeat_ngram,
    eval_sequence,
    plot_alpha_sweep_beam,
)
from new_dataset import (                                        # noqa: E402
    MEGContinuousTrialDataset,
    collate_continuous_trials,
    _load_onsets,
)
from splits import make_loso_splits                              # noqa: E402


# ===========================================================================
#  Random candidate generation (the ONLY conceptual difference vs. real)
# ===========================================================================

def make_random_candidates(
    N: int,
    V: int,
    k: int,
    valid_mask: List[bool],
    generator: torch.Generator,
) -> List[Optional[torch.Tensor]]:
    """
    Pre-draw one random k-subset of the vocabulary per valid position.

    Returns a list of length N: a LongTensor of k random vocab indices at valid
    positions, or None at invalid positions (which use the full vocab downstream,
    identical to the real script).  Drawn once per trial so all alphas share it.
    """
    k = min(k, V)
    out: List[Optional[torch.Tensor]] = []
    for t in range(N):
        if valid_mask[t]:
            perm = torch.randperm(V, generator=generator)[:k]
            out.append(perm.clone())
        else:
            out.append(None)
    return out


# ===========================================================================
#  Beam search — copy of fusion_beamsearch.beam_search_fusion with the
#  candidate-selection block swapped for the precomputed random sets.
# ===========================================================================

@torch.no_grad()
def beam_search_fusion_null(
    meg_scores:      torch.Tensor,   # (N, |V|)
    vocab:           List[str],
    word_texts:      List[str],      # ground truth — evaluation only
    valid_mask:      List[bool],
    rand_candidates: List[Optional[torch.Tensor]],  # per-position random k-subset
    tokenizer,
    model,
    device:          torch.device,
    alpha:           float,
    beam_width:      int = 5,
    top_k:           int = 5,
    normalization:   str = "logsoftmax",
    no_repeat_ngram: int = 0,
) -> Dict:
    """
    RANDOM-candidate null version of fusion_beamsearch.beam_search_fusion.

    Every line is intentionally identical to the real function except the valid
    branch of the candidate-selection block, which uses the precomputed random
    subset instead of meg_row.topk(k).
    """
    N = meg_scores.shape[0]
    V = len(vocab)
    k = min(top_k, V)

    # Pre-tokenize vocab words (full subword IDs, not just first token)
    word_to_tids: Dict[str, List[int]] = {}
    for w in vocab:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        word_to_tids[w] = ids if ids else [tokenizer.unk_token_id or 0]

    bos: List[int] = ([tokenizer.bos_token_id]
                      if tokenizer.bos_token_id is not None else [])
    beams = [{"history": [], "token_ids": list(bos), "cum_score": 0.0}]

    for t in range(N):
        # ── MEG candidates (RANDOM set instead of top-k) ──────────────────────
        meg_row = meg_scores[t].to(device).float()   # (|V|,)
        if valid_mask[t]:
            rand_indices  = rand_candidates[t].to(device)               # (k,)
            meg_cand_raw  = meg_row[rand_indices]                       # (k,)
            meg_cand_norm = _normalize_row(meg_cand_raw, normalization) # (k,)
            topk_indices  = rand_indices   # keep downstream variable name
        else:
            # No MEG signal: let LLM drive the candidate set (same as real)
            topk_indices  = torch.arange(V, device=device)
            meg_cand_norm = torch.zeros(V, device=device)

        cand_list = topk_indices.tolist()
        n_cands   = len(cand_list)

        # ── LLM scores per beam ───────────────────────────────────────────────
        # Phase 1: one forward pass over the beam's history (KV cached).
        # Phase 2: extend cache for multi-token candidates only.
        llm_cand_norm = torch.zeros(len(beams), n_cands, device=device)

        for bi, beam in enumerate(beams):
            h = beam["token_ids"]
            if not h:
                continue   # empty context → leave LLM contribution as zero

            hist_ids  = torch.tensor([h], dtype=torch.long, device=device)
            hist_out  = model(hist_ids, use_cache=True)
            past_kv   = hist_out.past_key_values
            last_logp = torch.log_softmax(hist_out.logits[0, -1, :], dim=-1)

            # Batch first-token score for all candidates at once
            first_tids = torch.tensor(
                [word_to_tids[vocab[wi]][0] for wi in cand_list],
                dtype=torch.long, device=device,
            )
            raw_list: List[float] = last_logp[first_tids].tolist()

            # Multi-token words: extend cache token-by-token
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
                    extra  += (torch.log_softmax(out.logits[0, -1, :], dim=-1)
                                [tids[step + 1]].item())
                raw_list[j] += extra

            raw = torch.tensor(raw_list, device=device)
            llm_cand_norm[bi] = _normalize_row(raw, normalization)

        # ── Expand beams ──────────────────────────────────────────────────────
        candidates: List[Dict] = []
        for bi, beam in enumerate(beams):
            blocked: List[Dict] = []
            for j, wi in enumerate(cand_list):
                word  = vocab[wi]
                fused = ((1.0 - alpha) * meg_cand_norm[j].item()
                         + alpha      * llm_cand_norm[bi, j].item())
                entry = {
                    "history":   beam["history"]   + [word],
                    "token_ids": beam["token_ids"] + word_to_tids[word],
                    "cum_score": beam["cum_score"] + fused,
                }
                if (no_repeat_ngram > 0
                        and _would_repeat_ngram(beam["history"], word,
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

    best    = beams[0]
    metrics = eval_sequence(best["history"], word_texts, valid_mask)
    return {
        "pred_sequence": best["history"],
        "cum_score":     float(best["cum_score"]),
        "metrics":       metrics,
    }


def sweep_alphas_beam_null(
    meg_scores:      torch.Tensor,
    vocab:           List[str],
    word_texts:      List[str],
    valid_mask:      List[bool],
    rand_candidates: List[Optional[torch.Tensor]],
    tokenizer,
    model,
    device:          torch.device,
    alphas:          List[float],
    beam_width:      int = 5,
    top_k:           int = 5,
    normalization:   str = "logsoftmax",
    no_repeat_ngram: int = 0,
) -> Dict[float, Dict]:
    """Run beam_search_fusion_null independently for each alpha.

    The same precomputed random candidate sets are reused across all alphas, so
    the ONLY thing that differs between alphas is the fusion weight — exactly as
    in the real script (where top-k is likewise fixed across alphas).
    """
    return {
        alpha: beam_search_fusion_null(
            meg_scores, vocab, word_texts, valid_mask, rand_candidates,
            tokenizer, model, device,
            alpha=alpha, beam_width=beam_width, top_k=top_k,
            normalization=normalization, no_repeat_ngram=no_repeat_ngram,
        )
        for alpha in alphas
    }


# ===========================================================================
#  Main (mirror of fusion_beamsearch.main with random candidates)
# ===========================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NULL] device={device}  subject={args.heldout_subject}  "
          f"llm={args.llm_name}  norm={args.normalization}  "
          f"beam_width={args.beam_width}  top_k={args.top_k}  seed={args.seed}")

    # Stage 1
    encoder, word_head, pooling_module, pooling_mode = fb.load_stage1_checkpoint(
        args.stage1_checkpoint_path, device
    )

    # Candidate bank
    teacher_cache = torch.load(args.teacher_cache_path, weights_only=False)
    bank_vectors, bank_word_types, type_to_id = fb.build_candidate_bank(
        teacher_cache, gpt2_mid_layer=args.gpt2_mid_layer
    )
    bank_vectors = bank_vectors.to(device)
    print(f"Bank: {bank_vectors.shape[0]} occurrences, {len(type_to_id)} unique types")

    # LLM
    tokenizer, llm_model = fb.load_fusion_llm(args.llm_name, device)

    # Vocab per poem (pre-computed once)
    vocab_cache: Dict[str, List[str]] = {}
    for poem in ("poem1", "poem2"):
        onsets = _load_onsets(poem)
        word_texts_poem = [e["word"].strip().lower() for e in onsets]
        vocab_cache[poem] = sorted(set(word_texts_poem))
        print(f"  {poem}: {len(word_texts_poem)} words, {len(vocab_cache[poem])} unique")

    # Test split
    splits  = make_loso_splits(args.heldout_subject)
    test_ds = MEGContinuousTrialDataset(
        splits["test"]["trials"],
        word_filter=splits["test"]["word_filter"],
        meg_base=args.meg_base,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=collate_continuous_trials)
    print(f"Test trials: {len(test_ds)}")

    # One CPU RNG for the whole run; draws advance deterministically per trial
    # (test_loader is shuffle=False, so re-running with the same seed reproduces).
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    # Accumulate metrics: {alpha: {metric: sum, n_trials: int}}
    acc = {a: {"bleu1": 0.0, "word_acc": 0.0, "n_valid": 0, "n_trials": 0}
           for a in ALPHA_GRID}
    trials_out: Dict[str, List] = {str(a): [] for a in ALPHA_GRID}
    n_trials = 0

    print(f"Running RANDOM-candidate beam-search fusion over {len(test_ds)} trials "
          f"× {len(ALPHA_GRID)} alphas ...")
    for batch in test_loader:
        session = int(batch["session"][0]) if "session" in batch else None

        z_word, valid_mask, word_texts, poem = fb.run_encoder_on_trial(
            batch, encoder, word_head, pooling_module, pooling_mode, device
        )

        vocab      = vocab_cache[poem]
        meg_scores = fb.meg_scores_to_type_level(
            z_word.to(device), bank_vectors, bank_word_types, vocab
        )                                    # (N_poem, |V|)

        valid_list = valid_mask.tolist()

        # Draw the random candidate sets ONCE for this trial; reuse across alphas.
        rand_candidates = make_random_candidates(
            N=meg_scores.shape[0], V=len(vocab), k=args.top_k,
            valid_mask=valid_list, generator=generator,
        )

        alpha_results = sweep_alphas_beam_null(
            meg_scores, vocab, word_texts, valid_list, rand_candidates,
            tokenizer, llm_model, device,
            alphas=ALPHA_GRID,
            beam_width=args.beam_width,
            top_k=args.top_k,
            normalization=args.normalization,
            no_repeat_ngram=args.no_repeat_ngram,
        )

        for alpha, res in alpha_results.items():
            m = res["metrics"]
            acc[alpha]["bleu1"]    += m["bleu1"]
            acc[alpha]["word_acc"] += m["word_acc"]
            acc[alpha]["n_valid"]  += m["n_valid"]
            acc[alpha]["n_trials"] += 1
            trials_out[str(alpha)].append({
                "pred_sequence": res["pred_sequence"],
                "word_texts":    word_texts,
                "valid_mask":    valid_list,
                "poem":          poem,
                "session":       session,
            })

        n_trials += 1
        if n_trials % 2 == 0 or n_trials == len(test_ds):
            print(f"  {n_trials}/{len(test_ds)} trials done ...")

    # Aggregate (averages across trials)
    results: Dict[str, Dict] = {}
    for alpha in ALPHA_GRID:
        nt = acc[alpha]["n_trials"]
        results[str(alpha)] = {
            "bleu1":    acc[alpha]["bleu1"]    / nt if nt > 0 else 0.0,
            "word_acc": acc[alpha]["word_acc"] / nt if nt > 0 else 0.0,
            "n_valid":  acc[alpha]["n_valid"],
            "n_trials": nt,
        }

    # Summary table
    print(f"\n=== NULL beam fusion — {args.heldout_subject}  "
          f"llm={args.llm_name}  norm={args.normalization}  "
          f"B={args.beam_width}  top_k={args.top_k}  seed={args.seed} ===")
    print(f"{'alpha':>6}  {'BLEU-1':>8}  {'word_acc':>9}")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        key = str(alpha) if str(alpha) in results else min(
            results.keys(), key=lambda kk: abs(float(kk) - alpha))
        r = results[key]
        print(f"{float(key):6.2f}  {r['bleu1']*100:7.2f}%  {r['word_acc']*100:8.2f}%")

    # Save JSON  (filename marked _null and seeded so it never overwrites the
    # real fusion_beamsearch.py output)
    os.makedirs(args.out_dir, exist_ok=True)
    llm_tag  = args.llm_name.replace("/", "_")
    out_path = os.path.join(
        args.out_dir,
        f"{args.heldout_subject}_beamfusion_null_B{args.beam_width}"
        f"_top{args.top_k}_{llm_tag}_{args.normalization}_seed{args.seed}.json",
    )
    output = {
        "heldout_subject": args.heldout_subject,
        "llm_name":        args.llm_name,
        "normalization":   args.normalization,
        "beam_width":      args.beam_width,
        "top_k":           args.top_k,
        "no_repeat_ngram": args.no_repeat_ngram,
        "null_baseline":   True,
        "seed":            args.seed,
        "n_trials":        n_trials,
        "results":         results,
        "trials":          trials_out,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved → {out_path}")

    if args.plot:
        for metric in ("bleu1", "word_acc"):
            plot_path = out_path.replace(".json", f"_{metric}.png")
            plot_alpha_sweep_beam(out_path, metric=metric, out_path=plot_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="RANDOM-candidate null baseline for MEG-guided beam-search "
                    "LLM fusion (contrastive_multimodal)."
    )
    p.add_argument("--heldout_subject", type=str, default="sub-01")
    p.add_argument("--stage1_checkpoint_path", type=str, default=None,
                   help="Path to stage1_best_*.pt. Defaults to "
                        "checkpoints/joint_annealed_exact/stage1_best_{subject}_joint_annealed_exact.pt")
    p.add_argument("--teacher_cache_path", type=str, default="teacher_cache.pt")
    p.add_argument("--gpt2_mid_layer", type=int, default=None,
                   help="Build the candidate bank from this cached GPT-2 layer "
                        "(hidden_states_word_aligned[L]) instead of the default h_mid. "
                        "Must match the real fusion run and the checkpoint's train.py --gpt2_mid_layer.")
    p.add_argument("--llm_name", type=str, default="gpt2")
    p.add_argument("--normalization", type=str, default="logsoftmax",
                   choices=["logsoftmax", "row_zscore"])
    p.add_argument("--beam_width", type=int, default=5,
                   help="Number of beams to maintain at each step.")
    p.add_argument("--top_k", type=int, default=5,
                   help="Number of RANDOM candidates considered per position. "
                        "At invalid positions the full vocab is used instead.")
    p.add_argument("--no_repeat_ngram", type=int, default=0,
                   help="Block repeated n-grams of this length (0=disabled).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the random candidate draws. Vary it to "
                        "average several independent null draws.")
    p.add_argument("--meg_base", type=str, default=None,
                   help="Path to icaed_Sai directory. Defaults to new_dataset.MEG_BASE.")
    p.add_argument("--out_dir", type=str, default="fusion_results_null")
    p.add_argument("--plot", action="store_true",
                   help="Save bleu1 and word_acc vs alpha plots alongside the JSON.")
    args = p.parse_args()

    if args.stage1_checkpoint_path is None:
        args.stage1_checkpoint_path = (
            f"checkpoints/joint_annealed_exact/"
            f"stage1_best_{args.heldout_subject}_joint_annealed_exact.pt"
        )

    main(args)
