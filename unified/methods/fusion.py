"""
fusion.py — Log-linear fusion of MEG scores with LLM next-word scores.

Two LLM scoring modes
---------------------
teacher_forced (default)
    At each position i the LLM conditions on the ground-truth tokens for
    words 0..i-1.  Scores are identical for every subject/session of the same
    poem, so they are computed once per poem and cached by the caller.

beam (--beam_width > 0)
    Maintains B beam histories of MEG-predicted words.  At each position t the
    LLM conditions on each beam's own predicted history (NOT ground truth) and
    scores the top-k MEG candidates.  Both MEG and LLM scores are normalized
    over the trial vocabulary before mixing.  Different alphas may produce
    different beam histories, so each alpha requires a separate beam search.

Fusion
------
    fused = (1 - alpha) * normalize(meg) + alpha * normalize(llm)

Both sides are normalized over the same trial vocabulary (|V| words), so the
denominator is identical and scores are directly comparable.

Normalization modes (--fusion_normalization)
--------------------------------------------
"logsoftmax"  (default)
    normalize = log_softmax(·, dim=0)  (over |V|)

"row_zscore"
    z = (x - mean(x)) / (std(x) + eps)  (over |V|)
    Removes the ~200x scale gap between MEG cosine similarities and LLM logits.

Public API
----------
load_fusion_llm(llm_name, device)                                   → (tokenizer, model)
compute_llm_scores(word_texts, vocab, ...)                          → Tensor(N, |V|)
beam_search_fusion(meg_scores, vocab, word_texts, valid_mask, ...)  → dict
sweep_alphas(meg, llm, vocab, words, mask, alphas, norm)            → ({alpha: metrics}, diag)
sweep_alphas_beam(meg, vocab, words, mask, tok, model, ...)         → {alpha: dict}
scale_diagnostics(meg_scores, llm_scores)                           → dict
fuse_scores(meg_scores, llm_scores, alpha, normalization)           → Tensor(N, |V|)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..evaluate import eval_option_a, eval_option_b


# ---------------------------------------------------------------------------
#  LLM loading
# ---------------------------------------------------------------------------

def load_fusion_llm(
    llm_name: str,
    device: torch.device,
) -> Tuple:
    """
    Load a frozen causal LM for use as the fusion language model.
    Returns (tokenizer, model).  Model is in eval mode with no gradients.
    """
    print(f"[fusion] Loading LLM: {llm_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(llm_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n = sum(p.numel() for p in model.parameters())
    print(f"  {llm_name}  {n:,} params  frozen")
    return tokenizer, model


# ---------------------------------------------------------------------------
#  LLM teacher-forced scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_llm_scores(
    word_texts: List[str],
    vocab: List[str],
    tokenizer,
    model,
    device: torch.device,
) -> torch.Tensor:
    """
    Teacher-forced LLM next-word scores over vocab.

    At position i, conditions on the ground-truth token sequence for
    words 0..i-1 and returns the logit for each vocab word (first subtoken).
    Position 0 has no preceding context → zero scores (uniform distribution).

    Parameters
    ----------
    word_texts : ground-truth word sequence for this trial (N words)
    vocab      : sorted unique words in this trial — same as predict() returns
    tokenizer  : from load_fusion_llm()
    model      : from load_fusion_llm(), frozen causal LM on device
    device     : target device

    Returns
    -------
    Tensor(N, |V|)  raw logits; scores[i] predicts word i from context 0..i-1
    """
    N = len(word_texts)
    V = len(vocab)

    # First-subtoken ID for each vocab word (with leading space for mid-BPE)
    vocab_tok_ids = []
    for w in vocab:
        # What token ID represents this word?
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(w, add_special_tokens=False)
        vocab_tok_ids.append(ids[0] if ids else (tokenizer.unk_token_id or 0))
    vocab_tok_ids_t = torch.tensor(vocab_tok_ids, dtype=torch.long, device=device)

    # Tokenise each word
    token_ids_per_word = []
    for w in word_texts:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if not ids:
            ids = [tokenizer.unk_token_id or 0]
        token_ids_per_word.append(ids)

    # Build full token sequence with optional BOS
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    all_ids = bos.copy()

    # boundary[i] = index in all_ids whose logit predicts word i's first token
    # logits[k] predicts token k+1, so boundary = len(all_ids) - 1 before appending word i
    boundaries: List[int] = []
    for ids in token_ids_per_word:
        boundaries.append(len(all_ids) - 1)   # -1 if no BOS → invalid (→ zeros)
        all_ids.extend(ids)

    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
    logits = model(input_ids).logits[0]   # (T, vocab_size)
    # For every token position, the LLM gives a score for every possible next token

    scores = torch.zeros(N, V)
    for i, b in enumerate(boundaries):
        if b < 0:
            continue   # no context for position 0 without BOS → leave as zeros
        scores[i] = logits[b, vocab_tok_ids_t].cpu()

    # For a poem1 trial, scores[i, j] is the LLM's log-score for unique word vocab[j]
    # being the next word at position i. The shape (56, |V|) has 56 rows
    # (one per word position in the sequence) and |V| columns 
    # (one per unique word in the poem).

    return scores   # (N, |V|)


# ---------------------------------------------------------------------------
#  Scale diagnostics
# ---------------------------------------------------------------------------

_DIAG_EPS = 1e-8

def scale_diagnostics(
    meg_scores: torch.Tensor,   # (N, |V|)
    llm_scores: torch.Tensor,   # (N, |V|)
) -> Dict:
    """
    Compute per-row (per-word-position) std across vocab candidates and
    summarise the LLM/MEG scale ratio.

    Normalization is performed independently per word position (row), not
    across positions, trials, or the full matrix.

    Returns
    -------
    dict with keys:
        mean_meg_row_std, mean_llm_row_std,
        mean_scale_ratio, median_scale_ratio, min_scale_ratio, max_scale_ratio
    """
    meg_std = meg_scores.float().std(dim=-1)          # (N,)
    llm_std = llm_scores.float().std(dim=-1)          # (N,)
    ratio   = llm_std / (meg_std + _DIAG_EPS)         # (N,)
    return {
        "mean_meg_row_std":   float(meg_std.mean()),
        "mean_llm_row_std":   float(llm_std.mean()),
        "mean_scale_ratio":   float(ratio.mean()),
        "median_scale_ratio": float(ratio.median()),
        "min_scale_ratio":    float(ratio.min()),
        "max_scale_ratio":    float(ratio.max()),
    }


# ---------------------------------------------------------------------------
#  Fusion
# ---------------------------------------------------------------------------

def fuse_scores(
    meg_scores: torch.Tensor,           # (N, |V|)
    llm_scores: torch.Tensor,           # (N, |V|)
    alpha: float,
    normalization: str = "logsoftmax",
) -> torch.Tensor:
    """
    Fuse MEG and LLM scores with configurable per-row normalization.

    normalization="logsoftmax"  (default, preserves original behavior)
        fused = (1-alpha)*log_softmax(meg) + alpha*log_softmax(llm)

    normalization="row_zscore"
        Each row is standardized across vocab candidates independently:
            z = (x - mean(x)) / (std(x) + eps)
        fused = (1-alpha)*meg_z + alpha*llm_z
        Removes the ~200x scale gap between cosine similarities and logits.
    """
    meg = meg_scores.float()
    llm = llm_scores.float()

    if normalization == "logsoftmax":
        n_meg = F.log_softmax(meg, dim=-1)
        n_llm = F.log_softmax(llm, dim=-1)
    elif normalization == "row_zscore":
        n_meg = (meg - meg.mean(dim=-1, keepdim=True)) / (meg.std(dim=-1, keepdim=True) + _DIAG_EPS)
        n_llm = (llm - llm.mean(dim=-1, keepdim=True)) / (llm.std(dim=-1, keepdim=True) + _DIAG_EPS)
    else:
        raise ValueError(f"Unknown normalization: {normalization!r}. "
                         f"Choose 'logsoftmax' or 'row_zscore'.")

    return (1.0 - alpha) * n_meg + alpha * n_llm


# ---------------------------------------------------------------------------
#  Alpha sweep
# ---------------------------------------------------------------------------

def sweep_alphas(
    meg_scores: torch.Tensor,
    llm_scores: torch.Tensor,
    vocab: List[str],
    word_texts: List[str],
    valid_mask: List[bool],
    alphas: List[float],
    normalization: str = "logsoftmax",
) -> tuple:
    """
    For each alpha value, fuse MEG and LLM scores and compute eval metrics.
    Scale diagnostics are computed once (independent of alpha).

    Returns
    -------
    results : {alpha: {"option_a": {...}, "option_b": {...}}}
    diag    : scale_diagnostics dict (MEG/LLM std ratio summary)
    """
    diag    = scale_diagnostics(meg_scores, llm_scores)
    results: Dict[float, Dict] = {}
    for alpha in alphas:
        fused     = fuse_scores(meg_scores, llm_scores, alpha, normalization)
        pred_top1 = [vocab[fused[i].argmax().item()] for i in range(len(word_texts))]
        a = eval_option_a(fused, vocab, word_texts, valid_mask)
        b = eval_option_b(pred_top1, word_texts, valid_mask)
        results[alpha] = {"option_a": a, "option_b": b}
    return results, diag


# ---------------------------------------------------------------------------
#  Beam-search fusion (MEG-guided LLM context)
# ---------------------------------------------------------------------------

def _normalize_row(scores: torch.Tensor, normalization: str) -> torch.Tensor:
    """Normalize a 1-D score vector over its own elements."""
    x = scores.float()
    if normalization == "logsoftmax":
        return F.log_softmax(x, dim=0)
    elif normalization == "row_zscore":
        return (x - x.mean()) / (x.std() + _DIAG_EPS)
    else:
        raise ValueError(f"Unknown normalization: {normalization!r}")


def _would_repeat_ngram(history: List[str], word: str, n: int) -> bool:
    """Return True if appending word to history would create a repeated n-gram."""
    if n <= 0 or len(history) < n - 1:
        return False
    ngram = tuple(history[-(n - 1):]) + (word,)
    for i in range(len(history) - n + 1):
        if tuple(history[i:i + n]) == ngram:
            return True
    return False


@torch.no_grad()
def beam_search_fusion(
    meg_scores:       torch.Tensor,    # (N, |V|)
    vocab:            List[str],
    word_texts:       List[str],       # ground truth — evaluation only, never used as context
    valid_mask:       List[bool],
    tokenizer,
    model,
    device:           torch.device,
    alpha:            float,
    beam_width:       int = 5,
    top_k:            int = 5,
    normalization:    str = "logsoftmax",
    no_repeat_ngram:  int = 0,
) -> Dict:
    """
    MEG-guided beam-search fusion.

    At each position t:
      1. Normalize MEG scores over the trial vocab (|V| words).
         At valid positions keep the top-k MEG candidates.
         At invalid positions (no MEG signal) expand over the full vocab so
         the LLM can select freely rather than from an arbitrary top-k subset.
      2. Batch the B beam histories and run a single LLM forward pass.
         Extract logits for the trial vocab only, then normalize over |V|.
         Both MEG and LLM are now on the same |V|-dimensional scale.
      3. fused = (1-alpha)*meg_norm[candidate] + alpha*llm_norm[candidate]
      4. B × n_candidates → keep top-B beams by cumulative fused score.

    no_repeat_ngram : int
        If > 0, block any candidate that would create a repeated n-gram of this
        length in the beam's history.  n=2 blocks immediate word repetition
        (e.g. "flash flash"); n=3 blocks repeated trigrams.  Falls back to the
        next-best candidate; if all candidates are blocked, the block is lifted
        for that beam so the search never stalls.

    Returns
    -------
    dict:
        pred_sequence : List[str]   best-beam predicted word sequence
        cum_score     : float       cumulative fused log-score
        option_b      : dict        word_accuracy, bleu1, wer vs ground truth
    """
    N = meg_scores.shape[0]
    V = len(vocab)
    k = min(top_k, V)

    # Full subword IDs per vocab word (scoring + beam context)
    word_to_tids: Dict[str, List[int]] = {}
    for w in vocab:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        word_to_tids[w] = ids if ids else [tokenizer.unk_token_id or 0]

    bos: List[int] = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

    # One beam with empty history
    beams: List[Dict] = [{"history": [], "token_ids": list(bos), "cum_score": 0.0}]

    for t in range(N):
        # ── MEG scores ────────────────────────────────────────────────────────
        meg_row = meg_scores[t].to(device).float()   # (|V|,)
        if valid_mask[t]:
            meg_norm     = _normalize_row(meg_row, normalization)
            topk_indices = meg_norm.topk(k).indices           # (k,)
        else:
            # No MEG signal: zero contribution; expand over full vocab so the
            # LLM (not an arbitrary MEG top-k) decides the candidate set.
            meg_norm     = torch.zeros(V, device=device)
            topk_indices = torch.arange(V, device=device)    # (V,)

        cand_list = topk_indices.tolist()   # vocab indices of candidates
        n_cands   = len(cand_list)

        # ── LLM scores: full-word log P(w | h_b) per beam ────────────────────
        # Optimised two-phase scoring:
        #   Phase 1 — run the beam history ONCE with use_cache=True to get both
        #             the KV cache and the last-position logit distribution.
        #             Single-token words are scored with a free index lookup.
        #   Phase 2 — for multi-token words only, extend the KV cache one token
        #             at a time.  The history is never recomputed per candidate.
        llm_cand_norm = torch.zeros(len(beams), n_cands, device=device)

        for bi, beam in enumerate(beams):
            h = beam["token_ids"]
            if not h:
                continue   # no context yet → leave as zeros (uniform)

            # Phase 1: one forward pass over history
            hist_ids = torch.tensor([h], dtype=torch.long, device=device)
            hist_out = model(hist_ids, use_cache=True)
            past_kv       = hist_out.past_key_values
            last_logprobs = F.log_softmax(hist_out.logits[0, -1, :], dim=-1)

            # Batch first-token score for all candidates in one index operation
            first_tids_t = torch.tensor(
                [word_to_tids[vocab[wi]][0] for wi in cand_list],
                dtype=torch.long, device=device,
            )
            raw_list: List[float] = last_logprobs[first_tids_t].tolist()

            # Phase 2: extra passes only for multi-token candidates
            for j, wi in enumerate(cand_list):
                tids = word_to_tids[vocab[wi]]
                if len(tids) == 1:
                    continue
                curr_past_kv = past_kv
                extra = 0.0
                for k_tok in range(len(tids) - 1):
                    inp = torch.tensor([[tids[k_tok]]], dtype=torch.long, device=device)
                    out = model(inp, past_key_values=curr_past_kv, use_cache=True)
                    curr_past_kv = out.past_key_values
                    extra += F.log_softmax(out.logits[0, -1, :], dim=-1)[tids[k_tok + 1]].item()
                raw_list[j] += extra

            raw = torch.tensor(raw_list, device=device)
            llm_cand_norm[bi] = _normalize_row(raw, normalization)

        # ── Expand beams ──────────────────────────────────────────────────────
        candidates: List[Dict] = []
        for bi, beam in enumerate(beams):
            blocked: List[Dict] = []   # fallback if all candidates are blocked
            for j, wi in enumerate(cand_list):
                word  = vocab[wi]
                fused = ((1.0 - alpha) * meg_norm[wi].item()
                         + alpha * llm_cand_norm[bi, j].item())
                entry = {
                    "history":   beam["history"] + [word],
                    "token_ids": beam["token_ids"] + word_to_tids[word],
                    "cum_score": beam["cum_score"] + fused,
                }
                if no_repeat_ngram > 0 and _would_repeat_ngram(beam["history"], word, no_repeat_ngram):
                    blocked.append(entry)
                else:
                    candidates.append(entry)
            # If every candidate for this beam was blocked, admit them anyway so
            # the search never stalls (this can happen with very small top_k).
            if not any(c["history"][:-1] == beam["history"] for c in candidates):
                candidates.extend(blocked)
        candidates.sort(key=lambda c: c["cum_score"], reverse=True)
        beams = candidates[:beam_width]

    best      = beams[0]
    b_metrics = eval_option_b(best["history"], word_texts, valid_mask)
    return {
        "pred_sequence": best["history"],
        "cum_score":     float(best["cum_score"]),
        "option_b":      b_metrics,
    }


def sweep_alphas_beam(
    meg_scores:      torch.Tensor,
    vocab:           List[str],
    word_texts:      List[str],
    valid_mask:      List[bool],
    tokenizer,
    model,
    device:          torch.device,
    alphas:          List[float],
    beam_width:      int = 5,
    top_k:           int = 5,
    normalization:   str = "logsoftmax",
    no_repeat_ngram: int = 0,
) -> Dict[float, Dict]:
    """
    Run beam_search_fusion independently for each alpha value.

    Each alpha may produce a different sequence because the beam history
    (which determines the LLM context) depends on which candidates win at
    each step, which in turn depends on alpha.
    """
    return {
        alpha: beam_search_fusion(
            meg_scores, vocab, word_texts, valid_mask,
            tokenizer, model, device,
            alpha=alpha, beam_width=beam_width, top_k=top_k,
            normalization=normalization, no_repeat_ngram=no_repeat_ngram,
        )
        for alpha in alphas
    }
