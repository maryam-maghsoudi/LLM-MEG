"""
fusion.py — Log-linear fusion of MEG scores with teacher-forced LLM next-word scores.

Core idea
---------
At each word position i the LLM is run teacher-forced on the ground-truth
tokens for words 0..i-1 and the logit for every vocab word is extracted
(first subtoken of each word).  The MEG scores from any of the three methods
are combined with these LLM scores via a configurable normalization + linear mix:

    fused = (1 - alpha) * normalize(meg_scores) + alpha * normalize(llm_scores)

alpha=0 → pure MEG, alpha=1 → pure LLM language model.

Normalization modes (--fusion_normalization)
--------------------------------------------
"logsoftmax"  (default)
    normalize = log_softmax(·, dim=-1)
    Product-of-experts interpretation; preserves the original behavior.

"row_zscore"
    For each word position independently, standardize across vocab candidates:
        z = (x - mean(x, dim=-1)) / (std(x, dim=-1) + eps)
    Removes the ~200x scale gap between MEG cosine similarities and LLM logits
    so that alpha has an interpretable effect across the full [0, 1] range.

LLM scores depend only on the poem text (not on the subject or session),
so they are computed once per poem and reused across trials.

Public API
----------
load_fusion_llm(llm_name, device)                          → (tokenizer, model)
compute_llm_scores(word_texts, vocab, ...)                 → Tensor(N, |V|)
scale_diagnostics(meg_scores, llm_scores)                  → dict
fuse_scores(meg_scores, llm_scores, alpha, normalization)  → Tensor(N, |V|)
sweep_alphas(meg, llm, vocab, words, mask, alphas, norm)   → ({alpha: metrics}, diag)
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
