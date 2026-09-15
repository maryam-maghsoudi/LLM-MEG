"""
metrics.py — shared top-k accuracy scoring for eval_stage1.py / eval_stage2.py.
"""

import torch
import torch.nn.functional as F


def topk_predictions(scores: torch.Tensor, k: int) -> torch.Tensor:
    """
    Returns the top-k candidate INDICES per query — (N, k),
    """
    k = min(k, scores.shape[1])
    return torch.topk(scores, k, dim=1).indices


def topk_accuracy_label_aware(scores: torch.Tensor, query_labels: torch.Tensor,
                               candidate_labels: torch.Tensor, k: int,
                               valid_mask: "torch.Tensor | None" = None) -> float:
    """
    scores           : (N_queries, N_candidates)
    query_labels     : (N_queries,) — each query's TRUE label
    candidate_labels : (N_candidates,) — each candidate's label
    valid_mask       : (N_queries,) bool, optional — excludes invalid queries entirely

    Returns the fraction of (valid) queries where at least one of the
    top-k highest-scoring candidates shares the query's label.
    """
    N_q, N_c = scores.shape
    k = min(k, N_c)
    topk_idx = torch.topk(scores, k, dim=1).indices          # (N_q, k)
    topk_labels = candidate_labels[topk_idx]                  # (N_q, k)
    hit = (topk_labels == query_labels.unsqueeze(1)).any(dim=1)   # (N_q,)
    if valid_mask is not None:
        hit = hit[valid_mask]
    return hit.float().mean().item() if hit.numel() > 0 else float("nan")


def topk_accuracy_from_scores(scores: torch.Tensor, true_indices: torch.Tensor, k: int,
                               valid_mask: "torch.Tensor | None" = None) -> float:
    """
    scores       : (N, C)
    true_indices : (N,) long — the single correct candidate INDEX per query


    """
    N_c = scores.shape[1]
    candidate_labels = torch.arange(N_c, device=scores.device)
    return topk_accuracy_label_aware(scores, true_indices, candidate_labels, k, valid_mask)


def chance_level(n_candidates: int, k: int) -> float:
    """
    Naive random-guessing chance level: k / n_candidates.


    """
    return k / n_candidates


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(N_a, D), (N_b, D) -> (N_a, N_b) cosine similarity.
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.T


def permutation_percentile(real_value: float, null_values) -> float:
    """
    Where does real_value fall relative to a null/shuffled distribution
    of the SAME metric? Returns the fraction of null_values >= real_value
    — a one-sided estimate: SMALLER means the real result sits further
    into the tail of the null distribution (less consistent with "no
    genuine signal, this is just what shuffled MEG achieves by chance").
    null_values: any sequence of floats (e.g. from many --shuffle_control
    runs with different seeds).
    """
    null_values = torch.as_tensor(list(null_values), dtype=torch.float32)
    return (null_values >= real_value).float().mean().item()


def per_label_topk_accuracy(topk_idx, query_labels, bank_labels):
    """
    Same "any occurrence of the correct type counts" rule as
    topk_accuracy_label_aware, broken out PER label instead of aggregated.
    topk_idx: (N, k) indices from topk_predictions(scores, k). Caller
    pre-filters to valid rows, same convention as topk_accuracy_label_aware.
    Returns (n_per_label, hits_per_label), both dicts keyed by label id.
    """
    topk_labels = bank_labels[topk_idx]
    hit = (topk_labels == query_labels.unsqueeze(1)).any(dim=1)
    n_per_label, hits_per_label = {}, {}
    for lbl in query_labels.unique().tolist():
        mask = query_labels == lbl
        n_per_label[lbl] = int(mask.sum())
        hits_per_label[lbl] = int(hit[mask].sum())
    return n_per_label, hits_per_label


def per_label_prediction_counts(topk_idx, bank_labels):
    """
    How often each bank label appears ANYWHERE in the given top-k
    predictions, across all queries -- independent of the true label.
    Precision-side companion to per_label_topk_accuracy: this is what
    quantifies over-prediction / skew, not recall.
    """
    topk_labels = bank_labels[topk_idx]
    counts = {}
    for lbl in topk_labels.unique().tolist():
        counts[lbl] = int((topk_labels == lbl).sum())
    return counts, topk_labels.shape[0]


def bank_label_counts(bank_labels):
    return {int(lbl): int((bank_labels == lbl).sum()) for lbl in bank_labels.unique()}


def chance_level_hypergeometric(count_in_bank, n_candidates, k):
    """
    Exact per-TYPE chance level -- corrects chance_level(n,k)=k/n, which
    ignores duplicate bank occurrences. A type with more bank duplicates
    has genuinely higher chance than the flat aggregate number implies.
    """
    if k >= n_candidates:
        return 1.0
    p_none = 1.0
    for i in range(k):
        p_none *= (n_candidates - count_in_bank - i) / (n_candidates - i)
    return 1.0 - p_none
