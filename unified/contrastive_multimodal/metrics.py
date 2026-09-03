"""
metrics.py — shared top-k accuracy scoring for eval_stage1.py / eval_stage2.py.

Pure functions only: everything here operates on already-computed score
tensors and label tensors. Building those tensors from real data (the
candidate bank from teacher_cache.py, the closed-vocab distribution from
train_stage2.py's machinery) is the eval scripts' job, not this file's —
keeps this file testable with zero dependency on real MEG/GPT-2 data.

topk_accuracy_label_aware   General case: a query is correct if ANY of its
                            top-k highest-scoring candidates shares its
                            LABEL — not just if one specific literal index
                            is in the top-k. This is the SAME underlying
                            idea as losses.py's multi_positive_contrastive_
                            loss (label equality, not index equality,
                            defines a match), showing up again at eval
                            time for the same reason: §8's candidate bank
                            has multiple OCCURRENCES per word TYPE, so more
                            than one candidate can legitimately count as
                            "correct" for a given query.
topk_accuracy_from_scores   Special case: exactly one correct candidate per
                            query, at a KNOWN index (e.g. Eval 2 — q_t is
                            already a distribution directly over token
                            ids). Implemented as topk_accuracy_label_aware
                            with each candidate labeled by its own index —
                            not a separate algorithm, the same one.
chance_level                Naive k/n_candidates — matches §8's own stated
                            numbers (~0.7% top-1, ~3.6% top-5 at ~140
                            candidates) exactly, so it's the established
                            convention to report against, not a shortcut
                            invented here. See its docstring for the
                            approximation it makes.
cosine_similarity_matrix    Small reusable utility for building Eval 1's
                            query-vs-candidate-bank score matrix.
permutation_percentile      Where a real result falls relative to a null/
                            shuffled distribution — shared machinery for
                            both eval scripts' --shuffle_control.
"""

import torch
import torch.nn.functional as F


def topk_predictions(scores: torch.Tensor, k: int) -> torch.Tensor:
    """
    Returns the top-k candidate INDICES per query — (N, k), NOT a boolean
    accuracy. topk_accuracy_from_scores / topk_accuracy_label_aware
    compute exactly this internally (torch.topk) and discard it; exposed
    separately here so eval scripts can build a qualitative prediction
    trace (ground truth vs. top-1 vs. top-k, per word) from the SAME
    ranking used for the accuracy number, without a second torch.topk call.
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
    top-k highest-scoring candidates shares the query's label. NaN if
    there are zero valid queries — callers should check for this rather
    than silently averaging it into a larger result.
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

    Thin call into topk_accuracy_label_aware with candidates labeled by
    their own index (candidate j's label == j), so "true label == some
    top-k candidate's label" collapses to exactly "true index is in the
    top-k" — the ordinary meaning of top-k accuracy.
    """
    N_c = scores.shape[1]
    candidate_labels = torch.arange(N_c, device=scores.device)
    return topk_accuracy_label_aware(scores, true_indices, candidate_labels, k, valid_mask)


def chance_level(n_candidates: int, k: int) -> float:
    """
    Naive random-guessing chance level: k / n_candidates.

    APPROXIMATION for the label-aware case: this ignores that some word
    types have more than one occurrence in the candidate bank, which
    technically raises the TRUE random-chance hit rate slightly above
    this number (a random top-k draw is a bit more likely to land on
    *some* occurrence of the right word than this simple ratio implies).
    Matches §8's own stated numbers exactly at n=140 though (1/140≈0.7%,
    5/140≈3.6%), so this is the established convention to report
    against — not a shortcut invented here.
    """
    return k / n_candidates


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(N_a, D), (N_b, D) -> (N_a, N_b) cosine similarity. Both sides
    L2-normalized here regardless of whether the caller already did —
    same defensive convention as losses.py's contrastive functions."""
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


if __name__ == "__main__":
    torch.manual_seed(0)
    print("=== metrics.py sanity check ===\n")

    # ------------------------------------------------------------------
    # 1. topk_accuracy_from_scores — hand-constructed, unambiguous case.
    # ------------------------------------------------------------------
    scores = torch.tensor([
        [0.9, 0.1, 0.05, 0.0],   # top-1 = idx0, top-2 = idx0,idx1
        [0.1, 0.2, 0.9, 0.0],    # top-1 = idx2
        [0.1, 0.1, 0.1, 0.9],    # top-1 = idx3
    ])
    true_indices = torch.tensor([0, 2, 0])   # query 2's true answer is idx0, but top-1 is idx3 -> miss
    acc_top1 = topk_accuracy_from_scores(scores, true_indices, k=1)
    acc_top2 = topk_accuracy_from_scores(scores, true_indices, k=2)
    assert abs(acc_top1 - 2 / 3) < 1e-6, f"expected 2/3, got {acc_top1}"
    print(f"[OK] topk_accuracy_from_scores: top1={acc_top1:.4f} (expected 2/3), top2={acc_top2:.4f}")

    # ------------------------------------------------------------------
    # 2. topk_accuracy_label_aware — the actual property being tested:
    #    a query counts as correct even when the LITERAL matching
    #    candidate index isn't in the top-k, as long as ANOTHER
    #    candidate sharing its label is. Mirrors losses.py's check #2.
    # ------------------------------------------------------------------
    # 4 candidates: idx0 and idx2 are both label "cat", idx1 and idx3 are both label "dog".
    candidate_labels = torch.tensor([10, 20, 10, 20])   # 10="cat", 20="dog"
    scores2 = torch.tensor([
        [0.1, 0.05, 0.9, 0.05],   # top-1 = idx2 (label "cat") -- query's OWN occurrence was idx0, but idx2 (same label) IS top-1
    ])
    query_labels = torch.tensor([10])   # this query's true label is "cat" (occurrence idx0's label)
    acc_label_aware = topk_accuracy_label_aware(scores2, query_labels, candidate_labels, k=1)
    assert acc_label_aware == 1.0, (
        f"expected a hit (idx2 shares the query's label even though idx0 wasn't top-1), got {acc_label_aware}"
    )
    print(f"[OK] topk_accuracy_label_aware: hit={acc_label_aware} "
          f"(correct even though the LITERAL matching candidate wasn't top-1)")

    # And the converse: same setup, but restrict the top-k to NOT include
    # any "cat"-labeled candidate at all -> must be a miss.
    scores3 = torch.tensor([[0.05, 0.9, 0.04, 0.01]])   # top-1 = idx1, label "dog" -- no "cat" anywhere near the top
    acc_miss = topk_accuracy_label_aware(scores3, query_labels, candidate_labels, k=1)
    assert acc_miss == 0.0, f"expected a miss, got {acc_miss}"
    print(f"[OK] topk_accuracy_label_aware: correctly misses when no same-label candidate is in top-k")

    # ------------------------------------------------------------------
    # 3. Cross-check: topk_accuracy_from_scores IS topk_accuracy_label_aware
    #    with identity-mapped candidate labels — verify they agree.
    # ------------------------------------------------------------------
    identity_labels = torch.arange(scores.shape[1])
    acc_via_label_aware = topk_accuracy_label_aware(scores, true_indices, identity_labels, k=1)
    assert abs(acc_via_label_aware - acc_top1) < 1e-6
    print(f"[OK] topk_accuracy_from_scores agrees exactly with the label-aware general case")

    # ------------------------------------------------------------------
    # 4. valid_mask correctly excludes queries.
    # ------------------------------------------------------------------
    bad_true_indices = torch.tensor([0, 2, 999])  # query 2's "true" index is nonsense — must be masked out, not crash-checked
    valid_mask = torch.tensor([True, True, False])
    # topk over 4 candidates never returns index 999, so query 2 would count as a miss if not excluded --
    # confirm excluding it changes the result vs. leaving it in.
    acc_with_bad = topk_accuracy_from_scores(scores, torch.tensor([0, 2, 0]), k=1, valid_mask=None)
    acc_excluded = topk_accuracy_from_scores(scores, torch.tensor([0, 2, 0]), k=1, valid_mask=valid_mask)
    assert abs(acc_excluded - 1.0) < 1e-6, f"with query 2 excluded, remaining 2 queries should both hit, got {acc_excluded}"
    print(f"[OK] valid_mask: full={acc_with_bad:.4f}  excluding query 2={acc_excluded:.4f}")

    # ------------------------------------------------------------------
    # 5. chance_level — matches §8's stated numbers at n=140.
    # ------------------------------------------------------------------
    c1 = chance_level(140, 1)
    c5 = chance_level(140, 5)
    assert abs(c1 - 0.00714) < 1e-4 and abs(c5 - 0.03571) < 1e-4
    print(f"[OK] chance_level(140, 1)={c1*100:.2f}%  chance_level(140, 5)={c5*100:.2f}%  "
          f"(matches the design spec's stated ~0.7% / ~3.6%)")

    # ------------------------------------------------------------------
    # 6. cosine_similarity_matrix — identical vectors -> 1.0, orthogonal -> 0.0.
    # ------------------------------------------------------------------
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    sim = cosine_similarity_matrix(v, v)
    assert torch.allclose(sim, torch.tensor([[1.0, 0.0], [0.0, 1.0]]), atol=1e-6)
    print(f"[OK] cosine_similarity_matrix: identical=1.0, orthogonal=0.0 as expected")

    # ------------------------------------------------------------------
    # 7. permutation_percentile — hand-checkable null distribution.
    # ------------------------------------------------------------------
    null_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    p = permutation_percentile(real_value=0.35, null_values=null_vals)
    assert abs(p - 2 / 5) < 1e-6, f"2 of 5 null values (0.4, 0.5) are >= 0.35, expected 0.4, got {p}"
    print(f"[OK] permutation_percentile(0.35, {null_vals}) = {p:.4f} (2/5 null values exceed it)")

    # ------------------------------------------------------------------
    # 8. topk_predictions — the raw indices agree with what the accuracy
    #    functions compute internally (same torch.topk, exposed directly).
    # ------------------------------------------------------------------
    preds = topk_predictions(scores, k=2)
    assert preds.tolist() == torch.topk(scores, 2, dim=1).indices.tolist()
    hit_from_preds = (preds == true_indices.unsqueeze(1)).any(dim=1).float().mean().item()
    assert abs(hit_from_preds - acc_top2) < 1e-6, "topk_predictions must agree with topk_accuracy_from_scores"
    print(f"[OK] topk_predictions matches the ranking topk_accuracy_from_scores computes internally")

    print("\n=== ALL CHECKS PASSED ===")
