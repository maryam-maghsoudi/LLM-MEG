"""
fusion_beamsearch.py — MEG-guided beam-search fusion for contrastive_multimodal.

At each position t:
  1. MEG encoder nominates the top-k candidates by cosine similarity (or the
     full vocab at invalid positions so the LLM can pick freely).
  2. For each surviving beam, run one LLM forward pass conditioned on that
     beam's own predicted history (NOT ground truth).  Extract log P(w | history)
     for each candidate using the KV cache; normalize over the trial vocab.
  3. fused = (1 - alpha) * normalize(meg) + alpha * normalize(llm)
  4. Retain the top-B beams by cumulative fused score.

Because each alpha may produce a different beam history — and therefore a
different LLM context — every alpha requires a separate beam search.

Evaluation: BLEU-1 and word accuracy on the best-beam sequence vs. ground
truth at valid positions.  R@k is not reported (beam search yields a sequence,
not per-position rankings).

Usage (run from inside contrastive_multimodal/):
    python fusion_beamsearch.py --heldout_subject sub-01
    python fusion_beamsearch.py --heldout_subject sub-01 \\
        --normalization row_zscore --beam_width 5 --top_k 5
    python fusion_beamsearch.py --heldout_subject sub-01 \\
        --no_repeat_ngram 2          # block immediate word repetition
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# run from inside contrastive_multimodal/
sys.path.insert(0, os.path.dirname(__file__))
from new_dataset import MEGContinuousTrialDataset, collate_continuous_trials, MEG_BASE
from new_models import MEGEncoder, WordProjectionHead
from pooling import WordAttentionPooling, pool_words
from splits import make_loso_splits

_EPS = 1e-8

# Same grid as teacher-forced for direct comparison
ALPHA_GRID = [round(a, 2) for a in (
    [i * 0.05 for i in range(19)]
    + [0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]
)]


# ===========================================================================
#  Stage 1 checkpoint loading (identical to fusion_teacher_forced.py)
# ===========================================================================

def load_stage1_checkpoint(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pooling_mode = ckpt.get("pooling_mode", "exact")
    encoder = MEGEncoder().to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.freeze()
    word_head = WordProjectionHead(encoder.backbone_dim).to(device)
    word_head.load_state_dict(ckpt["word_head"])
    word_head.eval()
    pooling_module = WordAttentionPooling(encoder.backbone_dim).to(device)
    if "pooling" in ckpt:
        pooling_module.load_state_dict(ckpt["pooling"])
    pooling_module.eval()
    print(f"Loaded Stage 1: {ckpt_path}  "
          f"(epoch={ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss','?'):.4f}, "
          f"pooling={pooling_mode})")
    return encoder, word_head, pooling_module, pooling_mode


def build_candidate_bank(teacher_cache):
    """
    Build the h_mid occurrence bank from teacher_cache.pt.
    Returns (bank_vectors, bank_word_types, type_to_id).
    """
    all_vecs, all_types = [], []
    type_to_id: Dict[str, int] = {}
    for poem_key, cache in teacher_cache.items():
        if not isinstance(cache, dict) or "h_mid" not in cache:
            continue
        h_mid = cache["h_mid"]           # (N_words, d)
        words = cache["word_texts"]
        for i, w in enumerate(words):
            wl = w.strip().lower()
            all_vecs.append(h_mid[i])
            all_types.append(wl)
            if wl not in type_to_id:
                type_to_id[wl] = len(type_to_id)
    bank_vectors = torch.stack(all_vecs, dim=0).float()   # (M, d)
    return bank_vectors, all_types, type_to_id


# ===========================================================================
#  Per-trial encoder forward pass
# ===========================================================================

def _move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [x.to(device) for x in v]
        else:
            out[k] = v
    return out


@torch.no_grad()
def run_encoder_on_trial(batch, encoder, word_head, pooling_module, pooling_mode, device):
    """Returns z_word (N, 128), valid_mask (N,), word_texts, poem."""
    batch = _move_batch(batch, device)
    z_dense = encoder(batch["meg_trial"])
    pooled, pool_valid = pool_words(
        pooling_mode, z_dense, batch["onset_samples"],
        offset_samples=batch["offset_samples"],
        trial_mask=batch["trial_mask"],
        attention_module=pooling_module,
    )
    z_word = word_head(pooled[0])
    combined_valid = (pool_valid[0] & batch["valid_mask"][0]).cpu()
    return z_word.cpu(), combined_valid, batch["word_texts"][0], batch["poem"][0]


# ===========================================================================
#  MEG score aggregation (occurrence → word-type)
# ===========================================================================

@torch.no_grad()
def meg_scores_to_type_level(z_word, bank_vectors, bank_word_types, vocab):
    """
    Cosine similarity of z_word against every bank occurrence, then MAX
    per word type → Tensor(N, |V|).

    z_word       : (N, 128)  L2-normalized (from WordProjectionHead)
    bank_vectors : (M, d)    NOT unit-norm; normalized here
    """
    bank_norm = F.normalize(bank_vectors.float(), dim=-1).to(z_word.device)
    sim = (z_word.float() @ bank_norm.T).cpu()   # (N, M)

    N, V = z_word.shape[0], len(vocab)
    word_to_vi = {w: i for i, w in enumerate(vocab)}
    type_scores = torch.full((N, V), -1.0)
    for j, wt in enumerate(bank_word_types):
        if wt not in word_to_vi:
            continue
        vi = word_to_vi[wt]
        type_scores[:, vi] = torch.max(type_scores[:, vi], sim[:, j])
    return type_scores   # (N, |V|)


# ===========================================================================
#  LLM loading
# ===========================================================================

def load_fusion_llm(llm_name: str, device):
    print(f"Loading LLM: {llm_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(llm_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  {sum(p.numel() for p in model.parameters()):,} params  frozen")
    return tokenizer, model


# ===========================================================================
#  Beam-search helpers
# ===========================================================================

def _normalize_row(scores: torch.Tensor, normalization: str) -> torch.Tensor:
    """Normalize a 1-D score vector over its own elements."""
    x = scores.float()
    if normalization == "logsoftmax":
        return F.log_softmax(x, dim=0)
    elif normalization == "row_zscore":
        return (x - x.mean()) / (x.std() + _EPS)
    else:
        raise ValueError(f"Unknown normalization: {normalization!r}")


def _would_repeat_ngram(history: List[str], word: str, n: int) -> bool:
    """Return True if appending word to history creates a repeated n-gram."""
    if n <= 0 or len(history) < n - 1:
        return False
    ngram = tuple(history[-(n - 1):]) + (word,)
    for i in range(len(history) - n + 1):
        if tuple(history[i:i + n]) == ngram:
            return True
    return False


# ===========================================================================
#  Beam search
# ===========================================================================

@torch.no_grad()
def beam_search_fusion(
    meg_scores:      torch.Tensor,   # (N, |V|)
    vocab:           List[str],
    word_texts:      List[str],      # ground truth — evaluation only
    valid_mask:      List[bool],
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
    MEG-guided beam-search fusion for one trial at one alpha value.

    At each position t:
      - Valid: top-k MEG candidates, normalized meg scores over |V|.
      - Invalid: full vocab candidates, zero MEG contribution.
    LLM is run once per beam with KV-cache; multi-token words extend the cache.

    Returns
    -------
    dict with pred_sequence, cum_score, metrics (bleu1, word_acc, n_valid).
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
        # ── MEG candidates ────────────────────────────────────────────────────
        meg_row = meg_scores[t].to(device).float()   # (|V|,)
        if valid_mask[t]:
            meg_norm     = _normalize_row(meg_row, normalization)   # (|V|,)
            topk_indices = meg_norm.topk(k).indices                 # (k,)
        else:
            # No MEG signal: let LLM drive the candidate set
            meg_norm     = torch.zeros(V, device=device)
            topk_indices = torch.arange(V, device=device)

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
            last_logp = F.log_softmax(hist_out.logits[0, -1, :], dim=-1)

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
                    extra  += (F.log_softmax(out.logits[0, -1, :], dim=-1)
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
                fused = ((1.0 - alpha) * meg_norm[wi].item()
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
    """Run beam_search_fusion independently for each alpha (different histories)."""
    return {
        alpha: beam_search_fusion(
            meg_scores, vocab, word_texts, valid_mask,
            tokenizer, model, device,
            alpha=alpha, beam_width=beam_width, top_k=top_k,
            normalization=normalization, no_repeat_ngram=no_repeat_ngram,
        )
        for alpha in alphas
    }


# ===========================================================================
#  Sequence-level evaluation
# ===========================================================================

def eval_sequence(pred_seq: List[str], ref_seq: List[str], valid_mask: List[bool]) -> Dict:
    """
    BLEU-1 and word accuracy at valid positions.

    pred_seq  : predicted word sequence (length N, one per position)
    ref_seq   : ground-truth word sequence (length N)
    valid_mask: True where MEG window was usable
    """
    hyp = [p for p, v in zip(pred_seq, valid_mask) if v]
    ref = [r for r, v in zip(ref_seq,  valid_mask) if v]
    n   = len(hyp)
    if n == 0:
        return {"word_acc": 0.0, "bleu1": 0.0, "n_valid": 0}
    word_acc = sum(h == r for h, r in zip(hyp, ref)) / n
    bleu1    = sentence_bleu(
        [ref], hyp, weights=(1.0,),
        smoothing_function=SmoothingFunction().method1,
    )
    return {"word_acc": word_acc, "bleu1": bleu1, "n_valid": n}


# ===========================================================================
#  Plotting
# ===========================================================================

def plot_alpha_sweep_beam(json_path: str, metric: str = "bleu1", out_path: str = None):
    """
    Plot metric vs alpha from a single-subject beam-search fusion JSON.

    metric : "bleu1" or "word_acc"
    """
    import matplotlib
    matplotlib.use("Agg" if out_path else "TkAgg")
    import matplotlib.pyplot as plt
    import numpy as np

    with open(json_path) as f:
        data = json.load(f)

    alphas = [float(a) for a in data["results"].keys()]
    values = [data["results"][str(a)][metric] * 100 for a in alphas]
    alphas_arr = np.array(alphas)
    values_arr = np.array(values)

    best_idx = int(np.argmax(values_arr))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas_arr, values_arr, color="darkorange", linewidth=2.0, label=data["heldout_subject"])
    ax.scatter([alphas_arr[best_idx]], [values_arr[best_idx]],
               color="darkorange", zorder=5, s=60,
               label=f"Best α={alphas_arr[best_idx]:.2f} → {values_arr[best_idx]:.1f}%")
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(0.01, ax.get_ylim()[0] + 0.3, "MEG only", fontsize=8, color="gray")
    ax.text(0.91, ax.get_ylim()[0] + 0.3, "LLM only", fontsize=8, color="gray")
    ax.set_xlabel("Alpha (LLM weight)")
    ax.set_ylabel(f"{metric} (%)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(
        f"Beam-search fusion — {metric}\n"
        f"Subject: {data['heldout_subject']}  "
        f"B={data['beam_width']}  top_k={data['top_k']}  "
        f"LLM: {data['llm_name']}  norm: {data['normalization']}",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot → {out_path}")
        plt.close()
    else:
        plt.show()


def plot_alpha_sweep_beam_multi(json_paths: List[str], metric: str = "bleu1",
                                 out_path: str = None, title: str = None):
    """
    Plot metric vs alpha from multiple subjects: thin lines per subject + mean line.
    Intended for use after all 13 LOSO runs complete.
    """
    import matplotlib
    matplotlib.use("Agg" if out_path else "TkAgg")
    import matplotlib.pyplot as plt
    import numpy as np

    all_alphas = None
    subject_curves: Dict[str, List[float]] = {}

    for path in json_paths:
        with open(path) as f:
            data = json.load(f)
        subj   = data["heldout_subject"]
        alphas = [float(a) for a in data["results"].keys()]
        values = [data["results"][str(a)][metric] * 100 for a in alphas]
        if all_alphas is None:
            all_alphas = alphas
        subject_curves[subj] = values

    alphas_arr = np.array(all_alphas)
    curves     = np.array(list(subject_curves.values()))
    mean_curve = curves.mean(axis=0)
    best_idx   = int(np.argmax(mean_curve))

    fig, ax = plt.subplots(figsize=(8, 5))
    for curve in subject_curves.values():
        ax.plot(alphas_arr, curve, color="darkorange", alpha=0.25, linewidth=1.0)
    ax.plot(alphas_arr, mean_curve, color="darkorange", linewidth=2.5,
            label=f"Mean (n={len(subject_curves)})")
    ax.scatter([alphas_arr[best_idx]], [mean_curve[best_idx]],
               color="darkorange", zorder=5, s=60,
               label=f"Best α={alphas_arr[best_idx]:.2f} → {mean_curve[best_idx]:.1f}%")
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(0.01, ax.get_ylim()[0] + 0.3, "MEG only", fontsize=8, color="gray")
    ax.text(0.91, ax.get_ylim()[0] + 0.3, "LLM only", fontsize=8, color="gray")
    ax.set_xlabel("Alpha (LLM weight)")
    ax.set_ylabel(f"{metric} (%)")
    ax.set_xlim(-0.02, 1.02)
    if title is None:
        with open(json_paths[0]) as f:
            meta = json.load(f)
        title = (f"Beam-search fusion — {metric}\n"
                 f"B={meta['beam_width']}  top_k={meta['top_k']}  "
                 f"LLM: {meta['llm_name']}  norm: {meta['normalization']}  "
                 f"n_subjects={len(subject_curves)}")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot → {out_path}")
        plt.close()
    else:
        plt.show()


# ===========================================================================
#  Main
# ===========================================================================

def _load_onsets(poem: str) -> List[Dict]:
    import json as _json
    onset_dir = Path("/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis"
                     "/contrastive_learning/onset_out")
    for cond in ("lis", "img"):
        p = onset_dir / f"{poem}{cond}_onsets.json"
        if p.exists():
            with open(p) as f:
                return _json.load(f)
    raise FileNotFoundError(f"No onset JSON found for poem={poem!r} in {onset_dir}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  subject={args.heldout_subject}  "
          f"llm={args.llm_name}  norm={args.normalization}  "
          f"beam_width={args.beam_width}  top_k={args.top_k}")

    # Stage 1
    encoder, word_head, pooling_module, pooling_mode = load_stage1_checkpoint(
        args.stage1_checkpoint_path, device
    )

    # Candidate bank
    teacher_cache = torch.load(args.teacher_cache_path, weights_only=False)
    bank_vectors, bank_word_types, type_to_id = build_candidate_bank(teacher_cache)
    bank_vectors = bank_vectors.to(device)
    print(f"Bank: {bank_vectors.shape[0]} occurrences, {len(type_to_id)} unique types")

    # LLM
    tokenizer, llm_model = load_fusion_llm(args.llm_name, device)

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

    # Accumulate metrics: {alpha: {metric: sum, n_trials: int}}
    acc = {a: {"bleu1": 0.0, "word_acc": 0.0, "n_valid": 0, "n_trials": 0}
           for a in ALPHA_GRID}
    n_trials = 0

    print(f"Running beam-search fusion over {len(test_ds)} trials "
          f"× {len(ALPHA_GRID)} alphas ...")
    for batch in test_loader:
        z_word, valid_mask, word_texts, poem = run_encoder_on_trial(
            batch, encoder, word_head, pooling_module, pooling_mode, device
        )

        vocab      = vocab_cache[poem]
        meg_scores = meg_scores_to_type_level(
            z_word.to(device), bank_vectors, bank_word_types, vocab
        )                                    # (N_poem, |V|)

        valid_list = valid_mask.tolist()

        # Each alpha is a separate beam search
        alpha_results = sweep_alphas_beam(
            meg_scores, vocab, word_texts, valid_list,
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
    print(f"\n=== Beam fusion — {args.heldout_subject}  "
          f"llm={args.llm_name}  norm={args.normalization}  "
          f"B={args.beam_width}  top_k={args.top_k} ===")
    print(f"{'alpha':>6}  {'BLEU-1':>8}  {'word_acc':>9}")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r = results[str(alpha)]
        print(f"{alpha:6.2f}  {r['bleu1']*100:7.2f}%  {r['word_acc']*100:8.2f}%")

    # Save JSON
    os.makedirs(args.out_dir, exist_ok=True)
    llm_tag  = args.llm_name.replace("/", "_")
    out_path = os.path.join(
        args.out_dir,
        f"{args.heldout_subject}_beamfusion_B{args.beam_width}"
        f"_top{args.top_k}_{llm_tag}_{args.normalization}.json",
    )
    output = {
        "heldout_subject": args.heldout_subject,
        "llm_name":        args.llm_name,
        "normalization":   args.normalization,
        "beam_width":      args.beam_width,
        "top_k":           args.top_k,
        "no_repeat_ngram": args.no_repeat_ngram,
        "n_trials":        n_trials,
        "results":         results,
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
        description="MEG-guided beam-search LLM fusion for contrastive_multimodal."
    )
    p.add_argument("--heldout_subject", type=str, default="sub-01")
    p.add_argument("--stage1_checkpoint_path", type=str, default=None,
                   help="Path to stage1_best_*.pt. Defaults to "
                        "checkpoints/joint_annealed_exact/stage1_best_{subject}_joint_annealed_exact.pt")
    p.add_argument("--teacher_cache_path", type=str, default="teacher_cache.pt")
    p.add_argument("--llm_name", type=str, default="gpt2")
    p.add_argument("--normalization", type=str, default="logsoftmax",
                   choices=["logsoftmax", "row_zscore"])
    p.add_argument("--beam_width", type=int, default=5,
                   help="Number of beams to maintain at each step.")
    p.add_argument("--top_k", type=int, default=5,
                   help="MEG top-k candidates considered per beam step. "
                        "At invalid positions the full vocab is used instead.")
    p.add_argument("--no_repeat_ngram", type=int, default=0,
                   help="Block repeated n-grams of this length (0=disabled). "
                        "n=2 prevents immediate word repetition.")
    p.add_argument("--meg_base", type=str, default=None,
                   help="Path to icaed_Sai directory. Defaults to new_dataset.MEG_BASE.")
    p.add_argument("--out_dir", type=str, default="fusion_results")
    p.add_argument("--plot", action="store_true",
                   help="Save bleu1 and word_acc vs alpha plots alongside the JSON.")
    args = p.parse_args()

    if args.stage1_checkpoint_path is None:
        args.stage1_checkpoint_path = (
            f"checkpoints/joint_annealed_exact/"
            f"stage1_best_{args.heldout_subject}_joint_annealed_exact.pt"
        )

    main(args)
