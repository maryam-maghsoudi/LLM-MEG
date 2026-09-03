"""
fusion_teacher_forced.py — LLM+MEG log-linear fusion for contrastive_multimodal.

Teacher-forced mode: LLM conditions on the ground-truth word prefix at each
position. LLM scores depend only on the poem text, not on the subject or
session, so they are computed ONCE per poem and reused across all test trials.

MEG scores: encoder → pooling → word_head → cosine similarity against the
full h_mid bank (117 occurrences), then aggregated to word-type level by
taking the MAX cosine similarity across all bank occurrences of each type.

Fusion formula:
    fused = (1 - alpha) * normalize(meg) + alpha * normalize(llm)

Both sides are normalized per-row over the poem vocabulary (|V| unique words)
before mixing, so the denominator is identical and scores are comparable.

Normalization modes (--normalization):
    logsoftmax  (default) — log_softmax over |V|. The LLM/MEG std ratio is
                typically ~100-200x, so LLM dominates at even small alpha.
    row_zscore  — z-score per row over |V|. Equalizes scale; gives a smoother
                alpha curve with a genuine interior optimum.

Usage (run from inside contrastive_multimodal/):
    python fusion_teacher_forced.py --heldout_subject sub-01
    python fusion_teacher_forced.py --heldout_subject sub-01 --normalization row_zscore
    python fusion_teacher_forced.py --heldout_subject sub-01 --llm_name gpt2-medium
"""

import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_stage1 import load_stage1_checkpoint, build_candidate_bank
from new_dataset import MEGContinuousTrialDataset, collate_continuous_trials, _load_onsets
from splits import make_loso_splits
from train import _move_batch, POEM_TO_ID
from pooling import pool_words

_EPS = 1e-8

# Coarse grid + fine grid near alpha=1 (same convention as unified/fuse_eval.py)
ALPHA_GRID = [round(a, 2) for a in (
    [i * 0.05 for i in range(19)]          # 0.00, 0.05, ..., 0.90
    + [0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]
)]


# ===========================================================================
#  LLM loading
# ===========================================================================

def load_fusion_llm(llm_name: str, device):
    """Load a frozen causal LM. Returns (tokenizer, model)."""
    print(f"[fusion] Loading LLM: {llm_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(llm_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in model.parameters())
    print(f"  {llm_name}  {n:,} params  frozen  device={device}")
    return tokenizer, model


# ===========================================================================
#  LLM teacher-forced scoring
# ===========================================================================

@torch.no_grad()
def compute_llm_scores(word_texts, vocab, tokenizer, model, device):
    """
    Teacher-forced LLM next-word scores over vocab.

    At position i, conditions on the ground-truth token sequence for words
    0..i-1 and scores each vocab word by its first-subtoken logit.
    Position 0 has no preceding context → zero scores (uniform).

    Parameters
    ----------
    word_texts : List[str]  — full word sequence for the poem (N words, with repeats)
    vocab      : List[str]  — sorted unique words in the poem (|V| entries)

    Returns
    -------
    Tensor(N, |V|)  raw logits (NOT normalized — normalization happens in fuse_scores)
    """
    N, V = len(word_texts), len(vocab)

    # First-subtoken ID per vocab word (with leading space for mid-BPE context)
    vocab_tok_ids = []
    for w in vocab:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(w, add_special_tokens=False)
        vocab_tok_ids.append(ids[0] if ids else (tokenizer.unk_token_id or 0))
    vocab_tok_ids_t = torch.tensor(vocab_tok_ids, dtype=torch.long, device=device)

    # Full subtoken sequence for the poem
    token_ids_per_word = []
    for w in word_texts:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        token_ids_per_word.append(ids if ids else [tokenizer.unk_token_id or 0])

    # Build input_ids with optional BOS
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    all_ids = bos.copy()
    boundaries = []   # boundaries[i] = token index whose logit predicts word i
    for ids in token_ids_per_word:
        boundaries.append(len(all_ids) - 1)   # -1 if no BOS → position 0 stays zeros
        all_ids.extend(ids)

    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
    logits = model(input_ids).logits[0]   # (T_tokens, vocab_size)

    scores = torch.zeros(N, V)
    for i, b in enumerate(boundaries):
        if b < 0:
            continue   # no context (position 0 without BOS) → leave as zeros
        scores[i] = logits[b, vocab_tok_ids_t].cpu()

    return scores   # (N, |V|)


# ===========================================================================
#  MEG score aggregation: occurrence-level → word-type-level
# ===========================================================================

def meg_scores_to_type_level(z_word, bank_vectors, bank_word_types, vocab):
    """
    Convert per-occurrence cosine similarities to per-word-type scores
    by taking the MAX similarity across all bank occurrences of each type.

    z_word          : (N, 128)  L2-normalized query embeddings (one per word position)
    bank_vectors    : (117, d)  h_mid bank — NOT unit-norm (raw JL-projected GPT-2
                      hiddens, norms range ~33–1259). Normalized here before dot product.
    bank_word_types : List[str]  word type per bank entry (117 entries, with repeats)
    vocab           : List[str]  sorted unique words in this poem (|V| entries)

    Returns Tensor(N, |V|) on CPU.

    Words in the bank that aren't in vocab (e.g. poem2 words during a poem1
    trial) are skipped. Any vocab word absent from the bank stays at -1.0
    (shouldn't happen — all 76 types have at least one bank occurrence).
    """
    # z_word is already L2-normalized (WordProjectionHead); normalize bank_vectors
    # so this is true cosine similarity — consistent with eval_stage1.py's
    # cosine_similarity_matrix which also normalizes both sides.
    bank_norm = F.normalize(bank_vectors.float(), dim=-1)   # (117, d)
    sim = (z_word @ bank_norm.T).cpu()                      # (N, 117)

    N, V = z_word.shape[0], len(vocab)
    word_to_vi = {w: i for i, w in enumerate(vocab)}

    type_scores = torch.full((N, V), -1.0)   # floor: cosine ∈ [-1, 1]
    for j, wt in enumerate(bank_word_types):
        if wt not in word_to_vi:
            continue
        vi = word_to_vi[wt]
        type_scores[:, vi] = torch.max(type_scores[:, vi], sim[:, j])

    return type_scores   # (N, |V|)


# ===========================================================================
#  Fusion and evaluation
# ===========================================================================

def fuse_scores(meg_scores, llm_scores, alpha, normalization):
    """
    Log-linear fusion: fused = (1-alpha)*normalize(meg) + alpha*normalize(llm)
    Both inputs are (N, |V|); output is (N, |V|).
    """
    meg = meg_scores.float()
    llm = llm_scores.float()
    if normalization == "logsoftmax":
        n_meg = F.log_softmax(meg, dim=-1)
        n_llm = F.log_softmax(llm, dim=-1)
    elif normalization == "row_zscore":
        n_meg = (meg - meg.mean(dim=-1, keepdim=True)) / (meg.std(dim=-1, keepdim=True) + _EPS)
        n_llm = (llm - llm.mean(dim=-1, keepdim=True)) / (llm.std(dim=-1, keepdim=True) + _EPS)
    else:
        raise ValueError(f"Unknown normalization: {normalization!r}")
    return (1.0 - alpha) * n_meg + alpha * n_llm


def eval_metrics(fused_scores, vocab, word_texts, valid_mask, ks=(1, 5)):
    """
    R@k, MRR, word accuracy, and BLEU-1 at valid positions where ground-truth is in vocab.

    R@k, MRR, word_acc are raw sums — caller divides by n_valid for rates.
    bleu1 is a per-trial sentence BLEU-1 rate (0–1) — caller divides by n_trials.
    """
    word_to_vi = {w: i for i, w in enumerate(vocab)}
    rk_sum = {k: 0 for k in ks}
    mrr_sum = 0.0
    acc_sum = 0
    n_valid = 0
    hyp, ref = [], []

    for i, (word, valid) in enumerate(zip(word_texts, valid_mask)):
        if not valid or word not in word_to_vi:
            continue
        n_valid += 1
        true_vi = word_to_vi[word]
        scores_i = fused_scores[i]
        rank = int((scores_i > scores_i[true_vi]).sum().item()) + 1
        for k in ks:
            rk_sum[k] += int(rank <= k)
        mrr_sum += 1.0 / rank
        acc_sum += int(rank == 1)
        hyp.append(vocab[int(scores_i.argmax().item())])
        ref.append(word)

    if hyp:
        bleu1 = sentence_bleu([ref], hyp, weights=(1.0,),
                              smoothing_function=SmoothingFunction().method1)
    else:
        bleu1 = 0.0

    return {
        "n_valid": n_valid,
        **{f"R@{k}": rk_sum[k] for k in ks},
        "MRR": mrr_sum,
        "word_acc": acc_sum,
        "bleu1": bleu1,
    }


def scale_diagnostics(meg_scores, llm_scores):
    """Per-row std ratio summary (LLM/MEG). Alpha-independent diagnostic."""
    meg_std = meg_scores.float().std(dim=-1)
    llm_std = llm_scores.float().std(dim=-1)
    ratio = llm_std / (meg_std + _EPS)
    return {
        "mean_meg_row_std":   float(meg_std.mean()),
        "mean_llm_row_std":   float(llm_std.mean()),
        "mean_scale_ratio":   float(ratio.mean()),
        "median_scale_ratio": float(ratio.median()),
    }


# ===========================================================================
#  Plotting
# ===========================================================================

def plot_alpha_sweep(
    json_paths,
    metric: str = "R@1",
    out_path: str = None,
    title: str = None,
):
    """
    Plot a fusion metric vs. alpha from one or more per-subject result JSONs.

    json_paths : str or List[str] — path(s) to *_fusion_*.json files
    metric     : one of "R@1", "R@5", "MRR", "word_acc", "bleu1"
    out_path   : if given, save figure to this path (PNG/PDF); else show interactively
    title      : optional figure title override

    One thin line per subject + a thick mean line. Vertical dashed lines mark
    alpha=0 (MEG-only) and alpha=1 (LLM-only). Best-alpha marker on the mean.
    """
    import matplotlib
    matplotlib.use("Agg" if out_path else "TkAgg")
    import matplotlib.pyplot as plt
    import numpy as np

    if isinstance(json_paths, str):
        json_paths = [json_paths]

    all_alphas = None
    subject_curves = {}   # subject → List[float]

    for path in json_paths:
        with open(path) as f:
            data = json.load(f)
        subj = data["heldout_subject"]
        alphas = [float(a) for a in data["results"].keys()]
        values = [data["results"][a][metric] * 100 for a in data["results"]]
        if all_alphas is None:
            all_alphas = alphas
        subject_curves[subj] = values

    alphas_arr = np.array(all_alphas)
    curves = np.array(list(subject_curves.values()))   # (n_subjects, n_alphas)
    mean_curve = curves.mean(axis=0)
    best_idx = int(np.argmax(mean_curve))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Per-subject lines
    for subj, curve in subject_curves.items():
        ax.plot(alphas_arr, curve, color="steelblue", alpha=0.25, linewidth=1.0)

    # Mean line
    ax.plot(alphas_arr, mean_curve, color="steelblue", linewidth=2.5,
            label=f"Mean (n={len(subject_curves)})")

    # Best-alpha marker on mean
    ax.scatter([alphas_arr[best_idx]], [mean_curve[best_idx]],
               color="steelblue", zorder=5, s=60,
               label=f"Best α={alphas_arr[best_idx]:.2f} → {mean_curve[best_idx]:.1f}%")

    # Reference lines
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(0.01, ax.get_ylim()[0] + 0.5, "MEG only", fontsize=8, color="gray")
    ax.text(0.91, ax.get_ylim()[0] + 0.5, "LLM only", fontsize=8, color="gray")

    ax.set_xlabel("Alpha (LLM weight)")
    ax.set_ylabel(f"{metric} (%)")
    ax.set_xlim(-0.02, 1.02)

    # Derive a default title from the first file's metadata
    if title is None:
        with open(json_paths[0]) as f:
            meta = json.load(f)
        title = (f"Fusion alpha sweep — {metric}\n"
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
#  Single-trial MEG forward pass
# ===========================================================================

def run_encoder_on_trial(batch, encoder, word_head, pooling_module, pooling_mode, device):
    """
    Run encoder → pooling → word_head for one trial (batch_size=1).

    Returns
    -------
    z_word     : Tensor(N_words, 128)  L2-normalized, one embedding per word position
    valid_mask : Tensor(N_words,) bool  True where pooling window AND onset are valid
    word_texts : List[str]             word sequence for this trial
    poem       : str
    """
    batch = _move_batch(batch, device)
    with torch.no_grad():
        z_dense = encoder(batch["meg_trial"])              # (1, T_out, 128)
        pooled, pool_valid = pool_words(
            pooling_mode, z_dense, batch["onset_samples"],
            offset_samples=batch["offset_samples"],
            trial_mask=batch["trial_mask"],
            attention_module=pooling_module,
        )                                                  # (1, N, 128), (1, N)
        z_word = word_head(pooled[0])                      # (N, 128)  — already L2-normalized
        combined_valid = (pool_valid[0] & batch["valid_mask"][0]).cpu()

    return z_word.cpu(), combined_valid, batch["word_texts"][0], batch["poem"][0]


# ===========================================================================
#  Main
# ===========================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  subject={args.heldout_subject}  "
          f"llm={args.llm_name}  norm={args.normalization}")

    # --- Stage 1 checkpoint ---
    encoder, word_head, pooling_module, pooling_mode, _ = load_stage1_checkpoint(
        args.stage1_checkpoint_path, device
    )

    # --- Candidate bank (117 occurrences, 76 types) ---
    teacher_cache = torch.load(args.teacher_cache_path, weights_only=False)
    bank_vectors, _, bank_word_types, type_to_id, _, _ = build_candidate_bank(teacher_cache)
    bank_vectors = bank_vectors.to(device)
    print(f"Bank: {bank_vectors.shape[0]} occurrences, {len(type_to_id)} unique types")

    # --- LLM ---
    tokenizer, llm_model = load_fusion_llm(args.llm_name, device)

    # --- Pre-compute LLM scores once per poem ---
    print("Pre-computing teacher-forced LLM scores (once per poem) ...")
    llm_cache = {}    # poem → Tensor(N_poem, |V_poem|)
    vocab_cache = {}  # poem → List[str]
    for poem in ("poem1", "poem2"):
        onsets = _load_onsets(poem)
        word_texts_poem = [e["word"].strip().lower() for e in onsets]
        vocab = sorted(set(word_texts_poem))
        llm_cache[poem] = compute_llm_scores(word_texts_poem, vocab, tokenizer, llm_model, device)
        vocab_cache[poem] = vocab
        print(f"  {poem}: {len(word_texts_poem)} words, {len(vocab)} unique → "
              f"llm_scores {tuple(llm_cache[poem].shape)}")

    # --- Test dataset ---
    splits = make_loso_splits(args.heldout_subject)
    test_ds = MEGContinuousTrialDataset(
        splits["test"]["trials"],
        word_filter=splits["test"]["word_filter"],
        meg_base=args.meg_base,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=collate_continuous_trials)
    print(f"Test trials: {len(test_ds)}")

    # --- Alpha sweep accumulator ---
    # Accumulate weighted sums; divide by total n_valid at the end
    acc = {a: {"R@1": 0, "R@5": 0, "MRR": 0.0, "word_acc": 0, "n_valid": 0, "bleu1_sum": 0.0}
           for a in ALPHA_GRID}
    diag_list = []
    n_trials = 0

    print(f"Running fusion over {len(test_ds)} trials ...")
    for batch in test_loader:
        z_word, valid_mask, word_texts, poem = run_encoder_on_trial(
            batch, encoder, word_head, pooling_module, pooling_mode, device
        )

        vocab = vocab_cache[poem]
        llm_scores = llm_cache[poem]         # (N_poem, |V|)

        meg_scores = meg_scores_to_type_level(
            z_word.to(device), bank_vectors, bank_word_types, vocab
        )                                    # (N_poem, |V|)

        valid_list = valid_mask.tolist()
        diag_list.append(scale_diagnostics(meg_scores, llm_scores))

        for alpha in ALPHA_GRID:
            fused = fuse_scores(meg_scores, llm_scores, alpha, args.normalization)
            m = eval_metrics(fused, vocab, word_texts, valid_list)
            for key in ("R@1", "R@5", "MRR", "word_acc"):
                acc[alpha][key] += m[key]
            acc[alpha]["n_valid"] += m["n_valid"]
            acc[alpha]["bleu1_sum"] += m["bleu1"]

        n_trials += 1
        if n_trials % 5 == 0:
            print(f"  {n_trials}/{len(test_ds)} trials done ...")

    # --- Aggregate metrics ---
    results = {}
    for alpha in ALPHA_GRID:
        n = acc[alpha]["n_valid"]
        results[alpha] = {
            k: (acc[alpha][k] / n if n > 0 else 0.0)
            for k in ("R@1", "R@5", "MRR", "word_acc")
        }
        results[alpha]["n_valid"] = n
        results[alpha]["bleu1"] = acc[alpha]["bleu1_sum"] / n_trials if n_trials > 0 else 0.0

    avg_diag = {k: sum(d[k] for d in diag_list) / len(diag_list) for k in diag_list[0]}

    # --- Print summary ---
    print(f"\n=== Fusion — {args.heldout_subject}  llm={args.llm_name}  norm={args.normalization} ===")
    print(f"{'alpha':>6}  {'R@1':>7}  {'R@5':>7}  {'MRR':>7}  {'word_acc':>9}  {'BLEU-1':>8}")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r = results[alpha]
        print(f"{alpha:6.2f}  {r['R@1']*100:6.2f}%  {r['R@5']*100:6.2f}%  "
              f"{r['MRR']*100:6.2f}%  {r['word_acc']*100:8.2f}%  {r['bleu1']*100:7.2f}%")
    print(f"\nScale: LLM/MEG std ratio = {avg_diag['mean_scale_ratio']:.1f}x "
          f"(median {avg_diag['median_scale_ratio']:.1f}x)  "
          f"[use --normalization row_zscore if LLM dominates at small alpha]")

    # --- Save ---
    os.makedirs(args.out_dir, exist_ok=True)
    llm_tag = args.llm_name.replace("/", "_")
    out_path = os.path.join(
        args.out_dir,
        f"{args.heldout_subject}_fusion_{llm_tag}_{args.normalization}.json",
    )
    output = {
        "heldout_subject": args.heldout_subject,
        "llm_name":        args.llm_name,
        "normalization":   args.normalization,
        "n_trials":        n_trials,
        "scale_diagnostics": avg_diag,
        "results": {str(a): results[a] for a in ALPHA_GRID},
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved → {out_path}")

    if args.plot:
        for metric in ("R@1", "bleu1"):
            plot_path = out_path.replace(".json", f"_{metric}.png")
            plot_alpha_sweep(out_path, metric=metric, out_path=plot_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Teacher-forced LLM+MEG fusion for contrastive_multimodal Stage 1."
    )
    p.add_argument("--heldout_subject", type=str, default="sub-01",
                   help="LOSO heldout subject (determines which checkpoint and test split to use).")
    p.add_argument("--stage1_checkpoint_path", type=str, default=None,
                   help="Path to stage1_best_*.pt. Defaults to "
                        "checkpoints/joint_annealed_exact/stage1_best_{subject}_joint_annealed_exact.pt")
    p.add_argument("--teacher_cache_path", type=str, default="teacher_cache.pt",
                   help="Path to teacher_cache.pt (built by teacher_cache.py).")
    p.add_argument("--llm_name", type=str, default="gpt2",
                   help="HuggingFace model name for teacher-forced LLM scoring. "
                        "Default: gpt2. Switch to e.g. gpt2-medium or "
                        "HuggingFaceTB/SmolLM2-360M without code changes.")
    p.add_argument("--normalization", type=str, default="logsoftmax",
                   choices=["logsoftmax", "row_zscore"],
                   help="Per-row normalization before mixing MEG and LLM scores. "
                        "row_zscore recommended when scale_ratio >> 10x.")
    p.add_argument("--meg_base", type=str, default=None,
                   help="Path to icaed_Sai preprocessed MEG directory. "
                        "Defaults to new_dataset.MEG_BASE.")
    p.add_argument("--out_dir", type=str, default="fusion_results",
                   help="Directory for output JSON files.")
    p.add_argument("--plot", action="store_true",
                   help="Save R@1 vs alpha plot alongside the JSON output.")
    args = p.parse_args()

    if args.stage1_checkpoint_path is None:
        args.stage1_checkpoint_path = (
            f"checkpoints/joint_annealed_exact/"
            f"stage1_best_{args.heldout_subject}_joint_annealed_exact.pt"
        )

    main(args)
