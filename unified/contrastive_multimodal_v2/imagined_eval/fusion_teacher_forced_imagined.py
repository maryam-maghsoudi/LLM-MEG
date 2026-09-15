"""
imagined_eval/fusion_teacher_forced_imagined.py
===============================================
Teacher-forced LLM+MEG log-linear fusion on PREDICTED-LISTENED (from imagined)
MEG, for v2.

This is the teacher-forced analogue of fusion_beamsearch_imagined.py: it reuses
fusion_teacher_forced.py's logic WHOLESALE (load_stage1_checkpoint,
build_candidate_bank, load_fusion_llm, compute_llm_scores,
meg_scores_to_type_level, fuse_scores, eval_metrics, scale_diagnostics,
run_encoder_on_trial, plot_alpha_sweep, ALPHA_GRID) and only changes where the
MEG comes from — pointing meg_base at predicted_npy/{mapping_key} instead of the
real listened icaed_Sai.

Pipeline: predicted-listened MEG (from the img->lis mapping model) -> the SAME
heldout subject's frozen Stage 1 decoder -> teacher-forced fusion (LLM scores
the ground-truth prefix). The LLM side depends only on the poem text, so it is
identical to the real-listened run; only the MEG (decoder input) differs.

The predicted-listened trials (written by predict_and_save.py) live under
    predicted_npy/{mapping_key}/{subj}/ses-{s}/meg/{subj}_sess-{s}_task-{poem}lis.npy
which is exactly the "...lis.npy" layout MEGContinuousTrialDataset's fast path
reads, so the fusion code needs no changes — only meg_base.

Output
------
Results go wherever --out_dir points. Each file is named to encode both the
mapping_key and the LLM/norm so imagined results never overwrite real-listened:
    {out_dir}/{subj}_fusion_{llm_tag}_{norm}_{mapping_key}.json

Usage (run from inside contrastive_multimodal_v2/imagined_eval/)
---------------------------------------------------------------
    python fusion_teacher_forced_imagined.py \
        --mapping_key RNN_full --subjects sub-01 \
        --out_dir fusion_tf_imagined --normalization row_zscore --plot

    # all 13 subjects
    python fusion_teacher_forced_imagined.py \
        --mapping_key RNN_full --out_dir fusion_tf_imagined
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

# fusion_teacher_forced.py (and new_dataset/splits) live one level up.
PARENT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PARENT)

import fusion_teacher_forced as ftf                                          # noqa: E402
from new_dataset import MEGContinuousTrialDataset, collate_continuous_trials  # noqa: E402
from splits import make_loso_splits                                          # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent

SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]


def run_one_subject(subj, predicted_root, stage1_ckpt, bank, llm_cache,
                    vocab_cache, device, args):
    """Teacher-forced fusion sweep for one subject on predicted-listened MEG.

    Returns the output dict (also written to JSON), or None if inputs missing.
    """
    bank_vectors, bank_word_types = bank

    if not os.path.exists(stage1_ckpt):
        print(f"  [skip] {subj}: stage1 checkpoint missing ({stage1_ckpt})")
        return None

    encoder, word_head, pooling_module, pooling_mode, _ = ftf.load_stage1_checkpoint(
        stage1_ckpt, device
    )

    # Test = heldout subject, both poems, all sessions — MEG comes from the
    # predicted-listened .npy under predicted_root (condition stays "lis").
    splits = make_loso_splits(subj)
    test_ds = MEGContinuousTrialDataset(
        splits["test"]["trials"], word_filter=splits["test"]["word_filter"],
        meg_base=predicted_root,
    )
    if len(test_ds) == 0:
        print(f"  [skip] {subj}: no predicted trials under {predicted_root}")
        return None
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=collate_continuous_trials)

    acc = {a: {"R@1": 0, "R@5": 0, "MRR": 0.0, "word_acc": 0, "n_valid": 0, "bleu1_sum": 0.0}
           for a in ftf.ALPHA_GRID}
    diag_list = []
    n_trials = 0

    print(f"  {subj}: {len(test_ds)} trials × {len(ftf.ALPHA_GRID)} alphas ...")
    for batch in test_loader:
        z_word, valid_mask, word_texts, poem = ftf.run_encoder_on_trial(
            batch, encoder, word_head, pooling_module, pooling_mode, device
        )
        vocab = vocab_cache[poem]
        llm_scores = llm_cache[poem]
        meg_scores = ftf.meg_scores_to_type_level(
            z_word.to(device), bank_vectors, bank_word_types, vocab
        )
        valid_list = valid_mask.tolist()
        diag_list.append(ftf.scale_diagnostics(meg_scores, llm_scores))

        for alpha in ftf.ALPHA_GRID:
            fused = ftf.fuse_scores(meg_scores, llm_scores, alpha, args.normalization)
            m = ftf.eval_metrics(fused, vocab, word_texts, valid_list)
            for key in ("R@1", "R@5", "MRR", "word_acc"):
                acc[alpha][key] += m[key]
            acc[alpha]["n_valid"] += m["n_valid"]
            acc[alpha]["bleu1_sum"] += m["bleu1"]

        n_trials += 1
        if n_trials % 5 == 0 or n_trials == len(test_ds):
            print(f"    {n_trials}/{len(test_ds)} trials done ...")

    results = {}
    for alpha in ftf.ALPHA_GRID:
        n = acc[alpha]["n_valid"]
        results[alpha] = {
            k: (acc[alpha][k] / n if n > 0 else 0.0)
            for k in ("R@1", "R@5", "MRR", "word_acc")
        }
        results[alpha]["n_valid"] = n
        results[alpha]["bleu1"] = acc[alpha]["bleu1_sum"] / n_trials if n_trials > 0 else 0.0

    avg_diag = {k: sum(d[k] for d in diag_list) / len(diag_list) for k in diag_list[0]}

    best_alpha = max(results.keys(), key=lambda a: results[a]["bleu1"])
    print(f"    best BLEU-1 α={best_alpha}: {results[best_alpha]['bleu1']*100:.2f}%  "
          f"(MEG-only α=0: {results[0.0]['bleu1']*100:.2f}%, "
          f"LLM-only α=1.0: {results[1.0]['bleu1']*100:.2f}%)  "
          f"R@1: α=0 {results[0.0]['R@1']*100:.2f}% → α=1 {results[1.0]['R@1']*100:.2f}%")

    llm_tag = args.llm_name.replace("/", "_")
    out_path = os.path.join(
        args.out_dir,
        f"{subj}_fusion_{llm_tag}_{args.normalization}_{args.mapping_key}.json",
    )
    output = {
        "heldout_subject": subj,
        "mapping_key":     args.mapping_key,      # imagined provenance
        "meg_source":      predicted_root,
        "llm_name":        args.llm_name,
        "normalization":   args.normalization,
        "n_trials":        n_trials,
        "scale_diagnostics": avg_diag,
        "results": {str(a): results[a] for a in ftf.ALPHA_GRID},
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"    saved -> {out_path}")

    if args.plot:
        for metric in ("R@1", "bleu1"):
            ftf.plot_alpha_sweep(out_path, metric=metric,
                                 out_path=out_path.replace(".json", f"_{metric}.png"))
    return output


def main(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    predicted_root = args.predicted_root or str(
        _THIS_DIR / "predicted_npy" / args.mapping_key
    )
    print(f"=== teacher-forced fusion (imagined)  mapping_key={args.mapping_key}  "
          f"norm={args.normalization}  device={device} ===")
    print(f"    predicted_root = {predicted_root}")
    print(f"    out_dir        = {args.out_dir}")

    # Candidate bank — built once, shared across subjects.
    teacher_cache = torch.load(args.teacher_cache_path, weights_only=False)
    bank_vectors, _, bank_word_types, type_to_id, _, _ = ftf.build_candidate_bank(
        teacher_cache, args.gpt2_mid_layer
    )
    bank = (bank_vectors.to(device), bank_word_types)
    print(f"Bank: {bank_vectors.shape[0]} occurrences, {len(type_to_id)} unique types")

    # LLM + teacher-forced scores per poem — identical to the real-listened run
    # (LLM conditions on ground-truth poem text, not on MEG), so computed once.
    tokenizer, llm_model = ftf.load_fusion_llm(args.llm_name, device)
    print("Pre-computing teacher-forced LLM scores (once per poem) ...")
    llm_cache: Dict[str, torch.Tensor] = {}
    vocab_cache: Dict[str, List[str]] = {}
    for poem in ("poem1", "poem2"):
        onsets = ftf._load_onsets(poem)
        word_texts_poem = [e["word"].strip().lower() for e in onsets]
        vocab = sorted(set(word_texts_poem))
        llm_cache[poem] = ftf.compute_llm_scores(word_texts_poem, vocab, tokenizer, llm_model, device)
        vocab_cache[poem] = vocab
        print(f"  {poem}: {len(word_texts_poem)} words, {len(vocab)} unique → "
              f"llm_scores {tuple(llm_cache[poem].shape)}")

    subjects = args.subjects if args.subjects else SUBJECTS
    done = 0
    for subj in subjects:
        stage1_ckpt = os.path.join(
            args.stage1_ckpt_dir, f"stage1_best_{subj}_{args.stage1_tag}.pt"
        )
        if run_one_subject(subj, predicted_root, stage1_ckpt, bank, llm_cache,
                           vocab_cache, device, args) is not None:
            done += 1
    print(f"\nDone. {done}/{len(subjects)} subjects -> {args.out_dir}")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Teacher-forced LLM+MEG fusion on predicted-listened (from imagined) MEG."
    )
    p.add_argument("--mapping_key", type=str, default="RNN_full",
                   help="Selects predicted_npy/{mapping_key} as the MEG source "
                        "(matches predict_and_save.py). Also tags the output filename.")
    p.add_argument("--predicted_root", type=str, default=None,
                   help="Explicit predicted .npy root (overrides predicted_npy/{mapping_key}).")
    p.add_argument("--subjects", type=str, nargs="*", default=None,
                   help="Subset of subjects (default: all 13).")
    p.add_argument("--stage1_ckpt_dir", type=str,
                   default=str(Path(__file__).resolve().parent.parent
                               / "checkpoints" / "joint_annealed_exact"),
                   help="Dir with stage1_best_{subj}_{tag}.pt checkpoints.")
    p.add_argument("--stage1_tag", type=str, default="joint_annealed_exact",
                   help="Checkpoint tag after the subject (e.g. joint_annealed_exact).")
    p.add_argument("--teacher_cache_path", type=str,
                   default=str(Path(__file__).resolve().parent.parent / "teacher_cache.pt"))
    p.add_argument("--gpt2_mid_layer", type=int, default=None,
                   help="Build the candidate bank from this cached GPT-2 layer "
                        "(hidden_states_word_aligned[L]) instead of the default h_mid. "
                        "Must match train.py --gpt2_mid_layer used for the checkpoint.")
    p.add_argument("--llm_name", type=str, default="gpt2")
    p.add_argument("--normalization", type=str, default="row_zscore",
                   choices=["logsoftmax", "row_zscore"])
    p.add_argument("--out_dir", type=str, required=True,
                   help="Where to write results.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--plot", action="store_true",
                   help="Save R@1 + bleu1 vs alpha plots alongside each JSON.")
    return p


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
