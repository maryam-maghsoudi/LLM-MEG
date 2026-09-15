"""
imagined_eval/fusion_beamsearch_imagined.py
===========================================
MEG-guided beam-search LLM fusion on PREDICTED-LISTENED (from imagined) MEG.

This is the beam-search analogue of eval_imagined.py: it reuses
fusion_beamsearch.py's logic WHOLESALE (load_stage1_checkpoint,
build_candidate_bank, load_fusion_llm, run_encoder_on_trial,
meg_scores_to_type_level, sweep_alphas_beam, ALPHA_GRID, plotting) and only
changes where the MEG comes from — pointing meg_base at
predicted_npy/{mapping_key} instead of the real listened icaed_Sai.

The predicted-listened trials (written by predict_and_save.py) live under
    predicted_npy/{mapping_key}/{subj}/ses-{s}/meg/{subj}_sess-{s}_task-{poem}lis.npy
which is exactly the "...lis.npy" layout MEGContinuousTrialDataset's fast path
reads, so the fusion code needs no changes — only meg_base.

Output
------
Results go wherever --out_dir points (NOT hardcoded to imagined_eval/). Each
file is named to encode both the mapping_key and the beam settings so imagined
results never overwrite the real-listened ones:
    {out_dir}/{subj}_beamfusion_B{B}_top{k}_{llm_tag}_{norm}_{mapping_key}.json

Usage (run from inside contrastive_multimodal/)
-----------------------------------------------
    python imagined_eval/fusion_beamsearch_imagined.py \
        --mapping_key RNN_full --subjects sub-01 \
        --out_dir /some/other/folder --normalization row_zscore --plot

    # all 13 subjects, custom folder
    python imagined_eval/fusion_beamsearch_imagined.py \
        --mapping_key RNN_full --out_dir /some/other/folder
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

# fusion_beamsearch.py (and new_dataset/splits) live one level up.
PARENT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PARENT)

import fusion_beamsearch as fb                                                # noqa: E402
from new_dataset import MEGContinuousTrialDataset, collate_continuous_trials  # noqa: E402
from splits import make_loso_splits                                          # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent

SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]


def run_one_subject(subj, predicted_root, stage1_ckpt, bank, llm, vocab_cache,
                    device, args):
    """Beam-search fusion sweep for one subject on predicted-listened MEG.

    Returns the output dict (also written to JSON), or None if inputs missing.
    """
    bank_vectors, bank_word_types, type_to_id = bank
    tokenizer, llm_model = llm

    if not os.path.exists(stage1_ckpt):
        print(f"  [skip] {subj}: stage1 checkpoint missing ({stage1_ckpt})")
        return None

    encoder, word_head, pooling_module, pooling_mode = fb.load_stage1_checkpoint(
        stage1_ckpt, device
    )

    # Test = heldout subject, both poems, all sessions — but MEG comes from the
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

    acc = {a: {"bleu1": 0.0, "word_acc": 0.0, "n_valid": 0, "n_trials": 0}
           for a in fb.ALPHA_GRID}
    trials_out: Dict[str, List] = {str(a): [] for a in fb.ALPHA_GRID}
    n_trials = 0

    print(f"  {subj}: {len(test_ds)} trials × {len(fb.ALPHA_GRID)} alphas ...")
    for batch in test_loader:
        session = int(batch["session"][0]) if "session" in batch else None
        z_word, valid_mask, word_texts, poem = fb.run_encoder_on_trial(
            batch, encoder, word_head, pooling_module, pooling_mode, device
        )
        vocab = vocab_cache[poem]
        meg_scores = fb.meg_scores_to_type_level(
            z_word.to(device), bank_vectors, bank_word_types, vocab
        )
        valid_list = valid_mask.tolist()

        alpha_results = fb.sweep_alphas_beam(
            meg_scores, vocab, word_texts, valid_list,
            tokenizer, llm_model, device,
            alphas=fb.ALPHA_GRID,
            beam_width=args.beam_width, top_k=args.top_k,
            normalization=args.normalization, no_repeat_ngram=args.no_repeat_ngram,
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
            print(f"    {n_trials}/{len(test_ds)} trials done ...")

    results: Dict[str, Dict] = {}
    for alpha in fb.ALPHA_GRID:
        nt = acc[alpha]["n_trials"]
        results[str(alpha)] = {
            "bleu1":    acc[alpha]["bleu1"]    / nt if nt > 0 else 0.0,
            "word_acc": acc[alpha]["word_acc"] / nt if nt > 0 else 0.0,
            "n_valid":  acc[alpha]["n_valid"],
            "n_trials": nt,
        }

    # Best-BLEU alpha for a quick console line
    best_alpha = max(results.keys(), key=lambda k: results[k]["bleu1"])
    print(f"    best BLEU-1 α={best_alpha}: {results[best_alpha]['bleu1']*100:.2f}%  "
          f"(MEG-only α=0: {results['0.0']['bleu1']*100:.2f}%, "
          f"LLM-only α=1.0: {results['1.0']['bleu1']*100:.2f}%)")

    llm_tag  = args.llm_name.replace("/", "_")
    out_path = os.path.join(
        args.out_dir,
        f"{subj}_beamfusion_B{args.beam_width}_top{args.top_k}_"
        f"{llm_tag}_{args.normalization}_{args.mapping_key}.json",
    )
    output = {
        "heldout_subject": subj,
        "mapping_key":     args.mapping_key,          # imagined provenance
        "meg_source":      predicted_root,
        "llm_name":        args.llm_name,
        "normalization":   args.normalization,
        "beam_width":      args.beam_width,
        "top_k":           args.top_k,
        "no_repeat_ngram": args.no_repeat_ngram,
        "n_trials":        n_trials,
        "results":         results,
        "trials":          trials_out,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"    saved -> {out_path}")

    if args.plot:
        for metric in ("bleu1", "word_acc"):
            fb.plot_alpha_sweep_beam(out_path, metric=metric,
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
    print(f"=== beam-search fusion (imagined)  mapping_key={args.mapping_key}  "
          f"device={device} ===")
    print(f"    predicted_root = {predicted_root}")
    print(f"    out_dir        = {args.out_dir}")

    # Candidate bank + LLM + vocab — built once, shared across subjects.
    teacher_cache = torch.load(args.teacher_cache_path, weights_only=False)
    bank_vectors, bank_word_types, type_to_id = fb.build_candidate_bank(teacher_cache)
    bank = (bank_vectors.to(device), bank_word_types, type_to_id)
    print(f"Bank: {bank_vectors.shape[0]} occurrences, {len(type_to_id)} unique types")

    llm = fb.load_fusion_llm(args.llm_name, device)

    vocab_cache: Dict[str, List[str]] = {}
    for poem in ("poem1", "poem2"):
        onsets = fb._load_onsets(poem)
        vocab_cache[poem] = sorted({e["word"].strip().lower() for e in onsets})

    subjects = args.subjects if args.subjects else SUBJECTS
    done = 0
    for subj in subjects:
        stage1_ckpt = os.path.join(
            args.stage1_ckpt_dir, f"stage1_best_{subj}_{args.stage1_tag}.pt"
        )
        if run_one_subject(subj, predicted_root, stage1_ckpt, bank, llm,
                           vocab_cache, device, args) is not None:
            done += 1
    print(f"\nDone. {done}/{len(subjects)} subjects -> {args.out_dir}")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Beam-search LLM fusion on predicted-listened (from imagined) MEG."
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
                   help="Checkpoint tag after the subject (e.g. joint_annealed_exact, "
                        "or joint_annealed_exact_shuffled for the shuffle-trained decoder).")
    p.add_argument("--teacher_cache_path", type=str,
                   default=str(Path(__file__).resolve().parent.parent / "teacher_cache.pt"))
    p.add_argument("--llm_name", type=str, default="gpt2")
    p.add_argument("--normalization", type=str, default="row_zscore",
                   choices=["logsoftmax", "row_zscore"])
    p.add_argument("--beam_width", type=int, default=5)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--no_repeat_ngram", type=int, default=0)
    p.add_argument("--out_dir", type=str, required=True,
                   help="Where to write results (any folder — NOT forced into imagined_eval/).")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--plot", action="store_true",
                   help="Save bleu1 + word_acc vs alpha plots alongside each JSON.")
    return p


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
