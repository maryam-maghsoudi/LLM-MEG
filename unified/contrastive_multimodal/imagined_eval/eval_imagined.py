"""
imagined_eval/eval_imagined.py
==============================
STAGE B of the imagined -> predicted-listened -> contrastive-decoder pipeline.

Feeds the predicted-listened .npy trials produced by predict_and_save.py into
each held-out subject's FROZEN Stage 1 contrastive decoder and scores word
retrieval, exactly as eval_stage1.py does for REAL listened MEG — the only
difference is where the MEG comes from (predicted, via meg_base) instead of
icaed_Sai.

This is a true zero-shot LOSO chain: for held-out subject S, both the img->lis
mapping model AND the Stage 1 decoder were trained without ever seeing S.

Reuses eval_stage1.py wholesale (load_stage1_checkpoint, build_candidate_bank,
run_stage1_forward, score_queries, trace helpers) — nothing about the scoring
path is re-implemented here.

Usage
-----
    # after: python predict_and_save.py --mapping_key RNN_full
    python eval_imagined.py --mapping_key RNN_full
    python eval_imagined.py --mapping_key CNN1D_full --subjects sub-01 sub-03
    python eval_imagined.py --mapping_key RNN_full --save_traces

Outputs (under ./results/{mapping_key}/)
----------------------------------------
    summary.json                         per-subject + aggregate top1/top5
    trace_{subject}.csv / .xlsx          (only with --save_traces)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# The contrastive_multimodal package uses plain (co-located) imports and is
# meant to run from its own directory. This script lives one level down, so
# put the parent on sys.path before importing any of its modules.
PARENT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PARENT)

from new_dataset import MEGContinuousTrialDataset, collate_continuous_trials   # noqa: E402
from splits import make_loso_splits                                            # noqa: E402
from new_controls import make_control                                          # noqa: E402
from metrics import topk_predictions, chance_level, permutation_percentile    # noqa: E402
from eval_stage1 import (                                                      # noqa: E402
    load_stage1_checkpoint,
    build_candidate_bank,
    run_stage1_forward,
    score_queries,
    build_prediction_trace,
    save_prediction_trace_csv,
    save_prediction_trace_excel,
)

SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]


def eval_one_subject(subj, predicted_root, stage1_ckpt, bank, device, save_dir,
                     n_shuffle_seeds=0):
    """Returns a per-subject metrics dict, or None if inputs are missing.

    If n_shuffle_seeds > 0, also runs a shuffle-time null: the predicted-MEG
    windows are re-paired to the wrong word onsets (make_control's
    "shuffle_time", same control eval_stage1.py uses for listened MEG),
    averaged over that many seeds. Adds shuffle_top1/top5 + permutation
    percentiles (fraction of null >= real) to the returned dict.
    """
    bank_vectors, bank_labels, bank_word_types, type_to_id = bank

    if not os.path.exists(stage1_ckpt):
        print(f"  [skip] {subj}: stage1 checkpoint missing ({stage1_ckpt})")
        return None

    encoder, word_head, pooling_module, pooling_mode, ckpt_subject = \
        load_stage1_checkpoint(stage1_ckpt, device)
    if ckpt_subject is not None and ckpt_subject != subj:
        print(f"  WARNING: checkpoint heldout={ckpt_subject!r} != requested {subj!r}")

    # Test = all sessions of the held-out subject, both poems. The predicted
    # trials live under predicted_root and are named "...lis.npy", so the
    # default condition="lis" picks them up transparently.
    test_trials = make_loso_splits(subj)["test"]["trials"]
    test_ds = MEGContinuousTrialDataset(
        test_trials, word_filter=None, condition="lis", meg_base=predicted_root,
    )
    if len(test_ds) == 0:
        print(f"  [skip] {subj}: no predicted trials found under {predicted_root}")
        return None

    loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                        collate_fn=collate_continuous_trials)

    z_word, word_text, poem_list, word_pos_list, session_list = run_stage1_forward(
        loader, encoder, word_head, pooling_module, pooling_mode, device
    )
    if z_word.shape[0] == 0:
        print(f"  [skip] {subj}: no valid word queries")
        return None

    accs, scores, valid, _ = score_queries(
        z_word.to(device), word_text, bank_vectors, bank_labels, type_to_id, ks=(1, 5)
    )
    n_cand = bank_vectors.shape[0]
    n_q    = int(valid.sum().item())

    print(f"  {subj}: top1={accs['top1']*100:.2f}%  top5={accs['top5']*100:.2f}%  "
          f"(n_queries={n_q}, chance@1={chance_level(n_cand,1)*100:.2f}%)")

    shuffle_metrics = {}
    if n_shuffle_seeds > 0:
        null_top1, null_top5 = [], []
        for seed in range(n_shuffle_seeds):
            shuf_ds = make_control(test_ds, "shuffle_time", seed=seed)
            shuf_loader = DataLoader(shuf_ds, batch_size=4, shuffle=False,
                                     collate_fn=collate_continuous_trials)
            z_s, wt_s, _, _, _ = run_stage1_forward(
                shuf_loader, encoder, word_head, pooling_module, pooling_mode, device
            )
            accs_s, _, _, _ = score_queries(
                z_s.to(device), wt_s, bank_vectors, bank_labels, type_to_id, ks=(1, 5)
            )
            null_top1.append(accs_s["top1"])
            null_top5.append(accs_s["top5"])
        shuffle_metrics = {
            "shuffle_top1": sum(null_top1) / len(null_top1),
            "shuffle_top5": sum(null_top5) / len(null_top5),
            "p_top1":       permutation_percentile(accs["top1"], null_top1),
            "p_top5":       permutation_percentile(accs["top5"], null_top5),
            "n_shuffle_seeds": n_shuffle_seeds,
        }
        print(f"      shuffle null ({n_shuffle_seeds} seeds): "
              f"top1={shuffle_metrics['shuffle_top1']*100:.2f}% (p={shuffle_metrics['p_top1']:.3f})  "
              f"top5={shuffle_metrics['shuffle_top5']*100:.2f}% (p={shuffle_metrics['p_top5']:.3f})")

    if save_dir is not None:
        preds5 = topk_predictions(scores, k=5)
        trace  = build_prediction_trace(
            word_text, poem_list, word_pos_list, session_list, preds5, bank_word_types
        )
        save_prediction_trace_csv(trace, os.path.join(save_dir, f"trace_{subj}.csv"))
        try:
            save_prediction_trace_excel(trace, os.path.join(save_dir, f"trace_{subj}.xlsx"))
        except ImportError:
            print("    (openpyxl not installed — skipped .xlsx trace)")

    return {
        "subject":      subj,
        "top1":         accs["top1"],
        "top5":         accs["top5"],
        "n_queries":    n_q,
        "n_candidates": n_cand,
        "pooling_mode": pooling_mode,
        **shuffle_metrics,
    }


def main(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    predicted_root = args.predicted_root or str(
        Path(__file__).parent / "predicted_npy" / args.mapping_key
    )
    print(f"=== eval_imagined  mapping_key={args.mapping_key}  device={device} ===")
    print(f"    predicted_root = {predicted_root}")

    # Candidate bank (h_mid over both poems) — identical for every subject,
    # built once. score_queries needs it on-device.
    teacher_cache = torch.load(args.teacher_cache_path, weights_only=False)
    bv, bl, bwt, type_to_id, _, _ = build_candidate_bank(teacher_cache)
    bank = (bv.to(device), bl.to(device), bwt, type_to_id)
    print(f"Candidate bank: {bv.shape[0]} occurrences, {len(type_to_id)} unique word types")

    out_dir = Path(args.out_root) / args.mapping_key
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = str(out_dir) if args.save_traces else None

    subjects = args.subjects if args.subjects else SUBJECTS
    per_subject = []
    for subj in subjects:
        stage1_ckpt = os.path.join(
            args.stage1_ckpt_dir, f"stage1_best_{subj}_{args.stage1_tag}.pt"
        )
        res = eval_one_subject(subj, predicted_root, stage1_ckpt, bank, device, trace_dir,
                               n_shuffle_seeds=(args.num_shuffle_seeds if args.shuffle_control else 0))
        if res is not None:
            per_subject.append(res)

    if not per_subject:
        print("\nNo subjects evaluated — nothing to summarize.")
        return

    mean_top1 = sum(r["top1"] for r in per_subject) / len(per_subject)
    mean_top5 = sum(r["top5"] for r in per_subject) / len(per_subject)
    n_cand    = per_subject[0]["n_candidates"]

    summary = {
        "mapping_key":     args.mapping_key,
        "stage1_tag":      args.stage1_tag,
        "predicted_root":  predicted_root,
        "n_subjects":      len(per_subject),
        "mean_top1":       mean_top1,
        "mean_top5":       mean_top5,
        "chance_top1":     chance_level(n_cand, 1),
        "chance_top5":     chance_level(n_cand, 5),
        "per_subject":     per_subject,
    }
    if args.shuffle_control and all("shuffle_top1" in r for r in per_subject):
        summary["mean_shuffle_top1"] = sum(r["shuffle_top1"] for r in per_subject) / len(per_subject)
        summary["mean_shuffle_top5"] = sum(r["shuffle_top5"] for r in per_subject) / len(per_subject)
        summary["num_shuffle_seeds"] = args.num_shuffle_seeds
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== AGGREGATE ({len(per_subject)} subjects, mapping={args.mapping_key}) ===")
    print(f"  mean top1 = {mean_top1*100:.2f}%   (chance {chance_level(n_cand,1)*100:.2f}%)")
    print(f"  mean top5 = {mean_top5*100:.2f}%   (chance {chance_level(n_cand,5)*100:.2f}%)")
    print(f"  saved -> {summary_path}")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Eval predicted-listened (from imagined) through Stage 1 contrastive decoder."
    )
    p.add_argument("--mapping_key", type=str, default="RNN_full",
                   help="Which predicted_npy/{mapping_key} subfolder to read (matches predict_and_save.py).")
    p.add_argument("--predicted_root", type=str, default=None,
                   help="Explicit path to predicted .npy root (overrides predicted_npy/{mapping_key}).")
    p.add_argument("--stage1_ckpt_dir", type=str,
                   default=str(Path(__file__).resolve().parent.parent / "checkpoints" / "joint_annealed_exact"),
                   help="Dir with stage1_best_{subj}_{tag}.pt checkpoints.")
    p.add_argument("--stage1_tag", type=str, default="joint_annealed_exact",
                   help="Suffix in the checkpoint filename after the subject.")
    p.add_argument("--teacher_cache_path", type=str,
                   default=str(Path(__file__).resolve().parent.parent / "teacher_cache.pt"))
    p.add_argument("--subjects", type=str, nargs="*", default=None,
                   help="Subset of subjects (default: all 13).")
    p.add_argument("--out_root", type=str,
                   default=str(Path(__file__).parent / "results"))
    p.add_argument("--save_traces", action="store_true",
                   help="Also write per-subject CSV + color-coded Excel prediction traces.")
    p.add_argument("--shuffle_control", action="store_true",
                   help="Also run a shuffle-time null (re-pair predicted MEG to wrong word onsets) "
                        "and add shuffle_top1/top5 + permutation percentiles to summary.json.")
    p.add_argument("--num_shuffle_seeds", type=int, default=20,
                   help="Seeds averaged for the shuffle null (only used with --shuffle_control).")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                   help="Compute device (default: auto = cuda if available, else cpu).")
    return p


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
