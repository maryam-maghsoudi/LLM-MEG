"""
analyze_stage1.py
=================
Analyze Stage 1 (retrieval) eval results in terms of word accuracy and BLEU-1,
with real-MEG vs shuffle-MEG vs zero-MEG comparison.

Data sources
------------
  Real MEG  : eval_results/{tag}/{subj}_trace.csv  — full per-word predictions
              (poem, session, word_pos, true_word, top1_pred, topk_preds,
               correct_top1, correct_topk). Both word accuracy AND BLEU-1 are
               computable from these.
  Controls  : slurm_logs/eval_{tag}/*_{subj}_*.out — the shuffle-MEG and
              zero-MEG runs only PRINTED top-1/top-5 retrieval accuracy; no
              per-word predictions were saved, so control BLEU-1 is NOT
              available and control "word accuracy" == top-1 accuracy.

Metrics
-------
  word_acc (top-1) : fraction of word positions whose top-1 retrieved type
                     equals the true word type (== mean of correct_top1).
  top5             : mean of correct_topk.
  BLEU-1           : corpus-level clipped unigram precision, aggregated over
                     each subject's 20 trials (poem x session). Reference and
                     hypothesis have equal length per trial, so the brevity
                     penalty is 1 and BLEU-1 reduces to clipped unigram
                     precision. (Real MEG only.)

Usage
-----
    python analyze_stage1.py                       # tag=joint_annealed_exact
    python analyze_stage1.py --tag joint_annealed_exact --plot
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]


# ---------------------------------------------------------------------------
#  Real-MEG metrics from trace CSVs
# ---------------------------------------------------------------------------

def _bleu1_from_trial(hyp, ref):
    """Clipped unigram counts for one trial. Returns (clipped_matches, n_hyp)."""
    ref_counts = Counter(ref)
    hyp_counts = Counter(hyp)
    clipped = sum(min(c, ref_counts[w]) for w, c in hyp_counts.items())
    return clipped, len(hyp)


def analyze_trace(csv_path):
    """Per-subject real-MEG metrics from one trace CSV."""
    rows = list(csv.DictReader(open(csv_path)))
    n = len(rows)
    n_top1 = sum(r["correct_top1"] == "True" for r in rows)
    n_top5 = sum(r["correct_topk"] == "True" for r in rows)

    # Group by trial (poem, session) to build ordered hyp/ref sequences for BLEU.
    trials = {}
    for r in rows:
        key = (r["poem"], r["session"])
        trials.setdefault(key, []).append((int(r["word_pos"]), r["top1_pred"], r["true_word"]))

    total_clip = total_hyp = 0
    for key, items in trials.items():
        items.sort(key=lambda x: x[0])          # order by word position
        hyp = [it[1] for it in items]
        ref = [it[2] for it in items]
        c, h = _bleu1_from_trial(hyp, ref)
        total_clip += c
        total_hyp  += h

    return {
        "word_acc": n_top1 / n,
        "top5":     n_top5 / n,
        "bleu1":    total_clip / total_hyp if total_hyp else 0.0,
        "n_words":  n,
        "n_trials": len(trials),
    }


# ---------------------------------------------------------------------------
#  Control metrics parsed from slurm logs
# ---------------------------------------------------------------------------

_RE_REAL_T1   = re.compile(r"^top1:\s*([\d.]+)%")
_RE_REAL_T5   = re.compile(r"^top5:\s*([\d.]+)%")
_RE_SHUF_T1   = re.compile(r"null top1:\s*mean=([\d.]+)%\s*real=([\d.]+)%\s*fraction of null >= real=([\d.]+)")
_RE_SHUF_T5   = re.compile(r"null top5:\s*mean=([\d.]+)%\s*real=([\d.]+)%\s*fraction of null >= real=([\d.]+)")
_RE_ZERO      = re.compile(r"top1=([\d.]+)%\s*top5=([\d.]+)%\s*\(real MEG")


def parse_control_log(log_path):
    """Extract real/shuffle/zero top-1/top-5 (%) from one eval .out log."""
    text = open(log_path).read()
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if (m := _RE_REAL_T1.match(line)):  out["real_top1"] = float(m.group(1))
        elif (m := _RE_REAL_T5.match(line)): out["real_top5"] = float(m.group(1))
        elif (m := _RE_SHUF_T1.search(line)):
            out["shuf_top1"], out["shuf_p1"] = float(m.group(1)), float(m.group(3))
        elif (m := _RE_SHUF_T5.search(line)):
            out["shuf_top5"], out["shuf_p5"] = float(m.group(1)), float(m.group(3))
        elif (m := _RE_ZERO.search(line)):
            out["zero_top1"], out["zero_top5"] = float(m.group(1)), float(m.group(2))
    return out


def find_control(tag, subj):
    logs = glob.glob(str(HERE / "slurm_logs" / f"eval_{tag}" / f"*_{subj}_*.out"))
    best = {}
    for lp in sorted(logs):                     # later files override earlier
        parsed = parse_control_log(lp)
        if "shuf_top1" in parsed or "zero_top1" in parsed:
            best = parsed
    return best


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(args):
    trace_dir = HERE / "eval_results" / args.tag
    per_subject = []

    for subj in SUBJECTS:
        csv_path = trace_dir / f"{subj}_trace.csv"
        if not csv_path.exists():
            print(f"  [skip] {subj}: no trace CSV")
            continue
        real = analyze_trace(str(csv_path))
        ctrl = find_control(args.tag, subj)
        per_subject.append({"subject": subj, **real, **ctrl})

    if not per_subject:
        print("No subjects found.")
        return

    def mean(k):
        vals = [r[k] for r in per_subject if k in r and r[k] is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    # ---- table ----
    print(f"\n=== Stage 1 retrieval analysis  (tag={args.tag}, {len(per_subject)} subjects) ===\n")
    print("Real MEG (from trace CSVs — word_acc=top-1, plus BLEU-1):")
    print(f"  {'subject':9s} {'word_acc%':>9s} {'top5%':>7s} {'BLEU-1%':>8s}")
    for r in per_subject:
        print(f"  {r['subject']:9s} {r['word_acc']*100:9.2f} {r['top5']*100:7.2f} {r['bleu1']*100:8.2f}")
    print(f"  {'MEAN':9s} {mean('word_acc')*100:9.2f} {mean('top5')*100:7.2f} {mean('bleu1')*100:8.2f}")

    print("\nControl comparison (top-1 = word accuracy; BLEU-1 not saved for controls):")
    print(f"  {'subject':9s} {'real_t1%':>8s} {'shuf_t1%':>8s} {'zero_t1%':>8s}   "
          f"{'real_t5%':>8s} {'shuf_t5%':>8s} {'zero_t5%':>8s}  {'p(shuf>=real)':>13s}")
    for r in per_subject:
        print(f"  {r['subject']:9s} "
              f"{r.get('real_top1',float('nan')):8.2f} {r.get('shuf_top1',float('nan')):8.2f} {r.get('zero_top1',float('nan')):8.2f}   "
              f"{r.get('real_top5',float('nan')):8.2f} {r.get('shuf_top5',float('nan')):8.2f} {r.get('zero_top5',float('nan')):8.2f}  "
              f"{r.get('shuf_p1',float('nan')):13.3f}")
    print(f"  {'MEAN':9s} "
          f"{mean('real_top1'):8.2f} {mean('shuf_top1'):8.2f} {mean('zero_top1'):8.2f}   "
          f"{mean('real_top5'):8.2f} {mean('shuf_top5'):8.2f} {mean('zero_top5'):8.2f}")

    # cross-check: trace-derived word_acc should match log real_top1
    disc = [(r["subject"], r["word_acc"]*100, r.get("real_top1"))
            for r in per_subject
            if r.get("real_top1") is not None and abs(r["word_acc"]*100 - r["real_top1"]) > 0.1]
    if disc:
        print("\n  NOTE: trace word_acc vs log real_top1 mismatch (>0.1pp):", disc)

    # ---- save summary json ----
    out_dir = HERE / "analysis"
    out_dir.mkdir(exist_ok=True)
    summary = {
        "tag": args.tag,
        "n_subjects": len(per_subject),
        "means": {k: mean(k) for k in
                  ["word_acc", "top5", "bleu1", "real_top1", "shuf_top1", "zero_top1",
                   "real_top5", "shuf_top5", "zero_top5"]},
        "per_subject": per_subject,
        "note_bleu1_controls": "BLEU-1 unavailable for shuffle/zero — per-word predictions were not saved by the control runs.",
    }
    with open(out_dir / f"stage1_analysis_{args.tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {out_dir / f'stage1_analysis_{args.tag}.json'}")

    if args.plot:
        make_plot(per_subject, mean, args.tag, out_dir)


def make_plot(per_subject, mean, tag, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    subs = [r["subject"] for r in per_subject]
    x = np.arange(len(subs))
    w = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # Left: word accuracy (top-1) real vs shuffle vs zero
    ax = axes[0]
    ax.bar(x - w, [r.get("real_top1", np.nan) for r in per_subject], w, label="real MEG", color="steelblue")
    ax.bar(x,     [r.get("shuf_top1", np.nan) for r in per_subject], w, label="shuffle MEG", color="darkorange")
    ax.bar(x + w, [r.get("zero_top1", np.nan) for r in per_subject], w, label="zero MEG", color="grey")
    ax.axhline(0.85, color="red", ls=":", lw=1, label="chance (0.85%)")
    ax.set_xticks(x); ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Word accuracy = top-1 (%)")
    ax.set_title(f"Word accuracy: real vs shuffle vs zero MEG\n"
                 f"mean real={mean('real_top1'):.2f}%  shuffle={mean('shuf_top1'):.2f}%  zero={mean('zero_top1'):.2f}%")
    ax.legend(fontsize=8)

    # Right: real-MEG word_acc vs BLEU-1 (controls have no BLEU-1)
    ax = axes[1]
    ax.bar(x - w/2, [r["word_acc"]*100 for r in per_subject], w, label="word acc (top-1)", color="steelblue")
    ax.bar(x + w/2, [r["bleu1"]*100 for r in per_subject], w, label="BLEU-1", color="seagreen")
    ax.set_xticks(x); ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("%")
    ax.set_title(f"Real MEG: word accuracy vs BLEU-1\n"
                 f"mean word_acc={mean('word_acc')*100:.2f}%  BLEU-1={mean('bleu1')*100:.2f}%")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = out_dir / f"stage1_analysis_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Analyze Stage 1 retrieval: word acc + BLEU-1, real vs controls.")
    p.add_argument("--tag", type=str, default="joint_annealed_exact")
    p.add_argument("--plot", action="store_true")
    return p


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
