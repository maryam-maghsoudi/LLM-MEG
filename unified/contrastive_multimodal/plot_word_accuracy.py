"""
plot_word_accuracy.py
=====================
Slopegraphs of Stage 1 word accuracy across the three MEG conditions —
real, shuffle, zero — with one connected line per held-out subject.
Two figures are produced: top-1 and top-5.

Data (reuses the tested parsing in analyze_stage1.py):
  real   : mean(correct_top1) / mean(correct_topk) from eval_results/{tag}/{subj}_trace.csv
  shuffle: null top-1 / top-5 mean over seeds, from slurm_logs/eval_{tag}/*.out
  zero   : zero-MEG top-1 / top-5, from the same logs

Usage
-----
    python plot_word_accuracy.py
    python plot_word_accuracy.py --tag joint_annealed_exact --out analysis/word_accuracy_slopegraph.png
"""

import argparse
from pathlib import Path

from analyze_stage1 import analyze_trace, find_control, SUBJECTS, HERE

CONDITIONS = ["Real MEG", "Shuffle MEG", "Zero MEG"]
N_CANDIDATES = 117


def chance_pct(k):
    """Top-k chance level (%) for a bank of N_CANDIDATES occurrences."""
    return 100.0 * k / N_CANDIDATES


def collect(tag):
    """Returns list of dicts with top-1 AND top-5 word accuracy (%) per subject."""
    trace_dir = HERE / "eval_results" / tag
    rows = []
    for subj in SUBJECTS:
        csv_path = trace_dir / f"{subj}_trace.csv"
        if not csv_path.exists():
            print(f"  [skip] {subj}: no trace CSV")
            continue
        real = analyze_trace(str(csv_path))
        ctrl = find_control(tag, subj)
        needed = ("shuf_top1", "zero_top1", "shuf_top5", "zero_top5")
        if any(k not in ctrl for k in needed):
            print(f"  [skip] {subj}: missing shuffle/zero control in logs")
            continue
        rows.append({
            "subject":      subj,
            "real_top1":    real["word_acc"] * 100.0,
            "shuffle_top1": ctrl["shuf_top1"],
            "zero_top1":    ctrl["zero_top1"],
            "real_top5":    real["top5"] * 100.0,
            "shuffle_top5": ctrl["shuf_top5"],
            "zero_top5":    ctrl["zero_top5"],
        })
    return rows


def make_plot(rows, tag, out_path, k):
    """Slopegraph for top-`k` accuracy (k=1 or 5)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cond_keys = [f"real_top{k}", f"shuffle_top{k}", f"zero_top{k}"]
    x = np.arange(len(CONDITIONS))
    cmap = plt.cm.get_cmap("tab20", len(SUBJECTS))

    fig, ax = plt.subplots(figsize=(9, 7))

    # One connected line per subject. Color is keyed to the subject's fixed
    # position in SUBJECTS (not enumerate order) so it stays identical across
    # figures / scripts even if some subjects are missing in one of them.
    for r in rows:
        color = cmap(SUBJECTS.index(r["subject"]))
        y = [r[c] for c in cond_keys]
        ax.plot(x, y, marker="o", color=color, lw=1.5, alpha=0.85, label=r["subject"])
        ax.annotate(r["subject"], (x[0], y[0]), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=7, color=color)

    # Mean line (bold black)
    mean_y = [np.mean([r[c] for r in rows]) for c in cond_keys]
    ax.plot(x, mean_y, marker="D", color="black", lw=3, zorder=10, label="MEAN")
    for xi, yi in zip(x, mean_y):
        ax.annotate(f"{yi:.2f}%", (xi, yi), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    # Chance reference
    ch = chance_pct(k)
    ax.axhline(ch, color="red", ls=":", lw=1.2, label=f"chance ({ch:.2f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, fontsize=11)
    ax.set_xlim(-0.5, 2.35)
    ax.set_ylabel(f"Word retrieval top-{k} (%)", fontsize=11)
    ax.set_title(f"Stage 1 top-{k} accuracy per subject: real vs shuffle vs zero MEG\n(tag={tag})",
                 fontsize=12)
    ax.legend(fontsize=7, ncol=2, loc="upper right", title="subject")
    ax.grid(axis="y", ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def _out_path_for_k(args, k):
    """Default: analysis/word_accuracy_slopegraph_{tag}_top{k}.png.
    If --out given, insert _top{k} before its suffix so both figures are distinct."""
    if args.out:
        base = Path(args.out)
        return base.with_name(f"{base.stem}_top{k}{base.suffix or '.png'}")
    return HERE / "analysis" / f"word_accuracy_slopegraph_{args.tag}_top{k}.png"


def main(args):
    rows = collect(args.tag)
    if not rows:
        print("No data collected.")
        return
    print(f"\n{'subject':9s} {'real1%':>7s} {'shuf1%':>7s} {'zero1%':>7s}   "
          f"{'real5%':>7s} {'shuf5%':>7s} {'zero5%':>7s}")
    for r in rows:
        print(f"{r['subject']:9s} {r['real_top1']:7.2f} {r['shuffle_top1']:7.2f} {r['zero_top1']:7.2f}   "
              f"{r['real_top5']:7.2f} {r['shuffle_top5']:7.2f} {r['zero_top5']:7.2f}")

    for k in (1, 5):
        out_path = _out_path_for_k(args, k)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        make_plot(rows, args.tag, out_path, k)


def build_arg_parser():
    p = argparse.ArgumentParser(description="Slopegraph of word accuracy: real vs shuffle vs zero MEG.")
    p.add_argument("--tag", type=str, default="joint_annealed_exact")
    p.add_argument("--out", type=str, default=None)
    return p


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
