"""
plot_imagined_accuracy.py
=========================
Slopegraphs of Stage 1 word retrieval accuracy from LISTENED (real MEG) to
IMAGINED (predicted-listened) MEG, one connected line per held-out subject.
Two figures are produced: top-1 (R@1) and top-5 (R@5).

Both conditions score against the SAME 117-occurrence candidate bank
(build_candidate_bank in eval_stage1.py), so their accuracies — and their
chance level — are directly comparable and share one y-axis.

Data
----
  listened : mean(correct_top1) / mean(correct_topk) from
             ../eval_results/{tag}/{subj}_trace.csv   (reuses analyze_stage1.analyze_trace)
  imagined : per-subject top1 / top5 from
             results/{mapping_key}/summary.json        (written by eval_imagined.py)

Colors are keyed to each subject's fixed position in SUBJECTS, matching
../plot_word_accuracy.py, so a given subject is the same color in both the
listened-vs-control and the listened-vs-imagined figures.

Usage
-----
    python plot_imagined_accuracy.py
    python plot_imagined_accuracy.py --mapping_key RNN_full --tag joint_annealed_exact
    python plot_imagined_accuracy.py --out results/RNN_full/wordacc_listened_vs_imagined.png
"""

import argparse
import json
import sys
from pathlib import Path

# analyze_stage1 (analyze_trace, SUBJECTS, HERE) lives one level up in
# contrastive_multimodal/, which uses plain co-located imports.
PARENT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PARENT)
from analyze_stage1 import analyze_trace, SUBJECTS, HERE   # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent


def collect(tag, mapping_key, results_suffix=""):
    """Returns (rows, chance_top1_pct, chance_top5_pct, has_shuffle).

    rows: list of dicts with listened + imagined (+ imagined-shuffle, if present)
    top-1/top-5 (%) per subject, for subjects present in BOTH the listened
    traces and the imagined summary.

    tag selects the listened traces (../eval_results/{tag}/); results_suffix is
    appended to the imagined results folder (results/{mapping_key}{suffix}/) so a
    shuffle-trained decoder (tag=..._shuffled, suffix=_shuffled) reads its own
    listened and imagined results, not the real decoder's.
    """
    trace_dir = HERE / "eval_results" / tag
    summary_path = _THIS_DIR / "results" / f"{mapping_key}{results_suffix}" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No imagined summary at {summary_path} — run eval_imagined.py "
            f"--mapping_key {mapping_key}"
            + (f" --out_suffix {results_suffix}" if results_suffix else "") + " first."
        )
    summary = json.loads(summary_path.read_text())
    imagined = {r["subject"]: r for r in summary["per_subject"]}
    chance1 = summary["chance_top1"] * 100.0
    chance5 = summary["chance_top5"] * 100.0
    # Shuffle column is only drawn if every plotted subject carries it.
    has_shuffle = all("shuffle_top1" in r for r in summary["per_subject"])

    rows = []
    for subj in SUBJECTS:
        csv_path = trace_dir / f"{subj}_trace.csv"
        if not csv_path.exists():
            print(f"  [skip] {subj}: no listened trace CSV")
            continue
        if subj not in imagined:
            print(f"  [skip] {subj}: not in imagined summary ({mapping_key})")
            continue
        real = analyze_trace(str(csv_path))
        img = imagined[subj]
        row = {
            "subject":       subj,
            "listened_top1": real["word_acc"] * 100.0,
            "imagined_top1": img["top1"] * 100.0,
            "listened_top5": real["top5"] * 100.0,
            "imagined_top5": img["top5"] * 100.0,
        }
        if has_shuffle:
            row["shuffle_top1"] = img["shuffle_top1"] * 100.0
            row["shuffle_top5"] = img["shuffle_top5"] * 100.0
        rows.append(row)
    return rows, chance1, chance5, has_shuffle


def make_plot(rows, tag, mapping_key, out_path, k, chance_pct, has_shuffle):
    """Slopegraph for top-`k` accuracy (k=1 or 5): listened -> imagined
    (-> imagined shuffle null, if available)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    conditions = ["Listened\n(real MEG)", f"Imagined\n({mapping_key})"]
    cond_keys = [f"listened_top{k}", f"imagined_top{k}"]
    if has_shuffle:
        conditions.append("Imagined\n(shuffle null)")
        cond_keys.append(f"shuffle_top{k}")
    x = np.arange(len(conditions))
    # Same colormap + fixed-subject indexing as ../plot_word_accuracy.py so
    # each subject keeps its color across the two scripts.
    cmap = plt.cm.get_cmap("tab20", len(SUBJECTS))

    fig, ax = plt.subplots(figsize=(8, 7))

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

    # Chance reference (identical bank for listened + imagined)
    ax.axhline(chance_pct, color="red", ls=":", lw=1.2, label=f"chance ({chance_pct:.2f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_xlim(-0.55, x[-1] + 0.4)
    ax.set_ylabel(f"Word retrieval top-{k} (%)", fontsize=11)
    ax.set_title(f"Stage 1 top-{k} accuracy per subject: listened vs imagined MEG\n"
                 f"(tag={tag}, mapping={mapping_key})", fontsize=12)
    ax.legend(fontsize=7, ncol=2, loc="upper right", title="subject")
    ax.grid(axis="y", ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def _out_path_for_k(args, k):
    """Default: results/{mapping_key}{results_suffix}/wordacc_listened_vs_imagined_{mapping_key}{results_suffix}_top{k}.png.
    If --out given, insert _top{k} before its suffix so both figures are distinct."""
    if args.out:
        base = Path(args.out)
        return base.with_name(f"{base.stem}_top{k}{base.suffix or '.png'}")
    name = f"{args.mapping_key}{args.results_suffix}"
    return _THIS_DIR / "results" / name / f"wordacc_listened_vs_imagined_{name}_top{k}.png"


def main(args):
    rows, chance1, chance5, has_shuffle = collect(args.tag, args.mapping_key, args.results_suffix)
    if not rows:
        print("No data collected.")
        return
    shuf_hdr = f" {'shuf_t1%':>8s} {'shuf_t5%':>8s}" if has_shuffle else ""
    print(f"\n{'subject':9s} {'lis_t1%':>8s} {'img_t1%':>8s}   {'lis_t5%':>8s} {'img_t5%':>8s}{shuf_hdr}")
    for r in rows:
        shuf = f" {r['shuffle_top1']:8.2f} {r['shuffle_top5']:8.2f}" if has_shuffle else ""
        print(f"{r['subject']:9s} {r['listened_top1']:8.2f} {r['imagined_top1']:8.2f}   "
              f"{r['listened_top5']:8.2f} {r['imagined_top5']:8.2f}{shuf}")
    if not has_shuffle:
        print("(no shuffle column — re-run eval_imagined.py with --shuffle_control to add it)")

    chance_by_k = {1: chance1, 5: chance5}
    for k in (1, 5):
        out_path = _out_path_for_k(args, k)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        make_plot(rows, args.tag, args.mapping_key, out_path, k, chance_by_k[k], has_shuffle)


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Slopegraph of word accuracy: listened vs imagined MEG (R@1 + R@5)."
    )
    p.add_argument("--tag", type=str, default="joint_annealed_exact",
                   help="Stage 1 checkpoint tag (selects ../eval_results/{tag}/ listened traces).")
    p.add_argument("--mapping_key", type=str, default="RNN_full",
                   help="Imagined mapping key (selects results/{mapping_key}{results_suffix}/summary.json).")
    p.add_argument("--results_suffix", type=str, default="",
                   help="Appended to the imagined results folder, matching eval_imagined.py --out_suffix "
                        "(e.g. '_shuffled'). Pair with --tag ..._shuffled to plot a shuffle-trained decoder.")
    p.add_argument("--out", type=str, default=None)
    return p


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
