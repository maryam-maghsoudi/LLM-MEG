#!/usr/bin/env python3
"""
plot_loss_curves.py — Training & validation loss curves for the temperature sweep.

Produces one figure per temperature, showing train (solid) and val (dashed)
loss curves for all 13 LOSO heldout subjects on the same axes.

Output: analyze_temp_sweep/figures/loss_curves_T{temp}.png

Run from llm_decoder/:
    python -m unified.analyze_temp_sweep.plot_loss_curves
    python -m unified.analyze_temp_sweep.plot_loss_curves --sweep_dir unified/sweep_temp_contrastive
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

_HERE     = Path(__file__).resolve().parent           # analyze_temp_sweep/
_UNIFIED  = _HERE.parent                              # unified/
_SWEEP    = _UNIFIED / "sweep_temp_contrastive"
_FIG_DIR  = _HERE / "figures"

SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09",
    "sub-10", "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot loss curves across temperature sweep")
    p.add_argument("--sweep_dir", default=str(_SWEEP),
                   help="Root of the temperature sweep (default: unified/sweep_temp_contrastive)")
    p.add_argument("--out_dir", default=str(_FIG_DIR),
                   help="Directory for output PNGs (default: analyze_temp_sweep/figures/)")
    p.add_argument("--model_tag", default="bert_base_uncased",
                   help="Model tag subdirectory (default: bert_base_uncased)")
    return p.parse_args()


def tag_to_label(tag: str) -> str:
    """Convert directory tag like 'temp_0_1' → display label 'T=0.1'."""
    num = tag.replace("temp_", "").replace("_", ".")
    # Handle ambiguous cases: 0.1.0 → 1.0, keep as-is otherwise
    return f"T={num}"


def load_histories(sweep_dir: Path, model_tag: str):
    """
    Discover all temperature directories and load history.json per subject.

    Returns list of dicts:
        { "tag": str, "label": str, "subjects": {sub: {"train": [...], "val": [...]}} }
    """
    temp_dirs = sorted(d for d in sweep_dir.iterdir()
                       if d.is_dir() and d.name.startswith("temp_"))
    if not temp_dirs:
        raise RuntimeError(f"No temp_* directories found in {sweep_dir}")

    results = []
    for tdir in temp_dirs:
        label = tag_to_label(tdir.name)
        subjects = {}
        base = tdir / "inference" / model_tag
        for sub in SUBJECTS:
            hist_path = base / f"loso_{sub}" / "history.json"
            if hist_path.exists():
                d = json.loads(hist_path.read_text())
                subjects[sub] = {
                    "train": d["train_loss"],
                    "val":   d["val_loss"],
                }
            else:
                print(f"  [warn] missing: {hist_path}")
        results.append({"tag": tdir.name, "label": label, "subjects": subjects})
        print(f"  {label}: loaded {len(subjects)}/13 subjects")

    return results


def plot_temperature(entry: dict, out_dir: Path):
    """One figure: train (solid) + val (dashed) for all subjects at one temperature."""
    label    = entry["label"]
    subjects = entry["subjects"]
    tag      = entry["tag"]

    n_subs  = len(subjects)
    colors  = cm.tab20(np.linspace(0, 1, max(n_subs, 1)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    ax_tr, ax_val = axes

    for i, (sub, hist) in enumerate(sorted(subjects.items())):
        c      = colors[i]
        epochs = range(1, len(hist["train"]) + 1)

        ax_tr.plot(epochs, hist["train"], color=c, linewidth=1.5, label=sub)
        ax_val.plot(epochs, hist["val"],  color=c, linewidth=1.5, label=sub,
                    linestyle="--")

    for ax, title in zip(axes, ["Train loss", "Val loss"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("InfoNCE loss")
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.7)
        ax.grid(True, alpha=0.3)
        # Reference line: log(64) = random baseline for bs=64
        ax.axhline(y=4.1589, color="black", linewidth=0.8, linestyle=":",
                   label="random (log 64)")

    fig.suptitle(f"InfoNCE loss curves — {label}  (all 13 LOSO subjects)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = out_dir / f"loss_curves_{tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def main():
    args    = parse_args()
    sweep   = Path(args.sweep_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sweep dir : {sweep}")
    print(f"Output    : {out_dir}")
    print()

    print("Loading histories ...")
    entries = load_histories(sweep, args.model_tag)

    print(f"\nPlotting {len(entries)} temperature figures ...")
    for entry in entries:
        plot_temperature(entry, out_dir)

    print(f"\nDone. {len(entries)} figures → {out_dir}/")


if __name__ == "__main__":
    main()
