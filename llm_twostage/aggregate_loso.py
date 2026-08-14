"""
aggregate_loso.py
=================
Collect per-subject Stage 2 eval results and run paired Wilcoxon signed-rank
tests comparing the real model vs. each control condition.

Controls compared:
  shuffle_time : time positions within each trial permuted
  zero         : z_t replaced with zeros throughout training

Usage
-----
python aggregate_loso.py
python aggregate_loso.py --metric next_word_agreement   # default
python aggregate_loso.py --metric mean_kl
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]

CONTROLS = {
    "shuffle_time": "_ctrl_shuffle_time",
    "zero":         "_ctrl_zero",
}

_HERE    = Path(__file__).parent
OUT_ROOT = _HERE / "out" / "HuggingFaceTB_SmolLM2-360M"


def load_metric(subj: str, suffix: str, metric: str) -> float | None:
    path = OUT_ROOT / f"{subj}{suffix}" / "eval_stage2.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get(metric)


def run_comparison(ctrl_name: str, ctrl_suffix: str, metric: str,
                   higher_is_better: bool) -> dict:
    print(f"\n{'='*60}")
    print(f"  Control: {ctrl_name}   metric: {metric}")
    print(f"{'='*60}")
    print(f"  {'subject':12s}  {'real':>8}  {ctrl_name:>12}  {'diff':>10}")
    print(f"  {'-'*48}")

    real_vals, ctrl_vals, diff_vals, missing = [], [], [], []

    for subj in SUBJECTS:
        real = load_metric(subj, "",           metric)
        ctrl = load_metric(subj, ctrl_suffix,  metric)
        if real is None or ctrl is None:
            missing.append(subj)
            print(f"  {subj:12s}  {'✗' if real is None else f'{real:.4f}':>8}  "
                  f"{'✗' if ctrl is None else f'{ctrl:.4f}':>12}  {'MISSING':>10}")
            continue
        diff = real - ctrl
        real_vals.append(real); ctrl_vals.append(ctrl); diff_vals.append(diff)
        print(f"  {subj:12s}  {real:8.4f}  {ctrl:12.4f}  {diff:+10.4f}")

    n = len(diff_vals)
    if n < 3:
        print(f"\n  Only {n} complete — skipping significance test.")
        if missing:
            print(f"  Missing: {missing}")
        return {}

    print(f"\n  {'MEAN':12s}  {np.mean(real_vals):8.4f}  {np.mean(ctrl_vals):12.4f}  {np.mean(diff_vals):+10.4f}")
    print(f"  {'STD':12s}  {np.std(real_vals):8.4f}  {np.std(ctrl_vals):12.4f}  {np.std(diff_vals):10.4f}")

    stat, p_val = wilcoxon(diff_vals, alternative="greater" if higher_is_better else "less")
    print(f"\n  Wilcoxon signed-rank (one-sided, n={n})")
    print(f"    statistic={stat:.1f}  p={p_val:.4f}", end="  ")
    if p_val < 0.001:
        print("→ p<0.001 ✓")
    elif p_val < 0.05:
        print("→ SIGNIFICANT at α=0.05 ✓")
    elif p_val < 0.10:
        print("→ marginal trend (p<0.10)")
    else:
        print("→ not significant")

    if missing:
        print(f"  (skipped {len(missing)} missing: {missing})")

    return {
        "n_subjects":    n,
        "real_mean":     float(np.mean(real_vals)),
        "ctrl_mean":     float(np.mean(ctrl_vals)),
        "diff_mean":     float(np.mean(diff_vals)),
        "diff_std":      float(np.std(diff_vals)),
        "wilcoxon_stat": float(stat),
        "wilcoxon_p":    float(p_val),
        "missing":       missing,
        "per_subject":   {
            s: {"real": r, "ctrl": c, "diff": d}
            for s, r, c, d in zip(
                [s for s in SUBJECTS if s not in missing],
                real_vals, ctrl_vals, diff_vals
            )
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metric", default="next_word_agreement",
                   choices=["next_word_agreement", "mean_kl"])
    args   = p.parse_args()
    metric = args.metric
    higher_is_better = (metric == "next_word_agreement")

    all_results = {"metric": metric}
    for ctrl_name, ctrl_suffix in CONTROLS.items():
        all_results[ctrl_name] = run_comparison(
            ctrl_name, ctrl_suffix, metric, higher_is_better
        )

    out_path = _HERE / "out" / "loso_stats.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Full results saved → {out_path}")


if __name__ == "__main__":
    main()
