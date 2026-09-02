"""
compare_controls.py — Compare inference Method 1 metrics: real vs. shuffle_time vs. zero.

Loads all 60 eval_results.json from:
    unified/out/inference/bert_base_uncased/{run_dir}/eval/eval_results.json

For each eval scheme (loso, session_cv, stimulus):
  - Collects per-fold/subject scalars for each control condition
  - Runs paired Wilcoxon signed-rank tests (real vs shuffle_time, real vs zero)
  - Prints a summary table and saves results to CSV + figures

Usage (from llm_decoder/ parent):
    python -m unified.method1_analysis.compare_controls
    python -m unified.method1_analysis.compare_controls --no_figs
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------

_HERE     = Path(__file__).parent
EVAL_ROOT = _HERE.parent / "out" / "inference" / "bert_base_uncased"
OUT_DIR   = _HERE / "results"

METRICS = ["R@1", "MRR", "word_accuracy", "BLEU1", "WER"]
METRIC_LABELS = {
    "R@1":           "R@1",
    "MRR":           "MRR",
    "word_accuracy": "Word Acc.",
    "BLEU1":         "BLEU-1",
    "WER":           "WER",
}
CONTROLS   = ["none", "shuffle_time", "zero"]
CTRL_LABEL = {"none": "Real", "shuffle_time": "Shuffle-time", "zero": "Zero MEG"}


# ---------------------------------------------------------------------------
#  Parsing
# ---------------------------------------------------------------------------

def _parse_dir_name(name: str):
    """
    Parse a run directory name into (eval_scheme, fold_id, control).

    Examples
    --------
    loso_sub-01                    → ('loso',       'sub-01',  'none')
    loso_sub-01_ctrl_shuffle_time  → ('loso',       'sub-01',  'shuffle_time')
    session_cv_fold0               → ('session_cv', 'fold0',   'none')
    session_cv_fold0_ctrl_zero     → ('session_cv', 'fold0',   'zero')
    stimulus_lines2                → ('stimulus',   'lines2',  'none')
    stimulus_lines4_ctrl_zero      → ('stimulus',   'lines4',  'zero')
    """
    ctrl = "none"
    m = re.search(r"_ctrl_(shuffle_time|zero)$", name)
    if m:
        ctrl = m.group(1)
        name = name[: m.start()]

    if name.startswith("loso_"):
        return "loso", name[len("loso_"):], ctrl
    if name.startswith("session_cv_"):
        return "session_cv", name[len("session_cv_"):], ctrl
    if name.startswith("stimulus_"):
        return "stimulus", name[len("stimulus_"):], ctrl

    raise ValueError(f"Cannot parse run dir: {name}")


def _load_summary(run_dir: Path) -> dict:
    """Load the scalar summary from eval_results.json."""
    path = run_dir / "eval" / "eval_results.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return {
        "R@1":           d["mean_R@1"],
        "MRR":           d["mean_MRR"],
        "word_accuracy": d["mean_accuracy"],
        "BLEU1":         d["mean_BLEU1"],
        "WER":           d["mean_WER"],
        "n_trials":      d["n_trials"],
    }


# ---------------------------------------------------------------------------
#  Load all results into a DataFrame
# ---------------------------------------------------------------------------

def load_all(eval_root: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(eval_root.iterdir()):
        if not run_dir.is_dir():
            continue
        try:
            scheme, fold_id, ctrl = _parse_dir_name(run_dir.name)
        except ValueError:
            continue
        summary = _load_summary(run_dir)
        if summary is None:
            print(f"  [warn] no eval results: {run_dir.name}")
            continue
        rows.append({
            "eval_scheme": scheme,
            "fold_id":     fold_id,
            "control":     ctrl,
            **summary,
        })
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} runs  ({df['eval_scheme'].value_counts().to_dict()})")
    return df


# ---------------------------------------------------------------------------
#  Statistics
# ---------------------------------------------------------------------------

def wilcoxon_paired(a: np.ndarray, b: np.ndarray):
    """Paired Wilcoxon signed-rank test. Returns (statistic, p_value)."""
    diffs = a - b
    if np.all(diffs == 0):
        return 0.0, 1.0
    stat, p = wilcoxon(diffs)
    return float(stat), float(p)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d = mean(diff) / std(diff)."""
    diffs = a - b
    sd = diffs.std(ddof=1)
    return float(diffs.mean() / sd) if sd > 0 else 0.0


# ---------------------------------------------------------------------------
#  Per-scheme analysis
# ---------------------------------------------------------------------------

def _get_vals(sub: pd.DataFrame, fold_ids: list, metric: str, control: str):
    """Return per-fold values for a metric+control. Returns None if any fold is missing."""
    rows = []
    for f in fold_ids:
        match = sub[(sub["fold_id"] == f) & (sub["control"] == control)]
        if match.empty:
            return None
        rows.append(match[metric].values[0])
    return np.array(rows)


def analyse_scheme(df: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """
    For one eval scheme, build a table with real model stats and, when available,
    paired Wilcoxon tests against shuffle_time and zero controls.
    Control columns are omitted silently when no control runs exist.
    """
    sub = df[df["eval_scheme"] == scheme]
    fold_ids = sorted(sub["fold_id"].unique())
    n = len(fold_ids)

    has_shuffle = "shuffle_time" in sub["control"].values
    has_zero    = "zero"         in sub["control"].values

    records = []
    for metric in METRICS:
        real_vals    = _get_vals(sub, fold_ids, metric, "none")
        shuffle_vals = _get_vals(sub, fold_ids, metric, "shuffle_time") if has_shuffle else None
        zero_vals    = _get_vals(sub, fold_ids, metric, "zero")         if has_zero    else None

        if real_vals is None:
            continue

        rec = {
            "metric":    METRIC_LABELS[metric],
            "real_mean": real_vals.mean(),
            "real_std":  real_vals.std(ddof=1),
            "n":         n,
            "_real":     real_vals,
        }

        if shuffle_vals is not None:
            stat_sh, p_sh = wilcoxon_paired(real_vals, shuffle_vals)
            rec.update({
                "shuffle_mean": shuffle_vals.mean(),
                "shuffle_std":  shuffle_vals.std(ddof=1),
                "W_vs_shuffle": stat_sh,
                "p_vs_shuffle": p_sh,
                "d_vs_shuffle": cohens_d(real_vals, shuffle_vals),
                "_shuffle":     shuffle_vals,
            })

        if zero_vals is not None:
            stat_z, p_z = wilcoxon_paired(real_vals, zero_vals)
            rec.update({
                "zero_mean": zero_vals.mean(),
                "zero_std":  zero_vals.std(ddof=1),
                "W_vs_zero": stat_z,
                "p_vs_zero": p_z,
                "d_vs_zero": cohens_d(real_vals, zero_vals),
                "_zero":     zero_vals,
            })

        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
#  Printing
# ---------------------------------------------------------------------------

SIG = {0.001: "***", 0.01: "**", 0.05: "*", 1.0: ""}

def _sig(p):
    for thresh, label in SIG.items():
        if p < thresh:
            return label
    return ""


def print_scheme_table(result_df: pd.DataFrame, scheme: str, n: int):
    has_shuffle = "shuffle_mean" in result_df.columns
    has_zero    = "zero_mean"    in result_df.columns

    print(f"\n{'='*80}")
    print(f"  {scheme.upper()}  (n={n} folds/subjects)"
          + ("" if (has_shuffle and has_zero) else "  [no controls]"))
    print(f"{'='*80}")

    hdr = f"  {'Metric':<12}  {'Real':>14}"
    if has_shuffle:
        hdr += f"  {'Shuffle':>14}  {'p(sh)':>8}  {'d(sh)':>6}"
    if has_zero:
        hdr += f"  {'Zero':>14}  {'p(zero)':>8}  {'d(zero)':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for _, row in result_df.iterrows():
        real_s = f"{row['real_mean']:.4f}±{row['real_std']:.4f}"
        line   = f"  {row['metric']:<12}  {real_s:>14}"
        if has_shuffle:
            sh_s   = f"{row['shuffle_mean']:.4f}±{row['shuffle_std']:.4f}"
            p_sh_s = f"{row['p_vs_shuffle']:.4f}{_sig(row['p_vs_shuffle'])}"
            line  += f"  {sh_s:>14}  {p_sh_s:>8}  {row['d_vs_shuffle']:>6.3f}"
        if has_zero:
            z_s   = f"{row['zero_mean']:.4f}±{row['zero_std']:.4f}"
            p_z_s = f"{row['p_vs_zero']:.4f}{_sig(row['p_vs_zero'])}"
            line += f"  {z_s:>14}  {p_z_s:>8}  {row['d_vs_zero']:>6.3f}"
        print(line)

    if has_shuffle or has_zero:
        print("  Significance: * p<0.05  ** p<0.01  *** p<0.001  (Wilcoxon signed-rank)")


# ---------------------------------------------------------------------------
#  Figures
# ---------------------------------------------------------------------------

def plot_scheme(result_df: pd.DataFrame, scheme: str, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not available — skipping figures")
        return

    has_shuffle = "shuffle_mean" in result_df.columns
    has_zero    = "zero_mean"    in result_df.columns

    colors = {"none": "#2196F3", "shuffle_time": "#FF9800", "zero": "#9E9E9E"}
    labels = CTRL_LABEL

    # Build the list of conditions to plot
    ctrls = ["none"]
    if has_shuffle:
        ctrls.append("shuffle_time")
    if has_zero:
        ctrls.append("zero")

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 4.5))
    fig.suptitle(f"Inference Method — {scheme.replace('_', ' ').title()}", fontsize=13)

    for ax, metric in zip(axes, METRICS):
        row = result_df[result_df["metric"] == METRIC_LABELS[metric]].iloc[0]

        raw_key = {"none": "_real", "shuffle_time": "_shuffle", "zero": "_zero"}
        data    = [row[raw_key[c]] for c in ctrls if raw_key[c] in row.index]
        xlabels = [labels[c] for c in ctrls if raw_key[c] in row.index]
        cols    = [colors[c] for c in ctrls if raw_key[c] in row.index]

        parts = ax.violinplot(data, positions=range(len(data)), showmedians=True,
                              showextrema=True)
        for pc, col in zip(parts["bodies"], cols):
            pc.set_facecolor(col)
            pc.set_alpha(0.6)
        for part in ("cmedians", "cmins", "cmaxes", "cbars"):
            if part in parts:
                parts[part].set_color("black")
                parts[part].set_linewidth(1.2)

        for i, (vals, col) in enumerate(zip(data, cols)):
            jitter = np.random.default_rng(42).uniform(-0.07, 0.07, len(vals))
            ax.scatter(i + jitter, vals, color=col, s=18, zorder=3, alpha=0.8)

        # Significance bars (only when controls are present)
        if len(data) > 1:
            y_max   = max(v.max() for v in data)
            y_range = y_max - min(v.min() for v in data)
            bar_h   = y_range * 0.07

            def _sig_bar(ax, x1, x2, y, p):
                sig = _sig(p)
                if not sig:
                    return y
                ax.plot([x1, x1, x2, x2], [y, y + bar_h * 0.5, y + bar_h * 0.5, y],
                        lw=1.2, color="black")
                ax.text((x1 + x2) / 2, y + bar_h * 0.55, sig,
                        ha="center", va="bottom", fontsize=10)
                return y + bar_h * 1.4

            y_bar = y_max + bar_h * 0.3
            if has_shuffle and "p_vs_shuffle" in row:
                y_bar = _sig_bar(ax, 0, ctrls.index("shuffle_time"), y_bar, row["p_vs_shuffle"])
            if has_zero and "p_vs_zero" in row:
                _sig_bar(ax, 0, ctrls.index("zero"), y_bar, row["p_vs_zero"])

        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(xlabels, fontsize=8, rotation=15, ha="right")
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"compare_controls_{scheme}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Figure → {out_path}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no_figs",   action="store_true")
    p.add_argument("--eval_root", default=None,
                   help="Root checkpoint directory containing loso_sub-* (and optionally "
                        "_ctrl_* sibling) subdirs. Default: unified/out/inference/bert_base_uncased")
    p.add_argument("--out_dir",   default=None,
                   help="Output directory for CSVs and figures. "
                        "Default: method1_analysis/results")
    return p.parse_args()


def main():
    args      = parse_args()
    eval_root = Path(args.eval_root) if args.eval_root else EVAL_ROOT
    out_dir   = Path(args.out_dir)   if args.out_dir   else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"eval_root : {eval_root}")
    print(f"out_dir   : {out_dir}")

    df = load_all(eval_root)

    all_results = {}
    for scheme in ["loso", "session_cv", "stimulus"]:
        sub = df[df["eval_scheme"] == scheme]
        if sub.empty:
            print(f"\n[skip] no data for {scheme}")
            continue

        n = sub[sub["control"] == "none"]["fold_id"].nunique()
        result_df = analyse_scheme(df, scheme)
        all_results[scheme] = result_df

        print_scheme_table(result_df, scheme, n)

        # Save CSV (drop internal _raw columns)
        raw_cols = [c for c in result_df.columns if c.startswith("_")]
        csv_df   = result_df.drop(columns=raw_cols)
        csv_path = out_dir / f"compare_controls_{scheme}.csv"
        csv_df.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"  CSV → {csv_path}")

        if not args.no_figs:
            plot_scheme(result_df, scheme, out_dir)

    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
