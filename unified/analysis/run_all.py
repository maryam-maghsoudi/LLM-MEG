"""
run_all.py — Orchestrate the full twostage analysis pipeline.

Steps
-----
1. Load all eval_results.json → trial-level tables.
2. Aggregate trials → fold-level (mean + std).
3. Stage 1 — validity: collect all 30 Wilcoxon tests, apply Holm-Bonferroni
   correction across all 30, save summary tables + figures.
4. Stage 2 — generalization: Kruskal-Wallis + pairwise Mann-Whitney U (real
   condition only), save summary table + figures.

Usage
-----
    cd llm_decoder/
    python -m unified.analysis.run_all
    python -m unified.analysis.run_all --out_dir unified/analysis/results
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from .load_data           import load_tables
from .aggregate           import aggregate_scalars, aggregate_recall_curves
from .stats_utils         import holm_bonferroni
from .stage1_validity     import run_stage1, stage1_summary_df, EVAL_SCHEMES
from .stage2_generalization import run_stage2, stage2_summary_df
from .plotting            import (
    plot_fig1_validity,
    plot_fig2_training_curves,
    plot_fig3_generalization_scalars,
    plot_fig4_generalization_recall,
)

_HERE = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser(description="Run twostage analysis pipeline")
    p.add_argument("--out_dir", default=str(_HERE), help="Root output directory")
    p.add_argument("--no_figs", action="store_true", help="Skip figure generation")
    p.add_argument("--n_boot",  type=int, default=1000, help="Bootstrap resamples")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    tbl_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n[1/4] Loading eval results ...")
    trials_df, recall_df = load_tables()
    if trials_df.empty:
        print("ERROR: no eval_results.json files found. Run evaluate.py first.")
        return

    # ── 2. Aggregate ──────────────────────────────────────────────────────────
    print("\n[2/4] Aggregating trial → fold level ...")
    fold_df        = aggregate_scalars(trials_df)
    fold_recall_df = aggregate_recall_curves(recall_df)

    # Save raw fold-level table for inspection
    fold_df.to_csv(tbl_dir / "fold_level_scalars.csv", index=False)
    print(f"  fold_df: {fold_df.shape}  (eval_scheme × fold_id × control)")
    _print_fold_counts(fold_df)

    # ── 3. Stage 1 — Validity ─────────────────────────────────────────────────
    print("\n[3/4] Stage 1 — Validity (real vs controls) ...")
    s1_records = run_stage1(fold_df)

    # Holm-Bonferroni across all 30 tests
    all_pvals = [r["p_value"] for r in s1_records]
    corrected = holm_bonferroni(all_pvals)
    for r, cp in zip(s1_records, corrected):
        r["p_corrected"] = float(cp)

    print(f"  Total tests: {len(s1_records)}  (expected 30)")
    _print_stage1_summary(s1_records)

    # Save per-scheme summary tables
    for scheme in EVAL_SCHEMES:
        df = stage1_summary_df(s1_records, scheme)
        csv_path = tbl_dir / f"stage1_summary_{scheme}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  → {csv_path}")
        try:
            tex_path = tbl_dir / f"stage1_summary_{scheme}.tex"
            df.to_latex(tex_path, index=False, float_format="%.4f")
        except Exception:
            pass

    # Save full Stage 1 records as JSON
    (tbl_dir / "stage1_records.json").write_text(
        json.dumps(s1_records, indent=2, default=str)
    )

    # Stage 1 figures
    if not args.no_figs:
        print("  Generating Stage 1 figures ...")
        for scheme in EVAL_SCHEMES:
            plot_fig1_validity(
                fold_df, fold_recall_df, scheme,
                out_dir=fig_dir, n_boot=args.n_boot,
            )
        print("  Generating training curve figures ...")
        for scheme in EVAL_SCHEMES:
            plot_fig2_training_curves(scheme, out_dir=fig_dir)

    # ── 4. Stage 2 — Generalization ───────────────────────────────────────────
    print("\n[4/4] Stage 2 — Generalization (across eval schemes) ...")
    s2_records = run_stage2(fold_df)
    _print_stage2_summary(s2_records)

    s2_df = stage2_summary_df(s2_records)
    csv_path = tbl_dir / "stage2_summary.csv"
    s2_df.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")
    try:
        s2_df.to_latex(tbl_dir / "stage2_summary.tex", index=False, float_format="%.4f")
    except Exception:
        pass

    (tbl_dir / "stage2_records.json").write_text(
        json.dumps(s2_records, indent=2, default=str)
    )

    if not args.no_figs:
        print("  Generating Stage 2 figures ...")
        plot_fig3_generalization_scalars(fold_df, out_dir=fig_dir)
        plot_fig4_generalization_recall(fold_recall_df, out_dir=fig_dir,
                                        n_boot=args.n_boot)

    print("\nDone. Outputs →", out_dir)


# ---------------------------------------------------------------------------
#  Console helpers
# ---------------------------------------------------------------------------

def _print_fold_counts(fold_df: pd.DataFrame):
    for scheme in fold_df["eval_scheme"].unique():
        sub = fold_df[fold_df["eval_scheme"] == scheme]
        n_folds = sub["fold_id"].nunique()
        ctrls   = sorted(sub["control"].unique())
        print(f"    {scheme}: {n_folds} folds, controls={ctrls}")


def _print_stage1_summary(records):
    print(f"\n  {'scheme':<12} {'metric':<12} {'contrast':<25} "
          f"{'none_mean':>10} {'ctrl_mean':>10} {'p_raw':>8} {'p_corr':>8} {'ES':>6}")
    print("  " + "-" * 87)
    for r in records:
        sig = "*" if (r["p_corrected"] or 1.0) < 0.05 else " "
        print(f"  {r['eval_scheme']:<12} {r['metric']:<12} {r['contrast']:<25} "
              f"{r['none_mean']:>10.4f} {r['ctrl_mean']:>10.4f} "
              f"{r['p_value']:>8.4f} {r['p_corrected'] or float('nan'):>8.4f} "
              f"{r['effect_size']:>6.3f}{sig}")


def _print_stage2_summary(records):
    seen = set()
    print(f"\n  {'metric':<12} {'comparison':<28} {'p_corr':>8} {'ES':>6} {'KW_p_corr':>10}")
    print("  " + "-" * 70)
    for r in records:
        key = (r["metric"], r["comparison"])
        if key in seen:
            continue
        seen.add(key)
        sig = "*" if r["p_corrected"] < 0.05 else " "
        print(f"  {r['metric']:<12} {r['comparison']:<28} "
              f"{r['p_corrected']:>8.4f} {r['effect_size']:>6.3f} "
              f"{r['kw_p_corrected']:>10.4f}{sig}")


if __name__ == "__main__":
    main()
