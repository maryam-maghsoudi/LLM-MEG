# Analysis Pipeline Spec: `llmtwostage` Evaluation

**Scope:** This analysis pipeline is for the **`llmtwostage`** method only (not `inference` or `interleaved`).

---

## 1. Background / Data Available

For `llmtwostage`, we have results across:

- **3 evaluation schemes:**
  - `LOSO` — leave-one-subject-out, 13 folds (fold = subject)
  - `session_cv` — 5-fold session cross-validation, 5 folds (fold = session)
  - `stimulus` — heldout stimulus (last 2 lines / last 4 lines of poems), 6 folds (fold = stimulus group)
- **3 controls per eval scheme:**
  - `none` — real training/decoding (i.e., the actual model, not a control)
  - `shuffle_time` — MEG time-shuffled control
  - `zero` — zeroed-MEG control
- **5 metrics per trial:**
  1. `recall_at_k` — a **curve** (recall value for each k), not a scalar
  2. `mrr` — scalar
  3. `word_acc` — scalar
  4. `bleu1` — scalar
  5. `wer` — scalar

So the full grid is: 3 eval schemes × 3 controls × (13, 5, or 6 folds depending on scheme) × N trials per fold × 5 metrics.

---

## 2. Goal

Build a reusable analysis pipeline that:
1. Loads all `llmtwostage` result files into one unified table.
2. Answers two distinct questions separately:
   - **Q1 (Validity):** Is the model decoding real signal, or performing at control-level? (real vs. `shuffle_time`, real vs. `zero`, *within* each eval scheme)
   - **Q2 (Generalization):** How does real-condition performance compare *across* the three eval schemes (LOSO vs. session_cv vs. stimulus)?
3. Produces summary tables, statistical test results, and figures for both questions.

Do not conflate Q1 and Q2 — they should be separate pipeline stages with separate outputs.

---

## 3. Data Model

### 3.1 Input format assumption
Results likely live as per-trial records (e.g. one CSV/JSON per eval_scheme × control run, or one big file — clarify with actual files when available). The pipeline should have a loader step that normalizes whatever the raw format is into the canonical long-format table below.

### 3.2 Canonical long-format table (trial-level)

One row per trial, per condition:

| column | type | description |
|---|---|---|
| `method` | str | always `"llmtwostage"` for this pipeline |
| `eval_scheme` | str | `LOSO`, `session_cv`, or `stimulus` |
| `fold_id` | str/int | subject id (LOSO), session id (session_cv), or stimulus-group id (stimulus) |
| `control` | str | `none`, `shuffle_time`, or `zero` |
| `trial_id` | str/int | unique trial identifier within the fold |
| `mrr` | float | scalar metric |
| `word_acc` | float | scalar metric |
| `bleu1` | float | scalar metric |
| `wer` | float | scalar metric |

### 3.3 Recall@k table (separate, long format)

Recall@k is a curve, so keep it in its own long table rather than cramming it into the trial table:

| column | type | description |
|---|---|---|
| `method` | str | always `"llmtwostage"` |
| `eval_scheme` | str | `LOSO`, `session_cv`, `stimulus` |
| `fold_id` | str/int | same as above |
| `control` | str | `none`, `shuffle_time`, `zero` |
| `trial_id` | str/int | unique trial identifier |
| `k` | int | rank cutoff |
| `recall` | float | recall@k value for this trial (already averaged over word positions within the trial by `evaluate.py`) |

From this, fold-level recall@k is `mean(recall)` grouped by `(eval_scheme, fold_id, control, k)`. No binary hit reconstruction needed — `evaluate.py` pre-aggregates within each trial.

---

## 4. Aggregation Rule

**Important:** Do not run statistics on raw trials. Trials within a fold are not independent (they share a subject/session/decoder instance).

Pipeline must aggregate trial-level metrics **up to the fold level first**:
- For each `(eval_scheme, fold_id, control)`, compute **mean and std** of `mrr`, `word_acc`, `bleu1`, `wer` across trials in that fold.
- For recall@k, compute mean `recall` across trials in that fold, for each `k`.

The within-fold std captures how consistently the model performs across trials (subjects/sessions/words) within each fold. Report it alongside the mean in summary tables and, where space allows, as error bars in figures.

This produces fold-level tables:
- Fold count per scheme: LOSO → 13 rows per control, session_cv → 5 rows per control, stimulus → 2 rows per control (`fold_id` = `"lines2"` or `"lines4"`).
- All statistical tests operate on these fold-level aggregates, treated as **paired samples** (each fold has a `none`, `shuffle_time`, and `zero` value).

---

## 5. Analysis Stage 1 — Validity (Q1): Real vs. Controls

For each `eval_scheme` independently:

For each scalar metric (`mrr`, `word_acc`, `bleu1`, `wer`):
- Paired comparison: `none` vs `shuffle_time`, and `none` vs `zero`, across folds.
- Use **Wilcoxon signed-rank test** (paired, non-parametric — safer than paired t-test given small fold counts, especially `stimulus` with n=6).
- Also compute an effect size: matched-pairs rank-biserial correlation (or Cohen's d as a secondary/parametric sanity check).
- Report: test statistic, p-value, effect size, and the raw paired means for both conditions.

For `recall_at_k`:
- Reduce the curve to a single scalar per fold via **AUC of the recall curve** (area under recall@k vs k, e.g. via trapezoidal rule over the k range tested).
- Run the same paired Wilcoxon test on this per-fold AUC scalar (`none` vs `shuffle_time`, `none` vs `zero`).
- Additionally, produce a qualitative plot: full recall@k curve (mean across folds) for `none`, `shuffle_time`, `zero`, with **bootstrap confidence intervals** computed by resampling folds (not trials) with replacement, e.g. 1000 resamples, 95% CI.

### 5.1 Multiple comparisons correction

Total tests = 3 eval schemes × 5 metrics (4 scalar + 1 recall-AUC) × 2 control contrasts (`none` vs `shuffle_time`, `none` vs `zero`) = 30 tests.

Apply **Holm-Bonferroni correction** across these 30 tests (or clearly document if correcting within a narrower family, e.g. per eval scheme, and justify why).

### 5.2 Output for Stage 1

Per `eval_scheme`, produce:
- **Summary table:** rows = metrics (`mrr`, `word_acc`, `bleu1`, `wer`, `recall_auc`), columns = `none` mean ± SEM (± std), `shuffle_time` mean ± SEM (± std), `zero` mean ± SEM (± std), p-value (corrected) for each contrast, effect size for each contrast. Report both SEM (across folds, for statistical precision) and std (within folds averaged, for interpretability of trial-level variability).
- **Figure 1 (single figure, 1 row × 5 subplots):**
  - **Subplot 1:** recall@k curves (3 lines: `none`/`shuffle_time`/`zero`) with bootstrap CI bands.
  - **Subplots 2–5:** one per scalar metric (`mrr`, `word_acc`, `bleu1`, `wer`) — 3 boxes per subplot (x-axis = control condition: `none`/`shuffle_time`/`zero`), with fold-level values overlaid as dots, and a line connecting each fold's dot across the three conditions (paired within-fold across controls). Within-fold std shown as vertical error bars on each dot.

---

## 6. Analysis Stage 2 — Generalization (Q2): Across Eval Schemes

Using **only the `none` (real) condition**, fold-level aggregates:

- For each scalar metric + `recall_auc`, compare distributions across `LOSO`, `session_cv`, `stimulus`.
- Since fold counts differ (13 vs 5 vs 6) and folds are *not* paired across schemes (different fold definitions entirely), use an **unpaired/independent-groups approach**: Kruskal-Wallis across the three schemes as an omnibus test, followed by pairwise Mann-Whitney U tests (with Holm-Bonferroni correction across the 3 pairwise comparisons) if the omnibus is significant.
- Do NOT treat these as paired samples — make this explicit in code comments, since it's a common mistake to import the Stage 1 pairing logic here.

### 6.1 Output for Stage 2

- **Summary table:** rows = metrics, columns = `LOSO` mean ± SEM (± std), `session_cv` mean ± SEM (± std), `stimulus` mean ± SEM (± std), Kruskal-Wallis p-value, pairwise p-values (corrected).
- **Figure 3:** small multiples (one panel per metric), x-axis = eval scheme, box/strip plot of fold-level real-condition values only, with within-fold std shown as error bars on each dot.
- **Figure 4:** recall@k curves, one line per eval scheme (real condition only), with bootstrap CI bands (resampling folds within each scheme).

---

## 7. Pipeline Structure (suggested code organization)

```
llmtwostage_analysis/
├── load_data.py        # raw result files -> canonical trial-level + recall-k long tables
├── aggregate.py         # trial-level -> fold-level aggregation
├── stage1_validity.py   # Q1: real vs shuffle_time vs zero, per eval scheme
├── stage2_generalization.py  # Q2: real condition across eval schemes
├── stats_utils.py       # wilcoxon, mann-whitney, kruskal-wallis, holm-bonferroni, bootstrap CI, effect sizes
├── plotting.py          # recall curves, box/strip plots
└── run_all.py           # orchestrates load -> aggregate -> stage1 -> stage2 -> save outputs
```

### Output artifacts
- `stage1_summary_<eval_scheme>.csv` × 3
- `stage2_summary.csv`
- `fig1_summary_<eval_scheme>.png` × 3 (5-subplot figure: recall@k curve + 4 scalar-metric panels)
- `fig3_scalar_metrics_across_schemes.png`
- `fig4_recall_curves_across_schemes.png`

---

## 8. Implementation Notes / Libraries

- `pandas` for data wrangling
- `scipy.stats` for `wilcoxon`, `mannwhitneyu`, `kruskal`
- `statsmodels.stats.multitest` (`multipletests` with `method='holm'`) for correction
- `numpy` for bootstrap resampling and AUC (`np.trapz`)
- `matplotlib` / `seaborn` for plotting