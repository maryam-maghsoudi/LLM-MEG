# Unified MEG Speech Decoder — Design Document

**Project**: Open-vocabulary MEG speech decoding using LLMs  
**Directory**: `llm_decoder/unified/`  
**Status**: Pre-implementation design

---

## 1. Overview

This pipeline unifies three MEG decoding methods under a common data interface, training protocol, and evaluation framework. The goal is a fair, head-to-head comparison of:

- **Method 1 — llm_inference**: Contrastive MEG encoder (InfoNCE) + LLM language model prior fused at inference
- **Method 2 — llm_twostage**: Stage 1 contrastive alignment to LLM hidden states + Stage 2 KL distillation via a causal GRU
- **Method 3 — interleaved**: MEG-derived soft tokens injected into a frozen LLM via `inputs_embeds`

All three methods are evaluated under three orthogonal generalization criteria, using identical train/val/test splits and the same two evaluation metrics.

---

## 2. Data

### 2.1 MEG recordings

- **Subjects**: 13 (`sub-01`, `sub-03` … `sub-17`)
- **Poems**: 2 (`poem1`, `poem2`)
- **Sessions per subject per poem**: 10 (indexed 0–9)
- **Conditions**: `lis` (listened), `img` (imagined) — **current pipeline uses `lis` only**; `img` decoding is a future extension
- **File format**: MNE `.fif` epoch files at `{MEG_BASE}/{subject}/ses-{session}/meg/`
- **Preprocessing**: collapse repetitions with `mean(axis=0)`, downsample 10× to 100 Hz, multiply by 1e12 (Tesla → picoTesla range), z-score per channel in float64, cast to float32

### 2.2 Stimuli

Both poems are excerpts from *A Visit from St. Nicholas*. Word onset timestamps come from WhisperX forced alignment in `contrastive_learning/onset_out/`.

**Poem 1** — 56 words, 12 lines:

| Lines | Words | Text |
|---|---|---|
| 1–2 | 0–9 | When out on the lawn / there arose such a clatter |
| 3–4 | 10–20 | I sprang from my bed / to see what was the matter |
| 5–6 | 21–29 | Away to the window / I flew like a flash |
| 7–8 | 30–38 | Tore open the shutters / and threw up the sash |
| 9–10 | 39–47 | The moon on the breast / of the new-fallen snow |
| 11–12 | 48–55 | Gave a lustre of midday / to objects below |

**Poem 2** — 61 words, 12 lines:

| Lines | Words | Text |
|---|---|---|
| 1–2 | 0–11 | He was dressed all in fur / from his head to his foot |
| 3–4 | 12–21 | And his clothes were all tarnished / with ashes and soot |
| 5–6 | 22–31 | A bundle of toys / he had flung on his back |
| 7–8 | 32–41 | And he looked like a peddler / just opening his pack |
| 9–10 | 42–50 | His eyes how they twinkled / his dimples how merry |
| 11–12 | 51–60 | His cheeks were like roses / his nose like a cherry |

### 2.3 MEG word windows

Each word occurrence is a window of MEG centered on the word's onset timestamp:

- **Pre-onset**: 100 ms → 10 samples at 100 Hz
- **Post-onset**: 300 ms → 30 samples at 100 Hz
- **Window size**: 40 samples total, shape `(155, 40)`
- Windows that fall outside the trial boundaries are discarded (marked invalid)

---

## 3. Evaluation Schemes

Three independent generalization tests. Each produces its own train/val/test split. All three methods receive **identical splits**.

### 3.1 Heldout Subject (LOSO)

Leave-one-subject-out cross-validation. 13 folds, one per subject.

| Split | Subjects | Poems | Sessions |
|---|---|---|---|
| Train | 12 non-heldout | both | 0–7 |
| Val | 12 non-heldout | both | 8–9 |
| Test | 1 heldout | both | 0–9 |

- Tests **cross-subject generalization**: can a model trained on other people decode a new individual's brain?
- Requires retraining all three methods 13 times

### 3.2 Heldout Trials — 5-fold Session CV

5-fold cross-validation over sessions. Each fold holds out 2 sessions from each poem simultaneously.

| Fold | Test sessions | Val sessions | Train sessions |
|---|---|---|---|
| 0 | [0, 1] | [8, 9] | [2, 3, 4, 5, 6, 7] |
| 1 | [2, 3] | [0, 1] | [4, 5, 6, 7, 8, 9] |
| 2 | [4, 5] | [2, 3] | [0, 1, 6, 7, 8, 9] |
| 3 | [6, 7] | [4, 5] | [0, 1, 2, 3, 8, 9] |
| 4 | [8, 9] | [6, 7] | [0, 1, 2, 3, 4, 5] |

Val = the previous fold's test sessions (cyclic). All 13 subjects appear in both train and test of every fold.

- Tests **cross-session generalization**: does the model generalise across repetitions of the same poem?

### 3.3 Heldout Stimulus

Hold out the last N lines of each poem by word position. Trains on training-line words across all subjects and sessions; tests on held-out-line words across all subjects and sessions.

| Variant | Train lines | Val lines | Test lines |
|---|---|---|---|
| Last 2 lines | 1–10 (words: p1:0–47, p2:0–50) | lines 1–10, sessions 8–9 | 11–12 (words: p1:48–55, p2:51–60) |
| Last 4 lines | 1–8 (words: p1:0–37, p2:0–41) | lines 1–8, sessions 8–9 | 9–12 (words: p1:38–55, p2:42–60) |

Val is a session-based carve from the training lines (sessions 8–9 of training-line words only), keeping trial split intact.

- Tests **cross-stimulus generalization**: can the model decode words it has never seen in training, using only MEG signal?
- This is the most stringent test — it directly probes whether MEG provides content information vs. the model memorizing word-order statistics

---

## 4. Train/Val Split Principle

**All splits are trial-based (never window-based).**

A trial is a `(subject, poem, session)` tuple. All word windows from a trial go exclusively to train, val, or test — they are never mixed. This prevents leakage caused by LLM hidden states being identical across sessions for the same word position (only the MEG side varies per trial).

---

## 5. Decoding Methods

### 5.1 Method 1 — LLM Inference (Contrastive + LLM Fusion)

**Concept**: Train a MEG encoder with InfoNCE against BERT word embeddings. At inference, combine per-word cosine similarities from the MEG encoder with next-word probabilities from a frozen LLM, weighted by a scalar `alpha`. The BERT text encoder can be replaced with other embedding models (e.g., LLM hidden states) in the future without changing the rest of the method.

**Architecture**:
```
MEGEncoder: (C=155, T=40) → (128,) L2-normalized
TextEncoder: BERT word embedding → Linear(768,256) → GELU → Dropout → Linear(256,128) L2-normalized

Training loss: InfoNCE(z_meg, z_text, τ=0.07)

Inference:
  sim_t     = cosine(z_meg_t, z_text_w) for each word w in vocab
  llm_t     = P(w | context, LLM)
  score_t   = (1 - alpha) * sim_t + alpha * log(llm_t)
  pred_t    = argmax_w score_t
```

**Trainable components**: MEGEncoder, TextEncoder projection head  
**Frozen at inference**: LLM (GPT-2 or SmolLM2)  
**Output**: ranked word list over eval vocabulary, raw scores returned for fusion

### 5.2 Method 2 — Two-Stage (InfoNCE + KL Distillation)

**Concept**: Replace BERT targets with contextual LLM hidden states. Stage 1 aligns MEG embeddings to occurrence-level LLM representations (context-dependent). Stage 2 trains a causal GRU to predict the LLM's next-word distribution from the sequence of MEG embeddings.

**Architecture**:

*Stage 1*:
```
MEGEncoder: (C=155, T=40) → (128,) L2-normalized
LLMTextProjection: h_mid_t (d_model) → Linear(d_model,256) → GELU → Dropout → Linear(256,128)

Training loss: InfoNCE(z_meg_t, z_text_t, τ=0.07)
  where z_text_t = mean-pool of LLM hidden states at layer hmid_layer over word's subword span
```

*Stage 2*:
```
Frozen MEGEncoder → z_1...z_T
GRUHead: GRU(input=128, hidden=256) → Linear(256, d_model)
  → frozen lm_head → q_t over full vocab → restrict to eval vocab

Training loss: KL(p_t || q_t)  where p_t = teacher LLM final-layer distribution (from cache)
```

**Trainable components**:
- Stage 1: MEGEncoder + LLMTextProjection
- Stage 2: GRUHead only (MEGEncoder frozen)

**Key difference from Method 1**: targets are occurrence-level (same word in different contexts has different z_text_t), not vocabulary-level BERT embeddings.

**LLM cache**: the frozen LLM is run once per poem offline (`cache_llm_hiddens.py`) and hidden states/logits are stored. The LLM is never loaded during training.

**Supported LLMs**: HuggingFaceTB/SmolLM2-360M (default), SmolLM2-1.7B, gpt2, Qwen/Qwen2-0.5B

### 5.3 Method 3 — Interleaved Soft Tokens

**Concept**: Map MEG windows to "soft tokens" in the LLM's embedding space via a trainable adapter MLP. Inject these soft tokens interleaved with real word token embeddings into the frozen LLM using `inputs_embeds`. Train the adapter to minimize cross-entropy on the text tokens.

**Architecture**:
```
MEGEncoder: (C=155, T=40) → (128,) L2-normalized  [frozen, from Method 2 Stage 1]
Adapter MLP: Linear(128,512) → GELU → Linear(512, n_soft × d_model)
  → reshape → (n_soft, d_model) soft tokens per word

Sequence (interleaved design):
  [soft(w1)] [tok(w1)...] [soft(w2)] [tok(w2)...] ...
  Loss masked to text-token positions only (soft positions get label=-100)

LLM: frozen GPT-2 or SmolLM2, receives full sequence via inputs_embeds
```

**Trainable components**: Adapter MLP only  
**Output**: LLM next-token probabilities at each text-token position, teacher-forced during training; autoregressive at inference

**Note on MEGEncoder**: the MEGEncoder used here is the one trained in Method 1 (InfoNCE against BERT embeddings), kept frozen. Only the Adapter MLP is trained. This isolates the adapter as the only variable in Method 3, and keeps the MEG representation consistent with the contrastive baseline.

---

## 6. Control Conditions

Two controls trained alongside each method to validate that MEG signal (not position statistics) drives performance.

| Control | Description | What it tests |
|---|---|---|
| **Zero MEG** | Replace all MEG windows with zero tensors | Whether the model learns purely from word-order / positional statistics without any neural signal |
| **Time-shuffled MEG** | Randomly permute word positions within each trial at load time | Whether the model uses word-specific MEG content, or just generic trial-level signal independent of word identity |

Controls are trained identically to the real model (same architecture, same splits, same hyperparameters). Performance significantly above controls is required before any claim of MEG-driven decoding.

---

## 7. File Structure

```
unified/
├── DESIGN.md                  ← this document
├── data/
│   ├── base_dataset.py        ← MEGWordDataset: one item per word occurrence
│   ├── splits.py              ← make_loso_splits(), make_session_cv_splits(), make_stimulus_splits()
│   └── controls.py            ← ZeroMEGWrapper, TimeShuffledMEGWrapper
├── methods/
│   ├── models.py              ← MEGEncoder, LLMTextProjection, GRUHead, Adapter (shared)
│   ├── train_inference.py     ← Method 1 training loop
│   ├── train_twostage.py      ← Method 2 Stage 1 + Stage 2 training loops
│   └── train_interleaved.py   ← Method 3 training loop
├── train.py                   ← unified entry point (delegates to method-specific scripts)
├── predict.py                 ← predict(subject, session, condition, method, ckpt) → scores + words
├── evaluate.py                ← Option A (ranking) + Option B (text) metrics
├── cache_llm_hiddens.py       ← (symlink or copy from llm_twostage) offline LLM caching
└── run/
    ├── submit_loso.sh         ← Slurm: all methods × LOSO
    ├── submit_session_cv.sh   ← Slurm: all methods × 5-fold
    └── submit_stimulus.sh     ← Slurm: all methods × stimulus split
```

---

## 8. Unified Training Entry Point

```
python train.py \
  --method      {inference, twostage, interleaved} \
  --eval_scheme {loso, session_cv, stimulus} \
  --heldout     sub-01              # loso only
  --fold        0                   # session_cv only (0–4)
  --n_lines     2                   # stimulus only (2 or 4)
  --control     {none, zero, shuffle_time} \
  --llm_name    HuggingFaceTB/SmolLM2-360M \
  --device      cuda \
  --out_root    unified/out/
```

Output directory: `out/{method}/{eval_scheme}/{heldout|fold_k|lines_N}/{control}/`

---

## 9. Inference API

```python
result = predict(
    subject   = "sub-01",
    session   = 3,
    condition = "lis",
    poem      = "poem1",
    method    = "twostage",
    ckpt_dir  = "out/twostage/loso/sub-01/none/",
)
# result = {
#     'words':       ["when", "out", ...],   # ground truth
#     'pred_top1':   ["when", "out", ...],   # argmax prediction
#     'scores':      Tensor(N, V),            # raw logits/scores per position (for fusion)
#     'vocab':       ["a", "and", ...],       # evaluation vocabulary
# }
```

The `scores` field is intentionally returned so that LLM-inference-style alpha fusion can be applied on top of any method's output in the future, without rerunning the model.

---

## 10. Evaluation Metrics

### Option A — Closed-Vocabulary Ranking

Vocabulary: all unique words appearing in the test split of each evaluation scheme (varies by split; typically 30–45 unique words per poem).

For each word position t in the test set, rank all vocab words by the method's score. Metrics:

- **R@k for k = 1 … |V|**: fraction of positions where the correct word is ranked within the top k, reported for every k up to the full vocabulary size |V|. This yields a full recall curve rather than three fixed cutoffs.
- **MRR**: mean reciprocal rank = mean(1 / rank_of_correct_word)

The recall curve makes it easy to visualise where a method gains over another across the full ranking, not just at arbitrary cutoffs. |V| is the number of unique words in the test split (typically 30–45).

Report per-poem and averaged. For stimulus split, also report separately for heldout lines vs. training lines.

### Option B — Open-Vocabulary Text Quality

Each method generates a word sequence via greedy decoding (or beam search) from its score distribution.

- **Word accuracy** (exact match per position, averaged)
- **BLEU-1** (unigram precision against ground-truth word sequence)
- **WER** (word error rate: substitutions + insertions + deletions / total words)

Report per-poem and averaged. For the stimulus split, report separately for heldout lines.

### Controls comparison

For each metric, report: real model vs. zero-MEG control vs. time-shuffled control. Report p-values from a paired Wilcoxon signed-rank test across folds/subjects.

---

## 11. LLM Fusion Extension (Future)

All three methods return raw `scores` tensors from `predict()`. An LLM-inference fusion layer can be applied on top of any method's scores without retraining:

```python
fused_scores = (1 - alpha) * method_scores + alpha * llm_prior_logits
```

where `llm_prior_logits` = log P(w | preceding words, frozen LLM). The alpha can be swept or tuned on the val set. This is a post-hoc addition requiring no changes to training.

---

## 12. Key Design Decisions (Summary)

| Decision | Choice | Reason |
|---|---|---|
| Train/val split | Trial-based | Prevents leakage from identical LLM targets across sessions |
| MEG window | [-100ms, +300ms] = 40 samples | Matches llm_twostage; consistent with onset-locked response |
| MEG preprocessing | ×1e12 before z-score | Float32 std underflows for raw Tesla-scale values |
| InfoNCE temperature | 0.07 | Inherited from contrastive pipeline |
| MEGEncoder for Method 3 | Shared from Method 1 (BERT InfoNCE) | Isolates adapter as the only variable; keeps MEG representation consistent with contrastive baseline |
| Eval vocabulary | Split-specific unique words | Avoids fixed closed-vocab assumption; varies naturally by split |
| Controls | Zero MEG, time-shuffled MEG | Tests position-only and word-unspecific baselines respectively |
| Val fold (session CV) | Previous fold, cyclic | Ensures val and test are always different sessions |
