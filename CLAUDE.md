# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Train with defaults (interleaved design, GPT-2, frozen encoder)
python train.py

# Train with specific options
python train.py --design upfront               # Design B baseline
python train.py --unfreeze_encoder             # Option B: joint encoder training
python train.py --llm_name gpt2-medium         # larger LLM
python train.py --out_dir out/run_01 --seed 0

# Evaluate (teacher-forced, all ablations)
python evaluate.py --adapter_ckpt out/train/best_adapter.pt

# Evaluate with specific ablation only
python evaluate.py --adapter_ckpt out/train/best_adapter.pt --ablation shuffle
# ablation choices: all, shuffle, random_soft, no_soft

# Autoregressive generation (true inference)
python inference.py --adapter_ckpt out/train/best_adapter.pt
python inference.py --adapter_ckpt out/train/best_adapter.pt --oracle_lengths --beam_size 5

# Dataset sanity check
python dataset.py

# LLM fusion (closed-vocabulary contrastive + GPT-2 hybrid, in llm_inference/)
cd llm_inference
python llm_fusion.py --subject sub-01 --session 3 --poem poem1 --cond lis
python llm_fusion.py --subject sub-01 --session 3 --poem poem1 --cond lis --sweep_alpha
```

Outputs are written to `out/` (training) and `llm_inference/fusion_out/` (fusion). All scripts accept `--out_dir` to override.

## Architecture

This is an **LLM-guided open-vocabulary MEG speech decoder** — a follow-up to a closed-vocabulary (76-word) contrastive decoder. The goal is to decode imagined speech from MEG brain recordings by conditioning a frozen LLM on MEG-derived "soft tokens."

### Data flow

```
MEG window (B, 155, 100)
    → MEGEncoder → (B, 128) L2-normalized embedding
    → Adapter (trainable MLP) → (B, n_soft, 768) soft tokens
    → injected into frozen GPT-2 via inputs_embeds
```

The MEG encoder (`meg_encoder.py`) mirrors the `MEGWordEncoderSmall` architecture from the upstream contrastive pipeline. Its pretrained checkpoint is loaded from `../contrastive_learning/compare_out/models/bert_wav2vec/meg_encoder.pt`. The encoder is kept **frozen** in the default Option A configuration to prevent overfitting.

The **adapter** (`adapter.py`) is the **only trainable component** in Option A: `Linear(128, 512) → GELU → Linear(512, n_soft × 768)`. Only the adapter's state dict is saved as `best_adapter.pt` / `final_adapter.pt`.

The **LLM** (`_FrozenLLM` in `model.py`) overrides `train()` to permanently stay in eval mode, preventing dropout noise from corrupting adapter gradients.

### Two sequence designs (both implemented in `model.py`)

**Design A — Interleaved** (`SEQUENCE_DESIGN = "interleaved"`, recommended):
```
[soft(w1)] [tok(w1)...] [soft(w2)] [tok(w2)...] ...
```
Loss is masked to text-token positions only. Each soft token directly precedes its word, giving clean credit assignment. This is the primary method.

**Design B — Upfront** (`"upfront"`, captioning-style baseline):
```
[soft(w1)] [soft(w2)] ... [soft(wN)] [tok(w1)...tok(wN)...]
```
All MEG evidence precedes all text; weaker credit assignment. More prone to the LLM ignoring soft tokens.

Both designs use `inputs_embeds` (not `input_ids`) to allow mixing soft tokens and real token embeddings in one sequence. Labels at soft-token positions are set to -100 (ignored by HuggingFace CE loss).

### Upstream contrastive pipeline (source of MEG encoder checkpoint)

The `../contrastive_learning/contrastive_word_meg.py` script trained the MEG encoder checkpoint this project loads. Understanding it is essential for knowing what the encoder expects.

**MEG file format and naming:**
```
{MEG_BASE}/{subject}/ses-{session}/meg/{subject}_sess-{session}_task-{condition}_meg-epo.fif
```
where `condition` = `"{poem_key}{cond_suffix}"` (e.g., `"poem1lis"`, `"poem2img"`). Files are MNE `.fif` epoch files; `epochs.get_data().mean(axis=0)` collapses repetitions to a single trial `(C, T_raw)`, then resampled 10× → `(155, T_ds)` and z-scored per channel.

**Flash removal:** The benchmark removes ~51 samples after each flash event (every 207 downsampled samples). `contrastive_word_meg.py` defaults `REMOVE_FLASHES = False` because flash removal causes ~91% of word windows to overlap removed regions, leaving only ~5 words per poem. **The llm_decoder dataset also skips flash removal** — consistent with this default.

**Two encoder sizes in the contrastive pipeline:**
- `MEGWordEncoderSmall` (~143k params): spatial C→32, 3 dilated temporal conv blocks, 128-d hidden — **this is the architecture whose checkpoint is loaded by `meg_encoder.py`**
- `MEGWordEncoder` (full, ~544k params): spatial C→64, 4 temporal conv blocks, 256-d hidden

**Window size discrepancy to be aware of:** The current `contrastive_word_meg.py` uses `WIN_POST_MS = 400` (giving `WIN_SIZE = 60` samples), reflected in its output directory name `contrastive_out_400ms`. The `llm_decoder/config.py` uses `WIN_POST_MS = 800` (`WIN_SIZE = 100`). The checkpoint at `compare_out/models/bert_wav2vec/meg_encoder.pt` was likely trained with the 800ms/100-sample window (matching what Architecture.md specifies) — not the 400ms version in the current script. **Do not re-train the contrastive encoder using the current script's 400ms window and expect it to work as a drop-in replacement.**

**Contrastive training**: NT-Xent (InfoNCE) loss at temperature 0.07 between L2-normalized MEG embeddings and frozen BERT word embeddings (bert-base-uncased, mean last-layer pooling). Text projection head `Linear(768,256)→GELU→Dropout→Linear(256,128)` is trained alongside the MEG encoder. Light data augmentation: `+0.02 * randn` Gaussian noise per batch.

**Phase 2 (imagined MEG):** An img→lis mapping model (CNN1D / UNet1D / etc. from the benchmark) translates imagined MEG to predicted listened MEG before feeding the frozen encoder. This mapping model lives in `../benchmark/`. The llm_decoder is designed to eventually replace this whole pipeline with end-to-end decoding.

### Dataset

`PoemTrialDataset` in `dataset.py` — one item = one full poem trial (subject × session × poem × condition). MEG data is loaded from `/fs/nexus-projects/brain_project/maryam_meg_dataset/icaed/`, downsampled 10× to 100 Hz, z-scored per channel, and sliced into per-word windows of [-200ms, +800ms] around onset timestamps. Onset timestamps come from `../contrastive_learning/onset_out/{poem}_word_onsets.json` (WhisperX forced alignment).

Words are tokenized with the LLM's tokenizer with a leading space (`" " + word`) to match mid-sentence BPE encoding. Token IDs are the same for all subjects/sessions of a given poem; only the MEG windows vary per trial.

`collate_trials` pads MEG windows and valid_masks across trials with different word counts.

### Data splits (critical — prevents LLM memorization)

The model must not see poem2 during training. With only two short poems, the LLM has strong priors that could make it memorize sequences rather than use MEG signal.

```
train : all subjects, poem1, sessions 0–7   (~104 trials)
val   : all subjects, poem1, sessions 8–9   (~26 trials)
test  : all subjects, poem2, all sessions
```

Configured in `config.py`: `TRAIN_POEMS`, `TEST_POEMS`, `TRAIN_SESSIONS`, `VAL_SESSIONS`.

### Evaluation

`evaluate.py` runs **teacher-forced** evaluation (ground-truth context provided). Metrics: `exact_match`, `bleu1`, `bert_sim` (BERT cosine similarity of predicted vs. true words), `restricted_R@k` / `restricted_MRR` (rank within the 76-word closed vocabulary, for comparison to the prior contrastive decoder).

Three ablations must pass before trusting any result:
- **shuffle**: MEG windows re-paired to random word positions — tests word-specificity of soft tokens
- **random_soft**: adapter outputs replaced with Gaussian noise at test time — tests whether LLM uses soft tokens at all
- **no_soft**: text-only LM, no soft tokens — measures language prior alone (upper bound for memorization)

`inference.py` runs true **autoregressive generation** (no ground-truth text). Supports `--oracle_lengths` (generate exactly N tokens per word where N comes from ground truth) and `--beam_size`.

### Key paths in `config.py`

| Variable | Value |
|---|---|
| `MEG_BASE` | `/fs/nexus-projects/brain_project/maryam_meg_dataset/icaed` |
| `ONSET_DIR` | `../contrastive_learning/onset_out/` |
| `MEG_CKPT` | `../contrastive_learning/compare_out/models/bert_wav2vec/meg_encoder.pt` |
| `OUT_DIR` | `./out/` |
| `LLM_NAME` | `"gpt2"` (change to `"gpt2-medium"` or `"microsoft/phi-2"`) |
| `LLM_D_MODEL` | `768` for GPT-2; update if switching LLMs |

### `llm_inference/` subdirectory

A separate, earlier-stage approach that fuses the existing contrastive decoder's cosine similarities with GPT-2's closed-vocabulary next-word probabilities via a weighted combination (controlled by `--alpha`). This does not use the adapter architecture — it operates on the 76-word closed vocabulary. Scripts: `llm_fusion.py`, `llm_fusion_kfold.py`, `inspect_fusion_predictions.py`. Results in `llm_inference/fusion_out/` and `llm_inference/predictions_out/`.
