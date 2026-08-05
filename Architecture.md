# LLM-Guided Imagined Speech Decoding — Architecture Specification

## Project context

This is a follow-up to a zero-shot imagined speech decoding paper (see main paper).
The existing system has three stages:
1. A mapping model that converts imagined MEG → predicted listened MEG
2. A contrastive decoder trained on listened MEG that retrieves words from a fixed 76-word vocabulary
3. A full pipeline: imagined MEG → mapping → predicted listened → contrastive decoder → word

The limitation of the existing system is the **closed vocabulary** (76 words from two short poems).
The goal of this new work is to replace or augment stage 2 with an LLM-guided decoder
that enables open-vocabulary decoding.

---

## What we already have

### Data
- MEG recordings from 13 subjects (sub-01 through sub-17, with gaps)
- Two conditions per subject: **listened** and **imagined**
- Two poems and two melodies as stimuli — only poems are used for word decoding
- 10 sessions per subject per condition
- MEG downsampled to 100 Hz, 155 channels, trials are ~27 seconds each
- Word-level onset timestamps extracted using WhisperX forced alignment
- For each word onset, a 1-second MEG window is extracted:
  [-200ms, +800ms] around onset → WIN_SIZE = 100 samples
- Vocabulary: 76 unique content words across the two poems

### Existing MEG encoder (contrastive decoder stage)
- Input: (B, C=155, T=100) MEG window
- Spatial conv (pointwise, C → 32)
- 3 dilated temporal conv blocks (kernel sizes 7,5,3 dilation 1,2,4)
- AdaptiveAvgPool → Linear → 128-d L2-normalized embedding
- Trained with NT-Xent contrastive loss against frozen BERT word embeddings
- Checkpoint: `compare_out/models/bert_wav2vec/meg_encoder.pt`

### Existing text encoder (BERT, used in contrastive decoder)
- BERT (bert-base-uncased) is used FROZEN as a feature extractor only
- For each of 76 words: tokenize → last hidden layer → mean pool → 768-d vector
- These 76 vectors are stored as a fixed buffer, never updated
- A trainable 2-layer MLP projection head maps 768-d → 128-d normalized embedding
- BERT's language modeling ability is NOT used — only its static word embeddings

### Existing evaluation
- Rank-based retrieval: MEG embedding vs all 76 word embeddings, cosine similarity
- Metrics: R@1, R@5, R@10, MRR, median rank, rank CDFs
- Evaluated in LOSO fashion (leave-one-subject-out)

---

## New architecture: LLM-guided open-vocabulary decoder

The core idea: instead of retrieving from a fixed 76-word vocabulary via contrastive
similarity, use an LLM to autoregressively generate the target word, conditioned on
MEG-derived "soft tokens" that are injected into the LLM's embedding space via a
small trainable adapter.

### Frozen LLM choice
- Use a small pretrained LLM loaded locally via HuggingFace transformers
- Recommended starting point: GPT-2 (d_model=768) or Phi-3-mini (d_model=2048)
- The LLM weights are COMPLETELY FROZEN throughout all training
- This is critical to prevent the LLM from memorizing the two poems

### Trainable adapter (projection layer)
- A small MLP: input_dim → d_model (LLM embedding dimension)
- input_dim is the MEG encoder's output dimension (128 if reusing existing encoder,
  or higher if using a new/extended encoder)
- This is the ONLY trainable component in the new system (plus optionally the MEG
  encoder if you choose to unfreeze it)
- Suggested architecture: Linear(128, 512) → GELU → Linear(512, d_model)
- Output: one or more soft-token vectors of shape (n_soft, d_model) per MEG window

### MEG encoder (two options)
- **Option A (easiest):** Reuse the existing contrastively-trained MEG encoder,
  frozen. Feed its 128-d output into the adapter.
- **Option B (more expressive):** Train a new MEG encoder from scratch jointly with
  the adapter. Could use the same architecture or a slightly deeper one.
- Recommend starting with Option A (frozen existing encoder) for the first experiment.

---

## Two sequence designs to implement and compare

### Design A — Interleaved (recommended primary method)

For each full poem trial (one subject, one session, one poem), construct one long
sequence by interleaving MEG soft tokens with their corresponding real text tokens:

```
[soft(word1)] [text(word1)] [soft(word2)] [text(word2)] ... [soft(wordN)] [text(wordN)]
```

**Step by step:**
1. For each word in the poem, extract its MEG window (already have onset timestamps)
2. Pass each MEG window through MEG encoder → adapter → soft token vector(s),
   shape (n_soft, d_model)
3. Tokenize each word using the LLM's tokenizer → integer token IDs
4. Look up each text token ID in the LLM's embedding table → text embedding vectors
5. Concatenate in order: [soft_tok(word1), text_emb(word1), soft_tok(word2),
   text_emb(word2), ...] → final tensor shape (seq_len, d_model)
6. Build a loss mask: 0 at soft-token positions, 1 at text-token positions
7. Pass the full embedding tensor into the LLM using `inputs_embeds` (NOT `input_ids`)
8. The LLM outputs a distribution at every position; compute cross-entropy loss
   ONLY at mask=1 positions (text tokens), where the target is the next real token
9. Backpropagate through loss → adapter weights only (LLM and optionally MEG
   encoder are frozen)

**Why this design:**
- At every word prediction, the model sees: the MEG evidence for the current word
  (just inserted as a soft token), plus all previous words' text and MEG evidence
- Forces the model to "check in" with brain evidence at each step
- Much harder to shortcut/memorize compared to Design B
- Directly analogous to how the model would work at inference time

**Inference:**
- At test time, no ground truth text is available
- Start with soft tokens only (all MEG windows for the poem are available offline)
- Autoregressively generate: after each soft token, let the LLM generate the
  text token rather than being given it
- Use greedy decoding or beam search (k=5 suggested)

---

### Design B — Upfront / captioning-style (secondary baseline)

All MEG soft tokens are placed at the beginning of the sequence (like an image
caption model sees the whole image before writing), then the LLM generates the
entire poem text:

```
[soft(word1)] [soft(word2)] ... [soft(wordN)] → [text: "He was dressed all in fur ..."]
```

**Step by step:**
1. For each word, extract MEG window → encoder → adapter → soft token vector(s)
2. Concatenate ALL soft tokens together: shape (N * n_soft, d_model)
3. Tokenize the ENTIRE poem text → text embedding sequence
4. Concatenate: [all soft tokens, all text embeddings]
5. Loss mask: 0 for soft-token positions, 1 for all text-token positions
6. Forward pass through LLM using `inputs_embeds`
7. Compute cross-entropy loss at text positions only, backpropagate to adapter

**Why this design:**
- Simpler to implement
- Mirrors image captioning (all visual evidence upfront, then generate)
- BUT: much harder credit assignment — the connection between a specific word's
  soft token and that word's prediction is indirect, mediated by attention across
  a long sequence
- More vulnerable to the LLM ignoring soft tokens and generating from text prior
  alone (especially dangerous with only 2 short poems)
- Use primarily as an ablation/baseline to compare against Design A

**Inference:**
- Feed all soft tokens, then autoregressively generate the full poem text

---

## Training setup

### Data split
- Train on seen subjects' listened MEG (same LOSO as main paper)
- Validate on held-out sessions of seen subjects
- Test on unseen subjects' listened MEG first, then full pipeline
  (imagined → mapping → decoder) as the final evaluation
- CRITICAL: hold out ENTIRE poems where possible (train on poem1, test on poem2)
  to prevent LLM memorizing the poem sequence — this is more important here than
  in the contrastive decoder because the LLM has much stronger language priors

### Loss
- Standard next-token cross-entropy, masked to text-token positions only
- No contrastive loss needed (though could add as auxiliary to keep soft tokens
  informative — worth ablating)

### Optimizer
- AdamW, LR = 1e-4 (lower than existing work due to frozen LLM)
- Consider linear warmup for first few epochs (transformers are sensitive to this)
- Early stopping on validation loss

### Batch construction
- One training example = one full poem trial (not one word window)
- Effective batch size will be small (maybe 4-8 full trials per batch)
- Given small dataset, expect ~13 subjects × 10 sessions × 2 poems = 260 training
  examples maximum

---

## Critical ablations to run (memorization sanity checks)

These ablations are essential before trusting any positive result, given the
tiny two-poem dataset. Run all of these:

1. **Shuffle ablation:** Permute MEG-to-word pairing randomly, retrain with same
   setup. If shuffled performance ≈ real performance, soft tokens are being ignored.

2. **Random soft token ablation:** At TEST time only, replace real soft tokens with
   random Gaussian noise vectors (same shape). If performance barely drops,
   the LLM is not using the brain signal at inference.

3. **No soft token baseline:** Run the frozen LLM with NO soft tokens at all —
   just the text prompt. This tells you what the LLM can do from language priors
   alone on your specific poems. If this is close to your full model, you have a
   memorization problem.

4. **Cross-poem generalization:** Train on poem1, test on poem2 (and vice versa).
   This is the cleanest test of whether the model is genuinely using MEG for
   open-vocabulary generalization rather than memorizing known sequences.

---

## Evaluation metrics

Since this is now open-vocabulary, rank-based metrics against 76 words no longer
fully apply. Use:

1. **Exact match accuracy** — fraction of words where the top-1 generated token
   matches the ground truth word exactly
2. **BERT similarity** — cosine similarity between the BERT embedding of the
   generated word and the ground truth word (soft correctness credit)
3. **BLEU-1** — unigram overlap between generated and ground truth poem text
4. **Restricted rank metric (sanity check)** — for generated candidates, check
   rank of true word among top-k beam search candidates; allows comparison to
   existing R@k metrics from the main paper
5. **WER (word error rate)** — for full poem generation, standard ASR metric

---

## File structure suggestion

```
llm_decoder/
    config.py              # all hyperparameters
    meg_encoder.py         # reuse or extend existing encoder
    adapter.py             # trainable MLP projection layer
    dataset.py             # poem-level interleaved sequence builder
    model.py               # full model: encoder + adapter + frozen LLM
    train.py               # training loop
    evaluate.py            # metrics + ablations
    inference.py           # autoregressive generation at test time
    design_b.py            # upfront/captioning-style variant
```

---

## Key references to cite

- Tang et al. 2023 (Nature Neuroscience) — semantic reconstruction from fMRI using
  encoding-model-guided LLM decoding (most relevant prior work)
- Défossez et al. 2023 (Nature Machine Intelligence) — contrastive MEG/EEG decoding
  (your existing decoder is closely related)
- NeuroLM — EEG tokenizer + LLM instruction tuning (closest to your new approach)
- BrainDEC — multimodal LLM for non-invasive brain-to-text decoding, explicitly
  compares interleaved vs captioning-style designs (directly supports Design A)
- Flamingo / CLIP+GPT literature — general soft-prompt/adapter conditioning approach
  that your architecture is based on