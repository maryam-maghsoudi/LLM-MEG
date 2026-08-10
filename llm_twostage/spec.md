# MEG-to-LLM two-stage decoding — implementation spec

## Context

This extends the existing `contrastive_word_meg.py` pipeline (contrastive MEG word
decoder trained against frozen BERT/GloVe word embeddings, evaluated via
rank-k accuracy against a vocabulary bank). That pipeline is the basis for a
submission on zero-shot imagined speech decoding from MEG (three-stage
pipeline: imagined-to-listened MEG mapping, contrastive word decoder trained on
listened MEG, zero-shot composition, evaluated leave-one-subject-out).

This spec describes a redesign of the contrastive word decoder: replacing the
frozen BERT/GloVe vocabulary-level embeddings with hidden states from a frozen
causal LLM (candidates: SmolLM, Qwen), and adding a second training stage that
distills the LLM's own next-word prediction behavior into a MEG-driven causal
head.

Dataset: 157-channel MEG from trained musicians, 8 conditions (4 stimuli x
listen/imagine), two Bach chorales and two poem excerpts — **this spec focuses
on the two poem stimuli** (`poem1`, `poem2`), 13 subjects, 10 sessions each,
76-word (unique) vocabulary (poem-specific, will change once LLM hidden states replace
the closed vocab lookup — see "Open vocabulary" note below).

---

## Stage 1 — contrastive alignment (replaces current `TextEncoder`)

```
LISTENED MEG  -> window [-200ms, +800ms] around word onset (100 samples @ 100Hz)
              -> MEGWordEncoder (existing CNN, small/full variant)
              -> z_t (128-d, L2-normalized)
                              |
                        InfoNCE (see below)
                              |
word sequence -> frozen causal LLM -> hidden state at a chosen MIDDLE layer
              -> hmid_t (one per word OCCURRENCE, not per word type)
```

Key differences from current code:

1. **Contextual, occurrence-level targets, not vocabulary-level.** Current
   `MEGWordDataset` keys pairs by `word_str` -> shared vocab index, so every
   occurrence of a word shares one BERT embedding. With `hmid_t` from a causal
   LLM, each occurrence has its own embedding (depends on `w_1...w_t`, not on
   future words, since the LLM is causal). Dataset must key on
   `(poem, subject, session, occurrence_id)`, not word string.
2. **Word -> token alignment.** LLM tokenizer does subword splitting. Build a
   word -> token-span map per stimulus by tokenizing the full poem text once
   and aligning to the word-onset list already produced by the onset pipeline
   (`onset_out/{poem}_word_onsets.json`). Pool token hidden states within a
   word's span (mean pooling, or last-subtoken) to get one `hmid_t` per word
   occurrence.
3. **Precompute and cache `hmid_t` offline**, once, for both poems, since the
   LLM and text never change during training. Same caching principle already
   used for BERT embeddings, just now sequence-level (per poem, per position)
   instead of vocab-level (per word type).
4. **Loss: InfoNCE**, not NT-Xent/symmetric loss. Single direction, MEG as
   query (matches the eval-time direction: MEG window -> rank against text
   bank):

   ```python
   def info_nce_loss(z_meg, z_text, temperature=TEMPERATURE):
       N = z_meg.shape[0]
       sim = z_meg @ z_text.T / temperature      # (N, N), MEG is query
       labels = torch.arange(N, device=z_meg.device)
       return F.cross_entropy(sim, labels)       # single direction only
   ```

   Note for Claude CLI: this drops the text->MEG direction that the current
   `nt_xent_loss` includes. Flagged as a design tradeoff (symmetric loss
   usually gives a slightly richer embedding space) but the person has chosen
   single-direction InfoNCE for this version. Keep the old `nt_xent_loss`
   function available (renamed, e.g. `nt_xent_loss_symmetric`) behind a config
   flag in case we want to A/B this later.

5. **Middle layer choice for `hmid_t` (`HMID_LAYER`) is a hyperparameter, not
   a guess.** Decide it via:
   - Cheap pass: linear/k-NN probe of word-occurrence identity from each
     candidate layer's hidden state (no MEG involved), sweep every ~4th layer,
     pick the layer(s) where word identity is most linearly separable.
     Middle-third-of-network layers are the prior, not a rule.
   - Confirm with 1-2 short Stage 1 training runs on the top candidate
     layers, pick by validation R@1/MRR on the actual MEG-to-text ranking
     task (the metric that matters), not by probe accuracy alone.
   - Cache activations for multiple candidate layers in the same offline LLM
     forward pass so candidates don't require separate LLM re-runs.

6. Training loop, dataset splitting (subjects/sessions), optimizer, and
   evaluation (`evaluate_ranking`) otherwise follow the existing code
   structure in `contrastive_word_meg.py`.

---

## Stage 2 — KL next-word distillation

```
z_1 ... z_T  (from FROZEN Stage 1 MEG encoder, eval() mode, no augmentation noise)
    -> Causal head (GRU), reads z_1...z_t causally
    -> y_1 ... y_T  (predicted next-word state)
    -> projection into LLM hidden dim (new learned layer, GRU hidden size
       almost certainly != LLM hidden size)
    -> q_t = lm_head(y_t)   [frozen LLM's own lm_head, closed/restricted vocab
                              — see "vocab scope" below]

word sequence -> frozen LLM, FINAL layer (pre-lm_head) -> hf_1 ... hf_T (true next-word state)
    -> p_t = lm_head(hf_t)  [teacher distribution]

Loss: KL(p_t || q_t), averaged over t = 1...T-1 (position T has no word T+1 to
      predict against)
```

Implementation details:

1. **GRU is a standard causal recurrence** over `z_t`:
   `h_t = GRU(z_t, h_{t-1})`, `h_0` = zero vector (or small learned param),
   reset at the start of every trial (poem recitation) — never carried across
   trials.
2. **`y_t` is a linear projection of `h_t`** into the LLM's hidden size before
   going through `lm_head`. This projection is a new trainable layer.
3. **Precompute `p_t` offline**, same caching pass as `hmid_t` (frozen LLM,
   frozen text, final layer this time instead of middle).
4. **Vocab scope for `p_t`/`q_t`: restrict to the closed poem/stimulus
   vocabulary**, not the full LLM vocab. Mask/renormalize the `lm_head`
   output to just the words that actually appear across both poems' stimulus
   sets. Full-vocab KL would be dominated by noise given the data volume (13
   subjects x 2 poems x 10 sessions).
5. **Data structure: full sequences in original word order, not shuffled
   windows.** New dataset class needed:
   ```python
   class MEGSequenceDataset(Dataset):
       # one item = one (subject, session, poem) trial
       # returns: meg_windows (T, C, WIN_SIZE) in word order, word_ids (T,)
   ```
   Batching across trials of different length `T` needs padding + a mask
   (poems have different word counts; sessions may drop words near edges from
   the windowing logic already in `onset_to_window_raw` /
   `onset_to_window_flash_removed`).
6. **Stage 1 encoder is frozen during Stage 2** (`requires_grad_(False)`,
   `.eval()` mode, no augmentation noise). New optimizer scoped only to
   `GRU.parameters()` + projection layer parameters — never combined with the
   Stage 1 optimizer.
7. Teacher forcing: the GRU reads `z_1...z_t` from MEG ground truth at every
   step, not its own previous predictions — avoids compounding errors during
   training. Note: this means Stage 2 as trained is not evaluated as a
   free-running generator; if free-running generation from imagined MEG is
   ever in scope, a decode-time procedure (greedy/beam search over `q_t`)
   would need to be added separately — out of scope for this spec.

---

## Epoch schedule

- `N_EPOCHS_STAGE1` is a **cap, not a fixed target** (previous "10" was a
  placeholder, not a real value).
- **Primary switch mechanism: early stopping already in the existing `train()`
  function**, tracking `PATIENCE` on Stage 1 validation InfoNCE loss. Stage 1
  runs until early stopping triggers (or the cap is hit); that epoch is the
  Stage 1 -> Stage 2 switch point.
- **Secondary: sweep `PATIENCE` and `N_EPOCHS_STAGE1` cap as hyperparameters**
  (e.g. cap in `{10, 20, 40}`, patience scaled accordingly), and compare by
  **Stage 2 final metrics**, not just Stage 1 metrics — Stage 1's only job is
  to serve Stage 2.
- After the switch: freeze the Stage 1 encoder, instantiate the Stage 2
  optimizer, continue the epoch counter (i.e. Stage 2 "epoch 11" in the
  original discussion, but really "epoch (Stage-1-stop-epoch + 1)").

---

## Train / val / test split

Two poems total, so poem cannot be the primary split axis (would lose
vocabulary diversity and confound "new stimulus" with "different
topic/words"). **Subject is the primary split axis (LOSO)**, matching the
existing NeurIPS benchmark's `--heldout_subject` structure. Both poems are
always included together in every split.

- **Test:** one held-out subject, all sessions, both poems. Never touched
  until final evaluation.
- **Train + val:** remaining 12 subjects. Val is carved out **by session**,
  not by shuffled word windows — e.g. sessions 0-7 -> train, sessions 8-9 ->
  val, for each of the 12 training subjects, both poems included.
  - Rationale: words within a session share session-level noise/drift.
    Validating on words from a session already seen in training would
    partly measure memorization of that session's noise rather than true
    generalization. Splitting by session tests generalization across time
    within a subject — closer to what the true (held-out-subject) test
    requires.
- **Repeat LOSO across all 13 subjects** for the final reported metric (mean
  +/- spread across folds), same structure as the existing img->lis benchmark
  models use.
- This split logic applies to **both Stage 1 and Stage 2** training (Stage 2
  uses `MEGSequenceDataset` over the same subject/session partition).

### Open vocabulary note

Because Stage 1/Stage 2 targets (`hmid_t`, `p_t`) come from a frozen LLM's
continuous representation space rather than a closed lookup table over the
training vocabulary, the model is not fundamentally restricted to words seen
during Stage 1 training — evaluation could in principle include word
occurrences that only appear in the held-out subject's data. This is a
genuine capability difference from the current BERT-vocab-index setup and
worth preserving/testing explicitly, not just an incidental side effect.

---

## Implementation checklist (for Claude CLI)

1. Offline caching script: word/token alignment per poem, run frozen LLM once
   per poem, extract and cache `hmid_t` at multiple candidate middle layers,
   and `hf_t` (final layer) -> `p_t` at the restricted closed vocab.
2. Layer-selection probe script (linear/k-NN probe over candidate layers).
3. Modify `MEGWordDataset` (or add a variant) to key on occurrence id and load
   cached `hmid_t` instead of BERT vocab embeddings.
4. Add `info_nce_loss` (single-direction), keep old symmetric loss available
   behind a flag.
5. Add `MEGSequenceDataset` (whole-trial, word-order, padded batches + mask).
6. Add GRU causal head + projection module for Stage 2.
7. Add Stage 2 training loop: frozen encoder forward (no grad, eval mode) ->
   GRU -> projection -> masked `lm_head` -> `KL(p_t || q_t)` averaged over
   `t=1..T-1`, masked appropriately for padding.
8. Wire epoch-cap + early-stopping switch between Stage 1 and Stage 2 in
   `main()`, with two separate optimizers that are never active together.
9. Implement LOSO subject split with session-based val carve-out, shared
   across Stage 1 and Stage 2 dataset construction.
10. Extend evaluation: keep `evaluate_ranking` for Stage 1; add a Stage 2
    metric (e.g. mean KL and/or next-word top-1 accuracy) on held-out trials.