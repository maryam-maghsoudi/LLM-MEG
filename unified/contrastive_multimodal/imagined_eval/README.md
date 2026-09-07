# imagined_eval

Evaluate the Stage 1 contrastive word decoder on **imagined** MEG, by first
mapping imagined → predicted-listened with the trained img→lis benchmark
models, then feeding the predicted-listened signal into the frozen decoder.

This is a true zero-shot LOSO chain: for held-out subject *S*, both the
img→lis mapping model **and** the Stage 1 decoder were trained without ever
seeing *S*.

```
imagined MEG (icaed/*_img_meg-epo.fif)
   │  mean-epochs → downsample ×10 → z-score          [predict_and_save.py]
   ▼
img→lis mapping model  (benchmark/no_flash_removal/loso_out/models/…)
   │  full-trial forward → predicted-listened → re-z-score
   ▼
predicted_npy/{mapping_key}/{subj}/ses-{s}/meg/{subj}_sess-{s}_task-{poem}lis.npy
   │  (155, 2700) float32, z-scored — icaed_Sai format
   ▼
frozen Stage 1 decoder  (checkpoints/joint_annealed_exact/…)   [eval_imagined.py]
   │  encoder → exact pooling → word_head → cosine vs h_mid bank
   ▼
top-1 / top-5 word retrieval  →  results/{mapping_key}/summary.json
```

## Why it works with zero changes to the decoder

`new_dataset.MEGContinuousTrialDataset` has a fast path that loads
`{meg_base}/{subj}/ses-{s}/meg/{subj}_sess-{s}_task-{poem}lis.npy` directly if
it exists. We write predicted-listened trials under exactly that name and
point `meg_base` at `predicted_npy/{mapping_key}`, so the predicted signal
transparently stands in for the real listened trial the decoder expects.

We use the **no-flash-removal** mapping models on purpose: the decoder was
trained on full (non-flash-removed) listened MEG, so those models' output is
already in the same time base and preprocessing (no flash-index
reconstruction needed).

**Design choice:** the mapping model's raw prediction is re-z-scored per
channel before saving, so the file matches the exactly-z-scored distribution
(`icaed_Sai`, mean≈0/std≈1 per channel) the encoder was trained on.

## Run

```bash
# STAGE A — map imagined → predicted-listened, save as .npy  (slow: loads .fif)
python predict_and_save.py --mapping_key RNN_full            # all 13 subjects
python predict_and_save.py --mapping_key CNN1D_full
python predict_and_save.py --mapping_key LinearLag_full      # ridge baseline

# STAGE B — feed predicted-listened into the frozen decoder  (fast, re-runnable)
python eval_imagined.py --mapping_key RNN_full
python eval_imagined.py --mapping_key RNN_full --save_traces # + per-subject CSV/xlsx
```

Both accept `--subjects sub-01 sub-03 …` to run a subset. `mapping_key` must
match between the two stages (it selects the `predicted_npy/{mapping_key}`
subfolder). Available mapping keys: `RNN_full`, `CNN1D_full`, `TCN_full`,
`UNet1D_full`, `ShallowMLP_full`, `LinearLag_full` (and the `_windowed`
variants).

## Outputs

- `predicted_npy/{mapping_key}/…` — predicted-listened trials (Stage A)
- `results/{mapping_key}/summary.json` — per-subject + aggregate top-1/top-5
  vs chance
- `results/{mapping_key}/trace_{subject}.{csv,xlsx}` — with `--save_traces`
