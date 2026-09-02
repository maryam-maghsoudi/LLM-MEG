#!/bin/bash
# Post-training analysis for temp_0_5 controls.
# Submitted with --dependency=afterok on all 26 control jobs.
#
# Steps:
#   1. Evaluate shuffle_time and zero controls (13 subjects each)
#   2. compare_controls.py  (real + both controls → Wilcoxon tests)
#   3. visualize_predictions.py
#   4. visualize_predictions_topk.py

set -e

WORKDIR=/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/llm_decoder
cd "$WORKDIR"

SWEEP=unified/sweep_temp_contrastive/temp_0_5/inference/bert_base_uncased
OUT=unified/method1_analysis/temp_0_5

SUBJECTS=(
  sub-01 sub-03 sub-04 sub-05 sub-06 sub-09
  sub-10 sub-11 sub-12 sub-13 sub-14 sub-16 sub-17
)

echo "============================================================"
echo "  Step 1 — Evaluate controls"
echo "============================================================"
for CONTROL in shuffle_time zero; do
  for SUB in "${SUBJECTS[@]}"; do
    CKPT=$SWEEP/loso_${SUB}_ctrl_${CONTROL}
    echo "  Evaluating $SUB  ctrl=$CONTROL ..."
    python -m unified.evaluate \
      --method inference --eval_scheme loso --heldout "$SUB" \
      --ckpt_dir "$CKPT" \
      --device cpu
  done
done

echo ""
echo "============================================================"
echo "  Step 2 — compare_controls"
echo "============================================================"
python -m unified.method1_analysis.compare_controls \
  --eval_root "$SWEEP" \
  --out_dir   "$OUT"

echo ""
echo "============================================================"
echo "  Step 3 — visualize_predictions"
echo "============================================================"
python -m unified.method1_analysis.visualize_predictions \
  --ckpt_root "$SWEEP" \
  --out_dir   "$OUT/predictions" \
  --device cpu

echo ""
echo "============================================================"
echo "  Step 4 — visualize_predictions_topk"
echo "============================================================"
python -m unified.method1_analysis.visualize_predictions_topk \
  --ckpt_root "$SWEEP" \
  --out_dir   "$OUT/predictions_topk" \
  --device cpu

echo ""
echo "All done. Results → $WORKDIR/$OUT"
