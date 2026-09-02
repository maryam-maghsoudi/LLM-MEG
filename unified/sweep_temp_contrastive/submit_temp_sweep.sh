#!/bin/bash
# Temperature sweep for Method 1 (contrastive InfoNCE), full LOSO.
#
# For each temperature × heldout subject, one Slurm job is submitted.
# Results are written to:
#   unified/sweep_temp_contrastive/temp_{T}/inference/bert_base_uncased/loso_{sub}/
#
# Usage:
#   bash unified/sweep_temp_contrastive/submit_temp_sweep.sh
#
# Override temperatures:
#   TEMPS="0.07 0.10 0.20" bash unified/sweep_temp_contrastive/submit_temp_sweep.sh

set -e

TEMPS=${TEMPS:-"0.03 0.05 0.07 0.10 0.15 0.20 0.30"}

SUBJECTS=(
  sub-01 sub-03 sub-04 sub-05 sub-06 sub-09
  sub-10 sub-11 sub-12 sub-13 sub-14 sub-16 sub-17
)

WORKDIR=/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/llm_decoder
SWEEP_ROOT=$WORKDIR/unified/sweep_temp_contrastive

EXCLUDE=legacygpu00,legacygpu02,legacygpu03,legacygpu04,legacygpu05,legacygpu06,legacygpu07,legacygpu09,legacygpu10,legacygpu11,legacygpu12,legacygpu13,legacygpu14,legacygpu18,legacygpu19,legacygpu20,legacygpu21,legacygpu22,legacygpu23,legacygpu24,legacygpu25,legacygpu26,legacygpu27,legacygpu28,legacygpu29,legacygpu30,legacygpu31,legacygpu32,legacygpu33,legacygpu34,legacygpu35,legacygpu36,legacygpu37,legacygpu38,legacygpu39,legacygpu40,legacygpu41,legacygpu42

SBATCH_BASE=(
  --partition=scavenger --account=scavenger --qos=scavenger
  --gres=gpu:rtxa5000:1 --cpus-per-task=4 --mem=32G --time=6:00:00
  "--exclude=$EXCLUDE"
)

read -ra TEMP_ARR <<< "$TEMPS"
total=$(( ${#TEMP_ARR[@]} * ${#SUBJECTS[@]} ))
echo "Temperatures : ${TEMP_ARR[*]}"
echo "Subjects     : ${#SUBJECTS[@]}"
echo "Total jobs   : $total"
echo ""

for TEMP in "${TEMP_ARR[@]}"; do
  # Format tag: replace "." with "p" for filesystem safety (0.10 -> 0p10)
  TEMP_TAG="temp_${TEMP//./_}"
  OUT_ROOT="$SWEEP_ROOT/$TEMP_TAG"
  LOGDIR="$SWEEP_ROOT/slurm_logs/$TEMP_TAG"
  mkdir -p "$LOGDIR"

  for SUBJ in "${SUBJECTS[@]}"; do
    jid=$(sbatch "${SBATCH_BASE[@]}" \
      --job-name="sweep_T${TEMP}_${SUBJ}" \
      --output="$LOGDIR/%j_${SUBJ}.out" \
      --error="$LOGDIR/%j_${SUBJ}.err" \
      --parsable \
      --wrap="cd $WORKDIR && python -m unified.train \
        --method inference \
        --eval_scheme loso \
        --heldout $SUBJ \
        --temperature $TEMP \
        --out_root $OUT_ROOT")
    echo "  T=$TEMP  $SUBJ → job $jid"
  done
done

echo ""
echo "All $total jobs submitted."
echo "Results → $SWEEP_ROOT/temp_*/"
