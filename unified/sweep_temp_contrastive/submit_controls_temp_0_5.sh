#!/bin/bash
# Submit shuffle_time and zero MEG control training jobs for temp_0_5.
#
# Outputs land in:
#   sweep_temp_contrastive/temp_0_5/inference/bert_base_uncased/loso_{sub}_ctrl_{control}/
#
# Usage (from llm_decoder/):
#   bash unified/sweep_temp_contrastive/submit_controls_temp_0_5.sh

set -e

TEMP=0.5
TEMP_TAG=temp_0_5
CONTROLS=(shuffle_time zero)

SUBJECTS=(
  sub-01 sub-03 sub-04 sub-05 sub-06 sub-09
  sub-10 sub-11 sub-12 sub-13 sub-14 sub-16 sub-17
)

WORKDIR=/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/llm_decoder
OUT_ROOT=$WORKDIR/unified/sweep_temp_contrastive/$TEMP_TAG
LOGDIR=$WORKDIR/unified/sweep_temp_contrastive/slurm_logs/$TEMP_TAG
mkdir -p "$LOGDIR"

EXCLUDE=legacygpu00,legacygpu02,legacygpu03,legacygpu04,legacygpu05,legacygpu06,legacygpu07,legacygpu09,legacygpu10,legacygpu11,legacygpu12,legacygpu13,legacygpu14,legacygpu18,legacygpu19,legacygpu20,legacygpu21,legacygpu22,legacygpu23,legacygpu24,legacygpu25,legacygpu26,legacygpu27,legacygpu28,legacygpu29,legacygpu30,legacygpu31,legacygpu32,legacygpu33,legacygpu34,legacygpu35,legacygpu36,legacygpu37,legacygpu38,legacygpu39,legacygpu40,legacygpu41,legacygpu42

SBATCH_BASE=(
  --partition=scavenger --account=scavenger --qos=scavenger
  --gres=gpu:rtxa5000:1 --cpus-per-task=4 --mem=32G --time=6:00:00
  "--exclude=$EXCLUDE"
)

total=$(( ${#CONTROLS[@]} * ${#SUBJECTS[@]} ))
echo "Temperature  : $TEMP  ($TEMP_TAG)"
echo "Controls     : ${CONTROLS[*]}"
echo "Subjects     : ${#SUBJECTS[@]}"
echo "Total jobs   : $total"
echo "Out root     : $OUT_ROOT"
echo ""

for CONTROL in "${CONTROLS[@]}"; do
  for SUBJ in "${SUBJECTS[@]}"; do
    jid=$(sbatch "${SBATCH_BASE[@]}" \
      --job-name="ctrl_T${TEMP}_${SUBJ}_${CONTROL}" \
      --output="$LOGDIR/%j_${SUBJ}_ctrl_${CONTROL}.out" \
      --error="$LOGDIR/%j_${SUBJ}_ctrl_${CONTROL}.err" \
      --parsable \
      --wrap="cd $WORKDIR && python -m unified.train \
        --method inference \
        --eval_scheme loso \
        --heldout $SUBJ \
        --temperature $TEMP \
        --control $CONTROL \
        --out_root $OUT_ROOT")
    echo "  $CONTROL  $SUBJ → job $jid"
  done
done

echo ""
echo "All $total jobs submitted."
echo "Results → $OUT_ROOT/inference/bert_base_uncased/loso_*_ctrl_*/"
