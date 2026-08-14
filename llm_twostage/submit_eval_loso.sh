#!/bin/bash
# After all LOSO training jobs complete, run evaluate.py for every subject.
# Submits: (a) real eval, (b) shuffle_time eval using the ctrl ckpt_dir.

set -e

ALL_SUBJECTS=(sub-01 sub-03 sub-04 sub-05 sub-06 sub-09 sub-10 sub-11 sub-12 sub-13 sub-14 sub-16 sub-17)
WORKDIR=/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/llm_decoder/llm_twostage
EXCLUDE=legacygpu00,legacygpu02,legacygpu03,legacygpu04,legacygpu05,legacygpu06,legacygpu07,legacygpu09,legacygpu10,legacygpu11,legacygpu12,legacygpu13,legacygpu14,legacygpu18,legacygpu19,legacygpu20,legacygpu21,legacygpu22,legacygpu23,legacygpu24,legacygpu25,legacygpu26,legacygpu27,legacygpu28,legacygpu29,legacygpu30,legacygpu31,legacygpu32,legacygpu33,legacygpu34,legacygpu35,legacygpu36,legacygpu37,legacygpu38,legacygpu39,legacygpu40,legacygpu41,legacygpu42

mkdir -p "$WORKDIR/slurm_logs"

SBATCH_BASE=(
  --partition=scavenger --account=scavenger --qos=scavenger
  --gres=gpu:rtxa5000:1 --cpus-per-task=4 --mem=32G --time=1:00:00
  "--exclude=$EXCLUDE"
  "--output=$WORKDIR/slurm_logs/%j_%x.out"
  "--error=$WORKDIR/slurm_logs/%j_%x.err"
)

for SUBJ in "${ALL_SUBJECTS[@]}"; do
  TAG=HuggingFaceTB_SmolLM2-360M

  # Real eval
  real_jid=$(sbatch "${SBATCH_BASE[@]}" \
    --job-name="eval_${SUBJ}_real" \
    --parsable \
    --wrap="cd $WORKDIR && python evaluate.py --heldout $SUBJ")
  echo "$SUBJ  eval real        → job $real_jid"

  # shuffle_time eval
  shtime_jid=$(sbatch "${SBATCH_BASE[@]}" \
    --job-name="eval_${SUBJ}_shtime" \
    --parsable \
    --wrap="cd $WORKDIR && python evaluate.py --heldout $SUBJ --ckpt_dir out/${TAG}/${SUBJ}_ctrl_shuffle_time")
  echo "$SUBJ  eval shuffle_time → job $shtime_jid"

  # zero eval
  zero_jid=$(sbatch "${SBATCH_BASE[@]}" \
    --job-name="eval_${SUBJ}_zero" \
    --parsable \
    --wrap="cd $WORKDIR && python evaluate.py --heldout $SUBJ --ckpt_dir out/${TAG}/${SUBJ}_ctrl_zero")
  echo "$SUBJ  eval zero        → job $zero_jid"
done
