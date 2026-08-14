#!/bin/bash
# Submit full LOSO: real training + shuffle_time control for all 13 subjects.
# sub-01 real training is already done; only shuffle_time is submitted for it.

set -e

SUBJECTS=(sub-03 sub-04 sub-05 sub-06 sub-09 sub-10 sub-11 sub-12 sub-13 sub-14 sub-16 sub-17)
WORKDIR=/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/llm_decoder/llm_twostage
EXCLUDE=legacygpu00,legacygpu02,legacygpu03,legacygpu04,legacygpu05,legacygpu06,legacygpu07,legacygpu09,legacygpu10,legacygpu11,legacygpu12,legacygpu13,legacygpu14,legacygpu18,legacygpu19,legacygpu20,legacygpu21,legacygpu22,legacygpu23,legacygpu24,legacygpu25,legacygpu26,legacygpu27,legacygpu28,legacygpu29,legacygpu30,legacygpu31,legacygpu32,legacygpu33,legacygpu34,legacygpu35,legacygpu36,legacygpu37,legacygpu38,legacygpu39,legacygpu40,legacygpu41,legacygpu42

mkdir -p "$WORKDIR/slurm_logs"

SBATCH_BASE=(
  --partition=scavenger --account=scavenger --qos=scavenger
  --gres=gpu:rtxa5000:1 --cpus-per-task=4 --mem=32G --time=4:00:00
  "--exclude=$EXCLUDE"
  "--output=$WORKDIR/slurm_logs/%j_%x.out"
  "--error=$WORKDIR/slurm_logs/%j_%x.err"
)

# ── sub-01: real already done; submit shuffle_time only ──────────────────────
jid=$(sbatch "${SBATCH_BASE[@]}" \
  --job-name=loso_sub-01_shtime \
  --parsable \
  --wrap="cd $WORKDIR && python train.py --heldout sub-01 --z_control shuffle_time --load_stage1 out/HuggingFaceTB_SmolLM2-360M/sub-01/stage1_best.pt")
echo "sub-01  shuffle_time → job $jid"

# ── remaining 12 subjects: real + shuffle_time (chained) ────────────────────
for SUBJ in "${SUBJECTS[@]}"; do
  real_jid=$(sbatch "${SBATCH_BASE[@]}" \
    --job-name="loso_${SUBJ}_real" \
    --parsable \
    --wrap="cd $WORKDIR && python train.py --heldout $SUBJ")
  echo "$SUBJ  real        → job $real_jid"

  shtime_jid=$(sbatch "${SBATCH_BASE[@]}" \
    --job-name="loso_${SUBJ}_shtime" \
    "--dependency=afterok:$real_jid" \
    --parsable \
    --wrap="cd $WORKDIR && python train.py --heldout $SUBJ --z_control shuffle_time --load_stage1 out/HuggingFaceTB_SmolLM2-360M/${SUBJ}/stage1_best.pt")
  echo "$SUBJ  shuffle_time → job $shtime_jid (after real $real_jid)"
done
