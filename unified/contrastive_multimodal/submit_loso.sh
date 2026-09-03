#!/bin/bash
# Submit Stage 1 LOSO training for all 13 subjects (new_contrastive pipeline).
#
# Usage:
#   bash submit_loso.sh [anneal_mode] [pooling_mode]
#
# Examples:
#   bash submit_loso.sh joint_annealed exact
#   bash submit_loso.sh llm_only exact
#   bash submit_loso.sh joint_annealed wide

set -e

ANNEAL_MODE=${1:-joint_annealed}
POOLING_MODE=${2:-exact}

SUBJECTS=(
  sub-01 sub-03 sub-04 sub-05 sub-06 sub-09
  sub-10 sub-11 sub-12 sub-13 sub-14 sub-16 sub-17
)

WORKDIR=/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/llm_decoder/unified/new_contrastive
LOGDIR=$WORKDIR/slurm_logs/loso_${ANNEAL_MODE}_${POOLING_MODE}
mkdir -p "$LOGDIR"

EXCLUDE=legacygpu00,legacygpu02,legacygpu03,legacygpu04,legacygpu05,legacygpu06,legacygpu07,legacygpu09,legacygpu10,legacygpu11,legacygpu12,legacygpu13,legacygpu14,legacygpu18,legacygpu19,legacygpu20,legacygpu21,legacygpu22,legacygpu23,legacygpu24,legacygpu25,legacygpu26,legacygpu27,legacygpu28,legacygpu29,legacygpu30,legacygpu31,legacygpu32,legacygpu33,legacygpu34,legacygpu35,legacygpu36,legacygpu37,legacygpu38,legacygpu39,legacygpu40,legacygpu41,legacygpu42

SBATCH_BASE=(
  --partition=scavenger --account=scavenger --qos=scavenger
  --gres=gpu:rtxa5000:1 --cpus-per-task=4 --mem=32G --time=6:00:00
  "--exclude=$EXCLUDE"
  "--output=$LOGDIR/%j_%x.out"
  "--error=$LOGDIR/%j_%x.err"
)

echo "anneal_mode=$ANNEAL_MODE  pooling_mode=$POOLING_MODE"
echo "Submitting ${#SUBJECTS[@]} LOSO jobs ..."

for SUBJ in "${SUBJECTS[@]}"; do
  jid=$(sbatch "${SBATCH_BASE[@]}" \
    --job-name="nc_loso_${SUBJ}_${ANNEAL_MODE}_${POOLING_MODE}" \
    --parsable \
    --wrap="cd $WORKDIR && python train.py \
      --heldout_subject $SUBJ \
      --anneal_mode $ANNEAL_MODE \
      --pooling_mode $POOLING_MODE \
      --out_dir $WORKDIR/checkpoints/${ANNEAL_MODE}_${POOLING_MODE}")
  echo "  $SUBJ → job $jid"
done

echo "Done. Logs: $LOGDIR"
