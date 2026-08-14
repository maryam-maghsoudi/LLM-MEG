#!/bin/bash
# Submit LOSO cross-validation for a given method (default: twostage).
#
# Usage:
#   bash submit_loso.sh [method] [control]
#
# Examples:
#   bash submit_loso.sh inference none
#   bash submit_loso.sh twostage none
#   bash submit_loso.sh twostage shuffle_time
#   bash submit_loso.sh interleaved none
#
# For method=interleaved, set MEG_ENC_CKPT to the Method 1 meg_encoder_best.pt.
#   MEG_ENC_CKPT=/path/to/meg_encoder_best.pt bash submit_loso.sh interleaved none

set -e

METHOD=${1:-twostage}
CONTROL=${2:-none}

SUBJECTS=(
  sub-01 sub-03 sub-04 sub-05 sub-06 sub-09
  sub-10 sub-11 sub-12 sub-13 sub-14 sub-16 sub-17
)

WORKDIR=/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/llm_decoder
UNIFIED=$WORKDIR/unified

EXCLUDE=legacygpu00,legacygpu02,legacygpu03,legacygpu04,legacygpu05,legacygpu06,legacygpu07,legacygpu09,legacygpu10,legacygpu11,legacygpu12,legacygpu13,legacygpu14,legacygpu18,legacygpu19,legacygpu20,legacygpu21,legacygpu22,legacygpu23,legacygpu24,legacygpu25,legacygpu26,legacygpu27,legacygpu28,legacygpu29,legacygpu30,legacygpu31,legacygpu32,legacygpu33,legacygpu34,legacygpu35,legacygpu36,legacygpu37,legacygpu38,legacygpu39,legacygpu40,legacygpu41,legacygpu42

LOGDIR=$UNIFIED/slurm_logs/loso_${METHOD}_ctrl_${CONTROL}
mkdir -p "$LOGDIR"

SBATCH_BASE=(
  --partition=scavenger --account=scavenger --qos=scavenger
  --gres=gpu:rtxa5000:1 --cpus-per-task=4 --mem=32G --time=6:00:00
  "--exclude=$EXCLUDE"
  "--output=$LOGDIR/%j_%x.out"
  "--error=$LOGDIR/%j_%x.err"
)

# Build extra args string
EXTRA_ARGS="--method $METHOD --eval_scheme loso --control $CONTROL"

if [[ "$METHOD" == "interleaved" ]]; then
  if [[ -z "$MEG_ENC_CKPT" ]]; then
    echo "ERROR: set MEG_ENC_CKPT for method=interleaved"
    exit 1
  fi
  EXTRA_ARGS="$EXTRA_ARGS --meg_enc_ckpt $MEG_ENC_CKPT"
fi

echo "Method=$METHOD  control=$CONTROL"
echo "Submitting ${#SUBJECTS[@]} LOSO jobs ..."

for SUBJ in "${SUBJECTS[@]}"; do
  jid=$(sbatch "${SBATCH_BASE[@]}" \
    --job-name="loso_${METHOD}_${SUBJ}_${CONTROL}" \
    --parsable \
    --wrap="cd $WORKDIR && python -m unified.train $EXTRA_ARGS --heldout $SUBJ")
  echo "  $SUBJ → job $jid"
done

echo "Done."
