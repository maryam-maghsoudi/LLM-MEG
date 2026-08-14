#!/bin/bash
# Submit heldout-stimulus evaluation for a given method.
#
# A single training run with the last N lines of each poem held out for testing.
# Run with both 2 and 4 heldout lines to measure generalization curve.
#
# Usage:
#   bash submit_stimulus.sh [method] [control] [n_lines]
#
# Examples:
#   bash submit_stimulus.sh twostage none 2
#   bash submit_stimulus.sh twostage none 4
#   bash submit_stimulus.sh twostage shuffle_time 2
#   bash submit_stimulus.sh interleaved none 2
#
# For method=interleaved, set MEG_ENC_CKPT first.

set -e

METHOD=${1:-twostage}
CONTROL=${2:-none}
N_LINES=${3:-2}

WORKDIR=/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/llm_decoder
UNIFIED=$WORKDIR/unified

EXCLUDE=legacygpu00,legacygpu02,legacygpu03,legacygpu04,legacygpu05,legacygpu06,legacygpu07,legacygpu09,legacygpu10,legacygpu11,legacygpu12,legacygpu13,legacygpu14,legacygpu18,legacygpu19,legacygpu20,legacygpu21,legacygpu22,legacygpu23,legacygpu24,legacygpu25,legacygpu26,legacygpu27,legacygpu28,legacygpu29,legacygpu30,legacygpu31,legacygpu32,legacygpu33,legacygpu34,legacygpu35,legacygpu36,legacygpu37,legacygpu38,legacygpu39,legacygpu40,legacygpu41,legacygpu42

LOGDIR=$UNIFIED/slurm_logs/stimulus_${METHOD}_ctrl_${CONTROL}_lines${N_LINES}
mkdir -p "$LOGDIR"

SBATCH_BASE=(
  --partition=scavenger --account=scavenger --qos=scavenger
  --gres=gpu:rtxa5000:1 --cpus-per-task=4 --mem=32G --time=6:00:00
  "--exclude=$EXCLUDE"
  "--output=$LOGDIR/%j_%x.out"
  "--error=$LOGDIR/%j_%x.err"
)

EXTRA_ARGS="--method $METHOD --eval_scheme stimulus --control $CONTROL --n_lines $N_LINES"

if [[ "$METHOD" == "interleaved" ]]; then
  if [[ -z "$MEG_ENC_CKPT" ]]; then
    echo "ERROR: set MEG_ENC_CKPT for method=interleaved"
    exit 1
  fi
  EXTRA_ARGS="$EXTRA_ARGS --meg_enc_ckpt $MEG_ENC_CKPT"
fi

echo "Method=$METHOD  control=$CONTROL  n_lines=$N_LINES"

jid=$(sbatch "${SBATCH_BASE[@]}" \
  --job-name="stim_${METHOD}_lines${N_LINES}_${CONTROL}" \
  --parsable \
  --wrap="cd $WORKDIR && python -m unified.train $EXTRA_ARGS")
echo "  → job $jid"

echo "Done."
