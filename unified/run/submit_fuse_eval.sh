#!/bin/bash
# Submit alpha-sweep LLM fusion evaluation for one method/scheme/norm combination.
#
# Usage:
#   bash submit_fuse_eval.sh [method] [eval_scheme] [control] [fusion_llm] [normalization] [closed_vocab_path]
#
# Examples:
#   bash submit_fuse_eval.sh inference loso none HuggingFaceTB/SmolLM2-360M logsoftmax
#   bash submit_fuse_eval.sh twostage session_cv none HuggingFaceTB/SmolLM2-360M row_zscore
#   bash submit_fuse_eval.sh inference loso none gpt2 row_zscore llm_twostage/cache/gpt2/vocab_info.json
#
# One job per combination; fuse_eval.py loops over all folds internally.

set -e

METHOD=${1:-inference}
EVAL_SCHEME=${2:-loso}
CONTROL=${3:-none}
FUSION_LLM=${4:-HuggingFaceTB/SmolLM2-360M}
NORM=${5:-logsoftmax}
CLOSED_VOCAB=${6:-}

WORKDIR=/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/llm_decoder
UNIFIED=$WORKDIR/unified

EXCLUDE=legacygpu00,legacygpu02,legacygpu03,legacygpu04,legacygpu05,legacygpu06,legacygpu07,legacygpu09,legacygpu10,legacygpu11,legacygpu12,legacygpu13,legacygpu14,legacygpu18,legacygpu19,legacygpu20,legacygpu21,legacygpu22,legacygpu23,legacygpu24,legacygpu25,legacygpu26,legacygpu27,legacygpu28,legacygpu29,legacygpu30,legacygpu31,legacygpu32,legacygpu33,legacygpu34,legacygpu35,legacygpu36,legacygpu37,legacygpu38,legacygpu39,legacygpu40,legacygpu41,legacygpu42

VOCAB_TAG=$([ -n "$CLOSED_VOCAB" ] && echo "_closed" || echo "")
LOGDIR=$UNIFIED/slurm_logs/fuse_eval_${METHOD}_${EVAL_SCHEME}_${NORM}_ctrl_${CONTROL}${VOCAB_TAG}
mkdir -p "$LOGDIR"

CLOSED_VOCAB_ARG=""
if [ -n "$CLOSED_VOCAB" ]; then
  CLOSED_VOCAB_ARG="--closed_vocab_path $WORKDIR/$CLOSED_VOCAB"
fi

CMD="cd $WORKDIR && python -m unified.fuse_eval \
  --method $METHOD \
  --eval_scheme $EVAL_SCHEME \
  --control $CONTROL \
  --fusion_llm_name $FUSION_LLM \
  --fusion_normalization $NORM \
  $CLOSED_VOCAB_ARG \
  --device cuda"

jid=$(sbatch \
  --partition=scavenger --account=scavenger --qos=scavenger \
  --gres=gpu:rtxa5000:1 --cpus-per-task=4 --mem=48G --time=4:00:00 \
  "--exclude=$EXCLUDE" \
  "--output=$LOGDIR/%j_%x.out" \
  "--error=$LOGDIR/%j_%x.err" \
  --job-name="fuse_${METHOD}_${EVAL_SCHEME}_${NORM}" \
  --parsable \
  --wrap="$CMD")

echo "Method=$METHOD  scheme=$EVAL_SCHEME  norm=$NORM  control=$CONTROL"
echo "Submitted job $jid  (logs: $LOGDIR)"
