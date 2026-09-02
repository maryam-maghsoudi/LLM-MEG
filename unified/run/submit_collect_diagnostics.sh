#!/bin/bash
# Submit collect_diagnostics.py as a single Slurm job.
#
# Loads all 13 subjects' MEG data once, then iterates over all
# 8 temperature × 13 heldout = 104 checkpoints and computes
# seen_single / seen_avg / unseen diagnostics for each.
#
# Output: unified/analyze_temp_sweep/sweep_diagnostics.json
#
# Usage (from llm_decoder/):
#   bash unified/run/submit_collect_diagnostics.sh

set -e

WORKDIR=/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/llm_decoder
LOGDIR=$WORKDIR/unified/analyze_temp_sweep/slurm_logs
mkdir -p "$LOGDIR"

EXCLUDE=legacygpu00,legacygpu02,legacygpu03,legacygpu04,legacygpu05,legacygpu06,legacygpu07,legacygpu09,legacygpu10,legacygpu11,legacygpu12,legacygpu13,legacygpu14,legacygpu18,legacygpu19,legacygpu20,legacygpu21,legacygpu22,legacygpu23,legacygpu24,legacygpu25,legacygpu26,legacygpu27,legacygpu28,legacygpu29,legacygpu30,legacygpu31,legacygpu32,legacygpu33,legacygpu34,legacygpu35,legacygpu36,legacygpu37,legacygpu38,legacygpu39,legacygpu40,legacygpu41,legacygpu42

jid=$(sbatch \
  --partition=scavenger --account=scavenger --qos=scavenger \
  --gres=gpu:rtxa5000:1 --cpus-per-task=4 --mem=32G --time=4:00:00 \
  "--exclude=$EXCLUDE" \
  --job-name="collect_diag_sweep" \
  --output="$LOGDIR/%j_collect_diagnostics.out" \
  --error="$LOGDIR/%j_collect_diagnostics.err" \
  --parsable \
  --wrap="cd $WORKDIR && python -m unified.analyze_temp_sweep.collect_diagnostics --device cuda")

echo "Submitted job $jid"
echo "Log: $LOGDIR/${jid}_collect_diagnostics.out"
echo "Output will be written to: $WORKDIR/unified/analyze_temp_sweep/sweep_diagnostics.json"
