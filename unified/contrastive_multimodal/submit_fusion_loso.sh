#!/bin/bash
# Submit teacher-forced LLM+MEG fusion for all 13 LOSO subjects.
# Usage:
#   bash submit_fusion_loso.sh [normalization] [llm_name]
#   e.g.: bash submit_fusion_loso.sh row_zscore gpt2

NORM=${1:-logsoftmax}
LLM=${2:-gpt2}

SUBJECTS=(sub-01 sub-03 sub-04 sub-05 sub-06 sub-09 sub-10
          sub-11 sub-12 sub-13 sub-14 sub-16 sub-17)

WORKDIR=/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/llm_decoder/unified/contrastive_multimodal
LOG_DIR="$WORKDIR/slurm_logs/fusion"
mkdir -p "$LOG_DIR"

for SUBJ in "${SUBJECTS[@]}"; do
    sbatch \
        --job-name="fuse_${SUBJ}" \
        --partition=scavenger \
        --account=scavenger \
        --gres=gpu:rtxa5000:1 \
        --mem=32G \
        --cpus-per-task=4 \
        --time=01:00:00 \
        --output="${LOG_DIR}/${SUBJ}_${NORM}.out" \
        --error="${LOG_DIR}/${SUBJ}_${NORM}.err" \
        --wrap="cd $WORKDIR && python fusion_teacher_forced.py \
            --heldout_subject $SUBJ \
            --llm_name $LLM \
            --normalization $NORM \
            --out_dir fusion_results \
            --plot"
    echo "Submitted fusion for $SUBJ (norm=$NORM, llm=$LLM)"
done
