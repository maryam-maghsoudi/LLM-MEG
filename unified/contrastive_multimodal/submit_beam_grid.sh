#!/bin/bash
# Grid search over MEG-guided beam-search fusion hyperparameters:
#   top_k (MEG candidates per step) x beam_width (B) x heldout subject.
#
# Grid:  k in {1,3,5,7,10,15,20,25,30}   B in {3,5,7,9}   13 LOSO subjects
#        => 9 * 4 * 13 = 468 jobs.
#
# Output JSONs are named  {subj}_beamfusion_B{B}_top{k}_gpt2_{norm}.json
# so every (k, B) combination writes to a distinct file (no overwrites).
#
# Usage:
#   bash submit_beam_grid.sh [normalization]     # default: row_zscore
#
# Wall time scales with B and k (beam-search cost grows ~linearly in B, and
# in k via multi-token candidate expansion). Base ~4h was the B=5/top=5, 29-
# alpha runtime; the grid now sweeps 35 alphas (0:0.03:1), so times are scaled
# up with headroom to avoid self-timeout (there is no mid-run checkpointing).

NORM=${1:-row_zscore}
LLM=gpt2

K_VALUES=(1 3 5 7 10 15 20 25 30)
B_VALUES=(3 5 7 9)
SUBJECTS=(sub-01 sub-03 sub-04 sub-05 sub-06 sub-09 sub-10
          sub-11 sub-12 sub-13 sub-14 sub-16 sub-17)

WORKDIR=/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/llm_decoder/unified/contrastive_multimodal
LOG_DIR="$WORKDIR/slurm_logs/beam"
mkdir -p "$LOG_DIR"

# Wall time (hours) as a function of B and k, rounded up. min 3h.
walltime_hours() {
    local B=$1 k=$2
    awk -v B="$B" -v k="$k" 'BEGIN{
        h = 4 * (B/5) * 1.2 * (1 + k/15);
        hi = int(h); if (h > hi) hi = hi + 1;   # ceil
        if (hi < 3) hi = 3;
        printf "%02d:00:00", hi;
    }'
}

n_submitted=0
for B in "${B_VALUES[@]}"; do
  for k in "${K_VALUES[@]}"; do
    TIME=$(walltime_hours "$B" "$k")
    for SUBJ in "${SUBJECTS[@]}"; do
        sbatch \
            --job-name="bf_${SUBJ}_B${B}_k${k}" \
            --partition=scavenger \
            --account=scavenger \
            --qos=scavenger \
            --gres=gpu:rtxa5000:1 \
            --mem=32G \
            --cpus-per-task=4 \
            --time="$TIME" \
            --output="${LOG_DIR}/${SUBJ}_B${B}_k${k}_${NORM}.out" \
            --error="${LOG_DIR}/${SUBJ}_B${B}_k${k}_${NORM}.err" \
            --wrap="cd $WORKDIR && python fusion_beamsearch.py \
                --heldout_subject $SUBJ \
                --llm_name $LLM \
                --normalization $NORM \
                --beam_width $B \
                --top_k $k \
                --out_dir fusion_results \
                --plot" >/dev/null
        n_submitted=$((n_submitted + 1))
    done
    echo "Submitted B=$B k=$k (13 subjects, time=$TIME)"
  done
done

echo ""
echo "Total jobs submitted: $n_submitted   (norm=$NORM, llm=$LLM)"
