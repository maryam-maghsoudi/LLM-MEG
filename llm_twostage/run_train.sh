#!/bin/bash
#SBATCH --job-name=meg_twostage
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --exclude=legacygpu00,legacygpu02,legacygpu03,legacygpu04,legacygpu05,legacygpu06,legacygpu07,legacygpu09,legacygpu10,legacygpu11,legacygpu12,legacygpu13,legacygpu14,legacygpu18,legacygpu19,legacygpu20,legacygpu21,legacygpu22,legacygpu23,legacygpu24,legacygpu25,legacygpu26,legacygpu27,legacygpu28,legacygpu29,legacygpu30,legacygpu31,legacygpu32,legacygpu33,legacygpu34,legacygpu35,legacygpu36,legacygpu37,legacygpu38,legacygpu39,legacygpu40,legacygpu41,legacygpu42

# Run from the llm_twostage directory
cd /fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/llm_decoder/llm_twostage

mkdir -p slurm_logs

# Default: sub-01, SmolLM2-360M; override via --export or sbatch args
HELDOUT=${HELDOUT:-sub-01}
LLM_NAME=${LLM_NAME:-HuggingFaceTB/SmolLM2-360M}

echo "Starting two-stage training"
echo "  Node      : $(hostname)"
echo "  GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Heldout   : ${HELDOUT}"
echo "  LLM       : ${LLM_NAME}"
echo ""

python train.py --heldout "${HELDOUT}" --llm_name "${LLM_NAME}"
