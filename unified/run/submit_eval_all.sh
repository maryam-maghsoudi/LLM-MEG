#!/bin/bash
# submit_eval_all.sh — run evaluate.py for all methods, eval schemes, and controls.
#
# Usage:
#   bash submit_eval_all.sh
#   bash submit_eval_all.sh --dry-run   # print sbatch commands without submitting

set -e

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

# ---------------------------------------------------------------------------
WORKDIR=/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/llm_decoder
UNIFIED=$WORKDIR/unified
BERT_TAG=bert_base_uncased
LLM_TAG=HuggingFaceTB_SmolLM2-360M

SUBJECTS=(
  sub-01 sub-03 sub-04 sub-05 sub-06 sub-09
  sub-10 sub-11 sub-12 sub-13 sub-14 sub-16 sub-17
)
N_FOLDS=5
STIMULUS_LINES=(2 4)
CONTROLS=(none shuffle_time zero)

EXCLUDE=legacygpu00,legacygpu02,legacygpu03,legacygpu04,legacygpu05,legacygpu06,legacygpu07,legacygpu09,legacygpu10,legacygpu11,legacygpu12,legacygpu13,legacygpu14,legacygpu18,legacygpu19,legacygpu20,legacygpu21,legacygpu22,legacygpu23,legacygpu24,legacygpu25,legacygpu26,legacygpu27,legacygpu28,legacygpu29,legacygpu30,legacygpu31,legacygpu32,legacygpu33,legacygpu34,legacygpu35,legacygpu36,legacygpu37,legacygpu38,legacygpu39,legacygpu40,legacygpu41,legacygpu42

LOGDIR=$UNIFIED/slurm_logs/eval_all
mkdir -p "$LOGDIR"

SBATCH_COMMON=(
  --partition=scavenger --account=scavenger --qos=scavenger
  --gres=gpu:rtxa5000:1 --cpus-per-task=4 --mem=16G --time=2:00:00
  "--exclude=$EXCLUDE"
  "--output=$LOGDIR/%j_%x.out"
  "--error=$LOGDIR/%j_%x.err"
)

submit() {
  local name="$1"; shift
  local cmd="$*"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] --job-name=$name --wrap=\"$cmd\""
  else
    jid=$(sbatch "${SBATCH_COMMON[@]}" --job-name="$name" --parsable --wrap="$cmd")
    echo "  submitted $name → job $jid"
  fi
}

EVAL="cd $WORKDIR && python -m unified.evaluate --device cuda"

_ctrl_suffix() { [[ "$1" == "none" ]] && echo "" || echo "_ctrl_$1"; }

total=0

# ---------------------------------------------------------------------------
#  LOSO — 13 subjects
# ---------------------------------------------------------------------------
echo "=== LOSO ==="
for SUBJ in "${SUBJECTS[@]}"; do
  for CTRL in "${CONTROLS[@]}"; do
    SUFFIX=$(_ctrl_suffix "$CTRL")

    submit "eval_inf_loso_${SUBJ}_${CTRL}" \
      "$EVAL --method inference --eval_scheme loso --heldout $SUBJ \
       --ckpt_dir $UNIFIED/out/inference/$BERT_TAG/loso_${SUBJ}${SUFFIX}"
    total=$((total + 1))

    submit "eval_ts_loso_${SUBJ}_${CTRL}" \
      "$EVAL --method twostage --eval_scheme loso --heldout $SUBJ \
       --ckpt_dir $UNIFIED/out/twostage/$LLM_TAG/loso_${SUBJ}${SUFFIX}"
    total=$((total + 1))

    submit "eval_il_loso_${SUBJ}_${CTRL}" \
      "$EVAL --method interleaved --eval_scheme loso --heldout $SUBJ \
       --ckpt_dir $UNIFIED/out/interleaved/$LLM_TAG/loso_${SUBJ}${SUFFIX}"
    total=$((total + 1))
  done
done

# ---------------------------------------------------------------------------
#  Session CV — 5 folds
# ---------------------------------------------------------------------------
echo ""
echo "=== Session CV ==="
for FOLD in $(seq 0 $(( N_FOLDS - 1 ))); do
  for CTRL in "${CONTROLS[@]}"; do
    SUFFIX=$(_ctrl_suffix "$CTRL")

    submit "eval_inf_scv_fold${FOLD}_${CTRL}" \
      "$EVAL --method inference --eval_scheme session_cv --fold $FOLD \
       --ckpt_dir $UNIFIED/out/inference/$BERT_TAG/session_cv_fold${FOLD}${SUFFIX}"
    total=$((total + 1))

    submit "eval_ts_scv_fold${FOLD}_${CTRL}" \
      "$EVAL --method twostage --eval_scheme session_cv --fold $FOLD \
       --ckpt_dir $UNIFIED/out/twostage/$LLM_TAG/session_cv_fold${FOLD}${SUFFIX}"
    total=$((total + 1))

    submit "eval_il_scv_fold${FOLD}_${CTRL}" \
      "$EVAL --method interleaved --eval_scheme session_cv --fold $FOLD \
       --ckpt_dir $UNIFIED/out/interleaved/$LLM_TAG/session_cv_fold${FOLD}${SUFFIX}"
    total=$((total + 1))
  done
done

# ---------------------------------------------------------------------------
#  Stimulus — n_lines = 2 and 4
# ---------------------------------------------------------------------------
echo ""
echo "=== Stimulus ==="
for NL in "${STIMULUS_LINES[@]}"; do
  for CTRL in "${CONTROLS[@]}"; do
    SUFFIX=$(_ctrl_suffix "$CTRL")

    submit "eval_inf_stim_lines${NL}_${CTRL}" \
      "$EVAL --method inference --eval_scheme stimulus --n_lines $NL \
       --ckpt_dir $UNIFIED/out/inference/$BERT_TAG/stimulus_lines${NL}${SUFFIX}"
    total=$((total + 1))

    submit "eval_ts_stim_lines${NL}_${CTRL}" \
      "$EVAL --method twostage --eval_scheme stimulus --n_lines $NL \
       --ckpt_dir $UNIFIED/out/twostage/$LLM_TAG/stimulus_lines${NL}${SUFFIX}"
    total=$((total + 1))

    submit "eval_il_stim_lines${NL}_${CTRL}" \
      "$EVAL --method interleaved --eval_scheme stimulus --n_lines $NL \
       --ckpt_dir $UNIFIED/out/interleaved/$LLM_TAG/stimulus_lines${NL}${SUFFIX}"
    total=$((total + 1))
  done
done

echo ""
echo "Total jobs: $total"
