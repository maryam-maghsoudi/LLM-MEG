#!/bin/bash
# submit_all.sh — submit all unified pipeline jobs.
#
# Coverage:
#   eval schemes : loso (13 subjects)  |  session_cv (5 folds)  |  stimulus (2+4 lines)
#   methods      : inference (Method 1)  |  twostage (Method 2)  |  interleaved (Method 3)
#   controls     : none  |  shuffle_time  |  zero
#
# Method 3 (interleaved) is always chained: it depends on Method 1 (control=none)
# for the same split completing first, so it can use that encoder checkpoint.
#
# Usage:
#   bash submit_all.sh
#   bash submit_all.sh --dry-run   # print sbatch commands without submitting

set -e

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

# ---------------------------------------------------------------------------
WORKDIR=/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/llm_decoder
UNIFIED=$WORKDIR/unified
BERT_TAG=bert_base_uncased   # matches train.py model_tag logic for bert-base-uncased

SUBJECTS=(
  sub-01 sub-03 sub-04 sub-05 sub-06 sub-09
  sub-10 sub-11 sub-12 sub-13 sub-14 sub-16 sub-17
)
N_FOLDS=5
STIMULUS_LINES=(2 4)
CONTROLS=(none shuffle_time zero)

EXCLUDE=legacygpu00,legacygpu02,legacygpu03,legacygpu04,legacygpu05,legacygpu06,legacygpu07,legacygpu09,legacygpu10,legacygpu11,legacygpu12,legacygpu13,legacygpu14,legacygpu18,legacygpu19,legacygpu20,legacygpu21,legacygpu22,legacygpu23,legacygpu24,legacygpu25,legacygpu26,legacygpu27,legacygpu28,legacygpu29,legacygpu30,legacygpu31,legacygpu32,legacygpu33,legacygpu34,legacygpu35,legacygpu36,legacygpu37,legacygpu38,legacygpu39,legacygpu40,legacygpu41,legacygpu42

LOGDIR=$UNIFIED/slurm_logs/all
mkdir -p "$LOGDIR"

SBATCH_COMMON=(
  --partition=scavenger --account=scavenger --qos=scavenger
  --gres=gpu:rtxa5000:1 --cpus-per-task=4 --mem=32G --time=6:00:00
  "--exclude=$EXCLUDE"
  "--output=$LOGDIR/%j_%x.out"
  "--error=$LOGDIR/%j_%x.err"
)

# ---------------------------------------------------------------------------
#  Helper: submit one job, return job ID
# ---------------------------------------------------------------------------
submit() {
  local name="$1"; shift
  local dep="$1";  shift      # "" or "afterok:JOBID"
  local cmd="$*"

  local dep_arg=()
  if [[ -n "$dep" ]]; then dep_arg=("--dependency=$dep"); fi

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] --job-name=$name ${dep_arg[*]} --wrap=\"$cmd\""
    echo "0"   # fake job ID
  else
    sbatch "${SBATCH_COMMON[@]}" \
      --job-name="$name" \
      "${dep_arg[@]}" \
      --parsable \
      --wrap="$cmd"
  fi
}

TRAIN="cd $WORKDIR && python -m unified.train"

total=0

# ---------------------------------------------------------------------------
#  LOSO — 13 subjects
# ---------------------------------------------------------------------------
echo "=== LOSO ==="
for SUBJ in "${SUBJECTS[@]}"; do
  # Method 1 and 2 can run in parallel; Method 3 waits for Method 1 (none).
  m1_none_jid=""

  for CTRL in "${CONTROLS[@]}"; do
    tag="${SUBJ}_${CTRL}"

    # Method 1 (inference)
    jid=$(submit "loso_inf_${tag}" "" \
      "$TRAIN --method inference --eval_scheme loso --heldout $SUBJ --control $CTRL")
    echo "  [M1] loso $SUBJ ctrl=$CTRL → job $jid"
    [[ "$CTRL" == "none" ]] && m1_none_jid=$jid
    total=$((total + 1))

    # Method 2 (twostage)
    jid=$(submit "loso_ts_${tag}" "" \
      "$TRAIN --method twostage --eval_scheme loso --heldout $SUBJ --control $CTRL")
    echo "  [M2] loso $SUBJ ctrl=$CTRL → job $jid"
    total=$((total + 1))
  done

  # Method 3 (interleaved) — all controls, all depend on Method 1 (none)
  ENC_CKPT=$UNIFIED/out/inference/$BERT_TAG/loso_${SUBJ}/meg_encoder_best.pt
  for CTRL in "${CONTROLS[@]}"; do
    tag="${SUBJ}_${CTRL}"
    jid=$(submit "loso_il_${tag}" "afterok:${m1_none_jid}" \
      "$TRAIN --method interleaved --eval_scheme loso --heldout $SUBJ --control $CTRL --meg_enc_ckpt $ENC_CKPT")
    echo "  [M3] loso $SUBJ ctrl=$CTRL → job $jid (after M1 none $m1_none_jid)"
    total=$((total + 1))
  done
done

# ---------------------------------------------------------------------------
#  Session CV — 5 folds
# ---------------------------------------------------------------------------
echo ""
echo "=== Session CV ==="
for FOLD in $(seq 0 $(( N_FOLDS - 1 ))); do
  m1_none_jid=""

  for CTRL in "${CONTROLS[@]}"; do
    tag="fold${FOLD}_${CTRL}"

    # Method 1
    jid=$(submit "scv_inf_${tag}" "" \
      "$TRAIN --method inference --eval_scheme session_cv --fold $FOLD --control $CTRL")
    echo "  [M1] fold $FOLD ctrl=$CTRL → job $jid"
    [[ "$CTRL" == "none" ]] && m1_none_jid=$jid
    total=$((total + 1))

    # Method 2
    jid=$(submit "scv_ts_${tag}" "" \
      "$TRAIN --method twostage --eval_scheme session_cv --fold $FOLD --control $CTRL")
    echo "  [M2] fold $FOLD ctrl=$CTRL → job $jid"
    total=$((total + 1))
  done

  ENC_CKPT=$UNIFIED/out/inference/$BERT_TAG/session_cv_fold${FOLD}/meg_encoder_best.pt
  for CTRL in "${CONTROLS[@]}"; do
    tag="fold${FOLD}_${CTRL}"
    jid=$(submit "scv_il_${tag}" "afterok:${m1_none_jid}" \
      "$TRAIN --method interleaved --eval_scheme session_cv --fold $FOLD --control $CTRL --meg_enc_ckpt $ENC_CKPT")
    echo "  [M3] fold $FOLD ctrl=$CTRL → job $jid (after M1 none $m1_none_jid)"
    total=$((total + 1))
  done
done

# ---------------------------------------------------------------------------
#  Stimulus — n_lines = 2 and 4
# ---------------------------------------------------------------------------
echo ""
echo "=== Stimulus ==="
for NL in "${STIMULUS_LINES[@]}"; do
  m1_none_jid=""

  for CTRL in "${CONTROLS[@]}"; do
    tag="lines${NL}_${CTRL}"

    # Method 1
    jid=$(submit "stim_inf_${tag}" "" \
      "$TRAIN --method inference --eval_scheme stimulus --n_lines $NL --control $CTRL")
    echo "  [M1] lines=$NL ctrl=$CTRL → job $jid"
    [[ "$CTRL" == "none" ]] && m1_none_jid=$jid
    total=$((total + 1))

    # Method 2
    jid=$(submit "stim_ts_${tag}" "" \
      "$TRAIN --method twostage --eval_scheme stimulus --n_lines $NL --control $CTRL")
    echo "  [M2] lines=$NL ctrl=$CTRL → job $jid"
    total=$((total + 1))
  done

  ENC_CKPT=$UNIFIED/out/inference/$BERT_TAG/stimulus_lines${NL}/meg_encoder_best.pt
  for CTRL in "${CONTROLS[@]}"; do
    tag="lines${NL}_${CTRL}"
    jid=$(submit "stim_il_${tag}" "afterok:${m1_none_jid}" \
      "$TRAIN --method interleaved --eval_scheme stimulus --n_lines $NL --control $CTRL --meg_enc_ckpt $ENC_CKPT")
    echo "  [M3] lines=$NL ctrl=$CTRL → job $jid (after M1 none $m1_none_jid)"
    total=$((total + 1))
  done
done

echo ""
echo "Total jobs submitted: $total"
