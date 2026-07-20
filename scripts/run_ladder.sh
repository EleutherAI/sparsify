#!/usr/bin/env bash
# Train the frozen-lookup ablation ladder: baseline, embed_skip, and gram_lookup at
# orders 0..4 -- one arm per GPU, all sharing k/expansion/budget/seed.
#   run_ladder.sh MODEL DATASET HOOKPOINT TABLE_DIR EXPANSION K MAX_EXAMPLES OUTPREFIX [EXTRA...]
set -euo pipefail

MODEL=${1:?}; DATASET=${2:?}; HOOKPOINT=${3:?}; TABLE=${4:?}
EXP=${5:?}; K=${6:?}; MAXEX=${7:?}; PREFIX=${8:?}
shift 8; EXTRA="$*"

PY="/opt/conda/bin/python -u -m sparsify"
BASE="$MODEL $DATASET --hookpoints $HOOKPOINT -k $K --expansion_factor $EXP --max_examples $MAXEX --log_to_wandb False $EXTRA"
LOGD="checkpoints/${PREFIX}_logs"; mkdir -p "$LOGD"

launch() {  # gpu name extra...
  local gpu=$1 name=$2; shift 2
  echo "CUDA_VISIBLE_DEVICES=$gpu $PY $BASE $* --run_name ${PREFIX}_${name}"
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_ALLOC_CONF=expandable_segments:True $PY $BASE "$@" \
      --run_name "${PREFIX}_${name}" > "$LOGD/${name}.log" 2>&1 &
}

launch 0 base
launch 1 embedskip --embed_skip True
launch 2 lookup0 --gram_lookup True --lookup_max_order 0 --gram_table_path "$TABLE"
launch 3 lookup1 --gram_lookup True --lookup_max_order 1 --gram_table_path "$TABLE"
launch 4 lookup2 --gram_lookup True --lookup_max_order 2 --gram_table_path "$TABLE"
launch 5 lookup3 --gram_lookup True --lookup_max_order 3 --gram_table_path "$TABLE"
launch 6 lookup4 --gram_lookup True --lookup_max_order 4 --gram_table_path "$TABLE"
wait
echo "### LADDER DONE (prefix=$PREFIX)"
