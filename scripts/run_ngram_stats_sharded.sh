#!/usr/bin/env bash
# Sharded multi-GPU n-gram statistics build.
#   run_ngram_stats_sharded.sh MODEL DATASET HOOKPOINT OUTDIR MAXTOKENS NGPU [EXTRA...]
# Builds the vocabulary once, accumulates NGPU row-shards in parallel (one GPU each),
# then merges + finalizes. Prints every launched command for reproducibility.
set -euo pipefail

MODEL=${1:?model}; DATASET=${2:?dataset}; HOOKPOINT=${3:?hookpoint}
OUTDIR=${4:?outdir}; MAXTOKENS=${5:?max_tokens}; NGPU=${6:?ngpu}
shift 6
EXTRA="$*"

PY="/opt/conda/bin/python -u -m sparsify.ngram_stats"
COMMON="$MODEL $DATASET --hookpoints $HOOKPOINT --output_dir $OUTDIR --vocab_dir $OUTDIR --max_tokens $MAXTOKENS --tau 30 $EXTRA"
LOGDIR="$OUTDIR/logs"; mkdir -p "$LOGDIR"

echo "### [1/3] VOCAB"
echo "CUDA_VISIBLE_DEVICES=0 $PY $COMMON --mode vocab"
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True $PY $COMMON --mode vocab 2>&1 | tee "$LOGDIR/vocab.log"

echo "### [2/3] ACCUM $NGPU shards"
pids=()
for i in $(seq 0 $((NGPU-1))); do
  echo "CUDA_VISIBLE_DEVICES=$i $PY $COMMON --mode accum --shard_id $i --num_shards $NGPU"
  CUDA_VISIBLE_DEVICES=$i PYTORCH_ALLOC_CONF=expandable_segments:True $PY $COMMON \
      --mode accum --shard_id "$i" --num_shards "$NGPU" > "$LOGDIR/accum_$i.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo "a shard failed; see $LOGDIR"; exit 1; }

echo "### [3/3] MERGE"
echo "CUDA_VISIBLE_DEVICES=0 $PY $COMMON --mode merge"
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True $PY $COMMON --mode merge 2>&1 | tee "$LOGDIR/merge.log"
echo "### DONE -> $OUTDIR"
