#!/usr/bin/env bash
# Once the 1.7B layers.17 n-gram table is finalized, train the ablation ladder and
# evaluate it. Fully autonomous: polls for the table, runs the ladder (one arm per GPU),
# then the Stage-6 eval.
set -uo pipefail
cd /mnt/ssd-1/lucia/sparsify

TABLE=ngram_tables/smollm17_l17/layers.17
CORPUS=/mnt/ssd-1/lucia/ngram_tables/corpus_1B
PY=/opt/conda/bin/python

echo "waiting for table $TABLE/table.json ..."
while [ ! -f "$TABLE/table.json" ]; do sleep 30; done
echo "table ready; R^2:"; $PY -c "import json;d=json.load(open('ngram_tables/smollm17_l17/rsquared.json'));b=d['layers.17']['by_order'];print({k:round(v['r2'],3) for k,v in b.items()})" 2>/dev/null || true

echo "### training ladder"
bash scripts/run_ladder.sh HuggingFaceTB/SmolLM2-1.7B "$CORPUS" layers.17 "$TABLE" 32 32 50000 l17 --batch_size 8

echo "### evaluating ladder"
PATHS=""
for a in base embedskip lookup0 lookup1 lookup2 lookup3 lookup4; do
  PATHS="$PATHS checkpoints/l17_${a}/layers.17"
done
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True $PY -u -m sparsify.ngram_eval \
    HuggingFaceTB/SmolLM2-1.7B "$CORPUS" --hookpoint layers.17 \
    --gram_table_path "$TABLE" --sae_paths $PATHS \
    --eval_tokens 2000000 --batch_size 8 --triviality_max_tokens 500000 \
    --output_dir ngram_tables/eval_l17
echo "### 1.7B LADDER + EVAL DONE"
