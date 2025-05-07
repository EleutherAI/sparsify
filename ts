#!/bin/bash
# WANDB_ENTITY=eleutherai uv run torchrun --nproc_per_node gpu -m sparsify \
# roneneldan/TinyStories-33M roneneldan/TinyStories \
# --transcode=True --skip_connection=True \
# --batch_size=8 --expansion_factor=32 --tp=4 \
# --hookpoints h.0.mlp h.1.mlp h.2.mlp h.3.mlp \
# --run_name non-clt-ts/$1 ${@:2}
# exit
WANDB_ENTITY=eleutherai uv run torchrun --nproc_per_node gpu -m sparsify \
roneneldan/TinyStories-33M roneneldan/TinyStories --ctx_len 128 \
--transcode=True --skip_connection=True \
--batch_size=8 --expansion_factor=128 \
--hookpoints h.0.mlp h.1.mlp h.2.mlp h.3.mlp \
--run_name clt-ts/$1 ${@:2} \
--cross_layer=4
exit
WANDB_ENTITY=eleutherai uv run python -m sparsify \
roneneldan/TinyStories-33M roneneldan/TinyStories \
--transcode=True --skip_connection=True \
--expansion_factor=32 \
--hookpoints h.0.mlp h.1.mlp h.2.mlp h.3.mlp \
--run_name clt-ts/$1 ${@:2}
