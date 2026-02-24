#!/bin/bash

INDEXES=(
  wiki_dpr_ivf_21m
  wiki_dpr_hnsw_21m
  wiki_dpr_flat_21m
)

for IDX in "${INDEXES[@]}"; do

  # no cache
  python pwrag/client/eval_dataset.py \
    dataset=nq \
    retriever/index=$IDX \
    retriever/cache=none \
    save_dir=results/nq/$IDX/none

  # proximity cache
  python pwrag/client/eval_dataset.py \
    dataset=nq \
    retriever/index=$IDX \
    retriever/cache=proximity \
    save_dir=results/nq/$IDX/proximity

done