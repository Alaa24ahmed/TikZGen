#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12354
export CUDA_VISIBLE_DEVICES=0

python eval.py \
  --trainset train_10k.parquet \
  --testset test.parquet \
  --output with-curriculum-learning_no_mc_10k_scores.json \
  --path ds_1.3b-base-with-curriculum-learning_10k_no_mc=samahadhoud/deepseek-coder-1.3b-base-with-curriculum-learning_10k_pvalue2_lambda0.3_tgrow8\
  --cache_dir evaluation/out_ds_1.3b-base-with-curriculum-learning_10k_no_mc