#!/bin/bash

output_prefix="./output_files/mcts_qwen_"
num_chunks=4

for i in {0..3}; do
     python mcts.py \
        --data_pths './ThinkLite-VL-hard-11k_default_train.parquet' \
        --output_file ${output_prefix}$((i+1)).parquet \
        --num-chunks $num_chunks \
        --chunk-idx $((i)) \
        --gpu-id $((i)) &
done