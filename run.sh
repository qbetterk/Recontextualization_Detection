#!/bin/bash
#
set -xue

if [ $HOSTNAME == "communication" ]; then
        # ****************************** script for communication ****************************** 
        export CUDA_VISIBLE_DEVICES=0
        export HF_DATASETS_CACHE=/local-storage/data/qkun/huggingface/datasets/
        export TRANSFORMERS_CACHE=/local-storage/data/qkun/huggingface/transformers/
        export HF_HOME=/local-storage/data/qkun/huggingface/
        proj_path=/local-storage/data/qkun/semafor/

elif [ $HOSTNAME == "coffee" ]; then
        # ****************************** script for coffee ****************************** 
        export CUDA_VISIBLE_DEVICES=3
        export HF_DATASETS_CACHE=/local/data/shared/huggingface_cache/datasets/
        export TRANSFORMERS_CACHE=/local/data/shared/huggingface_cache/transformers/
        export HF_CACHE_HOME=/local/data/shared/huggingface_cache
        export HF_HOME=/local/data/shared/huggingface_cache/
        proj_path=/local/data/qkun/semafor/
fi
python test.py

# model_name=flan-t5-small

# python train.py \
#         --model_name_or_path google/${model_name} \
#         --output_dir ${proj_path}/ckpt/${model_name} \
#         --train_file ./data_my/newsroom_sample.json \
#         --do_train \
#         --num_train_epochs 20 \
#         --lr_scheduler_type cosine \
#         --warmup_steps 1000 \
#         --save_strategy epoch \
#         --per_device_train_batch_size=64 \
#         --per_device_eval_batch_size=4 \
#         --auto_find_batch_size \
#         --save_total_limit 10 \
#         --overwrite_output_dir \
#         --log_level warning
#         # --debug_mode True \