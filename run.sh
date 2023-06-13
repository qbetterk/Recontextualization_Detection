#!/bin/bash
#
set -xue

if [ $HOSTNAME == "communication" ]; then
        # ****************************** script for communication ****************************** 
        export CUDA_VISIBLE_DEVICES=2
        export HF_DATASETS_CACHE=/local-storage/data/qkun/huggingface/datasets/
        export TRANSFORMERS_CACHE=/local-storage/data/qkun/huggingface/transformers/
        export HF_HOME=/local-storage/data/qkun/huggingface/
        proj_path=/local-storage/data/qkun/semafor/

elif [ $HOSTNAME == "coffee" ]; then
        # ****************************** script for coffee ****************************** 
        export CUDA_VISIBLE_DEVICES=5
        export HF_DATASETS_CACHE=/local/data/shared/huggingface_cache/datasets/
        export TRANSFORMERS_CACHE=/local/data/shared/huggingface_cache/transformers/
        export HF_CACHE_HOME=/local/data/shared/huggingface_cache
        export HF_HOME=/local/data/shared/huggingface_cache/
        proj_path=/local/data/qkun/semafor/
fi
python test.py

# # model_name=roberta-base
# model_name=google/flan-t5-base
# # model_name=EleutherAI/gpt-j-6B

# python train.py \
#         --model_name_or_path ${model_name} \
#         --output_dir ${proj_path}/ckpt/${model_name} \
#         --train_file ./data_my/newsroom_train.json \
#         --validation_file ./data_my/newsroom_val.json \
#         --do_train \
#         --do_eval \
#         --num_train_epochs 10 \
#         --lr_scheduler_type cosine \
#         --warmup_steps 100 \
#         --save_strategy epoch \
#         --per_device_train_batch_size=64 \
#         --per_device_eval_batch_size=32 \
#         --auto_find_batch_size \
#         --save_total_limit 1 \
#         --load_best_model_at_end \
#         --evaluation_strategy epoch \
#         --overwrite_output_dir \
#         --predict_with_generate \
#         --log_level warning