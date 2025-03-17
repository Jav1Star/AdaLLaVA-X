#!/bin/bash
export WANDB_PROJECT=adallava
RUN_NUM=""


original_model="liuhaotian/llava-v1.5-7b"

deepspeed ./src/adallava/train/train_mem.py \
    --deepspeed ./LLaVA/scripts/zero3.json \
    --model_name_or_path  ${original_model}\
    --version v1 \
    --data_path ./LLaVA-FineTune/llava_v1_5_mix665k.json \
    --image_folder ./LLaVA-FineTune/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --freeze_backbone False \
    --bf16 True \
    --num_prefix_layers 16 \
    --token_selecting "none" \
    --scheduler_type "L" \
    --output_dir checkpoints/ada-llava-L-v1.5-7b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --run_name ada-llava-L-v1.5-7b${RUN_NUM} \