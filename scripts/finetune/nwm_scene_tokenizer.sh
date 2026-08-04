#!/bin/bash
# Tokenizer finetune on nwm_scene from OXE-64 pretrain (BAIR recipe, reduced steps).
STEPS=${STEPS:-50005}
BS=${BS:-16}
accelerate launch --num_processes=1 --mixed_precision=bf16 train_tokenizer.py \
    --exp_name nwm_scene_tokenizer_ft --output_dir log_vqgan --seed 0 --mixed_precision bf16 \
    --model_type ctx_vqgan \
    --train_batch_size $BS --gradient_accumulation_steps 1 --disc_start 1000005 \
    --oxe_data_mixes_type nwm_scene --dataset_path /mnt/Data/nwm-baselines/ivideogpt \
    --resolution 64 --dataloader_num_workers 8 \
    --rand_select --video_stepsize 1 --segment_horizon 16 --segment_length 8 --context_length 1 \
    --pretrained_model_name_or_path pretrained_models/ivideogpt-oxe-64-act-free/tokenizer \
    --checkpointing_steps 5000 \
    --max_train_steps $STEPS
