#!/bin/bash
# Action-conditioned transformer finetune on nwm_scene (BAIR recipe, action_dim 5).
# Usage: TOKENIZER=<log_vqgan/.../unwrapped_model> bash scripts/finetune/nwm_scene_gpt.sh
STEPS=${STEPS:-50005}
BS=${BS:-16}
TOKENIZER=${TOKENIZER:?path to finetuned tokenizer unwrapped_model}
accelerate launch --num_processes=1 --mixed_precision=bf16 train_gpt.py \
    --exp_name nwm_scene_llama_ft --output_dir log_trm --seed 0 --mixed_precision bf16 \
    --vqgan_type ctx_vqgan \
    --pretrained_model_name_or_path "$TOKENIZER" \
    --config_name configs/llama/config.json --load_internal_llm --action_conditioned --action_dim 5 \
    --pretrained_transformer_path pretrained_models/ivideogpt-oxe-64-act-free/transformer \
    --per_device_train_batch_size $BS --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 --lr_scheduler_type cosine \
    --oxe_data_mixes_type nwm_scene --dataset_path /mnt/Data/nwm-baselines/ivideogpt \
    --resolution 64 --dataloader_num_workers 8 \
    --video_stepsize 1 --segment_length 16 --context_length 1 \
    --use_eval_dataset --use_fvd --use_frame_metrics \
    --weight_decay 0.01 --llama_attn_drop 0.1 --embed_no_wd \
    --checkpointing_steps 5000 \
    --max_train_steps $STEPS
