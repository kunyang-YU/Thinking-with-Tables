
# NCCL_DEBUG=INFO \
# REWARD_API_ADDRESS=0.0.0.0 \
# QWEN_API_PORT=8000 \
# VQA_WEIGHT=1 \
# FMT_WEIGHT=0.5 \
# CODE_WEIGHT=0.1 \
# MAX_PIXELS=3211264 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
deepspeed --master_port=29505 \
    scripts/rlhf_ds.py \
    --rlhf_type grpo \
    --model your_ckpt_path \
    --external_plugins your_code_path/TwT/src/rl/myrlhf/ms-swift/examples/train/grpo/plugin/twtplugin.py \
    --reward_funcs re_format \
    --resume_from_checkpoint your_ckpt_path \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 0.0.0.0 \
    --vllm_server_port 8000 \
    --vllm_server_timeout 60 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset \
        your_code_path/TwT/data/RL_data/WTQdata/FinQA/train.json \
        your_code_path/TwT/data/RL_data/WTQdata/TAT-QA/train.json \
        your_code_path/TwT/data/RL_data/MMData/paintings_train.json \
        your_code_path/TwT/data/RL_data/MMData/pawpularity_train.json \
        your_code_path/TwT/data/RL_data/MMData/skinca_train.json \
        your_code_path/TwT/data/RL_data/MMData/adoption_train2.json \
    --dataset_shuffle true \
    --train_dataloader_shuffle true \
    --max_pixels 2408448 \
    --max_length 10240 \
    --max_completion_length 10240 \
    --freeze_aligner false \
    --stop_words '</code>' '</answer>' \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --padding_side left \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1, "num_cycles": 0.5}' \
    --gradient_accumulation_steps 4 \
    --save_strategy 'steps' \
    --eval_strategy 'no' \
    --split_dataset_ratio 0 \
    --eval_steps 20000 \
    --save_steps 20 \
    --save_total_limit 100000 \
    --logging_steps 1 \
    --output_dir /data/yuky/rl_code_ckpt \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 8 \
    --num_generations 4 \
    --temperature 1.0 \
    --beta 0.01 \
    --top_p 0.9 \
    --top_k 50 \
    --repetition_penalty 1.05 \
    --deepspeed zero3 \
    --log_completions true \
    --report_to tensorboard \
    --loss_scale 'default+code' \
    --async_generate false \
    --num_iterations 1 \
    --overlong_filter true \
    --offload_optimizer true \
    --offload_model true \
    --attn_impl flash_attn \
    --vllm_enforce_eager true \


