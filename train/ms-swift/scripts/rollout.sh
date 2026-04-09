CUDA_VISIBLE_DEVICES=7 \
swift rollout \
    --model /ssd/yuky/TwT/src/output_dir/stage2-outputdir/v4-20251222-134020/checkpoint-696 \
    --external_plugins /ssd/yuky/TwT/src/rl/myrlhf/ms-swift/examples/train/grpo/plugin/twtplugin.py \
    --multi_turn_scheduler twt_scheduler \
    --vllm_gpu_memory_utilization 0.9 \
    --max_turns 10 \
    --vllm_max_model_len 20480
    # --stop_words '</code>' 