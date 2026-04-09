nproc_per_node=4

export NPROC_PER_NODE=$nproc_per_node
export OMP_NUM_THREADS=4

# nohup bash scripts/rm.sh > qwen_tool_all_data_180k_3epoch_4096_all_2round_maskstep1_code.log 2>&1 &
bsz=1
#501760
output_dir="output_dir"
MASTER_PORT=29501 \
FPS_MAX_FRAMES=10 \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,3,6,7 \
MAX_PIXELS=3211264 \
swift sft \
    --model your_code_path/TwT/model/Qwen3-vl-8b \
    --dataset \
        your_code_path/TwT/data/tableqa-preprocess/tableqa-python-1interaction.json \
        your_code_path/TwT/data/tableqa-preprocess/full-table-inter.json \
        your_code_path/TwT/data/multimodalQA-used/pawpularity.json \
        your_code_path/TwT/data/multimodalQA-used/adoption.json \
        your_code_path/TwT/data/multimodalQA-used/paintings.json \
        your_code_path/TwT/data/multimodalQA-used/skinca.json \
    --train_type full \
    --lora_rank 8 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --system your_code_path/TwT/src/prompt.txt \
    --num_train_epochs 3 \
    --per_device_train_batch_size $bsz \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --freeze_vit true \
    --gradient_accumulation_steps $(expr 16 / $bsz) \
    --save_strategy epoch \
    --max_length 15360 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --output_dir $output_dir \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --report_to wandb

# CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server     --model your_code_path/TwT/src/output_dir/stage2-outputdir/v4-20251222-134020/checkpoint-696    --served-model-name qwenvl-3-8b-2   --host 0.0.0.0     --port 8153   --tensor-parallel-size 1 --gpu-memory-utilization 0.9
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server     --model your_code_path/TwT/src/output_dir/sft_vl_8b/checkpoint-546    --served-model-name qwenvl-3-8b-sft    --host 0.0.0.0     --port 8102    --tensor-parallel-size 1 --gpu-memory-utilization 0.9