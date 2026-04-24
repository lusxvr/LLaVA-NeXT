#!/bin/bash
# Fine-tune LLaVA-7B on NextQA without any temporal resampler (baseline).
#
# What is trained:
#   - mm_projector      (MLP adapter)
#
# What is frozen:
#   - vision_tower      (SigLIP-so400m)
#   - LLM               (Qwen2-7B)
#
# Token budget:
#   Uses the default spatial_unpad pooling (~5824 tokens for 128 frames).
#   8192 context window; long videos may be truncated.

IMAGE_FOLDER=""
VIDEO_FOLDER="/data/wiedmann"
DATA_YAML="scripts/video/train/nextqa_experiment.yaml"

############### Prepare Envs #################
if [ -z "$CUDA_HOME" ] || [ ! -f "$CUDA_HOME/bin/nvcc" ]; then
    CUDA_HOME=$(find /opt/nvidia/hpc_sdk /usr/local -name "nvcc" 2>/dev/null | head -1 | sed 's|/bin/nvcc||')
fi
export CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
echo "Using CUDA_HOME: $CUDA_HOME"
MAX_JOBS=8 uv pip -v install flash-attn==2.5.7 --no-build-isolation
alias python=python3
############### Show Envs ####################
nvidia-smi

################ Model config ################
LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

export WANDB_PROJECT="llava-streaming-agg"

PREV_STAGE_CHECKPOINT="/data/wiedmann/hub/models--lmms-lab--llava-onevision-qwen2-7b-si"

PROMPT_VERSION="qwen_1_5"
RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-SI-nextqa_shuffled-baseline_lora_mlp"
echo "RUN_NAME: ${RUN_NAME}"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"

deepspeed --master_port 30005 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --image_folder "$IMAGE_FOLDER" \
    --video_folder $VIDEO_FOLDER \
    \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    \
    --mm_patch_merge_type spatial_unpad \
    --mm_newline_position no_token \
    \
    --mm_tunable_parts="mm_mlp_adapter" \
    \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir /data/wiedmann/llava-streaming/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --add_time_instruction False \
    --force_sample False \
    --val_split_fraction 0.01
exit 0;
