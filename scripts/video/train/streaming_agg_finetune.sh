#!/bin/bash
# Fine-tune LLaVA-7B with the StreamingStateAggregator as a temporal resampler.
#
# What is trained:
#   - vision_resampler  (StreamingStateAggregator, randomly initialised or from
#                        --mm_streaming_pretrained if provided)
#   - mm_projector      (adapts to aggregator output statistics)
#
# What is frozen:
#   - vision_tower      (SigLIP-so400m)
#   - LLM               (Qwen2-7B)
#
# Token budget:
#   S=512 state tokens per video vs. ~5824 for the spatial-pool baseline.
#   8192 context window fits 512 video tokens + multi-turn text comfortably.

IMAGE_FOLDER=""
VIDEO_FOLDER="/data/wiedmann/hub/datasets--lmms-lab--LLaVA-Video-178K/snapshots/6d8c562dc26d70042a0d9704d1cae58c94b89098/0_30_s_nextqa"
DATA_YAML="scripts/video/train/nextqa_experiment.yaml"

# Optional: path to a StreamingStateAggregator checkpoint from rep_sim linear
# probe training. Leave empty to start from random initialisation.
STREAMING_PRETRAINED=""

############### Prepare Envs #################
if [ -z "$CUDA_HOME" ] || [ ! -f "$CUDA_HOME/bin/nvcc" ]; then
    CUDA_HOME=$(find /opt/nvidia/hpc_sdk /usr/local -name "nvcc" 2>/dev/null | head -1 | sed 's|/bin/nvcc||')
fi
export CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
echo "Using CUDA_HOME: $CUDA_HOME"
#python3 -m pip install flash-attn --no-build-isolation
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

# Start from the pretrained LLaVA-7B one-vision checkpoint
PREV_STAGE_CHECKPOINT="/data/wiedmann/hub/models--lmms-lab--llava-onevision-qwen2-7b-ov/snapshots/0b07bf7565e244cf4f39982249eafe8cd799d6dd"

PROMPT_VERSION="qwen_1_5"
RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-streaming_agg_s512"
echo "RUN_NAME: ${RUN_NAME}"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"

deepspeed --master_port 30000 \
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
    --mm_resampler_type streaming_agg \
    --mm_streaming_input_dim 1152 \
    --mm_streaming_state_dim 1152 \
    --mm_streaming_num_state_tokens 512 \
    --mm_streaming_num_layers 4 \
    --mm_streaming_num_heads 8 \
    --mm_streaming_chunk_size 729 \
    --mm_streaming_pretrained "${STREAMING_PRETRAINED}" \
    \
    --mm_patch_merge_type spatial_unpad \
    --mm_newline_position no_token \
    \
    --mm_tunable_parts="mm_mlp_adapter,mm_vision_resampler" \
    \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir ./work_dirs/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --add_time_instruction False \
    --force_sample True
exit 0;
