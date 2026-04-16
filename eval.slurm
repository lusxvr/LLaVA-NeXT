#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --partition=hard 
#SBATCH -C A6000
#SBATCH --gpus-per-node=1
#SBATCH --time=47:59:00
#SBATCH --array=0-3
#SBATCH --output=/home/wiedmann/LLaVA-NeXT/logs/eval/%A-%a.out
#SBATCH --error=/home/wiedmann/LLaVA-NeXT/logs/eval/%A-%a.err

cd /home/wiedmann/LLaVA-NeXT
source .venv/bin/activate

# nextqa_mc_test
# longvideobench_val_v

# model_dir="lmms-lab/llava-onevision-qwen2-7b-ov"

RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-nextqa_shuffled-streaming_baseline_final_lora_agg_fc4_s4096"
BASE_CHECKPOINT="/data/wiedmann/hub/models--lmms-lab--llava-onevision-qwen2-7b-ov/snapshots/0b07bf7565e244cf4f39982249eafe8cd799d6dd"

output_dir="/data/wiedmann/llava-streaming/$RUN_NAME"

# Prefer the final model written to output_dir root by train.py on completion.
# Fall back to the latest checkpoint if the root model is absent (e.g. interrupted run).
if [[ -f "$output_dir/config.json" ]]; then
    model_dir="$output_dir"
else
    model_dir=$(ls -d "$output_dir/checkpoint-*" 2>/dev/null | sort -t- -k2 -n | tail -1)
fi

if [[ -z "$model_dir" ]]; then
    echo "ERROR: no model found under $output_dir" >&2
    exit 1
fi

frame_counts=(32 64 128 256)
echo "model_dir: $model_dir"

python -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=$model_dir,model_base=$BASE_CHECKPOINT,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=${frame_counts[$SLURM_ARRAY_TASK_ID]} \
    --tasks 'nextqa_mc_test,longvideobench_val_v'\
    --batch_size 1 \
    --output_path ./results/frames_${frame_counts[$SLURM_ARRAY_TASK_ID]}
