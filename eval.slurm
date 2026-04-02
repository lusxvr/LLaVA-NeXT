#!/bin/bash
#SBATCH --job-name=eval_nextqa
#SBATCH --nodes=1
#SBATCH --partition=hard 
#SBATCH -C A6000
#SBATCH --gpus-per-node=1
#SBATCH --time=47:59:00
#SBATCH --array=0-2
#SBATCH --output=/home/wiedmann/LLaVA-NeXT/logs/eval/%A-%a.out
#SBATCH --error=/home/wiedmann/LLaVA-NeXT/logs/eval/%A-%a.err

cd /home/wiedmann/LLaVA-NeXT
source .venv/bin/activate

# nextqa_mc_test
# longvideobench_val_v

# model_dir="lmms-lab/llava-onevision-qwen2-7b-ov"

RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-nextqa-streaming_agg_s512"
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

frame_counts=(32 64 128)
echo "model_dir: $model_dir"

python -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=$model_dir,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=${frame_counts[$SLURM_ARRAY_TASK_ID]} \
    --tasks nextqa_mc_test \
    --batch_size 1 \
    --output_path ./results/frames_${frame_counts[$SLURM_ARRAY_TASK_ID]}