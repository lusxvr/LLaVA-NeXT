#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --partition=hard 
#SBATCH -C A6000
#SBATCH --gpus-per-node=1
#SBATCH --time=47:59:00
#SBATCH --array=0
#SBATCH --output=/home/wiedmann/LLaVA-NeXT/logs/eval/%A-%a.out
#SBATCH --error=/home/wiedmann/LLaVA-NeXT/logs/eval/%A-%a.err

cd /home/wiedmann/LLaVA-NeXT
source .venv/bin/activate

# nextqa_mc_test
# longvideobench_val_v

python -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen \
    --tasks nextqa_mc_test \
    --batch_size 1 \
    --output_path ./results/ \
    #--limit 10