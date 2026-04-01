#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --partition=hard 
#SBATCH -C A6000
#SBATCH --gpus-per-node=2
#SBATCH --time=47:59:00
#SBATCH --array=0
#SBATCH --output=/home/wiedmann/LLaVA-NeXT/logs/train/%A-%a.out
#SBATCH --error=/home/wiedmann/LLaVA-NeXT/logs/train/%A-%a.err

cd /home/wiedmann/LLaVA-NeXT
source .venv/bin/activate

bash scripts/video/train/streaming_agg_finetune.sh