#!/bin/bash
#SBATCH -c 32
#SBATCH -t 1-04:00:00  # time in d-hh:mm:ss
#SBATCH --qos=grp_rwang133
#SBATCH --mem=200G
#SBATCH --gres=gpu:h100:4
#SBATCH -p general
#SBATCH -A grp_subbarao
#SBATCH --output=isft-%j.out
#SBATCH --error=isft-%j.err
eval "$(conda shell.bash hook)"
conda activate zero
cd /home/dkalwar/TinyZero/scripts
bash train_tiny_zero_isft.sh &
bash train_tiny_zero_grpo.sh &
wait