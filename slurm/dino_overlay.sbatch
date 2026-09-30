#!/bin/bash
#SBATCH --job-name=vicon-dino-overlay
#SBATCH --partition=ai
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/home/6991miam/VICONMapper/slurm/%x_%j.out
set -e
source /etc/profile.d/modules.sh
module load python/3.10.13 cuda11.8/toolkit/11.8.0
source /mmfs1/home/6991miam/VICONMapper/.venv/bin/activate
cd /mmfs1/home/6991miam/VICONMapper/src
export HF_HUB_OFFLINE=1  # compute nodes have no internet; weights are pre-cached
python model_overlays.py --models dinov3
