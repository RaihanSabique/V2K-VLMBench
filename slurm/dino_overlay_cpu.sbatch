#!/bin/bash
#SBATCH --job-name=vicon-dino-overlay-cpu
#SBATCH --partition=batch
#SBATCH --cpus-per-task=32

#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/mmfs1/home/6991miam/VICONMapper/slurm/%x_%j.out
set -e
source /etc/profile.d/modules.sh
module load python/3.10.13
source /mmfs1/home/6991miam/VICONMapper/.venv/bin/activate
cd /mmfs1/home/6991miam/VICONMapper/src
export HF_HUB_OFFLINE=1  # compute nodes have no internet; weights are pre-cached
# CPU fallback when every GPU partition is booked: one video per job (VIDEO=0|1)
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
python model_overlays.py --models dinov3 --video ${VIDEO:-0}
