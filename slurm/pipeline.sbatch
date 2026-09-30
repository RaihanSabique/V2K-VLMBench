#!/bin/bash
# Full pipeline: pretrained models (GPU) -> markers -> [DINOv3 gap fill] -> kinematics -> QA.
# DINOv3 runs only if your HF account has access (hf auth login once, after accepting
# the licence at https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).
#SBATCH --job-name=vicon-pipeline
#SBATCH --partition=ai
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mmfs1/home/6991miam/VICONMapper/slurm/%x_%j.out
set -e
source /etc/profile.d/modules.sh
module load python/3.10.13 cuda11.8/toolkit/11.8.0
source /mmfs1/home/6991miam/VICONMapper/.venv/bin/activate
cd /mmfs1/home/6991miam/VICONMapper/src
export HF_HUB_OFFLINE=1  # compute nodes have no internet; weights are pre-cached

python run_models.py --stages detect vitpose rtmpose sapiens2 mediapipe
python markers.py
python dino_track.py || echo "DINOv3 stage skipped (no access?) - marker gaps left as is"
python kinematics.py
python plots.py
python evaluate.py
python render.py
