#!/usr/bin/env bash
#SBATCH --job-name=yolo_new_v2
#SBATCH --mail-type=fail,end
#SBATCH --time=04:00:00
#SBATCH --partition=earth-4
#SBATCH --constraint=rhel8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32GB
#SBATCH --gres=gpu:l40s:2

echo "Start-Verzeichnis: $(pwd)"

source /cfs/earth/scratch/vollmflo/exercises-2025/init_micromamba.sh
micromamba activate ultralytics

echo "Genutztes Python: $(which python)"

cd ..
echo "Neues Verzeichnis: $(pwd)"

python python_files/eval_lopo_folds.py
