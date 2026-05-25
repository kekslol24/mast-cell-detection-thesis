#!/usr/bin/env bash
#SBATCH --partition=earth-4
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Port dynamisch wählen, falls 8888 belegt ist
# PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

PORT=8888


source /cfs/earth/scratch/vollmflo/exercises-2025/init_micromamba.sh
micromamba activate ultralytics

jupyter notebook --no-browser --port=${PORT} --ip=0.0.0.0

echo $(jupyter notebook list)