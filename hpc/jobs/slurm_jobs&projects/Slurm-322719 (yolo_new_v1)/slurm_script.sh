#!/usr/bin/env bash
#SBATCH --job-name=yolo_new_v1
#SBATCH --mail-type=fail,end
#SBATCH --time=02-00:00:00
#SBATCH --partition=earth-4
#SBATCH --constraint=rhel8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32GB
#SBATCH --gres=gpu:l40s:2


# Zeige aktuelles Verzeichnis vor dem Wechsel
echo "Start-Verzeichnis: $(pwd)"

source /cfs/earth/scratch/vollmflo/exercises-2025/init_micromamba.sh
micromamba activate ultralytics

# Prüfe, ob die Umgebung wirklich aktiv ist
echo "Genutztes Python: $(which python)"

# Gehe ein Verzeichnis hoch
cd ..
echo "Neues Verzeichnis: $(pwd)"

# export BG_RATIO=${BG_RATIO:-0}
# export FP_NEG_OVERSAMPLE=${FP_NEG_OVERSAMPLE:-3}

python ba_improved_comb.py
