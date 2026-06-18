#!/usr/bin/env bash
#SBATCH --job-name=yolo_new_v3
#SBATCH --mail-type=fail,end
#SBATCH --time=02-00:00:00
#SBATCH --partition=earth-4
#SBATCH --qos=earth-4.4d
#SBATCH --constraint=rhel8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
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
export N_GPUS=2
export MAX_PARALLEL=2


# python ba_improved_comb_CV.py

SLURM_JOB_NAME="yolo_new_v4_freeze10_copy" python eval_lopo_folds.py
