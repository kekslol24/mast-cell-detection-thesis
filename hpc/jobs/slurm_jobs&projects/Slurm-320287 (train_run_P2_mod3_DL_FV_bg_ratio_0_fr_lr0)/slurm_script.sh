#!/usr/bin/env bash
#SBATCH --job-name=train_run_P2_mod3_DL_FV_bg_ratio_0_fr_lr0
#SBATCH --mail-type=fail,end
#SBATCH --time=00-08:00:00
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

python ba_improved_P2.py
