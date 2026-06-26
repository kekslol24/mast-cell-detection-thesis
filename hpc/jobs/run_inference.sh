#!/usr/bin/env bash
#SBATCH --job-name=inference_run
#SBATCH --mail-type=fail,end
#SBATCH --time=00-00:05:00
#SBATCH --partition=earth-4
#SBATCH --constraint=rhel8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8GB
#SBATCH --gres=gpu:l40s:1


# Zeige aktuelles Verzeichnis vor dem Wechsel
echo "Start-Verzeichnis: $(pwd)"

source /cfs/earth/scratch/vollmflo/exercises-2025/init_micromamba.sh
micromamba activate ultralytics

# Prüfe, ob die Umgebung wirklich aktiv ist
echo "Genutztes Python: $(which python)"

# Gehe ein Verzeichnis hoch
cd ..
echo "Neues Verzeichnis: $(pwd)"

python python_files/inference.py
