#!/usr/bin/env bash
#SBATCH --job-name=run_gui
#SBATCH --mail-type=fail,end
#SBATCH --time=00-03:00:00
#SBATCH --partition=earth-4
#SBATCH --constraint=rhel8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --gres=gpu:l40s:1


# Zeige aktuelles Verzeichnis vor dem Wechsel
echo "Start-Verzeichnis: $(pwd)"

source /cfs/earth/scratch/vollmflo/exercises-2025/init_micromamba.sh
micromamba activate ultralytics

# Prüfe, ob die Umgebung wirklich aktiv ist
echo "Genutztes Python: $(which python)"

NODE_NAME=$(hostname)
echo "====================================================="
echo "Gradio läuft auf Knoten: $NODE_NAME"
echo "Port: 7860"
echo "====================================================="


python app.py
