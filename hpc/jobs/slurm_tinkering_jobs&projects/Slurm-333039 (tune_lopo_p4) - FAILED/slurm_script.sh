#!/usr/bin/env bash
#SBATCH --job-name=tune_lopo_p4
#SBATCH --mail-type=fail,end
#SBATCH --time=03-00:00:00
#SBATCH --partition=earth-4
#SBATCH --constraint=rhel8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32GB
#SBATCH --gres=gpu:l40s:1


# Zeige aktuelles Verzeichnis vor dem Wechsel
echo "Start-Verzeichnis: $(pwd)"

source /cfs/earth/scratch/vollmflo/exercises-2025/init_micromamba.sh
micromamba activate ultralytics

# Prüfe, ob die Umgebung wirklich aktiv ist
echo "Genutztes Python: $(which python)"

# cd from hpc/tinkering/ to hpc/ so `from ba_improved_comb import ...` resolves.
cd ..
echo "Neues Verzeichnis: $(pwd)"

export PYTHONPATH=$(pwd):$PYTHONPATH

python tinkering/tune_hyperpara.py
