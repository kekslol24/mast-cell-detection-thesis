#!/usr/bin/env bash
#SBATCH --job-name=tinker_final
#SBATCH --mail-type=fail,end
#SBATCH --time=00-08:00:00
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

# cd from hpc/tinkering/ to hpc/ so the script resolves data + imports identically
# to ba_tinker.py.
cd ..
echo "Neues Verzeichnis: $(pwd)"

# Recipe defaults inside ba_tinker_final.py reproduce the LOPO matrix winner
# `tinker_ls0.0_deg5_dfl1.5_fr10`. Override any knob via sbatch --export.
echo "FREEZE=${FREEZE:-default(10)} DEGREES=${DEGREES:-default(5.0)} \
DFL=${DFL:-default(1.5)} CLS_LOSS_WEIGHT=${CLS_LOSS_WEIGHT:-default(1.0)} \
LABEL_SMOOTHING=${LABEL_SMOOTHING:-default(0.0)} HSV_V=${HSV_V:-default(0.4)} \
MIXUP=${MIXUP:-default(0.0)} BG_RATIO=${BG_RATIO:-default(1)} \
FP_NEG_OVERSAMPLE=${FP_NEG_OVERSAMPLE:-default(1)} \
VAL_FRACTION=${VAL_FRACTION:-default(0.15)}"

export PYTHONPATH=$(pwd):$PYTHONPATH

python tinkering/ba_tinker_final.py
