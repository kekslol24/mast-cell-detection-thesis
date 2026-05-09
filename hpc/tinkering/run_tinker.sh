#!/usr/bin/env bash
#SBATCH --job-name=tinker_default
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

# cd from hpc/tinkering/ to hpc/ so the script resolves data + imports identically
# to the production run_training.sh.
cd ..
echo "Neues Verzeichnis: $(pwd)"

# Tinkering knobs come from sbatch --export=ALL,KNOB=VALUE. Defaults inside
# ba_tinker.py reproduce the P3-A baseline when nothing is set.
# TUNER_CFG=path/to/best_hyperparameters.yaml activates Step 3/4 mode (yaml
# is the hyperparameter base; explicitly env-set knobs override).
echo "TUNER_CFG=${TUNER_CFG:-(none)} \
FREEZE=${FREEZE:-default} LABEL_SMOOTHING=${LABEL_SMOOTHING:-default} \
DEGREES=${DEGREES:-default} DFL=${DFL:-default} HSV_V=${HSV_V:-default} \
MIXUP=${MIXUP:-default} CLS_LOSS_WEIGHT=${CLS_LOSS_WEIGHT:-default}"

python tinkering/ba_tinker.py
