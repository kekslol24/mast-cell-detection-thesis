#!/usr/bin/env bash
# Submits the hyperparameter tuner on Fold 4 (P4 holdout).
#
# Defaults: 30 iterations × 50 epochs, holdout=P4. Override per-invocation:
#   sbatch --export=ALL,TUNE_HOLDOUT=P3,TUNE_ITERATIONS=20,TUNE_EPOCHS=40 run_tune.sh

sbatch run_tune.sh
echo "Submitted: tune_lopo_p4 → runs/detect/tune/best_hyperparameters.yaml on completion"
