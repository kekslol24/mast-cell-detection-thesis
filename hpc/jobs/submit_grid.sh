#!/usr/bin/env bash
# Submits one SLURM job per (FP_NEG_OVERSAMPLE, BG_RATIO) combination.
# Output directories are named after the job: train_p2_fp{N}_bgr{M}/

for fp in 1 2 3; do
  for bgr in 0 1 2 3; do
    sbatch \
      --job-name="train_p2_fp${fp}_bgr${bgr}" \
      --export=ALL,FP_NEG_OVERSAMPLE=${fp},BG_RATIO=${bgr} \
      run_training.sh
    echo "Submitted: fp=${fp} bgr=${bgr} → train_p2_fp${fp}_bgr${bgr}/"
  done
done
