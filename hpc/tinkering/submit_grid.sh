#!/usr/bin/env bash
# Submits one SLURM job per (LABEL_SMOOTHING, DEGREES, DFL, FREEZE) combination.
# Output directories are named after the job, e.g.
#   tinker_ls0.1_deg5_dfl2.0_fr10/
#
# Full grid = 2 × 3 × 2 × 2 = 24 jobs. Each job is one full LOPO run (~1 day on
# 2× L40S), so the full grid is expensive. To narrow:
#   - fix one factor (e.g. `for fr in 10; do` → 12 jobs)
#   - drop a level (e.g. `for deg in 0 5; do` → 16 jobs)
#   - run a "vs P3-A baseline" probe by setting all factors to a single non-default
#     value at a time (4 jobs) before committing to the full grid.

LABEL_SMOOTHING_VALUES="0.0 0.1"
DEGREES_VALUES="0 5 10"
DFL_VALUES="1.5 2.0"
FREEZE_VALUES="0 10"

for ls in $LABEL_SMOOTHING_VALUES; do
  for deg in $DEGREES_VALUES; do
    for dfl in $DFL_VALUES; do
      for fr in $FREEZE_VALUES; do
        JOB_NAME="tinker_ls${ls}_deg${deg}_dfl${dfl}_fr${fr}"
        sbatch \
          --job-name="${JOB_NAME}" \
          --export=ALL,LABEL_SMOOTHING=${ls},DEGREES=${deg},DFL=${dfl},FREEZE=${fr} \
          run_tinker.sh
        echo "Submitted: ls=${ls} deg=${deg} dfl=${dfl} fr=${fr} → ${JOB_NAME}/"
      done
    done
  done
done
