---
run_id: P2-A
phase: Phase 2
slurm: 319399
job_name: "train p2 base nano"
script: ba_improved_P2.py
date: not recorded
status: corrupted (folds 2, 3, 5 unreliable — see SLURM collision bug below)
---

## Overview

First run on the P2 dataset (`1224151atypisch_normal`) using the generic COCO-pretrained
`yolo11n.pt` weights. Baseline to measure how much domain-specific pretrained weights
would help. Run concurrently with P2-B (SLURM 319400), which caused fold-workspace
collisions and corrupted results.

## Infrastructure Note: SLURM Concurrent Job Interference

SLURM does not snapshot the training script at submission time. Jobs execute whatever is
on disk when they start. When two jobs are submitted back-to-back and the script is
updated between submissions, later jobs can pick up newer configuration — wrong
`PROJECT_DIR`, different hyperparameters.

**Three independent collision paths existed:**

1. **`cv_temp` and `processed_data` path collision.** Temporary fold workspaces used
   shared names (`fold_0_workspace/`, etc.). Concurrent jobs skipped creating symlinks
   already present (pointing to the *other* job's data) and overwrote augmented images
   mid-run. Fixed in later runs by namespacing with `$SLURM_JOB_ID`:
   ```python
   TEMP_DIR      = os.path.abspath(f".cv_temp_{_JOB_ID}")
   PROCESSED_DIR = os.path.abspath(f".processed_data_{_JOB_ID}")
   ```

2. **`PROJECT_DIR` race condition.** Both jobs wrote to the same output directory.
   `fold_3/results.csv` contains interleaved epoch rows from two separate training runs.
   The final `val_model.val()` evaluated whichever `best.pt` was last written — a random
   mix. Fixed by deriving `PROJECT_DIR` from `$SLURM_JOB_NAME` (locked at submission):
   ```python
   PROJECT_DIR = f"./{os.environ.get('SLURM_JOB_NAME', 'local_run')}"
   ```

3. **Corrupted runs identified:** P2-A (319399) and P2-B (319400) folds 2, 3, 5 have
   identical results to 4 decimal places — proof both jobs trained on the same fold
   workspace data. Results for these folds are unreliable.

**Clean runs** (wide SLURM ID gaps, submitted in isolation): **P2-D (319416)** and
**P2-G (319934)**.

## Setup

| Parameter | Value |
|-----------|-------|
| Base model | `yolo11n.pt` (COCO pretrained) |
| Dataset | P2 — `1224151atypisch_normal` + FP negatives from `Task 6_1224151_negative.xlsx` |
| FP_NEG_OVERSAMPLE | 1 |
| BG_RATIO | 0 |
| Hyperparameters | YOLO defaults (no cfg override) |
| Augmentation | Pre-applied via `preprocess_image()` (see 01_P1_baseline.md for limitation) |

## Test Results

| Fold | mAP50-95 | mAP50  | Precision | Recall |
|------|----------|--------|-----------|--------|
| 1    | 0.3222   | 0.4405 | 0.8935    | 0.4205 |
| 2*   | 0.7112   | 0.8572 | 0.7222    | 0.8752 |
| 3*   | 0.4801   | 0.6221 | 0.5840    | 0.5939 |
| 4    | 0.3800   | 0.5084 | 0.4774    | 0.6044 |
| 5*   | 0.3678   | 0.4718 | 0.3802    | 0.5905 |
| **mean** | **0.452** | **0.580** | **0.612** | **0.617** |
| **std**  | **0.156** | **0.172** | — | — |

*Folds 2, 3, 5: corrupted by SLURM workspace collision with P2-B. Results unreliable.

## Conclusion

Results are largely uninterpretable due to the SLURM collision. The surviving folds (1, 4)
show weak performance (mAP50 0.44–0.51) consistent with generic COCO weights struggling
on a specialised morphology domain. The high fold variance (std=0.172) is partly artefactual
(corrupted folds) and partly genuine (same instability seen in P1). The collision bug
discovery here motivated the `$SLURM_JOB_ID` and `$SLURM_JOB_NAME` fixes applied to all
subsequent runs.

---

## Timeline Summary

**run_id:** P2-A | **date:** not recorded | **SLURM:** 319399
**HYPOTHESIS:** yolo11n.pt can be fine-tuned on P2 data as a baseline before domain-specific weights.
**CHANGE vs prior:** Switched from P1 dataset to P2 dataset; added FP negatives. Same generic yolo11n weights.
**RESULT:** Mean mAP50=0.580 ± 0.172, Recall=0.617. Folds 2/3/5 corrupted. — **inconclusive** (corrupted run).
**LEARNING:** SLURM concurrent job interference discovered and documented. `$SLURM_JOB_ID` and `$SLURM_JOB_NAME` namespacing fixes required before any results can be trusted. Domain-specific weights hypothesis not yet tested cleanly.
**VERDICT:** inconclusive / corrupted
