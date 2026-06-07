---
run_id: P2-H
phase: Phase 2
slurm: 320698
job_name: "train_p2_fp1_bgr0_new"
results_dir: (same results dir as DoE fp1_bgr0 cell — P2-H is the DoE control)
script: ba_improved_P2.py (BG_RATIO bug fixed)
date: not recorded (DoE grid submitted ~2026-05-06 based on log analysis date)
status: complete
---

## Overview

Clean implementation of the learning rate fix: `lr0=0.001` with `freeze=10` (backbone
frozen), BG_RATIO bug corrected, and pre-applied augmentation disabled. This run serves
simultaneously as the Phase 2 learning rate fix test and as the fp1_bgr0 control cell of
the DoE grid. It is the **Phase 2 reference configuration** for all subsequent comparisons.

## Infrastructure Note: Pre-applied Augmentation Disabled

As of this run, the pre-applied albumentations augmentation (see `01_P1_baseline.md`) is
**fully disabled**: the `preprocess_image()` block is commented out and `augment=False` is
passed to `model.train()`. Analysis (2026-05-07) confirmed that baking transforms to disk
before training exhausts augmentation's regularisation benefit after epoch 1 — every
subsequent epoch sees identical augmented copies. The fix (per-epoch on-the-fly augmentation
via `augment=True` + `mosaic=0.0`) was not yet enabled at this stage, making P2-H a clean
augmentation-free baseline. On-the-fly augmentation is the next planned intervention (P2-I,
which is eventually implemented as P3-A).

## Setup

| Parameter | Value |
|-----------|-------|
| Base model | `DL_Modell_FV.pt` (domain-specific) |
| Dataset | P2 |
| FP_NEG_OVERSAMPLE | 1 |
| BG_RATIO | 0 (bug fixed — background images now correctly passed to fold workers) |
| lr0 | **0.001** (key fix vs prior runs) |
| freeze | **10** (backbone layers 0–9 frozen) |
| patience | 100 |
| epochs | 5000 (early stopping active) |
| Augmentation | Disabled (pre-apply removed, `augment=False`) |

## Test Results

| Fold | mAP50-95 | mAP50  | Precision | Recall |
|------|----------|--------|-----------|--------|
| 1    | 0.5566   | 0.6482 | 0.9644    | 0.4537 |
| 2    | 0.3604   | 0.4562 | 0.9427    | 0.4250 |
| 3    | 0.7568   | 0.9401 | 0.8324    | 0.9118 |
| 4    | 0.3876   | 0.4784 | 0.9647    | 0.4292 |
| 5    | 0.4063   | 0.4991 | 0.9635    | 0.4660 |
| **mean** | **0.494** | **0.604** | **0.934** | **0.537** |
| **std**  | **0.166** | **0.194** | **0.057** | **0.210** |

## Fold-Level Notes

**Fold 3** is the dominant outlier: mAP50=0.940, recall=0.912, while all other folds sit at
mAP50 0.46–0.65 and recall 0.43–0.47. This is not a training instability (the lr fix worked —
no fold collapses at epochs 7/43 as in P2-G). It is a **persistent fold-split artefact**: the
patient(s) in fold 3's holdout are structurally easier to generalise to given this training
split. This pattern repeats in every subsequent DoE grid cell — fold 3 is always the outlier,
which is why the DoE grid's cross-fold std (~0.19) completely dominates any signal from the
FP or BG factors.

**Precision is very high (mean 0.934, std 0.057):** the lr0=0.001 fix produces a stable
model that classifies confidently. The problem has shifted from classification quality to
**generalisation** — the model knows what it has seen but struggles to transfer that to
unseen data.

## Conclusion

The `lr0=0.001 + freeze=10` configuration successfully prevents the fold-collapse seen in
P2-C and P2-G. All folds converge. Mean mAP50=0.604 matches P2-D (0.590) and bests P2-G
(0.530), confirming lr0=0.001 as the correct fine-tuning learning rate. However, the variance
target (std < 0.06 for mAP50) was **not met** — std remains 0.194, driven by fold 3's
outlier performance. The learning rate fix solved convergence stability but not generalisation.
This is now the reference configuration for the DoE grid.

---

## Timeline Summary

**run_id:** P2-H | **date:** not recorded | **SLURM:** 320698
**HYPOTHESIS:** lr0=0.001 + freeze=10 will stabilise fine-tuning from DL_Modell_FV.pt without catastrophic forgetting.
**CHANGE vs prior (P2-G):** lr0 0.01→0.001, freeze=10, BG_RATIO bug fixed, pre-applied augmentation disabled.
**RESULT:** Mean mAP50=0.604 ± 0.194, Recall=0.537. No fold collapses. — **improvement** (stability achieved).
**LEARNING:** lr0=0.001 is the correct fine-tuning rate for DL_Modell_FV.pt. The remaining high variance (std=0.194) is a fold-split artefact (fold 3 outlier), not training instability. The generalisation problem cannot be solved by LR alone — on-the-fly augmentation and more data are the next levers.
**VERDICT:** improvement / P2 reference configuration established
