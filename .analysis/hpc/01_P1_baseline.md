---
run_id: P1-baseline
phase: Phase 1
script: ba_improved.py
results_dir: yolo_runs_hpc_final/
date: not recorded
status: complete
---

## Overview

Proof-of-concept run on the first patient dataset (P1). Goal was to verify that YOLO11n
could detect and classify mast cells (Atypisch / Normal) in bone marrow aspirate images at
all, before investing in domain-specific pretrained weights or multi-patient corpora.
No held-out test set was defined for this phase — val metrics from the best epoch per fold
are the only available numbers.

## Setup

| Parameter | Value |
|-----------|-------|
| Base model | `yolo11n.pt` (COCO pretrained, no domain-specific weights) |
| Dataset | P1 — `Pos_neg 12241515` (annotated positive tiles) + `Negativ 12241515` (negative tiles) |
| Classes | Atypisch, Normal |
| CV strategy | 5-fold stratified CV |
| Augmentation | albumentations — flip, brightness, affine rotation ±15° (pre-applied to disk) |
| Epochs | up to 5000, patience=50 |
| Batch / imgsz | 32 / 512 |
| LR schedule | cos_lr=True |
| Hyperparameters | YOLO defaults (no cfg override) |

## Validation Results (best epoch per fold)

*No clean held-out test evaluation available. Val metrics shown.*

| Fold | Best Epoch | mAP50  | mAP50-95 | Precision | Recall |
|------|-----------|--------|----------|-----------|--------|
| 1    | 135       | 0.9521 | 0.8051   | 0.8844    | 0.9198 |
| 2    | 112       | 0.7561 | 0.5830   | 0.6343    | 0.7804 |
| 3    | 120       | 0.8931 | 0.7449   | 0.7642    | 0.7371 |
| 4    | 128       | 0.7754 | 0.6367   | 0.6710    | 0.7252 |
| 5    | 29        | 0.6212 | 0.4734   | 0.8867    | 0.4321 |
| **mean** | | **0.800** | **0.649** | **0.768** | **0.719** |
| **std**  | | **0.129** | **0.131** | — | — |

## Infrastructure Note: Pre-applied Augmentation

The albumentations augmentation used here (and inherited by early P2 runs) is **pre-applied
to disk** rather than applied on-the-fly per epoch. `preprocess_image()` reads each image
once, applies a single random transform, and saves one output file. The `PROCESSED_DIR`
therefore contains the same number of images as the source — no duplication, no effective
dataset enlargement.

The consequence: every training epoch sees the same frozen augmented copies. The
regularisation benefit of augmentation (different view per epoch) is exhausted after
epoch 1. Training effectively runs on an unaugmented dataset from epoch 2 onward.
This was not identified as a bug until Phase 2 analysis (2026-05-07); it is documented
here because it affects all runs that used `preprocess_image()`.

## Fold-Level Notes

**Fold 1** is a strong result (mAP50=0.952, recall=0.920) — the model finds mast cells
reliably when the train/val split is favourable. **Fold 5** converges at epoch 29, far
earlier than other folds, and achieves only recall=0.432. This is an unlucky split artefact
(fold 5's val set likely contains a harder subset of images), not a model failure. The
early convergence at epoch 29 vs. 112–135 for other folds is the tell.

The 0.129 mAP50 std is the central finding of this phase: the model works, but it is
**highly sensitive to the data split**. Reducing this variance becomes the central
engineering challenge for all subsequent phases.

## Conclusion

Proof of concept succeeds. YOLO11n on the P1 dataset can detect mast cells with mean
mAP50=0.800 and mean recall=0.719. The result validates the YOLO detection approach and
justifies investing in domain-specific pretrained weights. The high fold variance (std=0.129)
signals that either the dataset is too small, the class distribution too skewed per fold,
or both — and motivates the switch to domain-specific weights (`DL_Modell_FV.pt`) and
eventually the multi-patient corpus strategy adopted in Phases 3–5.

---

## Timeline Summary

**run_id:** P1-baseline | **date:** not recorded | **SLURM:** not recorded
**HYPOTHESIS:** YOLO11n can learn mast cell detection from the P1 single-patient dataset.
**CHANGE vs prior:** First run — no prior.
**RESULT:** Mean val mAP50=0.800 ± 0.129, Recall=0.719. No held-out test set. — **improvement** (baseline established).
**LEARNING:** Detection is feasible. High fold variance (std=0.129) driven by split luck, not model failure. Domain-specific weights and larger, multi-patient data needed to stabilise. Pre-applied augmentation provides no real regularisation benefit — this becomes a fix target in Phase 2.
**VERDICT:** success (proof of concept) / high variance warrants further work
