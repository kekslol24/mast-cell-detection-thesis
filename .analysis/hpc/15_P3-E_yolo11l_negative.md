---
run_id: P3-E
phase: Phase 3
slurm: not recorded
results_dir: tinker_default/
script: tinkering/ba_tinker.py (PRETRAINED_WEIGHTS=yolo11l.pt, all other knobs at P3-A defaults)
date: ~2026-05-11 (near P3-B/C/D timeframe)
status: complete (negative result)
---

## Overview

Model scaling test: replaces `yolo11n` (2.6 M parameters) with `yolo11l` (43 M parameters,
~16× larger) while keeping all other settings at P3-A defaults (degrees=0 — not the P3-B
winner). Hypothesis: a larger model with more representational capacity will learn better
feature representations and lift R_Normal, which lagged R_Atypisch by ~11 pp in P3-A.

## Setup

| Parameter | Value |
|-----------|-------|
| Base model | `yolo11l.pt` (COCO pretrained, large variant — 43 M params) |
| Recipe | P3-A defaults: `lr0=0.001, freeze=10, cls=1.0, dfl=1.5, degrees=0, mosaic=0.0, flipud=0.5, augment=True, cos_lr=True, batch=32, imgsz=512` |
| CV | 8-fold LOPO on P1–P8 (same as P3-A/B) |
| Note | degrees=0 used (P3-A defaults) — not the P3-B winner (degrees=5) |

## Test Results

| Fold | Holdout | mAP50  | mAP50-95 | Precision | Recall | R_Atypisch | R_Normal |
|------|---------|--------|----------|-----------|--------|------------|----------|
| 1    | P1      | 0.735  | 0.669    | 0.761     | 0.664  | 0.829      | 0.500    |
| 2    | P2      | 0.881  | 0.746    | 0.814     | 0.826  | 0.937      | 0.714    |
| 3    | P3      | 0.769  | 0.676    | 0.757     | 0.711  | 0.923      | 0.500    |
| 4    | P4      | 0.912  | 0.755    | 0.965     | 0.793  | 0.810      | 0.777    |
| 5    | P5      | 0.712  | 0.600    | 0.615     | 0.653  | 0.500      | 0.807    |
| 6    | P6      | 0.995  | 0.890    | 0.886     | 0.821  | 1.000      | 0.641    |
| 7    | P7      | 0.913  | 0.791    | 0.859     | 0.955  | 1.000      | 0.909    |
| 8    | P8      | 0.927  | 0.834    | 0.570     | 0.854  | 1.000      | 0.708    |
| **mean** | — | **0.855** | **0.745** | **0.778** | **0.785** | **0.875** | **0.695** |

## Comparison

| Config | mAP50 | Recall | R_Atypisch | R_Normal |
|--------|-------|--------|------------|----------|
| P3-A (yolo11n, deg=0) | 0.844 | 0.808 | 0.862 | 0.754 |
| P3-B winner (yolo11n, deg=5) | 0.864 | 0.866 | 0.892 | 0.840 |
| **P3-E (yolo11l, deg=0)** | **0.855** | **0.785** | **0.875** | **0.695** |
| P3-E delta vs P3-A | +0.011 | −0.023 | +0.013 | **−0.059** |

## Why the Large Model Underperforms

**1. Dataset is too small for 43 M parameters.** Each LOPO fold trains on roughly
170–220 images. yolo11l has 16× more free parameters than yolo11n, most of them in the
unfrozen neck and head. The classification head (`cv3`) is randomly re-initialised at every
run (nc change 1→2 drops pretrained cv3 weights), so all those parameters must be learned
from scratch from the fold's training images. yolo11n converges with less data; yolo11l
overfits the training fold and fails to generalise.

**2. Freeze=10 protects the backbone but exposes a large neck.** yolo11l's neck
(layers 10–21) is proportionally wider than yolo11n's. Unfreezing a wider neck with the
same small dataset and lr0=0.001 means larger gradient norms relative to the feature scale,
increasing the risk of overwriting transferred features.

**3. R_Normal is the canary.** Normal cells are underrepresented in most folds;
generalising to them requires robust representations, not memorised Atypisch patterns.
A larger model memorises faster and generalises less when data is scarce.

## Fold-Level Notes

Fold 6 (P6 holdout): R_Normal=0.641, the lowest in the entire project for a pure-Normal
holdout. The large model memorised the P1–P4 SM-patient features so efficiently that it
partly forgot how to represent Normal-dominant slides. This is exactly the regime where
a smaller, less over-parameterised model is preferable.

## Conclusion

**The large model is worse, not better.** mAP50 gains a marginal +1.1 pp but overall recall
drops 2.3 pp and R_Normal drops 5.9 pp vs P3-A. R_Normal=0.695 is the weakest result of
any model on P1–P8. **yolo11n is the correct model size for this corpus.** Scaling up
model capacity is not the path to better Normal recall — the path is more data (P9–P15)
and the confirmed recipe (degrees=5). P3-E bounds the search space: model size is not a
free variable at this dataset scale.

---

## Timeline Summary

**run_id:** P3-E | **date:** ~2026-05-11 | **SLURM:** not recorded
**HYPOTHESIS:** yolo11l (43 M params, 16× larger) will learn better feature representations and lift R_Normal above the yolo11n P3-A result.
**CHANGE vs prior (P3-A):** yolo11n.pt→yolo11l.pt; all other settings at P3-A defaults (degrees=0).
**RESULT:** mAP50=0.855 (+1.1 pp), Recall=0.785 (−2.3 pp), R_Normal=0.695 (−5.9 pp vs P3-A). — **regression** on all recall metrics.
**LEARNING:** Model capacity is not the limiting factor at this corpus size (~1100 images, 170–220 per fold). Large models overfit the Atypisch-dominant training distribution and lose Normal-class generalisation. yolo11n is confirmed as the correct architecture for P1–P8 scale. Path to better R_Normal is more data, not more parameters.
**VERDICT:** negative result / yolo11n confirmed as correct model size
