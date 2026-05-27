---
run_id: P4-C
phase: Phase 4
slurm: 337214
job_name: yolo_new_v3_freeze0
results_dir: yolo_new_v3_freeze0/
script: ba_improved_comb.py
date: not recorded (concurrent with P4-B)
status: complete
---

## Overview

Clean freeze=0 vs freeze=10 comparison on the P1–P15 corpus. P4-B is the reference
(degrees=5, freeze=10); P4-C tests whether unfreezing the full backbone on the larger corpus
(~1000–1300 training images per fold) yields a benefit over the frozen-backbone recipe.

The motivation for this comparison comes from the P3-B matrix: `freeze=10` beat `freeze=0`
by ~2 pp R_Normal on the P1–P8 corpus (~170–220 training images per fold). With 5–7× more
training images in P4, the "corpus too small to unfreeze" argument is weaker. P4-C tests
whether this intuition was correct.

## Setup

| Parameter | Value |
|-----------|-------|
| Base model | `yolo11n.pt` |
| degrees | 5 |
| dfl | 1.5 |
| freeze | **0** (backbone unfrozen — all layers trainable) |
| CV | 15-fold LOPO on P1–P15 |
| All other | Same as P4-B |

## Test Results

| Fold | Holdout | mAP50  | Recall | R_Atypisch | R_Normal |
|------|---------|--------|--------|------------|----------|
| 1    | P1      | 0.5754 | 0.5751 | 0.8645     | 0.2857   |
| 2    | P10     | 0.9525 | 1.0000 | 1.0000     | 1.0000   |
| 3    | P11     | 0.9195 | 0.8029 | 0.8824     | 0.7235   |
| 4    | P12     | 0.9487 | 0.9756 | 0.9513     | 1.0000   |
| 5    | P13     | 0.8764 | 1.0000 | 1.0000     | NaN*     |
| 6    | P14     | 0.9950 | 0.9897 | 0.9795     | 1.0000   |
| 7    | P15     | 0.9732 | 0.9409 | 0.9388     | 0.9431   |
| 8    | P2      | 0.9264 | 0.9118 | 0.9683     | 0.8554   |
| 9    | P3      | 0.8187 | 0.8343 | 0.8686     | 0.8000   |
| 10   | P4      | 0.9288 | 0.9562 | 0.9123     | 1.0000   |
| 11   | P5      | 0.6716 | 0.6310 | 0.3333     | 0.9286   |
| 12   | P6      | 0.9950 | 1.0000 | 1.0000     | 1.0000   |
| 13   | P7      | 0.9950 | 1.0000 | 1.0000     | 1.0000   |
| 14   | P8      | 0.8706 | 0.9308 | 1.0000     | 0.8616   |
| 15   | P9      | 0.8023 | 0.8442 | 0.7417     | 0.9467   |
| **mean** | — | **0.8833** | **0.8928** | **0.8960** | **0.8818** |

*Fold 5 (P13): per_class bug — true R_Normal = 1.0000. Excluded from R_Normal mean (n=14).

## freeze=10 vs freeze=0 Comparison

| Metric | P4-B freeze=10 | P4-C freeze=0 | Delta |
|--------|----------------|---------------|-------|
| mAP50 | 0.8866 | 0.8833 | **−0.33 pp** |
| Recall | 0.8945 | 0.8928 | **−0.17 pp** |
| R_Atypisch | 0.8996 | 0.8960 | **−0.36 pp** |
| R_Normal | 0.8816 | 0.8818 | **+0.02 pp** |

All deltas are within ±0.4 pp — far inside the ±2 pp threshold. Unfreezing the full
backbone on P1–P15 provides **no benefit** over the frozen-backbone recipe.

## Conclusion and Implication

**Decision: freeze=10 confirmed.** The "corpus too small to unfreeze" intuition holds even
at ~1000–1300 training images per fold. The backbone's COCO-pretrained features are
valuable enough that the marginal benefit of end-to-end adaptation is drowned out by the
noise of fine-tuning 3+ M parameters with a small dataset.

**Implication for P1–P53:** No freeze comparison run is needed at the larger corpus scale.
The P3-B winner recipe (`degrees=5, dfl=1.5, freeze=10, lr0=0.001, cls=1.0`) is confirmed
as the deployment recipe and will be used as-is for the P1–P53 baseline run.

This is a satisfying result methodologically: it means the recipe found at small corpus
scale (P1–P8, Phase 3) transfers directly to the larger corpus without needing re-tuning.
The domain properties (orientation invariance, localisation requirements, backbone feature
quality) are the same at any corpus size in this project.

---

## Timeline Summary

**run_id:** P4-C | **date:** not recorded | **SLURM:** 337214
**HYPOTHESIS:** At P1–P15 scale (~1000–1300 training images per fold), unfreezing the backbone (freeze=0) will outperform the frozen recipe (freeze=10) because the corpus has grown past the "too small to unfreeze" threshold.
**CHANGE vs prior (P4-B):** freeze 10→0. All else identical.
**RESULT:** All metrics within ±0.4 pp of P4-B. No meaningful difference. — **inconclusive** (null result / confirms freeze=10).
**LEARNING:** freeze=10 is correct regardless of corpus size at P1–P15 scale. No freeze comparison is needed for P1–P53. The P3-B recipe transfers across corpus scales without re-tuning. This is a methodologically useful negative result: the recipe is robust, not corpus-size-sensitive.
**VERDICT:** null result / freeze=10 confirmed as deployment recipe across scales
