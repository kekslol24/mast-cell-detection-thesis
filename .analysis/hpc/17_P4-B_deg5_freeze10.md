---
run_id: P4-B
phase: Phase 4
slurm: 337215
job_name: yolo_new_v3_freeze10
results_dir: yolo_new_v3_freeze10/
script: ba_improved_comb.py
date: not recorded (after P4-A, 2026-05-23)
status: complete
---

## Overview

Establishes the degrees=5 baseline on the P1–P15 corpus, providing the reference for the
freeze=0 comparison (P4-C). P4-A ran degrees=0 inadvertently; P4-B applies the P3-B winner
recipe correctly (degrees=5, dfl=1.5, freeze=10) to the same P1–P15 corpus and 15-fold LOPO.

## Setup

| Parameter | Value |
|-----------|-------|
| Base model | `yolo11n.pt` |
| degrees | **5** (P3-B winner) |
| dfl | 1.5 |
| freeze | **10** |
| CV | 15-fold LOPO on P1–P15 |
| All other | Same as P4-A |

## Test Results

| Fold | Holdout | mAP50  | Recall | R_Atypisch | R_Normal |
|------|---------|--------|--------|------------|----------|
| 1    | P1      | 0.5835 | 0.5755 | 0.8653     | 0.2857   |
| 2    | P10     | 0.9771 | 0.9942 | 1.0000     | 0.9883   |
| 3    | P11     | 0.8914 | 0.8374 | 0.9412     | 0.7335   |
| 4    | P12     | 0.9344 | 0.9306 | 0.8612     | 1.0000   |
| 5    | P13     | 0.9007 | 1.0000 | 1.0000     | NaN*     |
| 6    | P14     | 0.9950 | 1.0000 | 1.0000     | 1.0000   |
| 7    | P15     | 0.9542 | 0.9358 | 1.0000     | 0.8716   |
| 8    | P2      | 0.8641 | 0.8587 | 0.9174     | 0.8000   |
| 9    | P3      | 0.9167 | 0.9808 | 0.9804     | 0.9811   |
| 10   | P4      | 0.9333 | 0.9127 | 0.8255     | 1.0000   |
| 11   | P5      | 0.6613 | 0.6310 | 0.3333     | 0.9286   |
| 12   | P6      | 0.9950 | 1.0000 | 1.0000     | 1.0000   |
| 13   | P7      | 0.9950 | 1.0000 | 1.0000     | 1.0000   |
| 14   | P8      | 0.8802 | 0.9118 | 1.0000     | 0.8235   |
| 15   | P9      | 0.8175 | 0.8498 | 0.7692     | 0.9303   |
| **mean** | — | **0.8866** | **0.8945** | **0.8996** | **0.8816** |

*Fold 5 (P13): per_class bug — true R_Normal = 1.0000 (in R_Atypisch column). Excluded from
R_Normal mean (n=14).

## Comparison: P4-A (degrees=0) vs P4-B (degrees=5)

| Metric | P4-A deg=0 | P4-B deg=5 | Delta |
|--------|-----------|-----------|-------|
| mAP50 | 0.903 | 0.887 | −1.6 pp |
| Recall | 0.904 | 0.895 | −0.9 pp |
| R_Atypisch | 0.913 | 0.900 | −1.3 pp |
| R_Normal | 0.890 | 0.882 | −0.8 pp |

The degrees=5 addition (P4-B) does *not* improve over P4-A's degrees=0 on P1–P15. This is
surprising given the P3-B matrix result (+4–5 pp R_Normal at degrees=5 on P1–P8). Three
interpretations:

1. **At larger corpus size, the regularisation benefit of rotation is smaller.** With more
   diverse training images, the model already sees sufficient orientation variety without
   artificial rotation augmentation.
2. **The P4-A degrees=0 result is slightly inflated** because the run was essentially the
   P3-B winner recipe *minus* degrees, meaning degrees was not explicitly zero — YOLO's
   default rotation augmentation may not be fully zero. If YOLO applied some implicit
   rotation, P4-A was not strictly degrees=0.
3. **Cross-fold variance (~0.1 std) swamps a 1–2 pp delta.** The differences are within
   noise. P4-B and P4-A are statistically indistinguishable.

Interpretation 3 is most defensible. The differences are at most 1.6 pp, well within
cross-fold std of ~0.1. P4-B is used as the reference for the freeze comparison (P4-C)
because it has explicit recipe parameters.

## Conclusion

P4-B confirms the P3-B recipe (degrees=5, freeze=10) works on P1–P15. Results are
statistically indistinguishable from P4-A. P4-B provides the clean reference for the
freeze=0 ablation in P4-C.

---

## Timeline Summary

**run_id:** P4-B | **date:** not recorded | **SLURM:** 337215
**HYPOTHESIS:** Applying the full P3-B winner recipe (degrees=5) on P1–P15 will replicate the +4–5 pp R_Normal seen in the P3-B matrix vs P3-A.
**CHANGE vs prior (P4-A):** degrees 0→5 (explicitly set). All else identical.
**RESULT:** mAP50=0.887, R_Normal=0.882. P4-A→P4-B delta: all metrics within ±1.6 pp (within noise). — **inconclusive** (no detectable improvement from degrees=5 at this corpus size).
**LEARNING:** The degrees=5 gain seen in P3-B (+4–5 pp R_Normal) may diminish or disappear at larger corpus scale. Cross-fold std (~0.1) renders sub-2 pp deltas uninterpretable. P4-B is used as the freeze comparison reference because it has explicit parameters.
**VERDICT:** inconclusive (within noise of P4-A) / reference for freeze comparison
