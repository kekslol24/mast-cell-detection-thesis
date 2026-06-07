---
run_id: P2-B
phase: Phase 2
slurm: 319400
job_name: "train p2 mod nano"
script: ba_improved_P2.py
date: not recorded
status: corrupted (folds 2, 3, 5 share identical values with P2-A — SLURM workspace collision)
---

## Overview

Submitted concurrently with P2-A (319399), intending to test modified training settings
alongside the base configuration. Corrupted by the same SLURM workspace collision as P2-A.
The exact setting modifications were not recorded in the log.

## Setup

| Parameter | Value |
|-----------|-------|
| Base model | `yolo11n.pt` (COCO pretrained) |
| Dataset | P2 — same as P2-A |
| FP_NEG_OVERSAMPLE | 1 |
| BG_RATIO | 0 |
| Hyperparameters | Modified (exact changes not tracked in log) |

## Test Results

| Fold | mAP50-95 | mAP50  | Precision | Recall |
|------|----------|--------|-----------|--------|
| 1    | 0.3727   | 0.4985 | 0.5299    | 0.5114 |
| 2*   | 0.7112   | 0.8572 | 0.7222    | 0.8752 |
| 3*   | 0.4801   | 0.6221 | 0.5840    | 0.5939 |
| 4    | 0.4762   | 0.6219 | 0.5478    | 0.7638 |
| 5*   | 0.3678   | 0.4718 | 0.3802    | 0.5905 |
| **mean** | **0.482** | **0.614** | **0.553** | **0.667** |
| **std**  | **0.139** | **0.155** | — | — |

*Folds 2, 3, 5: identical to P2-A to 4 decimal places — confirmed workspace collision.

## Fold-Level Notes

Folds 2 and 3 are bit-for-bit identical to P2-A. This is the smoking gun for the SLURM
collision: two independent training runs cannot produce identical results unless they
trained on the same data. The surviving uncorrupted folds (1 and 4) show slightly better
mAP50 than P2-A (0.498 vs 0.441, 0.622 vs 0.508), suggesting the modified settings had
some effect — but with only two clean folds this cannot be confirmed.

## Conclusion

Results are uninterpretable for the same reason as P2-A. The marginal mean improvement
over P2-A (0.614 vs 0.580) is noise given the corruption. The only takeaway is that two
concurrent SLURM jobs sharing workspace paths will always produce corrupted fold results,
regardless of the hyperparameter changes being tested. Domain-specific weights (P2-C) are
the next meaningful experiment.

---

## Timeline Summary

**run_id:** P2-B | **date:** not recorded | **SLURM:** 319400
**HYPOTHESIS:** Modified training settings improve on P2-A baseline.
**CHANGE vs prior:** Same base as P2-A with unrecorded setting modifications.
**RESULT:** Mean mAP50=0.614 ± 0.155, Recall=0.667. Folds 2/3/5 corrupted (identical to P2-A). — **inconclusive** (corrupted run).
**LEARNING:** Reinforces the SLURM collision finding. Exact modifications not recorded — logging discipline required from here forward. Cannot draw conclusions about yolo11n on P2 from either P2-A or P2-B alone.
**VERDICT:** inconclusive / corrupted
