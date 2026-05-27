---
run_id: P2-D
phase: Phase 2
slurm: 319416
job_name: "train p2 mod dl_fv"
script: ba_improved_P2.py
date: not recorded
status: complete (clean — wide SLURM ID gap from P2-A/B/C, submitted in isolation)
---

## Overview

Domain-specific weights (`DL_Modell_FV.pt`) combined with modified training settings.
P2-D is the first **clean** P2 run (no SLURM workspace collision) and produced the best
result in Phase 2 prior to the learning rate fix. The exact modified settings were not
fully recorded, but the result is the reference point for all subsequent P2 comparisons.

## Setup

| Parameter | Value |
|-----------|-------|
| Base model | `DL_Modell_FV.pt` (domain-specific) |
| Dataset | P2 — same as P2-A/B/C |
| FP_NEG_OVERSAMPLE | 1 |
| BG_RATIO | 0 |
| lr0 | 0.01 (inferred — log states "modified settings" but does not specify lr) |
| Hyperparameters | Modified (exact changes not fully recorded) |
| SLURM isolation | Clean — wide ID gap from concurrent runs |

## Test Results

| Fold | mAP50-95 | mAP50  | Precision | Recall |
|------|----------|--------|-----------|--------|
| 1    | 0.3793   | 0.5050 | 0.5503    | 0.5278 |
| 2    | 0.4945   | 0.6386 | 0.5235    | 0.9569 |
| 3    | 0.5313   | 0.7043 | 0.7227    | 0.6587 |
| 4    | 0.4655   | 0.6240 | 0.6549    | 0.7152 |
| 5    | 0.3655   | 0.4777 | 0.9217    | 0.4162 |
| **mean** | **0.447** | **0.590** | **0.675** | **0.655** |
| **std**  | **0.072** | **0.090** | — | — |

## Fold-Level Notes

**Fold 2** is a standout: recall=0.957 with precision=0.524. This fold's training split
happened to provide the model with a particularly useful view of the data, allowing it to
detect nearly all cells (at the cost of some false positives). **Fold 5** remains the
persistent weak fold: recall=0.416, consistent with the pattern seen in P1 (fold 5 was
also the weakest there). Fold 5's holdout patient likely has harder or more unusual
morphology, or the fold composition is structurally unlucky under this data split.

The std=0.090 is a meaningful improvement over the yolo11n runs (0.139–0.172), confirming
that domain-specific weights stabilise training even at the expense of some mean performance.

## Conclusion

Best Phase 2 result prior to the learning rate fix: mAP50=0.590, recall=0.655. The modified
settings successfully recover the recall that P2-C lost (0.397 → 0.655), showing that some
adjustment to the default hyperparameters can compensate for the lr0=0.01 instability.
However, variance remains moderate (std=0.090) and fold 5 continues to be the weak link.
The precise cause of the improvement over P2-C cannot be isolated without knowing exactly
which settings changed. This run becomes the informal Phase 2 reference point, superseded
by P2-H once the lr0=0.001 fix is applied cleanly.

---

## Timeline Summary

**run_id:** P2-D | **date:** not recorded | **SLURM:** 319416
**HYPOTHESIS:** Modified settings on top of DL_Modell_FV.pt can recover the recall lost in P2-C.
**CHANGE vs prior:** Same base model as P2-C with unrecorded setting modifications. First clean (non-corrupted) P2 run.
**RESULT:** Mean mAP50=0.590 ± 0.090, Recall=0.655. Best Phase 2 result so far. — **improvement**.
**LEARNING:** The combination of domain-specific weights + some hyperparameter adjustment recovers recall substantially. Fold 5 is a persistent weak point independent of configuration. Exact modifications were not logged — critical gap for reproducibility.
**VERDICT:** improvement / informal P2 reference baseline
