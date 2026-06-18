---
run_id: P2-G
phase: Phase 2
slurm: 319934
job_name: not recorded
results_dir: yolo_runs_hpc_base_final_dl_fv_bg_ratio_2/
script: ba_improved_P2.py
date: not recorded
status: complete (clean — wide SLURM ID gap, submitted in isolation)
---

## Overview

First attempt to reduce false positives by adding unlabeled slide tiles (background images
with no mast cells) to training. The hypothesis was that exposing the model to confirmed
background texture would teach it to suppress detections on dense slide backgrounds,
reducing the precision/recall imbalance seen in P2-C. Background images were added at
BG_RATIO=2 (two unlabeled tiles per annotated image). This run also used lr0=0.01.

Note: the BG_RATIO mechanism had a silent bug at this point — see infrastructure note below.

## Infrastructure Note: BG_RATIO Bug (discovered 2026-05-06, fixed in P2-H)

`bg_paths` was sampled in `__main__` but **never passed into `fold_tasks`**, so background
images never reached the training workers. All runs with `BG_RATIO > 0` prior to the fix
were effectively `BG_RATIO = 0`. The "backgrounds" YOLO reported scanning were just FP
negatives × oversample (e.g., 330 × 3 = 990 entries from the FP Excel, not background tiles).

**Implication for P2-G:** This run's BG_RATIO=2 was silently ignored. The run is effectively
identical in data composition to P2-C (DL_Modell_FV.pt, FP_NEG_OVERSAMPLE=1, no background).
The performance difference vs P2-C (mAP50 0.530 vs 0.469) is therefore attributable to some
other factor — possibly the wide SLURM gap (cleaner job isolation), or script-state differences
at submission time. The BG_RATIO=2 effect was never measured by this run.

## Setup

| Parameter | Value |
|-----------|-------|
| Base model | `DL_Modell_FV.pt` (domain-specific) |
| Dataset | P2 |
| FP_NEG_OVERSAMPLE | 1 |
| BG_RATIO | 2 (claimed) / **0 (actual, due to bug)** |
| lr0 | 0.01 (YOLO default) |
| SLURM isolation | Clean — wide ID gap |

## Test Results

| Fold | mAP50-95 | mAP50  | Precision | Recall |
|------|----------|--------|-----------|--------|
| 1    | 0.3537   | 0.4683 | 0.9201    | 0.4178 |
| 2    | 0.4317   | 0.5649 | 0.4949    | 0.5617 |
| 3    | 0.3502   | 0.4814 | 0.8883    | 0.4215 |
| 4    | 0.3350   | 0.4576 | 0.8872    | 0.4406 |
| 5    | 0.5118   | 0.6799 | 0.7364    | 0.5795 |
| **mean** | **0.396** | **0.530** | **0.785** | **0.484** |
| **std**  | **0.075** | **0.090** | — | — |

## Fold-Level Notes

Validation training curves showed folds 1, 2, 5 reaching peak mAP50 ~0.945, but **folds 3
and 4 peaked at epochs 7 and 43 respectively and never improved** (training ran to the
patience limit). This early-plateau pattern is a **learning rate signature**: with lr0=0.01
and a domain-pretrained model, the model overshoots its optimal weights in the first
post-warmup steps. On unlucky fold splits (folds 3 and 4), there is not enough training
signal to recover. The learning rate, not the data ratio, is the primary bottleneck.

## Conclusion

Mixed outcome. Precision improved vs P2-D (0.785 vs 0.675) but recall dropped (0.484 vs
0.655) and mean mAP50 is lower (0.530 vs 0.590). The fold collapse in folds 3 and 4
confirms that lr0=0.01 is the core problem — it is not reliably correctable by data
composition changes. The BG_RATIO bug means no valid conclusion can be drawn about
background sampling from this run. The effective BG comparison was inadvertently done
later in the DoE grid (P2-H baseline with bug fixed).

---

## Timeline Summary

**run_id:** P2-G | **date:** not recorded | **SLURM:** 319934
**HYPOTHESIS:** Adding background tiles (BG_RATIO=2) will reduce false positives and improve the precision/recall balance.
**CHANGE vs prior:** Added BG_RATIO=2 to P2-C setup. (Actual effect: bug meant BG was never passed to training workers — effectively BG_RATIO=0.)
**RESULT:** Mean mAP50=0.530 ± 0.090, Recall=0.484. Precision up vs P2-D but recall down. Folds 3/4 collapsed early. — **regression** vs P2-D.
**LEARNING:** BG_RATIO bug invalidates any BG conclusions from this run. Fold early-collapse at epochs 7/43 is a learning rate signature — lr0=0.01 is incompatible with DL_Modell_FV.pt fine-tuning on unlucky splits. This conclusively motivates the lr0=0.001 fix.
**VERDICT:** regression / BG_RATIO bug discovered; lr=0.01 confirmed inadequate
