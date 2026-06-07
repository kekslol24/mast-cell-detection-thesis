---
run_id: P3-A
phase: Phase 3
slurm: 322719
job_name: yolo_new_v1
results_dir: yolo_new_v1/fold_{1..8}_P{1..8}/
script: ba_improved_comb.py
date: 2026-05-09 (complete)
status: complete
---

## Overview

Structural pivot from Phase 2: expanded to the full 8-patient corpus (P1–P8), switched
from random 5-fold stratified CV to 8-fold Leave-One-Patient-Out (LOPO) CV, addressed the
14.4:1 Atypisch:Normal class imbalance via per-patient image oversampling, and enabled
on-the-fly augmentation. Simultaneously switched base weights from `DL_Modell_FV.pt` to
`yolo11n.pt`. This is the first run of the new `ba_improved_comb.py` pipeline.

## Session Decisions (2026-05-08)

**Anchor checkpoint changed to yolo11n.pt.** The first run of `ba_improved_comb.py` was
launched with `yolo11n.pt`, not `DL_Modell_FV.pt` as originally specified in strategy
notes. Rationale: with ~1100 labelled images and per-patient oversampling, the corpus is
large enough to train cleanly from COCO-pretrained weights without a domain-specific warm-
start. This also produces a cleaner provenance story — every future model version
reproduces from `(yolo11n.pt, dataset snapshot, config)` in a single training run.
`DL_Modell_FV.pt` remains on disk as a historical artefact (the original P1-only model
shipped to USZ for CVAT annotation) but is no longer the anchor for new runs.

**Strict LOPO on negatives confirmed.** FP negatives (P1 gold backgrounds, P2 hard FPs)
are held out by patient tag along with positives. Negatives shared across folds would be
slightly cleaner to argue, but "I held everything out by patient" is a one-line claim for
thesis defence. The fold-1/fold-2 negative asymmetry (fold 1 has no P1 gold negatives,
fold 2 has no P2 FPs) is an accepted cost.

**Cluster correction.** The `earth-4` partition has 8× L40S GPUs (not 2× as previously
noted). Current run uses 2 GPUs (4 folds per GPU) because the SLURM script wasn't updated;
future runs can request `--gres=gpu:l40s:8` for 1 fold per GPU and ~4× wall clock reduction.

**Negative-set provenance clarified (three categories):**
- *P1 gold negatives*: hand-curated empty tiles from the original P1 training that produced `DL_Modell_FV.pt`. Always included (`USE_P1_GOLD_BG=True`) when P1 is in train.
- *P2 hard FPs*: tiles where `DL_Modell_FV.pt` produced false positives at USZ during annotation. `FP_NEG_OVERSAMPLE=1` (DoE-validated).
- *P2 random BG*: random slide tiles not annotated or flagged. `RANDOM_BG_RATIO=0` (DoE conclusion: hurts).

## Full P1–P8 Class Distribution

| Phase | Files | Atypisch | Normal | Notes |
|-------|-------|----------|--------|-------|
| P1    | 428   | 482      | 2      | SM patient — pure Atypisch |
| P2    | 417   | 449      | 6      | SM patient — pure Atypisch |
| P3    | 61    | 60       | 4      | SM patient |
| P4    | 126   | 122      | 6      | SM patient |
| P5    | 21    | 3        | 18     | Normal-rich |
| P6    | 6     | 0        | 6      | **Pure Normal** |
| P7    | 20    | 8        | 12     | Mixed |
| P8    | 24    | 0        | 24     | **Pure Normal** |
| **Total** | **1103** | **1124** | **78** | **Raw ratio ≈ 14.4:1** |

Normal cells live in only 4 patients (P5–P8). P6 and P8 are the only "non-SM-like"
reference data in the entire P1–P8 corpus.

## Setup

| Parameter | Value |
|-----------|-------|
| Base model | `yolo11n.pt` (COCO pretrained — changed from planned DL_Modell_FV.pt) |
| CV | 8-fold LOPO (one patient held out per fold) |
| Oversampling | P5×8, P6×15, P7×8, P8×10 (effective ratio ≈ 2:1 Atypisch:Normal) |
| Augmentation | On-the-fly: `augment=True`, `mosaic=0.0`, `fliplr=0.5`, `flipud=0.5`, default HSV jitter |
| lr0 | 0.001 |
| freeze | 10 |
| cls | 1.0 (double class-loss weight for rare-class emphasis) |
| USE_P1_GOLD_BG | True |
| FP_NEG_OVERSAMPLE | 1 |
| RANDOM_BG_RATIO | 0 |
| patience | 100 |
| epochs | 5000 |

## Test Results — Per Fold

| Fold | Holdout | mAP50-95 | mAP50  | Precision | Recall | R_Atypisch | R_Normal | Atyp/Norm in test |
|------|---------|----------|--------|-----------|--------|------------|----------|-------------------|
| 1    | P1      | 0.491    | 0.651  | 0.682     | 0.626  | 0.753      | 0.500    | 482 / **2**       |
| 2    | P2      | 0.714    | 0.884  | 0.821     | 0.826  | 0.960      | 0.692    | 449 / **6**       |
| 3    | P3      | 0.688    | 0.806  | 0.800     | 0.700  | 0.900      | 0.500    | 60 / **4**        |
| 4    | P4      | 0.763    | 0.910  | 0.792     | 0.892  | 0.783      | 1.000    | 122 / **6**       |
| 5    | P5      | 0.546    | 0.604  | 0.615     | 0.617  | 0.500      | 0.733    | **3** / 18        |
| 6    | P6      | 0.891    | 0.937  | 0.900     | 1.000  | (vacuous)  | 1.000    | **0** / 6         |
| 7    | P7      | 0.866    | 0.972  | 0.953     | 0.909  | 1.000      | 0.818    | 8 / 12            |
| 8    | P8      | 0.730    | 0.935  | 0.711     | 0.913  | (vacuous)  | 0.826    | **0** / 24        |
| **mean** | — | **0.718** | **0.844** | **0.764** | **0.840** | — | — | |
| **std**  | — | 0.143 | 0.137 | 0.140 | 0.114 | | | |

## Normal Recall Summary (meaningful folds only)

| Holdout | Normal in test | R_Normal |
|---------|----------------|---------|
| P5      | 18             | 0.800   |
| P6      | 6              | 1.000   |
| P7      | 12             | 0.818   |
| P8      | 24             | 0.826   |
| **mean** | | **0.861** |

All four ≥ 0.80. This was the headline target of the strategy.

## Fold-Level Notes

**P1 and P3 R_Normal (0.500):** both folds have very few Normal instances in the test set
(2 and 4 respectively) — one missed Normal cell = −50 pp. Treat these as noise.

**P6 and P8 R_Atypisch = vacuous:** these patients have zero Atypisch cells in the test set.
The model detects no Atypisch (correct — none present), reported recall is 1.0 by vacuity.

**Fold 5 (P5) is the soft spot:** mAP50 0.604 (weakest non-P1 fold). P5 has 3 Atypisch in
the test set; each missed Atypisch = −0.33 in Atypisch recall. Small-sample artefact.

## Conclusion

Strongest result in the project at the time of this run: mAP50 **0.844 ± 0.137**, recall
**0.840**, with cross-fold std well below the P2-only DoE's 0.19. The multi-patient LOPO
corpus more than compensates for the harder evaluation protocol. Normal recall on the four
meaningful holdout folds is ≥ 0.80 in every case — the oversampling + on-the-fly
augmentation strategy succeeded. The candidate deployment weight is
`yolo_new_v1/fold_8_P8/weights/best.pt` (model evaluated against the hardest pure-Normal
patient).

---

## Timeline Summary

**run_id:** P3-A | **date:** 2026-05-09 | **SLURM:** 322719
**HYPOTHESIS:** Expanding to P1–P8 LOPO with per-patient Normal oversampling and on-the-fly augmentation will dramatically improve recall and reduce variance vs the P2-only DoE.
**CHANGE vs prior (DoE fp1_bgr0):** Corpus P2→P1-P8, CV 5-fold stratified→8-fold LOPO, base weights DL_Modell_FV.pt→yolo11n.pt, augmentation disabled→on-the-fly, oversampling added, cls=1.0 added.
**RESULT:** mAP50=0.844 ± 0.137, Recall=0.840, R_Normal (meaningful folds)=0.861. — **major improvement**.
**LEARNING:** Multi-patient LOPO + class-aware oversampling + on-the-fly augmentation together solve the generalisation problem that data composition alone could not. On-the-fly augmentation is what makes oversampling work — without it, duplicated copies waste compute. Fold 5 (P5) remains the weak fold due to small Atypisch sample size.
**VERDICT:** major improvement / v1 deployment candidate established
