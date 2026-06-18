---
run_id: P2-E
phase: Phase 2
slurm: 319536
job_name: "train p2 base dl_fv hyperpara"
script: ba_improved_P2.py
date: not recorded
status: FAILED (catastrophic forgetting — all folds collapsed)
---

## Overview

Attempt to improve on P2-C/D by applying hyperparameters from YOLO's built-in tuner
(`model.tune()`), which had been run for 300 epochs × 100 iterations on the P2 data
(SLURM 269569). The hypothesis was that data-tuned hyperparameters would outperform
YOLO defaults. Instead, all five folds collapsed within 3–6 epochs.

## Setup

| Parameter | Value |
|-----------|-------|
| Base model | `DL_Modell_FV.pt` (domain-specific) |
| Dataset | P2 — same as P2-A/B/C/D |
| FP_NEG_OVERSAMPLE | 1 |
| BG_RATIO | 0 |
| cfg | `best_hyperparameters.yaml` from `model.tune()` (includes `lr0=0.01` — from-scratch recipe) |
| Tuner origin | SLURM 269569 — 300 epochs × 100 iterations on P2 data, **from-scratch recipe** |

## Test Results

| Fold | mAP50-95 | mAP50  | Precision | Recall |
|------|----------|--------|-----------|--------|
| 1    | 0.3292   | 0.4508 | 0.9033    | 0.4005 |
| 2    | 0.3205   | 0.4377 | 0.8816    | 0.4023 |
| 3    | 0.3215   | 0.4535 | 0.8962    | 0.4242 |
| 4    | 0.3308   | 0.4556 | 0.9092    | 0.4124 |
| 5    | 0.3243   | 0.4429 | 0.8856    | 0.4269 |
| **mean** | **0.325** | **0.448** | **0.895** | **0.413** |
| **std**  | **0.005** | **0.007** | — | — |

Training collapse evidence: validation `cls_loss` reached 5.8+ within 3–6 epochs.
Normal range for a healthy fine-tuning run: 0.5–1.5.

## Failure Analysis

**Root cause:** Recipe mismatch between tuner and deployment.

YOLO's built-in tuner (`model.tune()`) searches hyperparameters assuming the model trains
from scratch. The tuner that produced `best_hyperparameters.yaml` used `lr0=0.01`
and no freeze — the standard recipe for initialising from COCO weights. When these
hyperparameters are applied to fine-tuning from `DL_Modell_FV.pt` (a domain-specific
checkpoint), the high initial learning rate overwhelms the pretrained features before any
task-specific adaptation can occur. The randomly re-initialised `cv3` classification head
produces gradient spikes that propagate back through the neck and backbone at `lr0=0.01`,
erasing the domain knowledge.

**What the metrics show:** The near-zero variance (std=0.005) is *not* stability. It is all
five folds collapsing to the **same degenerate solution** — a model that detects something
but classifies poorly. Precision remains high (0.895) because the model is very conservative
(it only predicts when extremely confident), but recall at 0.413 means it misses 58% of cells.
The symptom is identical to P2-C but more severe (mAP50 0.448 vs 0.469 in P2-C).

**Corrected conclusion:** The original log entry stated "the YOLO tuner cannot be applied
to fine-tuning." This is too broad. The correct conclusion is: *a tuner whose search recipe
differs from the deployment recipe cannot safely contribute its output to deployment.* The
failure mechanism is the recipe mismatch, not the tuner per se. A tuner run on the *same*
recipe as deployment (same lr0, same freeze, same anchor) is structurally safe. This
reframing motivates the Phase 3 tinkering campaign's separate `tune_hyperpara.py` which
runs the tuner on the production recipe (P3-C).

## Conclusion

**Total failure.** The `cfg=CFG_PATH` argument must never be used when fine-tuning from
`DL_Modell_FV.pt` with hyperparameters calibrated for scratch training. This is noted as
a hard constraint in the `CLAUDE.md` configuration documentation. The tuner-generated
`best_hyperparameters.yaml` from SLURM 269569 is permanently retired for fine-tuning use.

---

## Timeline Summary

**run_id:** P2-E | **date:** not recorded | **SLURM:** 319536
**HYPOTHESIS:** YOLO tuner-optimised hyperparameters applied to DL_Modell_FV.pt will outperform YOLO defaults.
**CHANGE vs prior:** Added `cfg=best_hyperparameters.yaml` (from-scratch tuner output) to the P2-C setup.
**RESULT:** Mean mAP50=0.448 ± 0.005, Recall=0.413. All folds collapsed within 3–6 epochs (cls_loss > 5.8). — **total failure**.
**LEARNING:** Applying a from-scratch tuner's hyperparameters to a fine-tuning run causes catastrophic forgetting. The failure mechanism is recipe mismatch, not an inherent tuner limitation. The `cfg=best_hyperparameters.yaml` argument from SLURM 269569 is permanently retired. A tuner run on the production recipe would be safe — tested later in P3-C.
**VERDICT:** total failure / critical negative finding
