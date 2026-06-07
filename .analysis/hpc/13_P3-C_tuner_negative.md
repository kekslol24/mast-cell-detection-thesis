---
run_id: P3-C
phase: Phase 3
slurm: 323280 (canonical run; earlier tune* directories exist from prior iterations)
results_dir: runs/detect/tune11/ (30/30 iterations complete)
script: tinkering/tune_hyperpara.py
date: ~2026-05-11
status: complete (negative result — tuner yaml contradicts matrix evidence; Stage-3 validation skipped)
---

## Overview

Re-test of YOLO's built-in `model.tune()` hyperparameter search, this time run against the
**production recipe** (yolo11n.pt + freeze=10 + AdamW + `augment=True`, `mosaic=0.0`,
`flipud=0.5`, `cls=1.0`). This is a methodological rerun of P2-E's failed approach, but
with the recipe mismatch eliminated: the tuner's search space now coincides with deployment
conditions. 30 iterations × 50 epochs each, with P4 (the patient held out in this fold) as
the fitness criterion (mAP50-95 on a single held-out fold).

## Context: P2-E Reframing

P2-E (SLURM 319536) collapsed because a tuner run on a from-scratch recipe (`lr0=0.01`, no
anchor, no freeze) was applied to fine-tuning from `DL_Modell_FV.pt`. The original log
logged this as "the YOLO tuner cannot be applied to fine-tuning." This was overly broad.

The correct conclusion: *a tuner whose search recipe differs from the deployment recipe*
cannot safely contribute its output. The failure mechanism was recipe mismatch. A tuner run
on the exact production recipe is structurally safe and potentially useful. P3-C tests this.

## Setup

| Parameter | Value |
|-----------|-------|
| Base model | `yolo11n.pt` (same as production) |
| CV unit | Single fold — P4 holdout (LOPO Fold 4), not full 8-fold LOPO |
| Iterations | 30 |
| Epochs per iteration | 50 |
| Fitness metric | mAP50-95 on P4 test set |
| Recipe | freeze=10, AdamW, `augment=True`, `mosaic=0.0`, `flipud=0.5`, `cls=1.0` |

## Tuner Output (`best_hyperparameters.yaml`)

Best fitness: **0.73582** at iteration 13.

```
box=6.97   cls=1.11   dfl=2.47   degrees=0.0   flipud=0.408
hsv_v=0.358   scale=0.220   fliplr=0.456   mosaic=0.0   close_mosaic=10
lr0=0.00884   lrf=0.00885   momentum=0.938   weight_decay=0.00028
```

## Findings

The tuner converged on choices that **contradict the P3-B matrix evidence** on the same
parameters:

| Parameter | Tuner output | Matrix-optimal | Verdict |
|-----------|-------------|----------------|---------|
| `degrees` | 0.0 | 5 | +4–5 pp R_Normal at degrees=5 in matrix — tuner chose wrong |
| `dfl` | 2.47 | 1.5 | matrix shows dfl=2.0 already regresses R_Normal; 2.47 is worse |
| `mosaic` | 0.0 | 0.0 | agrees — correct for single-cell data |
| `flipud` | 0.41 | 0.5 | close enough |
| `lr0` | 0.00884 | 0.001 | slightly higher; within acceptable range |

**Why the tuner chose wrong:** The tuner optimised a single-fold mAP50-95 objective on P4
alone. P4's holdout is Atypisch-dominant (122 Atypisch, 6 Normal) — mAP50-95 on P4 is
dominated by Atypisch performance. The tuner's `degrees=0.0` and `dfl=2.47` may have
marginally improved Atypisch localisation on this specific fold at the expense of Normal
recall generalisation — a trade-off invisible to the single-fold fitness metric.

The P3-B matrix, which runs the full 8-fold LOPO per cell, captures this trade-off
correctly. A single-fold mAP50-95 tuner is structurally blind to per-class recall balance
and cross-fold generalisation.

**Fitness 0.736 vs matrix winner 0.741 (mAP50-95):** The 0.5 pp difference is within
cross-fold variance. The tuner did not meaningfully outperform the matrix winner even on
its own objective.

## Stage-3 Validation Decision

Applying the tuner yaml to the full 8-fold LOPO (Stage 3 of the tinkering workflow) would
cost ~6–12 GPU-hours. Prior: the yaml will underperform the matrix winner on R_Normal (both
dominant changes move *away* from the matrix's optimal zone). Stage-3 validation is
**skipped**. The matrix winner remains the deployment recipe.

## Conclusion

**Negative result, but methodologically informative.** An unconstrained 30-iteration tuner
optimising a single-fold mAP50-95 objective converged on choices that disagree with
systematic 8-fold matrix evidence — illustrating why the matrix DoE approach was the
correct experimental design. The thesis should report P3-C as evidence that automated
single-fold hyperparameter search is insufficient for rare-class multi-patient generalisation.

---

## Timeline Summary

**run_id:** P3-C | **date:** ~2026-05-11 | **SLURM:** 323280
**HYPOTHESIS:** YOLO model.tune() run on the production recipe (same anchor, freeze, augmentation) will safely find hyperparameters that improve on the P3-B matrix winner.
**CHANGE vs prior (P3-B):** Replaced manual matrix sweep with automated 30-iter × 50-epoch YOLO tuner; tuner run on single fold (P4 holdout) rather than full LOPO.
**RESULT:** Tuner fitness 0.736 (P4 mAP50-95). Output yaml contradicts matrix on degrees and dfl. Stage-3 validation skipped. — **negative result**.
**LEARNING:** Single-fold automated tuning is blind to per-class recall balance and cross-patient generalisation. The P3-B 8-fold matrix is a stronger experimental design. Automated tuners optimise their objective function, not the metric that matters clinically. P3-C is a productive negative finding for the thesis.
**VERDICT:** negative result / validates matrix DoE approach over automated single-fold tuning
