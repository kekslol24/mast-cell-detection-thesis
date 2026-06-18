---
run_id: P2-C
phase: Phase 2
slurm: 319401
job_name: "train p2 base dl_fv"
script: ba_improved_P2.py
date: not recorded
status: complete (not corrupted — separate SLURM ID from P2-A/B)
---

## Overview

First use of `DL_Modell_FV.pt` — the domain-specific pretrained weights trained by FV on
the P1 patient data. Hypothesis: domain-specific features should stabilise training and
improve mean mAP50 vs generic yolo11n. Default YOLO hyperparameters (lr0=0.01) retained.

## Infrastructure Note: Transfer Learning with nc Change 1→2

`DL_Modell_FV.pt` was trained on P1 with **nc=1** (single class). P2 training uses **nc=2**
(Atypisch, Normal). Ultralytics handles this silently, but the consequences are concrete:

A fresh `DetectionModel` is built with nc=2, then `intersect_dicts` loads checkpoint weights
only where key names AND tensor shapes match. The Detect head's classification branch (`cv3`)
outputs `nc` channels — its final `Conv2d(c3, nc, 1)` has shape mismatch (1→2) and is
**discarded and re-initialised randomly**. The box regression branch (`cv2`) is
nc-independent and transfers cleanly.

| Layer group | Transferred? |
|---|---|
| Backbone model.0–9 (frozen with `freeze=10`) | Yes |
| Neck model.10–21 | Yes |
| Detect `cv2` (bbox regression) | Yes |
| Detect `cv3` (classification, final conv) | **No — random init** |
| Detect `.dfl` | Always frozen |

**Training implications:**
- `cv3` starts with random weights → classification loss is high and noisy in early epochs.
- This gradient propagates back through the unfrozen neck.
- At `lr0=0.01` (YOLO default, calibrated for scratch training) the noisy gradient is large
  enough to **disrupt the pretrained neck feature pyramid** — likely the root cause of P2-C
  collapsing to mAP50=0.469 despite using domain-specific weights.
- At `lr0=0.001` the updates are small enough that the neck refines rather than gets
  overwritten. This is the fix validated in P2-H.
- Early-epoch metrics (ep. 1–50) are not representative — `cv3` is essentially random.
  Do not use them to judge run quality.

## Setup

| Parameter | Value |
|-----------|-------|
| Base model | `DL_Modell_FV.pt` (domain-specific, nc=1 → nc=2) |
| Dataset | P2 — same as P2-A/B |
| FP_NEG_OVERSAMPLE | 1 |
| BG_RATIO | 0 |
| lr0 | 0.01 (YOLO default) |
| freeze | not set (backbone not frozen) |
| Hyperparameters | YOLO defaults (no cfg override) |

## Test Results

| Fold | mAP50-95 | mAP50  | Precision | Recall |
|------|----------|--------|-----------|--------|
| 1    | 0.3088   | 0.4207 | 0.8706    | 0.3763 |
| 2    | 0.3787   | 0.5089 | 0.8824    | 0.3820 |
| 3    | 0.3697   | 0.4945 | 0.8567    | 0.4180 |
| 4    | 0.3445   | 0.4734 | 0.9014    | 0.4234 |
| 5    | 0.3156   | 0.4455 | 0.8896    | 0.3870 |
| **mean** | **0.344** | **0.469** | **0.880** | **0.397** |
| **std**  | **0.031** | **0.035** | — | — |

## Fold-Level Notes

All five folds converge to nearly the same values (std=0.031 vs 0.139–0.172 in yolo11n
runs). This is the stabilising effect of domain-specific pretrained features — the model
has a consistent prior and each fold finds a similar solution. However, **the solution it
finds is wrong**: precision is high (0.880) but recall is very low (0.397). The model
correctly classifies cells it detects, but misses more than 60% of them. This
high-precision/low-recall pattern is the signature of a learning rate that is too high
for fine-tuning: the pretrained features get partially overwritten, leaving the model
overly conservative.

## Conclusion

Domain-specific weights dramatically reduce variance (std=0.031 vs 0.139–0.172) — the
pretrained features stabilise training across folds. However, `lr0=0.01` is too high for
fine-tuning from a domain-specific checkpoint: the model achieves high precision but
catastrophically low recall (0.397). The next step is to retain `DL_Modell_FV.pt` but
reduce the learning rate. P2-D tests this with modified settings; P2-H is the clean fix
with `lr0=0.001`.

---

## Timeline Summary

**run_id:** P2-C | **date:** not recorded | **SLURM:** 319401
**HYPOTHESIS:** Domain-specific pretrained weights (DL_Modell_FV.pt) will outperform generic yolo11n on the P2 task.
**CHANGE vs prior:** Switched from `yolo11n.pt` to `DL_Modell_FV.pt`; kept lr0=0.01 default.
**RESULT:** Mean mAP50=0.469 ± 0.035, Recall=0.397. Lower mAP50 than yolo11n, near-zero recall variance. — **regression** (domain weights help stability, hurt recall).
**LEARNING:** `lr0=0.01` causes catastrophic partial forgetting when fine-tuning from domain-specific weights with nc mismatch. The randomly-reinitialised `cv3` head generates noisy gradients that overwrite pretrained neck features. The fix is `lr0=0.001`. The stability benefit (low std) is real and worth preserving.
**VERDICT:** regression on aggregate metrics / key insight unlocked about lr0
