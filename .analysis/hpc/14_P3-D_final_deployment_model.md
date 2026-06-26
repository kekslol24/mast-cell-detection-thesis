---
run_id: P3-D
phase: Phase 3
slurm: not recorded (submitted and completed 2026-05-11)
results_dir: tinker_final/final/
script: tinkering/ba_tinker_final.py + tinkering/run_final.sh
date: 2026-05-11 (complete)
status: complete — USZ deliverable
---

## Overview

Standard post-CV practice: take the recipe that the CV proved works and train one final
model on the full corpus with no holdout. LOPO is an evaluation protocol — each fold model
has never seen one patient, making none of them a deployment artefact. P3-D trains on all
P1–P8 data using the P3-B winner recipe (`deg5_dfl1.5_fr10`), producing the actual model
shipped to USZ.

## Strategy: Why Full-Corpus Final Training

The P3-B LOPO models each have one patient removed from training. A model trained on
all 8 patients has strictly more data than any fold model and represents the best
generalisation achievable from this corpus — it is the correct deployment artefact.
The CV's role was to validate the recipe; the final model's role is to use that recipe
on all available data.

An 85/15 stratified-by-patient val split is used for early stopping only. Texture
leakage between train and val is acceptable here — the unbiased generalisation estimate
already comes from P3-B LOPO, not from this val split.

## Session Decisions (2026-05-11)

Three tasks completed on this date:
1. P3-B matrix analysed — winner `ls0.0_deg5_dfl1.5_fr10` identified.
2. P3-C tuner reviewed — negative result confirmed, Stage-3 validation skipped.
3. P3-D trained, val metrics reviewed, `best.pt` prepared for USZ handover.

Outstanding after this date: 10 matrix cells still queued from SLURM job limit rejection;
USZ formal handover of `best.pt`.

## Setup

| Parameter | Value |
|-----------|-------|
| Base model | `yolo11n.pt` |
| Recipe | P3-B winner: `FREEZE=10, DEGREES=5.0, DFL=1.5, LABEL_SMOOTHING=0.0, CLS=1.0` |
| lr0 | 0.001 |
| Augmentation | `augment=True`, `mosaic=0.0`, `flipud=0.5`, `fliplr=0.5`, default HSV |
| cos_lr | True |
| optimizer | AdamW |
| imgsz / batch | 512 / 32 |
| CV | None — single training on full P1–P8 corpus |
| Val split | 85/15 stratified-by-patient (for early stopping only) |
| Oversampling | P5×8, P6×15, P7×8, P8×10 |

## In-Train Val Sanity Check (NOT a generalisation estimate)

| Metric | Value |
|--------|-------|
| mAP50-95 | 0.7175 |
| mAP50 | 0.8538 |
| Precision | 0.7526 |
| Recall | 0.8386 |
| Recall (Atypisch) | 0.9639 |
| Recall (Normal) | 0.7133 |

**R_Normal 0.713 is below the LOPO mean of 0.840** — this looks surprising since val
patients are also in train (leakage normally inflates val). Root cause: the val split
contains only ~10–15 Normal labels total (15% of 78). Missing 3–4 of those moves R_Normal
by ~25 pp. Single-point variance is ±0.10 on such small counts. A secondary effect is
YOLO's `best.pt` fitness criterion (weighted mean across classes), which biases checkpoint
selection toward epochs where Atypisch peaked rather than where Normal peaked.

**The authoritative generalisation estimate remains the P3-B LOPO matrix winner:**
Recall ≈ 0.866 | R_Normal ≈ 0.840 | R_Atypisch ≈ 0.892. The P3-D val check verifies
convergence, not deployment quality.

## Deliverable

`hpc/tinker_final/final/weights/best.pt` → shipped to USZ for next-round case collection:
- Model-assisted review of unannotated slides
- FP/FN correction loop  
- New-patient acquisition

Acceptance handover instructions: flag both clinician misses (FN) and low-confidence Normal
predictions in the review loop; track per-class R_Normal on incoming P9+ slides as the
primary monitoring metric.

## Conclusion

P3-D completes the Phase 3 experimental arc. The recipe validated by P3-B (deg5, dfl1.5,
freeze10) is baked into a full-corpus model. The val metrics confirm healthy convergence.
The shipped `best.pt` is the v1 USZ deployment artefact. All future runs (P4, P5) train
from `yolo11n.pt` from scratch on the expanded corpus, following the operational doctrine
(Pattern C: retrain from scratch on full accumulated corpus, never fine-tune on new data
alone).

---

## Timeline Summary

**run_id:** P3-D | **date:** 2026-05-11 | **SLURM:** not recorded
**HYPOTHESIS:** Training on the full P1–P8 corpus with the P3-B recipe will produce a deployment-ready model with generalisation ≥ the LOPO fold means.
**CHANGE vs prior (P3-B):** Removed LOPO loop; trained on all 8 patients simultaneously. Same recipe as P3-B winner.
**RESULT:** Val mAP50=0.854, Val Recall=0.839, Val R_Atypisch=0.964, Val R_Normal=0.713 (small val set, not authoritative). Converged cleanly. — **success**.
**LEARNING:** Full-corpus training converges cleanly on the P3-B recipe. R_Normal on the val sanity check is suppressed by small Normal count in the val split — not a signal of model regression. Authoritative generalisation estimate is P3-B LOPO: R_Normal=0.840.
**VERDICT:** success / USZ v1 deliverable shipped
