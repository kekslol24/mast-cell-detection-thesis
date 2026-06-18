---
run_id: P5-A
phase: Phase 5
slurm: TBD
job_name: TBD
results_dir: TBD
script: ba_improved_comb.py (P1–P53 extension with PATIENT_OVERSAMPLE_FIXED = {})
date: planned as of 2026-05-27
status: PLANNED — not yet submitted
---

## Overview

Full 53-patient corpus baseline run. 38 new patients (P16–P53) acquired from USZ, integrated
into `ba_improved_comb.py`. LOPO expands to 53 folds. Recipe is the confirmed P3-B winner
(`degrees=5, dfl=1.5, freeze=10, lr0=0.001, cls=1.0`). Per-patient oversampling fully
auto-computed (`PATIENT_OVERSAMPLE_FIXED = {}`).

## Dataset: P16–P53 Additions (2026-05-25)

38 patients added. Many are **pure-FP patients** (no mast cell annotations — only confirmed
false-positive Excel entries). A code fix applied 2026-05-27 ensures patients with empty
label files no longer receive a LOPO fold (they contribute FP negatives to other folds but
do not generate undefined test metrics).

| Patient class | Description | Effect on LOPO |
|--------------|-------------|----------------|
| Pure-FP patients | No labels/ folder OR all label files empty | No LOPO fold; FP negatives contributed to all other folds |
| Annotated patients | Has at least one non-empty label file | Gets a LOPO fold |

**Annotated P16–P53 (non-pure-FP):** P16, P17, P18, P24, P28, P29, P31, P37, P39, P40,
P43, P44, P45, P46, P47, P48, P50, P51, P52, P53 — exact count depends on empty-label
filtering applied at runtime.

## Updated Corpus Totals (P1–P53, annotated patients only)

| Metric | P1–P15 | P1–P53 | Delta |
|--------|--------|--------|-------|
| Files | 1564 | 1869 | +305 |
| Atypisch | 1288 | 1476 | +188 |
| Normal | 391 | 511 | +120 |
| Raw ratio | 3.3:1 | 2.9:1 | improving |

Normal share growing with new data — the new patients skew Normal-dominant.

## FP Negatives (P9–P53)

Total P9–P53 FP entries: 3637. Largest pools: P9 (729), P43 (198), P27 (160), P19 (142),
P42 (142). All Excel filenames confirmed present on disk.

## Code Changes Applied (2026-05-25 to 2026-05-27)

1. **`patient_dir.py`:** all 38 FP Excel filenames for P16–P53 filled in.
2. **`ba_improved_comb.py` — `PATIENT_FP_EXCELS` import added.** Latent `NameError` at
   line 567 fixed (symbol used but never imported — would have crashed FP-loading at runtime).
3. **`ba_improved_comb.py` — `PATIENT_OVERSAMPLE_FIXED = {}`:** Previously pinned P1–P8
   factors removed; `compute_oversample_factors()` now handles all patients.
4. **Empty-label guard added (2026-05-27):**
   ```python
   if not classes:
       continue   # empty label = no mast cells; skip from positives and LOPO folds
   ```
   Pure-FP patients with empty label files no longer get a LOPO fold with undefined metrics.

## Setup (planned)

| Parameter | Value |
|-----------|-------|
| Base model | `yolo11n.pt` |
| CV | LOPO, up to 53 folds (after empty-label filtering) |
| degrees | 5 |
| dfl | 1.5 |
| freeze | 10 |
| lr0 | 0.001 |
| cls | 1.0 |
| PATIENT_OVERSAMPLE_FIXED | {} (fully auto-computed) |
| FP negatives | All patients where Excel exists (P2 + P9–P53), ~3637 entries |

## Expected Outcomes

- mAP50 expected ≥ 0.90 (P4-A already at 0.903 on P1–P15; more Normal-heavy patients should
  further lift R_Normal).
- R_Normal expected to approach 0.90–0.93 given the new Normal-dominant patients (P51:
  5/90; P53: 0/5; P37: 0/2; P44: 0/2, etc.).
- Cross-fold std expected to decrease: 53 folds give a more stable mean estimate than 15.
- Folds for very small patients (P31: 1 file, P24: 1 file, P28: 1 file) will have extremely
  noisy metrics (1 missed cell = 100% recall drop) — report these as directional only.

## Unanswered Questions at P1–P53 Scale

- **`cls=1.0` necessity.** Raw ratio improved from 3.3:1 (P1–P15) to 2.9:1 (P1–P53). With
  better inherent balance, the double class-loss weight may be less necessary. Not tested —
  would require a separate run.
- **Oversampling necessity.** With P9 (195 Normal) and P51 (90 Normal) now in the corpus,
  the fixed P5–P8 factors (×8–15) may over-weight those patients relative to the global
  distribution. `compute_oversample_factors()` handles this dynamically, but the printed
  corpus summary should be verified at runtime.

## RESULT

**TBD** — run not yet submitted as of 2026-05-27.

## LEARNING

**TBD**

---

## Timeline Summary

**run_id:** P5-A | **date:** planned (not yet submitted as of 2026-05-27) | **SLURM:** TBD
**HYPOTHESIS:** Expanding from P1–P15 to P1–P53 (38 new patients, ~305 new images, ~120 new Normal annotations) will further improve R_Normal and reduce cross-fold variance.
**CHANGE vs prior (P4-B):** Corpus P1–P15→P1–P53; LOPO 15-fold→~53-fold; PATIENT_OVERSAMPLE_FIXED fully auto-computed; 3637 FP negatives from P2+P9–P53.
**RESULT:** TBD
**LEARNING:** TBD
**VERDICT:** TBD
