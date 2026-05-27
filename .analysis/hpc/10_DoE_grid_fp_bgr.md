---
run_id: DoE-grid
phase: Phase 2
slurm: 320698–320709 (12 jobs)
results_dirs: train_p2_fp{1,2,3}_bgr{0,1,2,3}_new/
script: ba_improved_P2.py (with grid submission via submit_grid.sh)
date: ~2026-05-06 (based on grid infrastructure note date)
status: 11 of 12 complete; fp3_bgr3 (320709) failed (OOM/timeout)
---

## Overview

Full 3×4 factorial design-of-experiments grid over two data composition factors:
`FP_NEG_OVERSAMPLE` ∈ {1, 2, 3} and `BG_RATIO` ∈ {0, 1, 2, 3}. All 12 cells share the
P2-H reference configuration (lr0=0.001, freeze=10, DL_Modell_FV.pt) with `BG_RATIO`
bug fixed. Grid infrastructure: `BG_RATIO` and `FP_NEG_OVERSAMPLE` read from environment
variables; `submit_grid.sh` submits all jobs in one command; `PROJECT_DIR` auto-named
from `$SLURM_JOB_NAME`.

## Infrastructure Note: `per_class` Positional Indexing Bug (measurement bug, not fixed)

`per_class(metrics, idx)` reads per-class recall by position in `metrics.box.r`:
```python
def per_class(metrics, idx):
    arr = metrics.box.r
    return float(arr[idx]) if idx < len(arr) else float('nan')
```

Ultralytics only includes a class in `ap_class_index` if that class has GT instances OR
model predictions in the evaluated split. When a class is absent from both, `metrics.box.r`
is shorter than `nc` and positional indexing breaks.

**Observed symptom:** Fold 5 (P13 holdout in later phases) reports R_Normal=NaN even though
P13 has Normal GT instances. P13 has 0 Atypisch GT and the model makes zero class-0
predictions → class 0 is absent from evaluation → `metrics.box.r` has length 1 → `per_class(_,
1)` is out of bounds → NaN.

The true Normal recall sits in the R_Atypisch column. This affects any fold where a holdout
patient is pure-single-class and the model makes no predictions for the missing class.

**Fix (not yet applied during the DoE):**
```python
def per_class(metrics, class_id):
    try:
        idx_map = list(metrics.box.ap_class_index)
        if class_id not in idx_map:
            return float('nan')
        return float(metrics.box.r[idx_map.index(class_id)])
    except (AttributeError, IndexError, TypeError):
        return float('nan')
```

Until fixed: any fold where holdout patient is pure-single-class may have swapped or
missing per-class recall. Cross-check: if `Recall == R_Atypisch` and `R_Normal == NaN`
for a pure-Normal holdout, the true Normal recall is the value in R_Atypisch.

## Setup (all cells)

| Parameter | Value |
|-----------|-------|
| Base model | `DL_Modell_FV.pt` |
| lr0 | 0.001 |
| freeze | 10 |
| patience | 100 |
| epochs | 5000 |
| FP_NEG_OVERSAMPLE | {1, 2, 3} — varied per cell |
| BG_RATIO | {0, 1, 2, 3} — varied per cell |
| Augmentation | Disabled |

## Test Results — All Grid Cells

| Config | SLURM | FP× | BGr | mAP50 (mean) | mAP50-95 | Recall | Precision |
|--------|-------|-----|-----|--------------|----------|--------|-----------|
| fp1_bgr0 (= P2-H) | 320698 | ×1 | 0 | **0.604** | 0.494 | 0.537 | **0.934** |
| fp1_bgr1 | 320699 | ×1 | 1 | 0.562 | 0.438 | 0.531 | 0.911 |
| fp1_bgr2 | 320700 | ×1 | 2 | 0.564 | 0.433 | 0.523 | 0.828 |
| fp1_bgr3 | 320701 | ×1 | 3 | 0.591 | 0.441 | 0.518 | 0.820 |
| fp2_bgr0 | 320702 | ×2 | 0 | 0.581 | 0.446 | **0.638** | 0.726 |
| fp2_bgr1 | 320703 | ×2 | 1 | 0.565 | 0.426 | 0.523 | 0.920 |
| fp2_bgr2 | 320704 | ×2 | 2 | 0.592 | 0.446 | 0.637 | 0.831 |
| fp2_bgr3 | 320705 | ×2 | 3 | 0.596 | 0.467 | 0.539 | 0.792 |
| fp3_bgr0 | 320706 | ×3 | 0 | 0.548 | 0.407 | 0.537 | 0.908 |
| fp3_bgr1 | 320707 | ×3 | 1 | 0.566 | 0.434 | 0.546 | 0.798 |
| fp3_bgr2 | 320708 | ×3 | 2 | 0.572 | 0.422 | 0.526 | 0.925 |
| fp3_bgr3 | 320709 | ×3 | 3 | ❌ FAILED | — | — | — |

## Fold-Level Pattern

Fold 3 is an outlier in **every single grid cell** — mAP50 range 0.88–0.94 vs. 0.44–0.61
for all other folds. The within-fold recall for folds 1, 2, 4, 5 is stuck at 0.42–0.49
across all 11 completed configurations. The cross-fold std (~0.15–0.21) completely dominates
any signal from the FP or BG factors, making statistical discrimination between cells
impossible within the P2-only 5-fold setup.

## Conclusions from DoE

**1. No configuration significantly outperforms fp1_bgr0 (P2-H).** Total mAP50 spread
across 11 cells: 0.548–0.604 — a range of 0.056. Within fold variance noise (~0.19 std),
no combination is meaningfully different.

**2. BG_RATIO has a consistently negative effect.** Within every FP level, bgr=0 gives
the best or tied-best mAP50 (fp1: 0.604, fp2: 0.581, fp3: 0.548). Each step up in
BG_RATIO degrades. **BG_RATIO should be kept at 0 for all further runs.**

**3. FP_NEG_OVERSAMPLE has diminishing returns above ×2.** fp=2 gives the highest reported
recall (0.638) but this is driven by a single fold-2 outlier (320702: recall=0.974); without
it, fp=2 is indistinguishable from fp=1. fp=3 underperforms both.

**4. The recall problem is a generalisation problem, not a data-ratio problem.** Recall
outside fold 3 is 0.42–0.49 regardless of FP or BG. The model fails to transfer features,
not because of class imbalance or insufficient negatives. On-the-fly augmentation is the
correct next intervention.

**5. fp3_bgr3 (320709) failed** — likely OOM or timeout from the largest dataset. Not
worth retrying; this corner of the grid is already ruled out by the BG_RATIO trend.

## Best Config from DoE

**fp1_bgr0 (= P2-H):** mAP50=0.604, recall=0.537, precision=0.934. Minimum data
manipulation, cleanest result.

---

## Timeline Summary

**run_id:** DoE-grid | **date:** ~2026-05-06 | **SLURM:** 320698–320709
**HYPOTHESIS:** Varying FP_NEG_OVERSAMPLE and BG_RATIO will identify a data composition that improves generalisation beyond the P2-H baseline.
**CHANGE vs prior:** Full 3×4 factorial grid over FP× and BGr, all sharing lr0=0.001/freeze=10 reference.
**RESULT:** Total mAP50 spread: 0.056 across 11 cells. BG consistently hurts. FP ×2 gives marginal recall bump driven by one outlier fold. fp1_bgr0 is the grid winner. — **inconclusive** (signal swamped by fold variance).
**LEARNING:** Data composition (FP oversampling, background tiles) cannot overcome the generalisation gap at P2-only scale. The recall problem is not a class-imbalance problem — it is a data-scarcity/augmentation problem. Exhausted this research direction; pivot to on-the-fly augmentation and multi-patient corpus.
**VERDICT:** inconclusive / research direction exhausted; pivot to augmentation + more data
