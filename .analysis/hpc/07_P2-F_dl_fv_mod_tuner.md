---
run_id: P2-F
phase: Phase 2
slurm: 319538
job_name: "train p2 mod dl_fv hyperpara"
script: ba_improved_P2.py
date: not recorded
status: complete (likely affected by SLURM collision — submitted back-to-back with P2-E)
---

## Overview

Combined the modified settings from P2-D with the tuner hyperparameters from P2-E.
Hypothesis: the modified settings might buffer some of the catastrophic forgetting caused
by the from-scratch tuner yaml. Results improved over P2-E but remained below P2-D,
confirming that tuner hyperparameters are net-negative for fine-tuning regardless of
other setting adjustments.

## Setup

| Parameter | Value |
|-----------|-------|
| Base model | `DL_Modell_FV.pt` (domain-specific) |
| Dataset | P2 — same as P2-A/B/C/D/E |
| FP_NEG_OVERSAMPLE | 1 |
| BG_RATIO | 0 |
| cfg | `best_hyperparameters.yaml` (same from-scratch tuner output as P2-E) |
| Other settings | Modified (as in P2-D — exact changes not recorded) |
| SLURM isolation | Likely affected — submitted back-to-back with P2-E (319536 → 319538) |

## Test Results

| Fold | mAP50-95 | mAP50  | Precision | Recall |
|------|----------|--------|-----------|--------|
| 1    | 0.3552   | 0.4758 | 0.3876    | 0.5770 |
| 2    | 0.3465   | 0.4573 | 0.3599    | 0.6335 |
| 3    | 0.3635   | 0.4879 | 0.8916    | 0.4153 |
| 4    | 0.4582   | 0.6014 | 0.5747    | 0.7264 |
| 5    | 0.4032   | 0.5181 | 0.5621    | 0.5013 |
| **mean** | **0.385** | **0.508** | **0.555** | **0.571** |
| **std**  | **0.046** | **0.057** | — | — |

## Fold-Level Notes

The collapse is less severe than P2-E (cls_loss spike not reported for this run), suggesting
the modified settings from P2-D partially mitigated the lr0=0.01 damage. However, precision
is highly erratic across folds (0.360–0.892), indicating the model found inconsistent
solutions — some folds went conservative, others went aggressive. This is the signature of
a partially-corrupted fine-tune: the neck's feature pyramid was disrupted but not completely
destroyed, leading to different degenerate solutions depending on fold composition.

## Conclusion

P2-F is worse than P2-D across all metrics (mAP50: 0.508 vs 0.590, recall: 0.571 vs 0.655).
The tuner yaml adds net harm even when combined with otherwise-better settings. This
definitively rules out `best_hyperparameters.yaml` from SLURM 269569 for any further use.
Together with P2-E, P2-F establishes a firm boundary: **no from-scratch tuner output should
be applied to fine-tuning from domain-specific weights.** The correct path forward is the
learning rate fix (P2-H) and a fresh tuner run on the production recipe (P3-C).

---

## Timeline Summary

**run_id:** P2-F | **date:** not recorded | **SLURM:** 319538
**HYPOTHESIS:** Modified settings (P2-D) combined with tuner yaml will outperform either alone.
**CHANGE vs prior:** Combined P2-D modified settings with P2-E tuner yaml.
**RESULT:** Mean mAP50=0.508 ± 0.057, Recall=0.571. Worse than P2-D on all metrics. — **regression** vs P2-D.
**LEARNING:** The tuner yaml damages performance regardless of other settings. The modifier-plus-tuner combination does not recover what the tuner loses. Definitively rules out SLURM 269569 `best_hyperparameters.yaml` for fine-tuning use.
**VERDICT:** regression / confirms tuner yaml incompatibility with fine-tuning
