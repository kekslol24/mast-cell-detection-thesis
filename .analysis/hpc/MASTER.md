# MASTER SYNTHESIS — Mast Cell Detection Pipeline
*Self-contained research document. No source file access required.*
*Generated 2026-05-27. Reflects experiment log state as of same date.*

---

## 1. PROJECT CONTEXT

### Problem

Systemic mastocytosis (SM) is diagnosed using WHO criteria, one of which is a minor
criterion: **>25% atypical mast cells** in a bone marrow aspirate. Counting and classifying
mast cells from slide images is a manual, time-consuming task performed by haematologists at
University Hospital Zurich (USZ). The goal of this bachelor's thesis (ZHAW, supervised by
Dr. Stefan Glüge, submission 2026-07-02) is to automate this classification using a YOLO11
object detection model.

Two cell classes are predicted:
- **Atypisch** — atypical mast cells (SM marker)
- **Normal** — normal mast cells (benign)

The WHO criterion requires reliable detection of *both* classes: under-detecting Atypisch
misses disease markers, but under-detecting Normal inflates the Atypisch ratio, potentially
producing false SM flags in borderline patients.

### Clinical Asymmetry

The training data comes almost entirely from confirmed SM patients, meaning Atypisch is the
dominant class. Normal cells are rare in the training corpus (see Section 3). This is the
inverse of the typical object detection imbalance assumption and drives several architectural
decisions throughout the project.

### Why YOLO11

YOLO11 (Ultralytics) was selected for its:
- Single-pass detection + classification on 512 px crops
- Mature fine-tuning infrastructure from pretrained COCO weights
- Support for custom anchor checkpoints (`DL_Modell_FV.pt`, a domain-specific pretrained
  model trained during the initial P1 phase)
- Active development and good documentation for the Ultralytics Python API

### Evaluation Metrics That Matter

| Metric | Role |
|--------|------|
| **Recall** | Primary — missing a cell is worse than a false alarm (clinical context) |
| **R_Normal** | Secondary — the WHO criterion requires both classes; Normal is the rare class |
| **R_Atypisch** | Expected to be high (majority class in training); monitored for regression |
| **mAP50** | Standard detection benchmark — used for run-to-run comparison |
| **mAP50-95** | Stricter localisation measure — reported but less clinically relevant |
| **Precision** | Monitored but lower priority than recall |

High recall at the cost of precision is the accepted trade-off for this clinical context.
The deployed confidence threshold (0.58) was tuned for this trade-off.

### Infrastructure

**Cluster:** ZHAW HPC, `earth-4` SLURM partition, 8× L40S GPUs per node.
(Note: early runs used only 2 GPUs because the SLURM script requested `--gres=gpu:l40s:2`;
the 8-GPU capacity was confirmed later.)

**Training launch:** `sbatch run_training.sh` from inside `hpc/jobs/`. The SLURM script
activates micromamba, `cd`s to `hpc/`, then runs the training Python script directly.
SLURM does not snapshot the script at submission — jobs run whatever is on disk at start
time, which caused SLURM concurrent job interference in early P2 runs (see Section 4).

**Local sync:** `hpc/` directory is synced to `/cfs/earth/scratch/vollmflo/BA/hpc/` on
the cluster.

---

## 2. PIPELINE

### Key Scripts

| Script | Role |
|--------|------|
| `ba_improved.py` | Phase 1 baseline. 5-fold stratified CV on P1 dataset, yolo11n, no fine-tuning. Not modified after Phase 1. |
| `ba_improved_P2.py` | Phase 2 script. Fine-tuning experiments on P2 dataset. Superseded by `ba_improved_comb.py`. |
| `ba_improved_comb.py` | **Main script from Phase 3 onward.** LOPO CV, multi-patient corpus, per-patient oversampling, generalised FP loading. Actively developed. |
| `tinkering/ba_tinker.py` | Copy of `ba_improved_comb.py` with seven training knobs made env-overridable. Used for Phase 3 tinkering matrix. |
| `tinkering/ba_tinker_final.py` | Single-run variant (no LOPO loop) for training the final deployment model. |
| `tinkering/tune_hyperpara.py` | LOPO-Fold-4 hyperparameter tuner. Runs `model.tune()` on the production recipe. |
| `inference.py` | Mass inference on a ZIP of new slide images. |
| `app/app.py` | Gradio UI (Analysis + Research tabs). |

### Data Flow: Training Pipeline (`ba_improved_comb.py`)

```
Annotated images (per-patient .jpeg + .txt label files)
    ↓
load_patient_images() — verifies label exists AND is non-empty (guard added 2026-05-27)
    ↓
pos_with_meta: [(img_path, patient_id, class_set), ...]
    ↓
FP negatives loaded from per-patient Excel files (PATIENT_FP_EXCELS)
via build_disk_index() + resolve_fp_filenames()
    ↓
compute_oversample_factors() — solves for per-patient repeat factor
targeting global Atypisch:Normal ratio ≈ 2.0:1 (see oversampling strategy)
    ↓
patients = sorted unique patient IDs in pos_with_meta
→ LOPO folds: one fold per patient
    ↓
train_fold(fold_idx) [in mp.Pool, one process per fold]:
    - train_paths = all pos except holdout + FP negatives for train patients
    - symlinks into cv_temp_{JOB_ID}/fold_N_workspace/
    - writes train.txt / val.txt / test.txt
    - model.train(data=yaml, ...)
    - model.val() on test.txt → metrics logged
    ↓
Summary table: TRAIN / VAL / TEST per-fold, mean ± std
```

**Key implementation details:**
- Workspace isolation by `$SLURM_JOB_ID` (prevents concurrent job collision on fold directories).
- `PROJECT_DIR` derived from `$SLURM_JOB_NAME` (locked at sbatch submission time).
- GPU assignment: `fold_idx % N_GPUS` (round-robin across available GPUs).
- `N_GPUS` from `torch.cuda.device_count()` or `N_GPUS` env var.

### Oversampling Strategy

With 14.4:1 Atypisch:Normal in the P1–P8 corpus, per-patient image-level oversampling is
applied: patient image paths are duplicated in `train.txt` (not the images themselves).
Combined with per-epoch random augmentation (`augment=True`), each duplicate sees different
pixels each epoch — the duplicates are not wasted.

`compute_oversample_factors()` solves iteratively from most Normal-heavy to least:
for each patient, find the smallest integer k such that the global weighted ratio reaches
the target (default 2.0). Factors are capped at 25.

In practice, once the large Normal-heavy patients (P8, P9) are assigned k=1 (their natural
weight is already sufficient), the algorithm assigns k=1 to most remaining patients and the
ratio stabilises at 1.5:1 to 2.0:1.

### Augmentation: On-the-fly vs Pre-applied

**Pre-applied augmentation (Phase 1 and early Phase 2):** `preprocess_image()` applies one
random transform to each image and saves to disk. Every training epoch sees the same frozen
augmented copy — the regularisation benefit is exhausted after epoch 1. This was identified
as ineffective in Phase 2 analysis and disabled in P2-H.

**On-the-fly augmentation (Phase 3 onward):** `augment=True` in `model.train()` applies
random transforms per-epoch. `mosaic=0.0` is mandatory — mosaic composites four images,
appropriate for scene detection but scrambles single-cell 512 px crops. `flipud=0.5` is
valid: bone marrow cells have no canonical orientation.

### Hyperparameter Tuning Approach

`tinkering/tune_hyperpara.py` runs YOLO's built-in `model.tune()` on a single LOPO fold
(P4 holdout, 30 iterations × 50 epochs). Critically, the tuner uses the **production
recipe** (yolo11n.pt, freeze=10, AdamW, `augment=True`, `mosaic=0.0`, `flipud=0.5`,
`cls=1.0`) so its output lives in the same hyperparameter regime as deployment.

The Phase 2 tuner failure (P2-E) occurred because the tuner had been run on a from-scratch
recipe (`lr0=0.01`, no freeze) and its output was applied to a fine-tuning run — a recipe
mismatch. Phase 3's tuner avoids this by construction.

### Transfer Learning: nc Change 1→2

`DL_Modell_FV.pt` was trained on P1 with nc=1 (one class). When fine-tuning on P2 with
nc=2, Ultralytics discards the Detect head's classification branch (`cv3`, final
`Conv2d(c3, nc, 1)`) because its shape changes (1→2) and re-initialises it randomly. The
box regression branch (`cv2`) and backbone transfer cleanly. The randomly-initialised `cv3`
generates noisy classification gradients in early epochs, which at `lr0=0.01` are large
enough to disrupt the pretrained neck — explaining P2-C's collapse to mAP50=0.469. At
`lr0=0.001` the updates are small enough to allow refinement rather than overwriting.

### Negative-Set Provenance

Three categories of negatives exist in this project:

| Category | Source | Role |
|----------|--------|------|
| P1 gold negatives | `old/P1/Negativ 12241515/` | Hand-curated empty tiles from original P1 training of DL_Modell_FV.pt. Always included when P1 is in train. |
| P2 hard FPs | `Task 6_1224151_negative.xlsx` (and P9–P53 Excels) | Tiles where the model produced false positives during USZ annotation. Confirmed empty by the clinician. High-value hard negatives. |
| Random BG | Unlabeled slide tiles from the image directory | Low-value. DoE confirmed this category uniformly hurts performance. `RANDOM_BG_RATIO=0` in all production runs. |

---

## 3. EXPERIMENTAL TIMELINE

### P1-baseline — Phase 1, ~2026-04 (date not recorded)

**HYPOTHESIS:** YOLO11n can learn mast cell detection from the P1 single-patient dataset.

First run. P1 dataset (`Pos_neg 12241515`, 428 images). yolo11n.pt (COCO pretrained).
5-fold stratified CV. Augmentation pre-applied to disk (later found ineffective). Default
YOLO hyperparameters.

**RESULT:** Val mAP50 = 0.800 ± 0.129, Recall = 0.719. No held-out test set. Fold 1
excellent (mAP50=0.952), Fold 5 collapses at epoch 29 (recall=0.432). **Verdict:
improvement** (baseline established).

**LEARNING:** Detection is feasible. High fold variance (std=0.129) driven by split luck.
Domain-specific weights and multi-patient data needed to stabilise. Pre-applied augmentation
provides no real regularisation — flagged for fixing.

---

### P2-A — Phase 2, date not recorded (SLURM 319399)

**HYPOTHESIS:** yolo11n.pt can serve as a P2 baseline before domain-specific weights.

First P2 run. Submitted concurrently with P2-B, triggering the SLURM workspace collision
bug: both jobs shared `fold_0_workspace/`, overwrote augmented images mid-run, and
interleaved training epochs in `results.csv`. Folds 2, 3, 5 are corrupted (bit-for-bit
identical to P2-B). Folds 1 and 4 survived as clean data.

**RESULT:** Mean mAP50=0.580 ± 0.172, Recall=0.617. Unreliable. **Verdict: inconclusive /
corrupted.**

**LEARNING:** SLURM concurrent job interference discovered. Fix: namespace temporary
directories with `$SLURM_JOB_ID`; derive `PROJECT_DIR` from `$SLURM_JOB_NAME`.

---

### P2-B — Phase 2, date not recorded (SLURM 319400)

**HYPOTHESIS:** Modified settings improve on P2-A.

Concurrent with P2-A — same collision. Folds 2, 3, 5 identical to P2-A to four decimal
places. Exact modifications not recorded in the log.

**RESULT:** Mean mAP50=0.614 ± 0.155, Recall=0.667. Corrupted. **Verdict: inconclusive /
corrupted.**

**LEARNING:** Reinforces SLURM collision finding. Logging discipline for setting changes
required going forward.

---

### P2-C — Phase 2, date not recorded (SLURM 319401)

**HYPOTHESIS:** Domain-specific pretrained weights (DL_Modell_FV.pt) will outperform
generic yolo11n on the P2 task.

First use of `DL_Modell_FV.pt`. lr0=0.01 (YOLO default). Clean run (wide SLURM ID gap).
The nc change 1→2 randomly re-initialises `cv3`; at lr0=0.01, the noisy classification
gradients partially overwrite pretrained neck features.

**RESULT:** Mean mAP50=0.469 ± 0.031, Recall=0.397, Precision=0.880. Lower mAP50 than
yolo11n, but variance collapsed (std 0.031 vs 0.139–0.172). **Verdict: regression on
aggregate metrics / key insight about lr0.**

**LEARNING:** Domain weights stabilise training (low std) but lr0=0.01 causes catastrophic
partial forgetting for fine-tuning. Fix: lr0=0.001. The high-precision/low-recall signature
(0.880 / 0.397) is the fingerprint of a learning rate too high for fine-tuning.

---

### P2-D — Phase 2, date not recorded (SLURM 319416)

**HYPOTHESIS:** Modified settings on DL_Modell_FV.pt can recover recall lost in P2-C.

First clean, uncorrupted P2 run with domain-specific weights. Modified settings (exact
changes not recorded). Wide SLURM gap from P2-A/B/C — no collision.

**RESULT:** Mean mAP50=0.590 ± 0.090, Recall=0.655. Fold 2: recall=0.957; Fold 5:
recall=0.416. **Verdict: improvement / informal P2 reference baseline.**

**LEARNING:** Some combination of hyperparameter adjustments recovers recall from P2-C's
0.397 to 0.655. The exact modifications were not logged — a reproducibility gap. Fold 5
is a persistent weak split in every run, unrelated to configuration.

---

### P2-E — Phase 2, date not recorded (SLURM 319536) — FAILED

**HYPOTHESIS:** YOLO tuner-optimised hyperparameters applied to DL_Modell_FV.pt will
outperform YOLO defaults.

Applied `best_hyperparameters.yaml` from a tuner run on a **from-scratch recipe**
(SLURM 269569, lr0=0.01, no anchor). All five folds collapsed within 3–6 epochs
(val `cls_loss` reaching 5.8+, normal range 0.5–1.5). Near-zero variance (std=0.005)
is all folds converging to the same degenerate solution.

**RESULT:** Mean mAP50=0.448 ± 0.005, Recall=0.413. **Verdict: total failure.**

**LEARNING:** Recipe mismatch between tuner and deployment causes catastrophic forgetting.
The `cfg=best_hyperparameters.yaml` from SLURM 269569 is permanently retired. Correct
conclusion (not "tuners are incompatible with fine-tuning" but "tuner recipe must match
deployment recipe") motivates the Phase 3 re-run of the tuner on the production recipe.

---

### P2-F — Phase 2, date not recorded (SLURM 319538)

**HYPOTHESIS:** Modified settings (P2-D) combined with the tuner yaml will outperform
either alone.

Combined P2-D modifications with the P2-E tuner yaml. Slightly better than P2-E but worse
than P2-D on all metrics.

**RESULT:** Mean mAP50=0.508 ± 0.057, Recall=0.571. **Verdict: regression vs P2-D.**

**LEARNING:** The tuner yaml damages performance regardless of other settings. Definitively
rules out SLURM 269569 `best_hyperparameters.yaml` for fine-tuning use.

---

### P2-G — Phase 2, date not recorded (SLURM 319934)

**HYPOTHESIS:** Adding background tiles (BG_RATIO=2) will suppress false positives and
improve the precision/recall balance.

`DL_Modell_FV.pt`, BG_RATIO=2, lr0=0.01. Clean run (wide SLURM gap). The BG_RATIO bug
was present: `bg_paths` was not passed to `fold_tasks`, so zero background tiles actually
reached training. Folds 3 and 4 collapsed at epochs 7 and 43 (early-stopping signature of
lr0=0.01 on unlucky splits).

**RESULT:** Mean mAP50=0.530 ± 0.090, Recall=0.484. **Verdict: regression vs P2-D.**

**LEARNING:** BG_RATIO bug invalidates any background conclusions. Fold early-collapse is
a learning rate signature — lr0=0.01 is incompatible with domain-specific fine-tuning on
unlucky splits. Conclusively motivates the lr0=0.001 fix.

---

### P2-H — Phase 2, ~2026-05-06 (SLURM 320698)

**HYPOTHESIS:** lr0=0.001 + freeze=10 will stabilise fine-tuning from DL_Modell_FV.pt.

Key changes from P2-G: lr0 0.01→0.001, freeze=10, BG_RATIO bug fixed (bg_paths now passed
to fold workers), pre-applied augmentation disabled (`augment=False`). No fold collapses.
P2-H is simultaneously the fp1_bgr0 control cell of the DoE grid.

**RESULT:** Mean mAP50=0.604 ± 0.194, Recall=0.537, Precision=0.934. Fold 3 outlier
(mAP50=0.940), all other folds 0.46–0.65. **Verdict: improvement / P2 reference
configuration established.**

**LEARNING:** lr0=0.001 is the correct fine-tuning rate for DL_Modell_FV.pt. Remaining
high variance (std=0.194) is a fold-split artefact (fold 3 structural outlier), not
training instability. Generalisation requires on-the-fly augmentation and more data.

---

### DoE Grid (3×4 Factorial) — Phase 2, ~2026-05-06 (SLURM 320698–320709)

**HYPOTHESIS:** Varying FP_NEG_OVERSAMPLE ∈ {1,2,3} and BG_RATIO ∈ {0,1,2,3} will
identify a data composition that improves generalisation beyond the P2-H baseline.

Full factorial grid (12 cells, 11 completed — fp3_bgr3 OOM/timeout). All share the P2-H
reference config (lr0=0.001, freeze=10, DL_Modell_FV.pt, BG bug fixed). Grid infrastructure:
env-var-overridable knobs, `submit_grid.sh`, auto-named PROJECT_DIR.

**RESULT:** Total mAP50 spread across 11 cells: 0.548–0.604 (range 0.056). BG_RATIO
consistently hurts within every FP level. fp=2 gives the highest recall (0.638) but driven
by one fold-2 outlier. fp1_bgr0 (= P2-H) is the grid winner. **Verdict: inconclusive /
research direction exhausted.**

**LEARNING:** Data composition (FP oversampling, background tiles) cannot overcome the
generalisation gap at P2-only scale. Cross-fold std (~0.19) swamps any signal from the
factors. The recall problem is a data-scarcity/augmentation problem, not a class-imbalance
problem. Pivot: multi-patient corpus + on-the-fly augmentation.

---

### P3-A — Phase 3, 2026-05-09 (SLURM 322719)

**HYPOTHESIS:** Expanding to P1–P8 LOPO with per-patient Normal oversampling and on-the-fly
augmentation will dramatically improve recall and reduce variance.

Structural pivot. New corpus: 8 patients (1103 images, 1124 Atypisch, 78 Normal; raw ratio
14.4:1). New CV protocol: 8-fold LOPO. New oversampling: P5×8, P6×15, P7×8, P8×10 (effective
ratio ≈ 2:1). Augmentation: on-the-fly (`augment=True`, `mosaic=0.0`, `flipud=0.5`). Base
weights switched from DL_Modell_FV.pt to **yolo11n.pt** (cleaner provenance; corpus large
enough to train from COCO). `cls=1.0` (double class-loss weight).

Session decisions embedded: strict LOPO on negatives (patient tag held out consistently);
three-category negative provenance clarified; earth-4 cluster confirmed as 8×L40S.

**RESULT:** Mean mAP50=0.844 ± 0.137, Recall=0.840. R_Normal on meaningful folds: P5=0.800,
P6=1.000, P7=0.818, P8=0.826 (mean 0.861). **Verdict: major improvement.**

**LEARNING:** Multi-patient LOPO + class-aware oversampling + on-the-fly augmentation together
solve the generalisation problem that data composition alone could not. On-the-fly augmentation
is what makes oversampling effective — frozen augmented copies waste compute. Fold 5 (P5)
remains the soft spot (3 Atypisch in test, each miss = −33 pp).

---

### P3-B — Phase 3, ~2026-05-09 to 2026-05-11 (SLURM 321743–321758)

**HYPOTHESIS:** A systematic 4-factor training-time knob sweep will find a configuration
that lifts R_Normal above the P3-A baseline of 0.795 without sacrificing R_Atypisch.

4-factor matrix (2×3×2×2 = 24 cells; 14 run): label_smoothing ∈ {0.0,0.1}, degrees ∈
{0,5,10}, dfl ∈ {1.5,2.0}, freeze ∈ {0,10}. `ba_tinker.py` with env-overridable knobs.
Default cell (ls=0.0, deg=0, dfl=1.5, fr=10) ≡ P3-A as a sanity check.

**RESULT:**
- Winner: `ls0.0_deg5_dfl1.5_fr10` — Recall=0.866 (+5.8 pp), R_Normal=0.840 (+8.6 pp),
  mAP50=0.864 (+2.0 pp) vs P3-A.
- `dfl=2.0` consistently regresses R_Normal across all other knobs.
- `degrees=5` is the sweet spot (0 loses ~4 pp R_Normal; 10 adds nothing further).
- `freeze=10` beats `freeze=0` in every recall-relevant slice.
- `label_smoothing=0.1` is a bit-for-bit no-op (env-var likely didn't propagate to YOLO's
  loss path, or the implementation is inert in this Ultralytics version).

**Verdict: improvement / production recipe confirmed.**

**LEARNING:** degrees=5 is the single biggest lever. dfl tilt and freeze depth are secondary
but directional. label_smoothing is irrelevant for this task/version. Recipe
`deg5_dfl1.5_fr10` adopted as the production configuration for all subsequent phases.

---

### P3-C — Phase 3, ~2026-05-11 (SLURM 323280) — Negative Result

**HYPOTHESIS:** YOLO `model.tune()` run on the production recipe will safely find
hyperparameters that improve on the P3-B matrix winner.

30 iterations × 50 epochs, P4 holdout, mAP50-95 as fitness. Tuner output: `degrees=0.0,
dfl=2.47, lr0=0.00884`. Both dominant moves contradict the P3-B matrix: degrees=0 loses
~4 pp R_Normal vs the matrix-optimal 5; dfl=2.47 regresses R_Normal even more than dfl=2.0
(already ruled out in P3-B). Tuner fitness 0.736 is within noise of P3-B winner's 0.741.

Stage-3 validation (full 8-fold LOPO on the tuner yaml) skipped: ~6–12 GPU-hours cost,
high prior probability of underperforming on R_Normal. **Verdict: negative result.**

**LEARNING:** Single-fold automated tuning is blind to per-class recall balance and cross-
patient generalisation. The tuner optimised Atypisch performance on P4 (Atypisch-dominant
fold), sacrificing Normal recall generalisation. The P3-B 8-fold matrix is the stronger
experimental design. P3-C is a productive negative finding for the thesis.

---

### P3-D — Phase 3, 2026-05-11 — USZ Deliverable

**HYPOTHESIS:** Training on full P1–P8 with the P3-B recipe produces a deployment-ready
model with generalisation ≥ the LOPO fold means.

Post-CV standard practice: take the validated recipe and train one model on all data with
no holdout. 85/15 stratified-by-patient val split for early stopping only (leakage
acceptable — unbiased estimate is from P3-B LOPO). Single training on 1 L40S.

**RESULT:** Val metrics (not authoritative): mAP50=0.854, Recall=0.839, R_Atypisch=0.964,
R_Normal=0.713. R_Normal suppressed by small val Normal count (~10–15 labels = ±0.25 pp
sensitivity per missed cell). Authoritative estimate remains P3-B LOPO: R_Normal=0.840.
**Verdict: success / USZ v1 deliverable shipped.**

`tinker_final/final/weights/best.pt` → shipped to USZ for model-assisted annotation of
next patient cohort (P9–P15, later expanded to P16–P53).

**LEARNING:** Full-corpus training converges cleanly. The val R_Normal dip is a small-
sample artefact, not a regression signal. The `best.pt` checkpoint selection criterion
(YOLO's weighted fitness) biases toward Atypisch epochs — this is an accepted limitation
given the imbalanced val split.

---

### P3-E — Phase 3, ~2026-05-11 — Negative Result

**HYPOTHESIS:** yolo11l (43 M parameters, 16× larger than yolo11n) will learn better
feature representations and lift R_Normal.

P3-A defaults (degrees=0) with yolo11l.pt. 8-fold LOPO on P1–P8.

**RESULT:** mAP50=0.855 (+1.1 pp vs P3-A), Recall=0.785 (−2.3 pp), R_Normal=0.695
(−5.9 pp — worst R_Normal result in the entire project). **Verdict: regression / negative
result.**

**LEARNING:** The corpus (~1100 images, 170–220 per fold) is too small for 43 M parameters.
The randomly re-initialised `cv3` head has proportionally more free parameters to learn from
scratch. The larger model memorises Atypisch patterns and loses Normal-class generalisation.
`yolo11n` is confirmed as the correct model size. Scaling capacity is not the path to better
R_Normal — more data is.

---

### P4-A — Phase 4, 2026-05-23 (SLURM 334000)

**HYPOTHESIS:** Expanding from P1–P8 to P1–P15 (7 new patients including P9 with 195
Normal annotations) will further improve R_Normal generalisation.

P1–P15 corpus (1564 files, 1288 Atypisch, 391 Normal; raw ratio 3.3:1; effective ratio 1.5:1
after auto oversampling). Generalised FP loading from 8 Excel files (P2 + P9–P15; 1434 FP
entries total). 15-fold LOPO. **Note:** ran degrees=0 inadvertently (DEGREES env var not yet
added to the script at time of submission). Equivalent to P3-A recipe on larger corpus.

**RESULT:** mAP50=0.903, Recall=0.904, R_Atypisch=0.913, R_Normal=0.890. vs P3-A (degrees=0,
P1–P8): +5.9 pp mAP50, +9.6 pp Recall, +13.6 pp R_Normal. **Verdict: major improvement.**

**LEARNING:** Corpus expansion is the dominant lever at this stage. P9 (195 Normal
annotations) is the single highest-value patient in the corpus. The degrees=0 vs degrees=5
gap from P3-B remains unmeasured on P1–P15 — P4-B quantifies it.

---

### P4-B — Phase 4, date not recorded (SLURM 337215)

**HYPOTHESIS:** Applying the full P3-B winner recipe (degrees=5) on P1–P15 will replicate
the +4–5 pp R_Normal seen in the P3-B matrix vs P3-A.

Identical to P4-A but with `degrees=5` explicitly set. 15-fold LOPO.

**RESULT:** mAP50=0.887, Recall=0.895, R_Normal=0.882. Delta vs P4-A: all metrics within
±1.6 pp. **Verdict: inconclusive (within noise of P4-A).**

**LEARNING:** The degrees=5 benefit observed in P3-B (+4–5 pp R_Normal) diminishes or
disappears at larger corpus scale. Cross-fold std (~0.1) makes sub-2 pp deltas uninterpretable.
P4-B used as freeze comparison reference for P4-C because it has explicit recipe parameters.

---

### P4-C — Phase 4, date not recorded (SLURM 337214) — Null Result

**HYPOTHESIS:** At P1–P15 scale (~1000–1300 training images per fold), unfreezing the full
backbone (freeze=0) will outperform the frozen recipe (freeze=10).

Identical to P4-B except freeze=10→0. 15-fold LOPO.

**RESULT:** All metrics within ±0.4 pp of P4-B (freeze=10). R_Normal delta: +0.02 pp.
**Verdict: null result / freeze=10 confirmed.**

**LEARNING:** The "corpus too small to unfreeze" intuition holds at P1–P15 scale (~1000–1300
training images per fold). Backbone features from COCO pretraining remain more valuable than
end-to-end adaptation at this data scale. No freeze comparison needed for P1–P53. The P3-B
recipe transfers across corpus scales without re-tuning.

---

### P5-A — Phase 5, planned as of 2026-05-27 — Not Yet Run

**HYPOTHESIS:** Expanding from P1–P15 to P1–P53 (38 new patients, ~305 new annotated images,
~120 new Normal annotations, 3637 FP negatives) will further improve R_Normal and reduce
cross-fold variance.

Full 53-patient corpus. Recipe: P3-B winner confirmed (`degrees=5, dfl=1.5, freeze=10,
lr0=0.001, cls=1.0`). LOPO up to 53 folds (exact count after empty-label filtering). Fully
auto-computed oversampling (`PATIENT_OVERSAMPLE_FIXED = {}`). P16–P53 include many pure-FP
patients (no mast cell annotations, only FP Excel entries); a 2026-05-27 code fix ensures
these do not generate undefined LOPO folds.

**RESULT:** TBD — **not yet submitted.**

---

## 4. CROSS-CUTTING FINDINGS

### What Consistently Helped

1. **Multi-patient LOPO over single-patient or stratified k-fold CV.** The jump from P2-only
   5-fold (mAP50 ~0.60) to P1–P8 8-fold LOPO (mAP50 0.844) was the single largest
   performance gain in the project. Holding out entire patients captures real biological
   inter-patient heterogeneity; random stratified splits leak patient texture into validation.

2. **lr0=0.001 for fine-tuning from domain-specific weights.** Any run at lr0=0.01 with
   DL_Modell_FV.pt either collapsed (P2-C, P2-E, P2-G folds 3/4) or achieved sub-optimal
   recall (P2-C: 0.397). lr0=0.001 resolved all collapse cases and matched or exceeded the
   best lr0=0.01 results.

3. **On-the-fly augmentation (`augment=True`, `mosaic=0.0`).** Enabled in P3-A alongside
   other changes. Pre-applied augmentation (Phase 1 and early Phase 2) provided no real
   regularisation benefit — transforms were frozen to disk, effective from epoch 1 only.
   On-the-fly transforms make oversampled duplicates visually distinct per epoch.

4. **freeze=10 (backbone frozen).** Confirmed in P3-B matrix and P4-C ablation. Prevents
   the large number of backbone parameters from being adapted on a small dataset where
   there is insufficient signal to outperform COCO-pretrained features.

5. **degrees=5 rotation augmentation.** The single biggest training-time lever in the P3-B
   matrix: +4–5 pp R_Normal vs degrees=0 at P1–P8 scale. Bone marrow cells have no canonical
   orientation — rotation regularises without distorting morphological cues. Effect diminishes
   (but does not reverse) at larger corpus scale (P4-A/B delta within noise).

6. **corpus expansion over data-ratio manipulation.** The DoE grid (Phase 2) showed that
   FP oversampling and background sampling cannot overcome a 0.19 cross-fold std. Adding
   real patients (Phase 3–4) delivered +9.6 pp recall where data-ratio tuning delivered zero.

### What Didn't Help

1. **BG_RATIO > 0 (random background tiles).** Consistently hurt mAP50 and recall within
   every FP level in the DoE grid. The negative effect is specific to the random background
   category — P1 gold negatives and P2 hard FPs are different assets and were never ablated
   by the DoE.

2. **FP_NEG_OVERSAMPLE above ×1.** fp=2 showed marginal recall improvement (0.638 vs 0.537)
   but this was driven by one fold-2 outlier in SLURM 320702; without the outlier, fp=2 is
   indistinguishable from fp=1. fp=3 underperforms both on mAP50.

3. **Tuner hyperparameters from a from-scratch recipe (P2-E, P2-F).** Caused catastrophic
   forgetting. A tuner run on the production recipe (P3-C) was safe but still contradicted
   the P3-B matrix evidence on degrees and dfl — single-fold tuning is blind to per-class
   recall balance.

4. **dfl=2.0.** Consistently regressed R_Normal vs dfl=1.5 across all degree and freeze
   combinations in the P3-B matrix. Tightening localisation loss tilts the balance away from
   rare-class classification discrimination.

5. **Larger model (yolo11l, P3-E).** 16× more parameters caused worse Normal-class
   generalisation (R_Normal 0.695 vs 0.754 for yolo11n on P1–P8). The corpus is too small
   for 43 M parameters.

6. **Pre-applied albumentations augmentation (Phase 1, early Phase 2).** Transforms baked
   to disk before training. Each epoch sees identical frozen copies — no regularisation
   benefit after epoch 1.

### Surprises and Unexpected Behaviour

1. **P3-A (LOPO) did not increase variance vs the P2 DoE.** The intuition was that harder
   evaluation (true held-out patients vs random splits) would *raise* std. Instead, std
   dropped from ~0.19 (P2 DoE) to 0.137 (P3-A), because the larger and more diverse training
   corpus more than compensated for the harder protocol.

2. **DL_Modell_FV.pt was abandoned in Phase 3.** The Phase 2 work used it as the anchor
   throughout. In Phase 3, the corpus was large enough (1100+ images) to train cleanly from
   the generic `yolo11n.pt` without a domain-specific warm-start — producing a cleaner
   provenance chain with no improvement sacrifice.

3. **label_smoothing=0.1 is a no-op.** Cells with ls=0.0 and ls=0.1 produce bit-identical
   fold metrics in the P3-B matrix. Either the env-var didn't propagate to YOLO's loss path
   or the implementation is inert in this Ultralytics version. This was discovered, not
   anticipated.

4. **The P3-B tuner (P3-C) contradicted the matrix on key parameters.** After the P2-E
   failure was reframed as a recipe mismatch, a "safe" tuner run was expected to provide
   useful incremental improvements. Instead, it converged on degrees=0 and dfl=2.47 —
   both moves in the wrong direction according to the P3-B matrix. The explanation is that
   single-fold mAP50-95 tuning on an Atypisch-dominant fold optimises for Atypisch at the
   expense of Normal.

5. **P4-A (degrees=0 on P1–P15) matched P4-B (degrees=5 on P1–P15).** The +4–5 pp R_Normal
   benefit of degrees=5 seen in P3-B (P1–P8 scale) did not replicate cleanly at P1–P15
   scale. The effect may diminish with a more diverse corpus, or may be within noise (std ~0.1).

### Failure Modes

**SLURM concurrent job interference (structural, fixed):** Concurrent jobs shared temporary
workspace paths (`fold_N_workspace/`, `PROCESSED_DIR`) and the output `PROJECT_DIR`. Corrupted
folds 2, 3, 5 of P2-A and P2-B; likely affected P2-E and P2-F. Direct evidence: folds with
identical results to 4 decimal places, and `results.csv` files with interleaved epoch rows
from two different training runs. Fixed by: `$SLURM_JOB_ID` namespacing for temp dirs;
`$SLURM_JOB_NAME` derivation for `PROJECT_DIR`.

**BG_RATIO bug (data pipeline, fixed 2026-05-06):** `bg_paths` was sampled in `__main__` but
not passed to `fold_tasks`. Background images never reached training workers. All BG_RATIO > 0
runs before P2-H were effectively BG_RATIO=0. Invalidated all previous BG_RATIO comparisons
(only the DoE grid with the fix is authoritative for BG conclusions).

**Per-class positional indexing bug (measurement, not fixed):** `per_class(metrics, idx)`
indexes `metrics.box.r` by position rather than by `ap_class_index`. When a class is absent
from both GT and predictions in a fold, Ultralytics returns a shorter array and positional
indexing misassigns or returns NaN. Affected symptom: Fold 5 (P13 holdout) reports
R_Normal=NaN in every run; the true Normal recall (0.954) appears in the R_Atypisch column.
Any pure-single-class holdout patient may be affected.

**Pre-applied augmentation (design limitation, disabled):** Not a bug — a design choice that
proved ineffective. Disabled in P2-H and replaced with on-the-fly augmentation in P3-A.

**OOM / timeout (fp3_bgr3, SLURM 320709):** The largest DoE cell (3× FP + 3 BG per annotated
image) failed with no TEST output. Not retried — this corner of the grid was already ruled out
by the BG_RATIO trend.

---

## 5. FINAL STATE

### Best Completed Run: P4-B (SLURM 337215)

Configuration: P1–P15 corpus, 15-fold LOPO, yolo11n.pt, `degrees=5, dfl=1.5, freeze=10,
lr0=0.001, cls=1.0, mosaic=0.0, flipud=0.5, augment=True`.

| Metric | Value |
|--------|-------|
| mAP50 | 0.887 |
| mAP50-95 | — (not aggregated in table) |
| Recall | 0.895 |
| Precision | — |
| R_Atypisch | 0.900 |
| **R_Normal** | **0.882** |
| Cross-fold std (mAP50) | ~0.14 |

(P4-B and P4-A are statistically indistinguishable — P4-B is cited as the production recipe
because all parameters are explicit.)

### Per-Class Performance

**R_Atypisch (0.900):** Strong across nearly all folds. Weakest in folds where the holdout
patient has very few Atypisch instances: Fold 11 (P5 holdout: 3 Atypisch, recall=0.333 —
one miss = −33 pp) and Fold 15 (P9 holdout: 4 Atypisch, recall=0.769 — one miss = −25 pp).
These are small-sample artefacts, not model regressions.

**R_Normal (0.882):** Acceptable. Still trails R_Atypisch by ~2 pp. The gap has narrowed
from 11 pp (P3-A) to 2 pp over the course of Phase 3–4 development. The remaining gap is
partially a measurement artefact (per_class bug excludes P13 from the mean) and partially
genuine. The weakest reliable R_Normal measurements are from folds where the holdout patient
has very few Normal instances (P1: 2 Normal in test = 0.286; P3: 4 Normal = values noisy).

**Fold 1 (P1 holdout) is the persistent weakness:** mAP50 ~0.58–0.63 and R_Normal ~0.286
across all runs. P1 uses the old `Pos_neg` annotation regime (different from P2–P15's
annotated format) and has only 2 Normal instances in the test set. This is structural — not
improvable by hyperparameter tuning. More P1-regime patients would help.

### Shipped Deployment Model

`tinker_final/final/weights/best.pt` (P3-D) was shipped to USZ as the v1 model. It is the
full-corpus P1–P8 model trained on the P3-B winner recipe. The authoritative generalisation
estimate is P3-B LOPO: Recall=0.866, R_Normal=0.840, R_Atypisch=0.892.

P4-B represents the updated evaluation with the expanded P1–P15 corpus, but no corresponding
full-corpus P1–P15 final model has been trained yet (the equivalent of P3-D for Phase 4).

### Known Limitations

1. **All training data comes from SM patients.** P1–P4 are nearly pure-Atypisch; P6 and P8
   are the only pure-Normal (non-SM-like) reference patients. Generalisation to non-SM
   patients presenting with Normal-dominant slides is an out-of-distribution problem.
   P4-A/B empirically validate this to some extent (P6, P8 holdout folds achieve R_Normal
   ≥ 0.82–1.00), but the regime is underrepresented in training.

2. **Very small annotated counts for some patients.** P13 (7 images), P14 (7 images),
   P24/P28/P29/P31 (1 image each). These patients' holdout folds have extreme metric
   sensitivity (1 missed cell = full recall loss). They contribute FP negatives reliably
   but their test metrics are directional only.

3. **Per-class positional indexing bug** causes R_Normal to report as NaN for pure-single-class
   holdout patients where the model makes no predictions for the absent class. Currently
   affects P13 in every run. The true value is in the R_Atypisch column.

4. **label_smoothing is inert** in this Ultralytics version. This rare-class calibration lever
   is unavailable until a version where it is correctly implemented. No workaround identified.

5. **The P5-A run (P1–P53) has not been submitted.** The current best results (P4-B/P4-C)
   are on 15 patients. 38 additional patients have been integrated into the script and their
   data verified, but the training run outcome is unknown.

---

## 6. UNRUN IDEAS / FUTURE WORK

### High Priority (quantified plans exist)

**P5-A: P1–P53 full corpus baseline.** Script ready, data verified, FP Excels filled in,
code fixes applied (empty-label guard, PATIENT_FP_EXCELS import). Recipe confirmed. Single
run required. Expected: mAP50 > 0.90, R_Normal > 0.89. (See `19_P5-A_P1-P53_planned.md`.)

**Full-corpus final model (P1–P15 or P1–P53).** Equivalent of P3-D but trained on the full
expanded corpus. The P4-B recipe is the candidate. This would be the v2 USZ deployment model.

### Medium Priority (considered in log; not run)

**cls=1.0 ablation at expanded scale.** With effective ratio now 1.5:1 (P1–P15) and moving
toward 1:1 (P1–P53 with Normal-heavy new patients), the double class-loss weight may no
longer be necessary. Requires one additional LOPO run with cls=0.5 (YOLO default) vs cls=1.0
on P1–P15 or P1–P53.

**Remaining P3-B matrix cells (10 cells).** The `ls=0.1, degrees ∈ {5,10}` block and two
`dfl=2.0` cells not run due to `QOSMaxSubmitJobPerUserLimit`. Assessment: not decision-
relevant. label_smoothing is empirically inert; degrees=5 zone is already mapped.

**Requesting 8 GPUs in SLURM (`--gres=gpu:l40s:8`).** earth-4 has 8× L40S confirmed.
Current runs use 2. Moving to 8 GPUs for P5-A would reduce wall clock by ~4× (15→53 folds
on more GPUs simultaneously, less contention per GPU).

### Low Priority / Speculative

**Synthetic Normal tile generation.** Cropping patches centred on Normal cells from P5–P8
to artificially expand the Normal training set. Assessed in Phase 3 strategy notes and
deemed unnecessary given oversampling+augmentation results. Would revisit only if R_Normal
stalls below 0.85 after P5-A.

**Per-class loss weights.** YOLO11 does not expose per-class loss weights cleanly. The global
classification loss weight `cls` (default 0.5, currently 1.0) is the available proxy.
Increasing further (cls=2.0) or applying class-specific weighting via a custom loss function
would require Ultralytics source modification.

**`mosaic=True` with scale=0.5.** Standard mosaic is inappropriate for single-cell 512 px
crops. A patched mosaic that composites only crops of the same patient (preserving stain
distribution) is a hypothetical option. Not explored; not planned.

**External validation set (P6 or P8 as held-out).** Withholding one pure-Normal patient
entirely as a post-hoc external validation. LOPO already provides this per-fold, but a
dedicated held-out set that never informed any configuration decision would be cleaner for
thesis reporting. Feasible now that the corpus has 53 patients — withholding one patient is
less costly.

---

## APPENDIX A: Full Runs Table

| Run | SLURM | Phase | Base Model | FP× | BGr | lr0 | freeze | mAP50 (mean±std) | Recall | R_Normal | Verdict |
|-----|-------|-------|-----------|-----|-----|-----|--------|------------------|--------|----------|---------|
| P1-baseline | — | 1 | yolo11n | — | — | 0.01 | — | 0.800 ± 0.129 (val) | 0.719 | — | success (poc) |
| P2-A | 319399 | 2 | yolo11n | ×1 | 0 | 0.01 | — | 0.580 ± 0.172 | 0.617 | — | corrupted |
| P2-B | 319400 | 2 | yolo11n (mod) | ×1 | 0 | 0.01 | — | 0.614 ± 0.155 | 0.667 | — | corrupted |
| P2-C | 319401 | 2 | DL_FV | ×1 | 0 | 0.01 | — | 0.469 ± 0.035 | 0.397 | — | regression |
| P2-D | 319416 | 2 | DL_FV (mod) | ×1 | 0 | 0.01 | — | 0.590 ± 0.090 | 0.655 | — | improvement |
| P2-E | 319536 | 2 | DL_FV+tuner | ×1 | 0 | 0.01 | — | 0.448 ± 0.007 | 0.413 | — | FAILED |
| P2-F | 319538 | 2 | DL_FV+mod+tuner | ×1 | 0 | 0.01 | — | 0.508 ± 0.057 | 0.571 | — | regression |
| P2-G | 319934 | 2 | DL_FV | ×1 | 2* | 0.01 | — | 0.530 ± 0.090 | 0.484 | — | regression |
| P2-H | 320698 | 2 | DL_FV | ×1 | 0 | 0.001 | 10 | 0.604 ± 0.194 | 0.537 | — | improvement |
| DoE fp1_bgr0 | 320698 | 2 | DL_FV | ×1 | 0 | 0.001 | 10 | 0.604 ± 0.194 | 0.537 | — | grid ref |
| DoE fp1_bgr1 | 320699 | 2 | DL_FV | ×1 | 1 | 0.001 | 10 | 0.562 ± 0.181 | 0.531 | — | regression |
| DoE fp1_bgr2 | 320700 | 2 | DL_FV | ×1 | 2 | 0.001 | 10 | 0.564 ± 0.172 | 0.523 | — | regression |
| DoE fp1_bgr3 | 320701 | 2 | DL_FV | ×1 | 3 | 0.001 | 10 | 0.591 ± 0.174 | 0.518 | — | regression |
| DoE fp2_bgr0 | 320702 | 2 | DL_FV | ×2 | 0 | 0.001 | 10 | 0.581 ± 0.184 | 0.638 | — | ≈ ref |
| DoE fp2_bgr1 | 320703 | 2 | DL_FV | ×2 | 1 | 0.001 | 10 | 0.565 ± 0.163 | 0.523 | — | regression |
| DoE fp2_bgr2 | 320704 | 2 | DL_FV | ×2 | 2 | 0.001 | 10 | 0.592 ± 0.180 | 0.637 | — | ≈ ref |
| DoE fp2_bgr3 | 320705 | 2 | DL_FV | ×2 | 3 | 0.001 | 10 | 0.596 ± 0.172 | 0.539 | — | ≈ ref |
| DoE fp3_bgr0 | 320706 | 2 | DL_FV | ×3 | 0 | 0.001 | 10 | 0.548 ± 0.185 | 0.537 | — | regression |
| DoE fp3_bgr1 | 320707 | 2 | DL_FV | ×3 | 1 | 0.001 | 10 | 0.566 ± 0.183 | 0.546 | — | regression |
| DoE fp3_bgr2 | 320708 | 2 | DL_FV | ×3 | 2 | 0.001 | 10 | 0.572 ± 0.186 | 0.526 | — | regression |
| DoE fp3_bgr3 | 320709 | 2 | DL_FV | ×3 | 3 | 0.001 | 10 | ❌ FAILED | — | — | OOM |
| P3-A | 322719 | 3 | yolo11n | ×1 | 0 | 0.001 | 10 | 0.844 ± 0.137 | 0.840 | 0.861 | major improvement |
| P3-B (winner) | 321743–58 | 3 | yolo11n | ×1 | 0 | 0.001 | 10, deg=5 | 0.864 ± ~0.14 | 0.866 | 0.840 | improvement |
| P3-C | 323280 | 3 | yolo11n | ×1 | 0 | 0.001 | 10 | tuner only | — | — | negative |
| P3-D | — | 3 | yolo11n | ×1 | 0 | 0.001 | 10, deg=5 | val 0.854 | val 0.839 | 0.713† | USZ deliverable |
| P3-E | — | 3 | yolo11l | ×1 | 0 | 0.001 | 10 | 0.855 ± ~0.14 | 0.785 | 0.695 | regression |
| P4-A | 334000 | 4 | yolo11n | ×1 | 0 | 0.001 | 10, deg=0 | 0.903 ± 0.100 | 0.904 | 0.890 | major improvement |
| P4-B | 337215 | 4 | yolo11n | ×1 | 0 | 0.001 | 10, deg=5 | 0.887 ± ~0.14 | 0.895 | 0.882 | **best complete** |
| P4-C | 337214 | 4 | yolo11n | ×1 | 0 | 0.001 | 0, deg=5 | 0.883 ± ~0.14 | 0.893 | 0.882 | null result |
| P5-A | TBD | 5 | yolo11n | ×1 | 0 | 0.001 | 10, deg=5 | TBD | TBD | TBD | planned |

*P2-G BG_RATIO=2 was claimed; actual=0 due to BG_RATIO bug (fixed in P2-H).
†P3-D R_Normal=0.713 is a val-split estimate, not LOPO; authoritative R_Normal from P3-B LOPO = 0.840.

---

## APPENDIX B: Glossary

| Term | Definition |
|------|-----------|
| **mAP50** | Mean Average Precision at IoU threshold 0.50. Standard object detection benchmark. Reported per-fold and as mean±std across folds. |
| **mAP50-95** | Mean AP averaged over IoU thresholds 0.50–0.95 in steps of 0.05. Stricter localisation measure; values are always lower than mAP50. |
| **Recall** | Fraction of all GT cell instances detected: TP / (TP + FN). The primary clinical metric — missing a cell is worse than a false alarm. |
| **R_Atypisch** | Per-class recall for class 0 (Atypisch mast cells). |
| **R_Normal** | Per-class recall for class 1 (Normal mast cells). The rare class; the WHO criterion requires both classes to be reliable. |
| **Precision** | TP / (TP + FP). High precision = few false alarms. Secondary metric in this clinical context. |
| **LOPO CV** | Leave-One-Patient-Out Cross-Validation. Each fold holds out all images from one patient; the model is evaluated exclusively on that patient's data. Prevents patient-level texture leakage between train and val. |
| **DL_Modell_FV.pt** | Domain-specific YOLO11n checkpoint pretrained by FV on the P1 patient dataset (nc=1, single class). Used as the fine-tuning anchor in Phases 1–2. Replaced by `yolo11n.pt` in Phase 3. |
| **yolo11n.pt** | YOLO11 nano (2.6 M parameters) pretrained on COCO. Current anchor for all Phase 3–5 runs. |
| **yolo11l.pt** | YOLO11 large (43 M parameters) pretrained on COCO. Tested in P3-E; found to underperform yolo11n at this corpus size. |
| **nc** | Number of classes. DL_Modell_FV.pt has nc=1; all Phase 2+ runs use nc=2 (Atypisch, Normal). The nc change causes `cv3` to be randomly re-initialised. |
| **cv3** | The Detect head's classification branch (`Conv2d(c3, nc, 1)`). Randomly re-initialised when nc changes between checkpoint and training config. Responsible for early-epoch classification noise. |
| **cv2** | The Detect head's bounding-box regression branch. nc-independent; transfers cleanly from any checkpoint. |
| **freeze=N** | Freeze the first N backbone layers (model.0 through model.N-1). `freeze=10` freezes layers 0–9 (the full yolo11n backbone), leaving the neck and head trainable. |
| **FP_NEG_OVERSAMPLE** | How many times the confirmed false-positive negative tiles are repeated in `train.txt`. Default: 1 (no oversampling). DoE-validated range: 1–2. |
| **BG_RATIO** | Ratio of random unlabeled background tiles to annotated images in training. DoE result: uniformly hurts; `RANDOM_BG_RATIO=0` in all production runs. Distinct from P1 gold negatives and P2 hard FPs. |
| **P1 gold negatives** | Hand-curated empty tiles from the original P1 training. High-quality, always included when P1 is in train. |
| **P2 hard FPs** | Confirmed false-positive tiles from the USZ annotation workflow. The model's known failure cases — high-value hard negatives. |
| **SLURM_JOB_ID** | SLURM's unique numeric ID for each submitted job. Used to namespace temporary directories and prevent concurrent job workspace collisions. |
| **SLURM_JOB_NAME** | Human-readable job name set at `sbatch` submission via `--job-name`. Locked at submission time — used to derive `PROJECT_DIR` so concurrent jobs write to different output directories. |
| **submit_grid.sh** | Shell script that submits all cells of a factorial grid as independent SLURM jobs in one command. Passes cell configuration via environment variables (`FREEZE`, `DEGREES`, etc.). |
| **DFL** | Distribution Focal Loss weight (`dfl` in YOLO config). Controls the localisation loss contribution. Matrix finding: `dfl=2.0` regresses R_Normal vs `dfl=1.5` in this domain. |
| **cls** | Classification loss weight (`cls` in YOLO config). Default 0.5; set to 1.0 in all Phase 3+ runs to invest more model capacity in class discrimination. |
| **mosaic** | YOLO data augmentation that composites four images into one training sample. Appropriate for scene detection (COCO), inappropriate for single-cell crops — must be set to 0.0 for this task. |
| **flipud** | Vertical flip probability in YOLO augmentation. Set to 0.5 (default is 0) because bone marrow cells have no canonical vertical orientation. |
| **per_class bug** | Measurement bug in `ba_improved_comb.py`: `per_class(metrics, idx)` indexes `metrics.box.r` by position, not by `ap_class_index`. When a class is absent from both GT and predictions, the array is shorter than nc and positional indexing misassigns recall values. Fix known but not yet applied. |
| **P_N** | Patient N — the unique patient identifier used throughout the project. P1–P4: SM patients (Atypisch-dominant). P5–P8: Normal-rich/pure-Normal. P9–P15: second USZ acquisition batch. P16–P53: third batch (many are pure-FP patients with no mast cell annotations). |
| **pure-FP patient** | A patient with no mast cell annotations — only confirmed false-positive entries in their Excel file. They do not receive a LOPO fold (their test set would have zero GT boxes) but their FP entries are added to training for all other folds as hard negatives. |
| **WHO 25% criterion** | WHO minor criterion for SM: >25% atypical mast cells in bone marrow aspirate. The clinical decision boundary that the Atypisch/Normal ratio must cross reliably for diagnostic utility. |
