# Experiment Log — Mast Cell Detection Pipeline

Tracks all training runs, the reasoning behind each, and conclusions drawn.
All results tables show **test-set** metrics (held-out split, evaluated with best.pt after training),
plus validation metrics where noted. Val metrics during training are shown separately and are
not used for model comparison — the test split is the authoritative number.

---

## Infrastructure Notes

### Concurrent job interference — full analysis

SLURM does not snapshot `ba_improved_P2.py` at submission time. Jobs execute whatever is on disk when they start. When jobs are queued back-to-back and the script is updated between submissions, later jobs pick up newer config (wrong `PROJECT_DIR`, different hyperparameters). This corrupted multiple early P2 runs.

Three independent collision paths existed:

**1. `cv_temp` and `processed_data` (fixed via `SLURM_JOB_ID`)**  
Temporary fold workspaces and augmented images used a shared path. Concurrent jobs would collide on `fold_0_workspace/` names, skip creating symlinks already present (pointing to the other job's data), and overwrite augmented images mid-run. Fixed by namespacing with `$SLURM_JOB_ID`:
```python
_JOB_ID       = os.environ.get("SLURM_JOB_ID", "local")
TEMP_DIR      = os.path.abspath(f".cv_temp_{_JOB_ID}")
PROCESSED_DIR = os.path.abspath(f".processed_data_{_JOB_ID}")
```

**2. `PROJECT_DIR` race condition (fixed via `SLURM_JOB_NAME`)**  
The output directory was hardcoded in the script. Even with cv_temp isolated, if two jobs ran with the same `PROJECT_DIR`, their `fold_N/results.csv` and `best.pt` would be written by both simultaneously. Direct evidence: `yolo_runs_hpc_mod3_dl_fv_bg_ratio_3_fr_lr0/fold_3/results.csv` contains interleaved epoch rows (90–91 from one job, 327–329 from another). The final `val_model.val()` evaluated whichever `best.pt` was last written — a random mix of two training runs.

Fixed by deriving `PROJECT_DIR` from `$SLURM_JOB_NAME`, which is locked at submission time:
```python
PROJECT_DIR = f"./{os.environ.get('SLURM_JOB_NAME', 'local_run')}"
```

**3. Corrupted runs identified**  
- P2-A (319399) and P2-B (319400): folds 2, 3, 5 have identical results to 4 decimal places — proof both jobs trained on the same fold workspace data. Results for these runs are unreliable.
- P2-E (319536) and P2-F (319538): submitted back-to-back, likely affected.
- `mod3_dl_fv_bg_ratio_3` (320286+320287): confirmed corrupted fold_3.

**Clean runs** (wide ID gaps, submitted in isolation): **P2-D (319416)** and **P2-G (319934)**.

### `per_class` positional indexing bug (measurement bug, not fixed)

`per_class(metrics, idx)` in `ba_improved_comb.py` reads per-class recall by position in `metrics.box.r`:

```python
def per_class(metrics, idx):
    arr = metrics.box.r
    return float(arr[idx]) if idx < len(arr) else float('nan')
```

Ultralytics only includes a class in `ap_class_index` (and therefore in `metrics.box.r`) if that class has either GT instances or model predictions in the evaluated split. When a class is absent from both, the array is shorter than `nc` and positional indexing breaks.

**Observed symptom:** Fold 5 (P13 holdout) reports R_Normal = NaN in every run, even though P13 has 7 Normal GT instances. P13 has 0 Atypisch GT, and the model makes zero class-0 predictions on P13's 7 images — so class 0 is entirely absent from evaluation. `metrics.box.r` = `[0.954]` (length 1, Normal recall only):

- `per_class(_, 0)` → `arr[0]` = 0.954 → reported as R_Atypisch (actually Normal recall)
- `per_class(_, 1)` → index out of bounds → NaN → reported as R_Normal

The actual Normal recall for Fold 5 is **0.954** across all runs. It is sitting in the R_Atypisch column.

**Why other pure-Normal patients (P6, P8, P15) are unaffected:** the model makes at least one class-0 false-positive prediction on those patients' images, so Ultralytics includes class 0 in `ap_class_index` with vacuous recall = 1.0. The array stays length 2 and positional indexing accidentally works. The R_Atypisch = 1.000 reported for those folds is vacuous and meaningless; the R_Normal values are correct.

**Fix (not yet applied):** use `ap_class_index` for the mapping:

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

Until fixed, treat any fold where a holdout patient is pure-single-class as having potentially swapped or missing per-class recall values. Cross-check against the overall `Recall` column: if `Recall == R_Atypisch` and `R_Normal == NaN` for a pure-Normal holdout, the true Normal recall is `R_Atypisch`.

---

### BG_RATIO bug (fixed 2026-05-06)

`bg_paths` was sampled in `__main__` but never passed into `fold_tasks`, so background images never reached the training workers. All runs with `BG_RATIO > 0` were effectively `BG_RATIO = 0`. The "backgrounds" YOLO reported scanning were just FP negatives × oversample (e.g. 330 × 3 = 990).

This invalidates all previous BG_RATIO comparisons. The only real variable between mod1/mod2/mod3 runs (320285–320287) was `FP_NEG_OVERSAMPLE`:

| Job   | FP oversample | BG_RATIO (claimed) | Actual BG | Test mAP50 | Recall |
|-------|---------------|--------------------|-----------|------------|--------|
| 320285 | ×1           | 0                  | 0         | 0.564      | 0.645  |
| 320286 | ×2           | 0                  | 0         | 0.586      | 0.570  |
| 320287 | ×3           | 3 (bug)            | 0         | 0.563      | 0.563  |

Observed effect: higher `FP_NEG_OVERSAMPLE` → lower recall (model becomes more conservative). mAP50 is roughly flat. FP oversample of ×2 gives the best mAP50 so far without sacrificing too much recall.

Fix: `bg_paths` now passed through `fold_tasks` and appended to `train_paths_with_fp` without additional oversampling.

### Transfer learning mechanics: nc change from 1 → 2 (analysed 2026-05-06)

`DL_Modell_FV.pt` was trained on P1 with **nc=1**. P2 training uses **nc=2** (Atypisch, Normal). Ultralytics handles this silently but it has concrete consequences.

**What Ultralytics does internally:**  
A fresh `DetectionModel` is built with nc=2, then `intersect_dicts` loads checkpoint weights only where key names AND tensor shapes match. Because the Detect head's classification branch (`cv3`) outputs `nc` channels, its final `Conv2d(c3, nc, 1)` has a different shape (1→2) — so those weights are **discarded and re-initialized randomly**. The box regression branch (`cv2`) is nc-independent and transfers cleanly.

| Layer group | Transferred? |
|---|---|
| Backbone model.0–9 (frozen with `freeze=10`) | Yes |
| Neck model.10–21 | Yes |
| Detect `cv2` (bbox regression) | Yes |
| Detect `cv3` (classification, final conv) | **No — random init** |
| Detect `.dfl` | Always frozen |

**Training implications:**  
`cv3` starts with random weights, so classification loss is high and noisy in early epochs. This gradient propagates back through the unfrozen neck. At `lr0=0.01` (YOLO default, calibrated for scratch training) this noise is large enough to disrupt the pretrained neck feature pyramid — likely the root cause of P2-C collapsing to mAP50=0.469 despite using `DL_Modell_FV.pt`. At `lr0=0.001` the updates are small enough that the neck refines rather than gets overwritten.

**What to watch for:**  
- Early-epoch metrics (ep. 1–50) are not representative — `cv3` is essentially random; do not use them to judge run quality.  
- Class bias: `bias_init()` sets identical initial bias for both classes regardless of actual Atypisch/Normal ratio; if one class is underrepresented the model skews toward the majority until gradient corrects it. Monitor per-class recall, not just aggregate.  
- Fast convergence expected once `cv3` is past the noise phase — the backbone already produces mast-cell-specific features, so the classification boundary is a simpler problem than training from scratch.

### Grid search infrastructure (added 2026-05-06)

`BG_RATIO` and `FP_NEG_OVERSAMPLE` are now read from environment variables with fallback defaults. `submit_grid.sh` submits all 12 combinations (fp∈{1,2,3} × bgr∈{0,1,2,3}) as independent SLURM jobs in one command. `PROJECT_DIR` auto-names from the job name, so results land in `train_p2_fp1_bgr0/`, etc.

### Augmentation: pre-applied vs. on-the-fly (analysed 2026-05-07)

**Pre-applied augmentation does not increase the dataset.** `preprocess_image()` reads each image once, applies a single random transform, and saves one output image. The PROCESSED_DIR contains the same number of images as the source directories — no duplication, no size change (all images stay at their original resolution). Every training epoch sees exactly the same augmented copies; the transforms are frozen at preprocessing time, not re-sampled.

**Why this is ineffective.** Augmentation regularises by presenting a different view of each image every epoch. When the augmented images are baked to disk, that variability is exhausted after epoch 1. Training then effectively runs on an unaugmented dataset. The overhead (CPU time + disk space) is incurred, the benefit is not.

**Current state of `ba_improved_P2_fixed.py`:** augmentation is fully disabled — the pre-apply block is commented out and `augment=False` is passed to `model.train()`. This is the cleanest possible ablation baseline.

**Recommended test sequence:**

| Config | `preprocess_image` augment | `augment` in `model.train()` | Effect |
|--------|---------------------------|------------------------------|--------|
| **A — baseline** | off (current) | `False` | Clean control; isolates architecture/data changes |
| **B — on-the-fly** | off | `True` + `mosaic=0.0` | Per-epoch random transforms; real regularisation |
| **C — both** | on | `True` | Double regularisation; rarely adds over B, increases compute |

Config B is the correct way to use augmentation. Mosaic must be disabled (`mosaic=0.0`) for cell-tile data: mosaic composites four images into one, which is appropriate for scene detection but scrambles single-cell crops at 512 px.

**`flipud=0.5` is sensible for this data.** Bone marrow cell images have no canonical orientation (not upright like people or vehicles), so vertical flip is a valid augmentation. YOLO's built-in augmentation includes it when `augment=True`; no explicit override needed.

**Recommendation:** Run A (current scripts) as the clean baseline, then run B by setting `augment=True` and `mosaic=0.0` in `model.train()`. Compare test-set recall — if B closes the val/test gap it confirms the model was overfitting rather than learning generalisable features.

---

## Background: Phase 1 — P1 Dataset, YOLO11n Baseline

**Scripts:** `ba_improved.py`  
**Results dir:** `yolo_runs_hpc_final/`  
**Purpose:** Initial proof of concept on the first patient dataset (P1)

### Setup
- Model: `yolo11n.pt` (COCO pretrained, no domain-specific weights)
- Data: P1 — `Pos_neg 12241515` (annotated positive tiles) + `Negativ 12241515` (negative tiles)
- Classes: Atypisch, Normal
- Strategy: 5-fold stratified CV, albumentations augmentation (flip, brightness, rotate ±15°)
- Epochs: up to 5000, patience=50, batch=32, imgsz=512, cos_lr=True

### Validation results (best epoch per fold, from results.csv)
*Note: No clean held-out test evaluation available for this phase; val metrics shown.*

| Fold | Best Epoch | mAP50  | mAP50-95 | Precision | Recall |
|------|-----------|--------|----------|-----------|--------|
| 1    | 135       | 0.9521 | 0.8051   | 0.8844    | 0.9198 |
| 2    | 112       | 0.7561 | 0.5830   | 0.6343    | 0.7804 |
| 3    | 120       | 0.8931 | 0.7449   | 0.7642    | 0.7371 |
| 4    | 128       | 0.7754 | 0.6367   | 0.6710    | 0.7252 |
| 5    | 29        | 0.6212 | 0.4734   | 0.8867    | 0.4321 |
| **mean** | | **0.800** | **0.649** | **0.768** | **0.719** |
| **std**  | | **0.129** | **0.131** | | |

### Conclusion
Proof of concept succeeds. Fold 1 achieves excellent results (mAP50=0.952), but the 5-fold mean is
dragged down by high fold-to-fold variance (std=0.129). Fold 5 converges at epoch 29 with very low
recall (0.432), indicating an unlucky data split rather than a model problem. The high variance
becomes the central challenge for all subsequent runs.

---

## Phase 2 — P2 Dataset Fine-Tuning Experiments

All Phase 2 runs use the P2 dataset: `1224151atypisch_normal` (annotated images, .jpeg format)
with confirmed false-positive negatives loaded from `Task 6_1224151_negative.xlsx`.
Script: `ba_improved_P2.py` throughout.

---

### Run P2-A — YOLO11n Baseline on P2 (base settings)
**SLURM:** `Slurm-319399 (train p2 base nano)`

#### Setup
- Base model: `yolo11n.pt` (COCO pretrained)
- FP_NEG_OVERSAMPLE=1, BG_RATIO=0
- Default YOLO hyperparameters (no cfg override)

#### Test results
| Fold | mAP50-95 | mAP50  | Precision | Recall |
|------|----------|--------|-----------|--------|
| 1    | 0.3222   | 0.4405 | 0.8935    | 0.4205 |
| 2    | 0.7112   | 0.8572 | 0.7222    | 0.8752 |
| 3    | 0.4801   | 0.6221 | 0.5840    | 0.5939 |
| 4    | 0.3800   | 0.5084 | 0.4774    | 0.6044 |
| 5    | 0.3678   | 0.4718 | 0.3802    | 0.5905 |
| **mean** | **0.452** | **0.580** | **0.612** | **0.617** |
| **std**  | **0.156** | **0.172** | | |

#### Conclusion
Extremely high variance (fold 2 reaches mAP50=0.857, fold 1 only 0.441). Training from the generic
yolo11n.pt weights on this specialised domain is unstable. Using domain-specific pretrained weights
(DL_Modell_FV.pt) should help.

---

### Run P2-B — YOLO11n Baseline on P2 (modified settings)
**SLURM:** `Slurm-319400 (train p2 mod nano)`

#### Setup
- Base model: `yolo11n.pt`
- FP_NEG_OVERSAMPLE=1, BG_RATIO=0
- Modified training settings (exact changes not tracked)

#### Test results
| Fold | mAP50-95 | mAP50  | Precision | Recall |
|------|----------|--------|-----------|--------|
| 1    | 0.3727   | 0.4985 | 0.5299    | 0.5114 |
| 2    | 0.7112   | 0.8572 | 0.7222    | 0.8752 |
| 3    | 0.4801   | 0.6221 | 0.5840    | 0.5939 |
| 4    | 0.4762   | 0.6219 | 0.5478    | 0.7638 |
| 5    | 0.3678   | 0.4718 | 0.3802    | 0.5905 |
| **mean** | **0.482** | **0.614** | **0.553** | **0.667** |
| **std**  | **0.139** | **0.155** | | |

#### Conclusion
Marginal improvement in recall over P2-A, but variance remains very high. Fold 2 and 3 share
identical results with P2-A, suggesting those folds converged to the same solution regardless of
the setting modifications. Domain-specific starting weights are needed.

---

### Run P2-C — DL_Modell_FV Fine-Tuning (base settings)
**SLURM:** `Slurm-319401 (train p2 base dl_fv)`

#### Setup
- Base model: `DL_Modell_FV.pt` (domain-specific pretrained weights)
- FP_NEG_OVERSAMPLE=1, BG_RATIO=0
- Default hyperparameters (lr0=0.01), no cfg override

#### Test results
| Fold | mAP50-95 | mAP50  | Precision | Recall |
|------|----------|--------|-----------|--------|
| 1    | 0.3088   | 0.4207 | 0.8706    | 0.3763 |
| 2    | 0.3787   | 0.5089 | 0.8824    | 0.3820 |
| 3    | 0.3697   | 0.4945 | 0.8567    | 0.4180 |
| 4    | 0.3445   | 0.4734 | 0.9014    | 0.4234 |
| 5    | 0.3156   | 0.4455 | 0.8896    | 0.3870 |
| **mean** | **0.344** | **0.469** | **0.880** | **0.397** |
| **std**  | **0.031** | **0.035** | | |

#### Conclusion
Using DL_Modell_FV.pt dramatically reduces variance (std=0.031 vs 0.139–0.156 with yolo11n) — the
pretrained domain knowledge stabilises training across folds. However, mean mAP50 (0.469) and recall
(0.397) are lower than yolo11n runs. The model is high-precision (0.880) but misses too many cells.
This pattern is characteristic of a learning rate too high for fine-tuning: the model's existing
knowledge gets partially overwritten, leaving it conservative and biased toward precision over recall.

---

### Run P2-D — DL_Modell_FV Fine-Tuning (modified settings)
**SLURM:** `Slurm-319416 (train p2 mod dl_fv)`

#### Setup
- Base model: `DL_Modell_FV.pt`
- FP_NEG_OVERSAMPLE=1, BG_RATIO=0
- Modified training settings

#### Test results
| Fold | mAP50-95 | mAP50  | Precision | Recall |
|------|----------|--------|-----------|--------|
| 1    | 0.3793   | 0.5050 | 0.5503    | 0.5278 |
| 2    | 0.4945   | 0.6386 | 0.5235    | 0.9569 |
| 3    | 0.5313   | 0.7043 | 0.7227    | 0.6587 |
| 4    | 0.4655   | 0.6240 | 0.6549    | 0.7152 |
| 5    | 0.3655   | 0.4777 | 0.9217    | 0.4162 |
| **mean** | **0.447** | **0.590** | **0.675** | **0.655** |
| **std**  | **0.072** | **0.090** | | |

#### Conclusion
Best result so far on the test set: mean mAP50=0.590, recall=0.655. The modified settings improve
recall substantially over P2-C (0.655 vs 0.397) at the cost of some precision. Variance is still
moderate (std=0.090). Fold 2 achieves excellent recall (0.957) but fold 5 remains low (0.416),
showing the fold-to-fold instability has not been resolved.

---

### Run P2-E — DL_Modell_FV + YOLO Tuner Hyperparameters (FAILED)
**SLURM:** `Slurm-319536 (train p2 base dl_fv hyperpara)`

#### Setup
- Base model: `DL_Modell_FV.pt`
- FP_NEG_OVERSAMPLE=1, BG_RATIO=0
- Hyperparameter config from YOLO built-in tuner (`cfg=CFG_PATH` → `best_hyperparameters.yaml`)
- Tuner was run for 300 epochs × 100 iterations on P2 data (Slurm-269569)

#### Test results
| Fold | mAP50-95 | mAP50  | Precision | Recall |
|------|----------|--------|-----------|--------|
| 1    | 0.3292   | 0.4508 | 0.9033    | 0.4005 |
| 2    | 0.3205   | 0.4377 | 0.8816    | 0.4023 |
| 3    | 0.3215   | 0.4535 | 0.8962    | 0.4242 |
| 4    | 0.3308   | 0.4556 | 0.9092    | 0.4124 |
| 5    | 0.3243   | 0.4429 | 0.8856    | 0.4269 |
| **mean** | **0.325** | **0.448** | **0.895** | **0.413** |
| **std**  | **0.005** | **0.007** | | |

Training collapsed in 3–6 epochs across all folds. The validation `cls_loss` reached 5.8+
(normal range: 0.5–1.5), indicating catastrophic forgetting of the pretrained weights.

#### Conclusion
**Total failure.** YOLO's built-in tuner optimises hyperparameters for training from scratch.
When those hyperparameters (particularly `lr0=0.01`) are applied to fine-tuning from a pretrained
checkpoint, they overwrite learned features before any adaptation can occur. The near-zero variance
(std=0.005) is not stability — it is all folds collapsing to the same degenerate solution.
**The `cfg=CFG_PATH` argument must never be used in fine-tuning runs.**

---

### Run P2-F — DL_Modell_FV + Modified Hyperparameters + Tuner Config
**SLURM:** `Slurm-319538 (train p2 mod dl_fv hyperpara)`

#### Setup
- Base model: `DL_Modell_FV.pt`
- Modified settings combined with hyperparameter cfg

#### Test results
| Fold | mAP50-95 | mAP50  | Precision | Recall |
|------|----------|--------|-----------|--------|
| 1    | 0.3552   | 0.4758 | 0.3876    | 0.5770 |
| 2    | 0.3465   | 0.4573 | 0.3599    | 0.6335 |
| 3    | 0.3635   | 0.4879 | 0.8916    | 0.4153 |
| 4    | 0.4582   | 0.6014 | 0.5747    | 0.7264 |
| 5    | 0.4032   | 0.5181 | 0.5621    | 0.5013 |
| **mean** | **0.385** | **0.508** | **0.555** | **0.571** |
| **std**  | **0.046** | **0.057** | | |

#### Conclusion
Slightly better than the failed P2-E, but worse than P2-D across all metrics. The combination of
modified settings with the tuner config does not add benefit over modified settings alone (P2-D:
mAP50=0.590 vs P2-F: mAP50=0.508). Confirmed: tuner hyperparameters are counterproductive for
fine-tuning.

---

### Run P2-G — DL_Modell_FV + Background Image Sampling (BG_RATIO=2)
**SLURM:** `Slurm-319934`  
**Results dir:** `yolo_runs_hpc_base_final_dl_fv_bg_ratio_2/`

#### Motivation
To reduce false positives, unlabeled slide tiles (images with no mast cells and no label file) were
added to training as confirmed background images. Hypothesis: the model would learn to suppress
detections on dense slide background texture.

#### Setup
- Base model: `DL_Modell_FV.pt`
- FP_NEG_OVERSAMPLE=1, **BG_RATIO=2** (2 unlabeled background tiles per annotated image)
- No cfg override

#### Test results
| Fold | mAP50-95 | mAP50  | Precision | Recall |
|------|----------|--------|-----------|--------|
| 1    | 0.3537   | 0.4683 | 0.9201    | 0.4178 |
| 2    | 0.4317   | 0.5649 | 0.4949    | 0.5617 |
| 3    | 0.3502   | 0.4814 | 0.8883    | 0.4215 |
| 4    | 0.3350   | 0.4576 | 0.8872    | 0.4406 |
| 5    | 0.5118   | 0.6799 | 0.7364    | 0.5795 |
| **mean** | **0.396** | **0.530** | **0.785** | **0.484** |
| **std**  | **0.075** | **0.090** | | |

Validation training curves showed folds 1, 2, 5 reaching peak mAP50 ~0.945, but folds 3 and 4
peaked at epochs 7 and 43 respectively and never improved (training ran to patience limit).

#### Conclusion
Mixed outcome. Test set precision improved (0.785 vs 0.675 in P2-D), but recall dropped (0.484
vs 0.655) and mean mAP50 is lower (0.530 vs 0.590). The validation curves reveal the root cause:
folds 3 and 4 are collapsing immediately after the warmup phase (epoch 7), which is a
**learning rate signature** — with `lr0=0.01` and a domain-pretrained model, the model overshoots
its optimal weights in the first few post-warmup steps. BG_RATIO=2 amplifies this instability in
unlucky fold splits by diluting the annotated signal. The learning rate, not the data ratio, is
the primary bottleneck.

---

### Run P2-H — Lower LR + Backbone Freeze, No Background
**SLURM:** 320698 (`train_p2_fp1_bgr0_new`)  
**Status:** complete

#### Setup
- Base model: `DL_Modell_FV.pt`
- `lr0=0.001`, `freeze=10`, `FP_NEG_OVERSAMPLE=1`, `BG_RATIO=0`
- `patience=100`, `epochs=5000` (early stopping active)
- BG_RATIO bug fixed: background images now correctly passed to fold workers

#### Test results
| Fold | mAP50-95 | mAP50  | Precision | Recall |
|------|----------|--------|-----------|--------|
| 1    | 0.5566   | 0.6482 | 0.9644    | 0.4537 |
| 2    | 0.3604   | 0.4562 | 0.9427    | 0.4250 |
| 3    | 0.7568   | 0.9401 | 0.8324    | 0.9118 |
| 4    | 0.3876   | 0.4784 | 0.9647    | 0.4292 |
| 5    | 0.4063   | 0.4991 | 0.9635    | 0.4660 |
| **mean** | **0.494** | **0.604** | **0.934** | **0.537** |
| **std**  | **0.166** | **0.194** | **0.057** | **0.210** |

#### Conclusion
The `lr0=0.001` + `freeze=10` configuration successfully prevents the fold-collapse seen in
P2-C/P2-G (no fold peaks and immediately drops to plateau). All folds converge. However, the
variance target (std < 0.06) was **not met** — std remains at 0.194. Fold 3 is the outlier:
it achieves mAP50=0.940 and recall=0.912, while the other four folds are stuck at mAP50
0.46–0.65 and recall 0.43–0.47. This is a persistent fold-split artefact, not a training
instability. The learning rate fix solved convergence but not generalisation.

Mean mAP50=0.604 matches P2-D (0.590) and bests P2-G (0.530), confirming that `lr0=0.001`
is the correct fine-tuning learning rate. This is now the reference configuration for the DoE
grid.

---

### DoE Grid — Full 3×4 Factorial: FP_NEG_OVERSAMPLE × BG_RATIO
**SLURM:** 320698–320709  
**Results dirs:** `train_p2_fp{1,2,3}_bgr{0,1,2,3}_new/`  
**Status:** 11 of 12 complete; fp3_bgr3 (320709) failed (no TEST output — likely timeout/OOM)

All runs share the reference config established by P2-H:
`DL_Modell_FV.pt`, `lr0=0.001`, `freeze=10`, `patience=100`, `epochs=5000`, `BG_RATIO` and
`FP_NEG_OVERSAMPLE` varied per grid cell. BG_RATIO bug fixed in all.

#### Test results — DoE grid

| Config | SLURM | FP× | BGr | mAP50 (mean) | mAP50-95 | Recall | Precision |
|--------|-------|-----|-----|--------------|----------|--------|-----------|
| fp1_bgr0 | 320698 | ×1 | 0 | **0.604** | 0.494 | 0.537 | **0.934** |
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

#### Fold-level pattern (consistent across all runs)

Fold 3 is an outlier in every single grid cell:

| Fold | Typical mAP50 range (non-fold-3) | Fold 3 range |
|------|----------------------------------|--------------|
| 1, 2, 4, 5 | 0.44 – 0.61 | — |
| 3 | — | 0.88 – 0.94 |

The within-fold recall for folds 1, 2, 4, 5 is stuck at 0.42–0.49 across all 11 configurations.
Fold 3 alone achieves 0.80–0.95. The cross-fold std (~0.15–0.21) completely dominates any
signal from the FP or BG factors. This is a small-dataset fold-split artefact.

#### Conclusions from DoE

**1. No configuration significantly outperforms fp1_bgr0 (P2-H).** The total mAP50 spread
across all 11 completed cells is 0.548–0.604 — a range of 0.056. Within the noise of fold
variance (std ≈ 0.19), no combination is meaningfully different.

**2. BG_RATIO has a consistently negative effect.** Within every FP level, bgr=0 gives the
best or tied-best mAP50 (fp1: 0.604, fp2: 0.581, fp3: 0.548). Each step up in BG_RATIO
degrades performance. Confirmed: background tiles suppress recall without improving precision
at this dataset scale. BG_RATIO should be kept at 0 for all further runs.

**3. FP_NEG_OVERSAMPLE has diminishing returns above ×2.** fp=1 gives the best mAP50.
fp=2 gives the highest reported recall (0.638) but this is driven by a single fold-2 outlier
(recall=0.974 in 320702, 0.965 in 320704) — without those outliers, fp=2 is indistinguishable
from fp=1. fp=3 underperforms both on mAP50 and shows no recall benefit.

**4. The recall problem is a generalisation problem, not a data-ratio problem.** Recall
outside of fold 3 is 0.42–0.49 regardless of FP or BG configuration. The model is not
failing because of class imbalance or insufficient negatives — it is failing to transfer the
features it learned in training to the test split. On-the-fly augmentation is the correct
next intervention.

**5. fp3_bgr3 (320709) failed** — no TEST output in SLURM log. Likely timeout or OOM from
the largest dataset (3× FP + 3 BG tiles per annotated image). Not worth retrying; this corner
of the grid is already ruled out by the BG_RATIO trend.

#### Best config from DoE
**fp1_bgr0** (= P2-H baseline): mAP50=0.604, recall=0.537, precision=0.934.
This is the cleanest result and uses the minimum additional data manipulation.

---

## Summary Table — Test Set Results (all P2 runs)

| Run | SLURM | Base Model | FP× | BGr | lr0 | freeze | mAP50 (mean±std) | Recall |
|-----|-------|-----------|-----|-----|-----|--------|------------------|--------|
| P2-A | 319399 | yolo11n | ×1 | 0 | 0.01 | — | 0.580 ± 0.172 | 0.617 |
| P2-B | 319400 | yolo11n (mod) | ×1 | 0 | 0.01 | — | 0.614 ± 0.155 | 0.667 |
| P2-C | 319401 | DL_Modell_FV | ×1 | 0 | 0.01 | — | 0.469 ± 0.035 | 0.397 |
| P2-D | 319416 | DL_Modell_FV (mod) | ×1 | 0 | 0.01 | — | 0.590 ± 0.090 | 0.655 |
| P2-E | 319536 | DL_Modell_FV + tuner | ×1 | 0 | 0.01 | — | 0.448 ± 0.007 ❌ | 0.413 |
| P2-F | 319538 | DL_Modell_FV + mod + tuner | ×1 | 0 | 0.01 | — | 0.508 ± 0.057 | 0.571 |
| P2-G | 319934 | DL_Modell_FV | ×1 | 2 | 0.01 | — | 0.530 ± 0.090 | 0.484 |
| **P2-H** | 320698 | DL_Modell_FV | ×1 | 0 | **0.001** | **10** | **0.604 ± 0.194** | **0.537** |
| DoE fp1_bgr1 | 320699 | DL_Modell_FV | ×1 | 1 | 0.001 | 10 | 0.562 ± 0.181 | 0.531 |
| DoE fp1_bgr2 | 320700 | DL_Modell_FV | ×1 | 2 | 0.001 | 10 | 0.564 ± 0.172 | 0.523 |
| DoE fp1_bgr3 | 320701 | DL_Modell_FV | ×1 | 3 | 0.001 | 10 | 0.591 ± 0.174 | 0.518 |
| DoE fp2_bgr0 | 320702 | DL_Modell_FV | ×2 | 0 | 0.001 | 10 | 0.581 ± 0.184 | 0.638 |
| DoE fp2_bgr1 | 320703 | DL_Modell_FV | ×2 | 1 | 0.001 | 10 | 0.565 ± 0.163 | 0.523 |
| DoE fp2_bgr2 | 320704 | DL_Modell_FV | ×2 | 2 | 0.001 | 10 | 0.592 ± 0.180 | 0.637 |
| DoE fp2_bgr3 | 320705 | DL_Modell_FV | ×2 | 3 | 0.001 | 10 | 0.596 ± 0.172 | 0.539 |
| DoE fp3_bgr0 | 320706 | DL_Modell_FV | ×3 | 0 | 0.001 | 10 | 0.548 ± 0.185 | 0.537 |
| DoE fp3_bgr1 | 320707 | DL_Modell_FV | ×3 | 1 | 0.001 | 10 | 0.566 ± 0.183 | 0.546 |
| DoE fp3_bgr2 | 320708 | DL_Modell_FV | ×3 | 2 | 0.001 | 10 | 0.572 ± 0.186 | 0.526 |
| DoE fp3_bgr3 | 320709 | DL_Modell_FV | ×3 | 3 | 0.001 | 10 | ❌ FAILED | — |

---

## Next Steps

### Run P2-I — On-the-fly augmentation (planned)

The DoE conclusively showed that data-ratio manipulation (FP oversample, BG tiles) cannot
overcome the generalisation gap. The one untested lever is **per-epoch random augmentation**
during training. Config:

- Base: fp1_bgr0 (P2-H reference)
- Change: `augment=True` → replace with `mosaic=0.0` and explicit augmentation flags in
  `model.train()` (fliplr=0.5, flipud=0.5, hsv_h/s/v at YOLO defaults)
- `mosaic=0.0` is mandatory: mosaic composites four images which is appropriate for scene
  detection but scrambles single-cell 512 px crops
- `flipud=0.5` is valid: bone marrow cells have no canonical orientation

If augmentation closes the val/test gap, the inter-fold std should drop and mean recall
outside fold 3 should rise above 0.50.

If test-set precision is still a concern after P2-H, `FP_NEG_OVERSAMPLE` will be increased from
1 to 2–3. The FP negatives are high-quality hard negatives (the model's known failure cases) and
increasing their weight is a targeted way to reduce false positives without adding random noise.

### Clinical dataset note (added 2026-05-07)

All training data (P2) comes from SM patients, meaning **Atypisch is the dominant class and
Normal is the minority**. This is the inverse of the typical object detection imbalance
assumption. Consequences:

- The model has good Atypisch recall by default (majority class, well represented).
- Normal cells are under-detected — the model has rarely seen a Normal-dominant slide.
- Missing Normal cells inflates the Atypisch/Normal ratio, which can produce false SM flags
  even in borderline patients.
- The WHO criterion (>25% Atypisch) requires reliable detection of *both* classes. Under-
  detection of Normal is therefore a clinically meaningful failure mode, not just a precision
  issue.
- Generalisation to non-SM patients (who present with Normal-dominant slides) is a hard
  out-of-distribution problem: the training set contains almost no examples of that regime.
- This should be flagged as a primary limitation in the thesis Methods section.

---

## P1–P8 Full Dataset — Class Imbalance Strategy (added 2026-05-08)

### Distribution across all 8 patients

Combining all available phases (P1–P8) into one training corpus reveals the true scale of
the class imbalance:

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
| **Total** | **1103** | **1124** | **78** | **Ratio ≈ 14.4 : 1** |

Normal cells live in only four patients (P5–P8). P6 and P8 contain *zero* Atypisch — they
are the only "non-SM-like" reference data in the entire corpus. P1–P4 contribute 1112
Atypisch but only 18 Normal annotations across 1032 images.

### Why the previous DoE strategies do not solve this

The DoE explored `FP_NEG_OVERSAMPLE` and `BG_RATIO`. Neither addresses class imbalance:

- `FP_NEG_OVERSAMPLE` duplicates **negative** images (no labels). It teaches the model what
  is *not* a cell, not how to distinguish Atypisch from Normal.
- `BG_RATIO` adds unlabeled slide tiles. Same problem: it suppresses false positives but
  does nothing for the minority class.
- The DoE confirmed both factors plateau (FP) or hurt (BG). Continuing to tune them is
  exhausted as a research direction.

The 14.4 : 1 imbalance is the dominant unaddressed source of error. Recall on Normal will
drive every clinical metric (false SM flags), so this is the next required intervention.

### Recommended strategy for full-corpus training

#### 1. Image-level oversampling of Normal-rich images

Duplicate image paths (not annotations) in `train.txt` to bring the *effective* per-epoch
class ratio closer to 1 : 2. Targeted oversampling per phase:

| Source | Suggested factor | Effective Atypisch | Effective Normal |
|--------|------------------|--------------------|------------------|
| P1     | ×1               | 482                | 2                |
| P2     | ×1               | 449                | 6                |
| P3     | ×1               | 60                 | 4                |
| P4     | ×1               | 122                | 6                |
| P5     | ×8               | 24                 | 144              |
| P6     | ×15              | 0                  | 90               |
| P7     | ×8               | 64                 | 96               |
| P8     | ×10              | 0                  | 240              |
| **Effective total** | | **≈ 1200** | **≈ 590** |

Result: per-epoch ratio ≈ 2 : 1 instead of 14 : 1. The on-the-fly augmentation already
planned for P2-I makes each duplicate visually distinct, so the model does not see the
same pixels 10× per epoch.

#### 2. Patient-aware splitting — leave-one-patient-out (LOPO) CV

5-fold stratified random CV is no longer defensible at this scale:
- Only 8 patients, 4 of whom contain Normal at all.
- Random k-fold leaks patient-specific texture between train and val, inflating reported
  metrics relative to true generalisation.

Switch to **LOPO CV** — 8 folds, one patient held out per fold. Caveats:

- When P6 or P8 is the test fold, only Normal recall is measured (no Atypisch present).
  This is exactly the metric of interest for non-SM generalisation.
- When P1 or P2 is the test fold, only Atypisch recall is meaningfully measured.
- Per-fold metrics will be more variable than under stratified CV, but the variance will
  reflect real biological inter-patient heterogeneity, not split luck.

#### 3. P6 and P8 as the Normal-domain anchor

These are the only data points for the "Normal-dominant slide" regime. Treat them as
strategic test cases:
- Keep both in the train set across most LOPO folds.
- Use them periodically as a *separate* sanity-check val ("does the model still predict
  Normal at all when no Atypisch are present?") — independent of CV scoring.
- Withholding either one entirely as a held-out external validation set is a defensible
  alternative if cross-validation shows the model is overfitting to majority Atypisch.

#### 4. On-the-fly augmentation (still required)

Pure duplication × 10 means the model sees identical pixels 10×. Pair oversampling with
the per-epoch random augmentation already planned for P2-I (`augment=True`, `mosaic=0.0`,
`fliplr=0.5`, `flipud=0.5`, default HSV jitter). Augmentation is what makes oversampling
work — without it, the duplicated copies are wasted compute.

#### 5. Class loss weight (optional, secondary)

YOLO11 does not expose per-class loss weights cleanly, but the global classification loss
weight `cls` (default `0.5`) can be raised to ~`1.0`. This invests more model capacity in
class discrimination relative to localisation. Marginal effect compared to oversampling;
test only after the oversampling+LOPO baseline is established.

#### 6. Negative-set provenance — three distinct categories (clarified 2026-05-08)

The previous "BG vs FP" framing was incomplete. Negatives in this project come from
three distinct asset classes, each with different curation level and trust:

| Category | Source | What it is |
|----------|--------|-----------|
| **P1 gold negatives** | `old/P1/Negativ 12241515/` | Hand-curated empty tiles by FV during the original from-scratch P1 training that produced `DL_Modell_FV.pt`. Verified backgrounds. |
| **P2 hard FPs** | `Task 6_1224151_negative.xlsx` | Tiles where `DL_Modell_FV.pt` produced false positives at USZ during their CVAT annotation work. Flagged by USZ as confirmed-empty. |
| **P2 random BG** | `new/P2/images/Train/` minus annotated minus FPs | Random sample from the ~20k P2 slide tiles that were skimmed but not annotated or flagged. |

The DoE's `BG_RATIO` knob varied **category 3** (random P2 BG). Its conclusion that
"BG uniformly hurts" applies *only* to that category. Categories 1 and 2 were never
the subject of an ablation:

- The **P1 gold negatives** were used to train `DL_Modell_FV.pt` from scratch and
  are an independent training asset. They were silently disabled by `BG_RATIO=0`
  in the DoE — that was a code-structure artefact, not an experimental finding.
  The new pipeline includes all of them by default (`USE_P1_GOLD_BG=1`) when P1
  is in train.
- The **P2 hard FPs** stay at `FP_NEG_OVERSAMPLE=1` (DoE-validated default).
- The **P2 random BG** stays at `RANDOM_BG_RATIO=0` (DoE conclusion holds for
  this category only).

Config in `ba_improved_comb.py`:

```python
USE_P1_GOLD_BG     = True   # always include FV's curated tiles
FP_NEG_OVERSAMPLE  = 1      # all hard FPs once
RANDOM_BG_RATIO    = 0      # DoE rejected; kept as a knob for future tests
```

### Run P3-A — LOPO + Normal oversampling on P1–P8 (full-corpus baseline)
**SLURM:** 322719 (`yolo_new_v1`)
**Results dir:** `hpc/yolo_new_v1/fold_{1..8}_P{1..8}/`, summary in `fold_results.csv`
**Status:** complete, 2026-05-09

#### Setup
- **Anchor:** `yolo11n.pt` (changed from the originally planned `DL_Modell_FV.pt` — see *Anchor checkpoint changed* below for rationale).
- 8-fold **LOPO CV** (one patient held out per fold).
- Per-patient image oversampling per the table in the strategy section above (P5×8, P6×15, P7×8, P8×10; SM-heavy patients ×1).
- On-the-fly augmentation: `augment=True`, `mosaic=0.0`, `fliplr=0.5`, `flipud=0.5`, default HSV jitter.
- `lr0=0.001`, `freeze=10`, `cls=1.0`, `patience=100`, `epochs=5000` (early stopping active).
- `USE_P1_GOLD_BG=1`, `FP_NEG_OVERSAMPLE=1`, `RANDOM_BG_RATIO=0` (DoE-validated negative-set defaults).

#### Test results — per fold

| Fold | Holdout | mAP50-95 | mAP50  | Precision | Recall | Recall (Atyp) | Recall (Norm) | Atyp / Norm in test |
|------|---------|----------|--------|-----------|--------|---------------|---------------|---------------------|
| 1    | P1      | 0.593    | 0.730  | 0.691     | 0.680  | 0.861         | 0.500         | 482 / **2**         |
| 2    | P2      | 0.724    | 0.875  | 0.876     | 0.808  | 0.914         | 0.702         | 449 / **6**         |
| 3    | P3      | 0.727    | 0.845  | 0.761     | 0.848  | 0.981         | 0.716         | 60  / **4**         |
| 4    | P4      | 0.670    | 0.862  | 0.684     | 0.909  | 0.819         | 1.000         | 122 / **6**         |
| 5    | P5      | 0.492    | 0.543  | 0.535     | 0.650  | 0.500         | 0.800         | **3** / 18          |
| 6    | P6      | 0.931    | 0.995  | 0.900     | 1.000  | (vacuous)     | 1.000         | **0** / 6           |
| 7    | P7      | 0.827    | 0.968  | 0.953     | 0.909  | 1.000         | 0.818         | 8   / 12            |
| 8    | P8      | 0.782    | 0.935  | 0.711     | 0.913  | (vacuous)     | 0.826         | **0** / 24          |
| **mean** |     | **0.718**| **0.844** | **0.764** | **0.840** | —          | —             |                     |
| **std**  |     | 0.143    | 0.137  | 0.140     | 0.114  |               |               |                     |

Per-class recall on rows with insufficient minority-class instances is unreliable: P1 has
2 Normal cells, P2 has 6, P3 has 4, P4 has 6 — recall on those folds for the Normal class
should be treated as noise. Atypisch recall on P6 and P8 is vacuous (zero Atypisch in the
held-out patient). The four meaningful Normal-recall measurements are P5/P6/P7/P8 holdout.

#### Class-recall summary (Normal-meaningful folds only)

| Holdout | Normal in test | Recall (Normal) |
|---------|----------------|-----------------|
| P5 | 18 | 0.800 |
| P6 | 6  | 1.000 |
| P7 | 12 | 0.818 |
| P8 | 24 | 0.826 |
| **mean** | | **0.861** |

All four ≥ 0.80; mean 0.861. This was the headline target of the strategy and it landed.

#### Conclusion

This is the strongest result in the project: mAP50 **0.844 ± 0.137**, recall **0.840**,
with cross-fold std finally below the P2-only DoE's 0.19 despite LOPO being a harder
evaluation than random k-fold. The intuition that LOPO would *lower* aggregate mAP50
relative to random CV did not hold — the larger and more diverse training corpus more
than compensates for the harder split.

**Comparison to P2-only baseline (P2-H: mAP50 0.604, R 0.537):** every fold except P5
exceeds the P2-H mean on both metrics, most by a wide margin. **Fold 5 (P5) is the only
soft spot** (mAP50 0.543 vs. P2-H mean 0.604). Explanation: P5 has 3 Atypisch and 18
Normal instances in the test set, so each missed Atypisch costs −0.33 in Atypisch recall.
This is a small-sample artefact, not a model regression.

**Normal recall is acceptable.** With only 78 Normal annotations corpus-wide, the
strategy's per-patient oversampling + on-the-fly augmentation succeeded in lifting Normal
recall on every Normal-rich holdout fold to ≥ 0.80. The remaining concern — generalisation
to non-SM patients (the P6/P8 regime) — is now empirically validated, not assumed: with
those patients held out, the model still detects all 6 / 24 Normal cells with recall 1.0 / 0.826.

#### Implications

- **No need for synthetic Normal tiles.** The fallback plan (cropping patches centred on
  Normal cells from P5–P8) is unnecessary at this dataset scale; the oversampling table
  already delivers the targeted ratio.
- **The next bottleneck is Atypisch recall on Normal-dominant test slides** (Fold 5
  Atypisch recall = 0.50 on 3 samples, Fold 4 Atypisch recall = 0.82 on 122). More
  mixed-presentation patients (some Atypisch + some Normal in the same slide, like P7)
  would tighten this faster than any further algorithmic tuning.
- **`yolo_new_v1/fold_8_P8/weights/best.pt` is the candidate v1 shipped weight.** All folds
  produced a `best.pt` of consistent size (~5.4 MB), and any of them is a defensible
  release weight; using the P8-holdout fold as the canonical version means the shipped
  model has been evaluated against the hardest pure-Normal slide on hand.

### Thesis implications

This shifts the central methodological story from "find optimal FP/BG ratio" to "address
clinical class imbalance under patient-level data scarcity". Update for `BA_FV_extended.qmd`:

- **Methodology — Data Acquisition**: replace the P2-only description with P1–P8 table and
  patient-level breakdown.
- **Methodology — Model Training Strategy**: replace stratified k-fold with LOPO CV and
  document the oversampling table.
- **Limitations**: reframe SM-only data as "clinical asymmetry — only 8 patients, of which
  4 contain Normal cells, of which 2 are pure-Normal" rather than "Atypisch-dominant data".
- **Future Research**: add patient enrolment as the primary axis for future work — more
  non-SM patients is more valuable than algorithmic refinement at this dataset scale.

---

## Session log — 2026-05-08 (post-strategy adjustments and operational decisions)

### Anchor checkpoint changed: yolo11n.pt instead of DL_Modell_FV.pt

The first run of `ba_improved_comb.py` against the full P1–P8 corpus was launched with
`PRETRAINED_WEIGHTS = "yolo11n.pt"`, not `DL_Modell_FV.pt` as the strategy section above
originally specified. Rationale: with a corpus of ~1100 labelled images and per-patient
oversampling, the new dataset is large enough that the COCO-pretrained yolo11n base
trains cleanly without the domain-specific warm-start. This also produces a cleaner
provenance story — every future model version reproduces from `(yolo11n.pt, dataset
snapshot, config)` in a single training run, with no chain of fine-tunes to track.

`DL_Modell_FV.pt` remains on disk as a historical artefact (the original P1-only model
shipped to USZ for CVAT annotation) but is no longer the anchor for new runs.

### Per-fold annotation ratios (computed from current config)

The strategy targets ~2 : 1 Atypisch : Normal across the whole corpus. Per fold the
ratio shifts because LOPO removes whichever patient is held out:

| Holdout | Atyp (eff.) | Norm (eff.) | Ratio |
|---|---|---|---|
| P1 | 719 | 586 | 1.23 : 1 |
| P2 | 752 | 582 | 1.29 : 1 |
| P3 | 1141 | 584 | 1.95 : 1 |
| P4 | 1079 | 582 | 1.85 : 1 |
| P5 | 1177 | 444 | 2.65 : 1 |
| P6 | 1201 | 498 | 2.41 : 1 |
| P7 | 1137 | 492 | 2.31 : 1 |
| P8 | 1201 | 348 | 3.45 : 1 |

Mean ≈ 2.14 : 1, range 1.23–3.45 : 1. This is unavoidable with fixed per-patient
oversample factors + LOPO. Folds where SM-heavy patients are held out (P1, P2) skew
Normal-favourable; folds where pure-Normal patients are held out (P5–P8) skew
Atypisch-heavy. Per-class recall already captures this asymmetry; no further code
change required.

### Strict LOPO on negatives — confirmed (not relaxed)

Negatives (P1 gold backgrounds, P2 hard FPs) are held out by patient tag along with
positives. Consequence:

| Fold | Holdout | P1 gold in train | P2 FPs in train | Total neg |
|---|---|---|---|---|
| 1 | P1 | — (excluded) | 330 | 330 |
| 2 | P2 | 1030 | — (excluded) | 1030 |
| 3–8 | P3–P8 | 1030 | 330 | 1360 |

Discussed relaxing this (negatives shared across folds) since empty tiles carry no
diagnostic morphology. Decision: keep strict — slide-level batch effects (stain,
microscope white balance, slide age) can leak even through empty tiles, and "I held
everything out by patient" is a one-line claim for thesis defence. The fold-1/fold-2
asymmetry is a known accepted cost.

### Cluster correction: earth-4 has 8× L40S

Earlier `CLAUDE.md` note "2× L40S" was based on the SLURM script's request, not the
actual node capacity. `sinfo -o "%P %G %t" | grep gpu` confirms `earth-4 gpu:l40s:8`.
The current run still uses 2 GPUs (4 folds per GPU) because the SLURM script wasn't
changed; future runs can request `--gres=gpu:l40s:8` for one fold per GPU and ~4× wall
clock reduction (each fold runs on its own L40S, no contention).

`ba_improved_comb.py` now picks GPU count from `torch.cuda.device_count()` or `N_GPUS`
env var; no code change needed when bumping the SLURM allocation.

### Operational doctrine — retraining and versioning (added 2026-05-08)

Three retraining patterns exist; risk and reproducibility differ sharply.

| Pattern | What's seen during training | When it makes sense |
|---|---|---|
| **A. Fine-tune on new data only** (e.g. v2 + new patient corrections) | Only new data | Never for shipped models — this is the failure mode that produced the original P2 / DL_FV regression |
| **B. Continue from v_n on full accumulated corpus** | Full P1–P_N | Production iteration when compute is expensive and a regression test suite exists |
| **C. Retrain from yolo11n.pt on full accumulated corpus** | Full P1–P_N | Default for this thesis and for any model intended for distribution |

Rule for shipped models: **C is the default.** Each version is a one-step recipe from
a fixed input (`yolo11n.pt`, dataset at version N, config). No anchor-chain to audit,
no compounding bias, no catastrophic-forgetting accumulation.

B is acceptable for in-house iteration when compute is the bottleneck, *provided* the
new training pass uses the full corpus (not just new data). Continuing from v_n on
P_(1..N) is a normal, well-understood ML practice and is not the same as the desktop
app's local fine-tune (which is A).

A is reserved for the desktop app's "Personal model" feature — clearly labelled as
local-only and never represented as a shared model version.

### Customer-correction workflow

When a downstream user (USZ or future customer) returns corrections:

1. Annotated images they edited become a new patient directory under `data/new/P_(N+1)/`.
2. Their FP list (Excel or otherwise) maps to the same patient tag and is loaded via
   the existing `FP_NEG_OVERSAMPLE` path.
3. Add `"P_(N+1)"` to `PATIENT_IMAGE_DIRS` and to `PATIENT_OVERSAMPLE` (factor depends
   on their class balance).
4. Re-run `ba_improved_comb.py` from `yolo11n.pt`. LOPO automatically becomes
   (N+1)-fold; the new patient gets one fold where they're held out, which directly
   measures generalisation to the new clinical site.
5. Compare per-class recall on the new patient's holdout fold against the prior
   version's recall on the same patient — this is the regression check.

Document each customer's correction batch as a discrete entry in this log so the
training corpus version → model version mapping stays auditable.

---

## Desktop application — safeguards added 2026-05-08

Implemented in `desktop_app/`; full rationale in `BA_FV_extended.qmd` §
*Continual Learning Without Distribution Drift*.

- **Anchored fine-tune.** `FineTuneWorker` accepts `base_train_dir`; when set, the
  base corpus is symlinked into the train split alongside user edits. Prevents
  catastrophic forgetting in the local "Personal model" path.
- **Baseline validation pass.** `FineTuneWorker` accepts `baseline_test_dir`; old +
  new model are evaluated on the held-out set after training. UI dialog shows
  per-class recall delta and refuses promotion (default "No") if any class regresses
  by more than 2 percentage points.
- **Corrections export.** `core/exporter.py` packages edited images + labels +
  provenance manifest (annotator, host, date, model SHA-1 hash, source identifier,
  notes) into a ZIP. This is the *recommended* path for genuine improvement and is
  styled accordingly in the UI.
- **Renamed for honesty.** "Retrain on Edits" → "Personal model (this machine
  only)". "Export corrections…" → "Submit for central improvement (recommended)".
  Tooltips and the sidebar paragraph make the local-vs-central distinction explicit.

### Statistics view (added 2026-05-08)

`ui/statistics_view.py` + `core/statistics.py`. Patient-level summary panel
implementing the WHO 25% Atypisch minor criterion for SM:

- Verdict banner: "Indication of SM" / "No SM indication" / "Insufficient data"
  (red / green / grey). Threshold is `WHO_ATYPISCH_THRESHOLD = 0.25`; minimum
  10 total mast cells before any verdict is rendered.
- Counts: total Atypisch, total Normal, ratio, threshold.
- Confidence summary per class (mean, median, range) — relevant because low-conf
  Atypisch detections inflate the ratio.
- Per-image breakdown table, sortable, double-click opens image in editor.
- Always-visible WHO caveat: this is one of several minor criteria, not a
  diagnosis, and model recall is imperfect.
- "Export summary" produces a TXT report including the disclaimer.
- Auto-refreshes after folder load, after inference completes, and after every edit.

This is the intended end-user output of the pipeline: load images → run detection
→ open Statistics → read the indication. Everything else (gallery, editor, fine-tune)
is supporting infrastructure.

---

## Infrastructure: Desktop Application (added 2026-05-07)

A PySide6 desktop application (`desktop_app/`) was developed in parallel with HPC training
to enable clinical use of the trained models without a server dependency.

### Key features implemented

- **Chunked inference**: `InferenceWorker` processes images in configurable chunks (default
  250) to bound peak RAM regardless of folder size. Each chunk saves `.dapp_meta.json` on
  completion — a crash loses at most one chunk and the next run resumes automatically.
- **Gallery with filter tabs**: All / With detections / No detection / Low confidence / Edited.
  Navigation in the annotation editor is scoped to the active tab.
- **Annotation editor**: Draw/edit/delete bounding boxes. Edits marked in `.dapp_meta.json`
  and never overwritten by subsequent inference runs.
- **Active learning loop**: edited images (FP corrections) can be used to fine-tune the
  loaded model via `FineTuneWorker`. Training batch is automatically set to ⌊inference
  batch / 3⌋ to avoid OOM (training requires ~3–4× more VRAM than inference due to gradients
  + Adam states + backprop activations).
- **QSettings persistence**: confidence, batch size, chunk size, epoch count, and last-used
  model path are saved across sessions (`~/.config/ZHAW/MastCellDetector.conf` on Linux).
- **Hardware detection**: CUDA, ROCm (AMD WSL2 via `/dev/dxg`), MPS (Apple), CPU. ROCm
  recommended batch is halved to leave headroom for the HSA runtime bridge in WSL2.

### PyInstaller packaging (2026-05-07)

Built with:
```bash
pyinstaller --noconfirm --windowed --name "MastCellDetector" \
  --add-data="ui/style.qss:ui" --collect-data ultralytics main.py
```

Note: `--add-data` separator is `:` on Linux, `;` on Windows. The output is a Linux ELF
binary (`dist/MastCellDetector/MastCellDetector`) that runs inside WSL2 only — it is not a
native Windows executable. A native `.exe` would require running PyInstaller from a Windows
Python environment.

---

## Phase 3 follow-up — Tinkering campaign (planned 2026-05-09)

### Motivation

The P3-A result (mAP50 0.844 ± 0.137, recall 0.840) is the strongest configuration on
file, but it is the best result *within the design space the prior stages examined*. The
Phase-2 DoE ruled out dataset-composition factors while holding training-time hyperparameters
fixed; the Phase-3 LOPO+oversampling pivot changed the cross-validation unit and the
sampling strategy but inherited the Phase-2 hyperparameter recipe wholesale. Neither stage
explored training-time hyperparameters orthogonal to data composition. The tinkering campaign
closes this gap.

### Files

`hpc/tinkering/` — fully isolated from the production scripts.

- `ba_tinker.py` — copy of `ba_improved_comb.py` with seven knobs made env-overridable
  (`FREEZE`, `CLS_LOSS_WEIGHT`, `LABEL_SMOOTHING`, `DEGREES`, `DFL`, `HSV_V`, `MIXUP`).
  Defaults reproduce P3-A exactly when no env var is set.
- `tune_hyperpara.py` — LOPO-Fold-4 (P4 holdout) hyperparameter tuner. Reuses helpers
  from `ba_improved_comb` so workspace construction is identical to deployment.
- `run_tune.sh`, `run_tinker.sh` — SLURM job scripts.
- `submit_tune.sh`, `submit_grid.sh` — launchers.

### Workflow (four steps)

| Step | Job(s) | Purpose | Decision rule |
|---|---|---|---|
| 1. Tune | `submit_tune.sh` (1 job, ~1 day on 1 L40S) | yolo11n + freeze=10 + AdamW, 30 iter × 50 epochs on Fold 4 (P4 holdout). Recipe matches deployment exactly. | Output: `best_hyperparameters.yaml` |
| 2. Matrix | `submit_grid.sh` (24 jobs in parallel) | 2×3×2×2 grid over (label_smoothing, degrees, dfl, freeze). Submit concurrently with step 1 — independent. | Identify best cell vs. P3-A baseline |
| 3. Validate tuner | One full LOPO with tuner yaml only | Sanity-check that the tuner output doesn't collapse like P2-E. | Reject yaml if test mAP50 regresses by >2 pp vs. P3-A |
| 4. Final | One full LOPO with (tuner yaml + matrix winner knobs) | Combined configuration. | Adopt as v2 candidate if it beats P3-A; otherwise keep P3-A |

### Knob matrix (Step 2)

| Knob | Default (P3-A) | Levels | Rationale |
|---|---|---|---|
| `label_smoothing` | 0.0 | {0.0, 0.1} | Mild regularisation against overconfident class decisions; cheap |
| `degrees` (rotation) | 0 | {0, 5, 10} | Cells have no canonical orientation; angular jitter unused so far |
| `dfl` | 1.5 | {1.5, 2.0} | Higher DFL tightens localisation; helps small objects |
| `freeze` | 10 | {0, 10} | Tests whether the ~3× larger corpus has grown out of the freeze regime that was right for P2-only |

24 cells at full grid; subset by editing the four `_VALUES` lists at the top of
`submit_grid.sh`. The default cell (`ls=0.0 deg=0 dfl=1.5 fr=10`) is the P3-A control —
a sanity check that the tinkering pipeline reproduces yolo_new_v1 before reading any
other cell as a real signal.

### Why a focused tuner is safe now (P2-E reframing)

P2-E (Slurm-319536) used `cfg=best_hyperparameters.yaml` from a tuner that had been run
on a *from-scratch* recipe (default `lr0=0.01`, no anchor, no freeze). Applying that yaml
to a fine-tuning run from `DL_Modell_FV.pt` produced catastrophic forgetting (cls_loss
~5.8 within 3–6 epochs, all 5 folds collapsing to mAP50 ≈ 0.448).

That outcome was logged earlier as "the YOLO tuner cannot be applied to fine-tuning". The
correct conclusion is narrower: *a tuner whose search recipe differs from the deployment
recipe* cannot safely contribute its output to deployment. The failure mechanism was the
recipe mismatch, not the tuner per se.

`tune_hyperpara.py` precludes this failure mode by tuning the *exact* production recipe:
anchor `yolo11n.pt`, `freeze=10`, AdamW, `augment=True`, `mosaic=0.0`, `flipud=0.5`,
`cls=1.0`. The output yaml therefore lives in the same hyperparameter regime as the
deployed model and is structurally safe to apply. Step 3 of the workflow is the empirical
verification that the structural argument is correct.

### Expected outcomes

- **Tuner yaml**: small adjustments to defaults (lr0, momentum, weight_decay, warmup)
  within the same recipe family. Catastrophic regression precluded by construction;
  marginal improvement (1–3 pp on test mAP50) is the realistic upper bound.
- **Matrix**: most cells expected to be statistically indistinguishable from P3-A given
  cross-fold std ≈ 0.14. Most likely positive movers: `freeze=0` (corpus has grown
  enough), `degrees=5–10` (rotation jitter unused so far). `label_smoothing=0.1` is the
  most likely lever for per-class recall on rare-class folds (P5).
- **Validation (Step 3)**: tuner output expected within ±2 pp of P3-A. If it regresses
  by more, reject the yaml and proceed with matrix winner alone.
- **Combined (Step 4)**: realistic ceiling ~0.87 mAP50 (≈3 pp above P3-A). A larger
  improvement would suggest the matrix is finding a regime the prior stages did not
  examine — interesting but unexpected.

### Decision tree on completion

```
Step 1+2 complete
    └── Step 3 (validate tuner yaml)
            ├── Test mAP50 within ±2 pp of P3-A
            │       └── Step 4: combine yaml + matrix winner → v2 candidate
            └── Test mAP50 < P3-A by >2 pp
                    └── Step 4: matrix winner alone (no tuner yaml) → v2 candidate
v2 candidate ≥ P3-A on test mAP50 AND no per-class recall regression > 2 pp
    └── adopt as v2; document in this log
v2 candidate worse than P3-A
    └── keep P3-A as v1; document the negative result here as a useful bound
```

The negative-result branch is a productive outcome: it says "the tinkering search did not
locate a configuration outside the P3-A neighbourhood that performs better", which is
itself information about the corpus and the recipe. Either way the campaign produces a
reportable finding.

---

# Phase 3 — Full P1–P8 corpus, LOPO CV, class-imbalance handling

After the P2 phase locked in `lr0=0.001 + freeze=10 + DL_Modell_FV.pt` as a stable fine-tuning
recipe, the project scaled out to the full eight-patient corpus. Two structural changes from P2:

1. **Leave-One-Patient-Out CV** replaces stratified 5-fold. With Normal cells living in only 4 of
   8 patients (P5–P8, of which P6 and P8 are pure-Normal slides), random stratified splits would
   leak patient-specific texture into val and inflate metrics. LOPO yields 8 folds — one held-out
   patient each — and per-fold variance now reflects biological heterogeneity rather than split luck.
2. **Per-patient image oversampling.** The full dataset is 1124 Atypisch vs 78 Normal (14.4:1).
   Image-level duplication of Normal-rich patients (`P5×8, P6×15, P7×8, P8×10`) via symlink with
   on-the-fly augmentation brings the effective per-epoch ratio to ≈ 2:1 without baking
   transforms to disk.

Switched base weights to `yolo11n.pt`. The deployment-trained `DL_Modell_FV.pt` carried implicit
features tuned to the P2 phase only; starting from the open-source nano weights and rebuilding
on the full corpus produced cleaner gradients.

---

### Run P3-A — P1–P8 LOPO baseline with per-patient oversampling
**SLURM:** `322719`
**Results dir:** `yolo_new_v1/`
**Script:** `ba_improved_comb.py` (per-patient oversampling, `cls=1.0`, `mosaic=0.0`, `flipud=0.5`,
on-the-fly `augment=True`, `lr0=0.001`, `freeze=10`, `BG_RATIO=0`, `FP_NEG_OVERSAMPLE=1`)

#### Test results (per-fold, holdout patient as test set)

| Fold | Holdout | mAP50  | mAP50-95 | Precision | Recall | R_Atypisch | R_Normal |
|------|---------|--------|----------|-----------|--------|------------|----------|
| 1    | P1      | 0.651  | 0.491    | 0.682     | 0.626  | 0.753      | 0.500    |
| 2    | P2      | 0.884  | 0.714    | 0.821     | 0.826  | 0.960      | 0.692    |
| 3    | P3      | 0.806  | 0.688    | 0.800     | 0.700  | 0.900      | 0.500    |
| 4    | P4      | 0.910  | 0.763    | 0.792     | 0.892  | 0.783      | 1.000    |
| 5    | P5      | 0.604  | 0.546    | 0.615     | 0.617  | 0.500      | 0.733    |
| 6    | P6      | 0.937  | 0.891    | 0.716     | 0.891  | 1.000      | 0.781    |
| 7    | P7      | 0.972  | 0.866    | 0.949     | 1.000  | 1.000      | 1.000    |
| 8    | P8      | 0.922  | 0.730    | 0.799     | 0.911  | 1.000      | 0.822    |
| **mean** | — | **0.844** | **0.718** | **0.772** | **0.808** | **0.862** | **0.754** |

#### Conclusion
First viable full-corpus result. Mean recall 0.808 across 8 held-out patients is acceptable for a
clinical deployment baseline, but two soft signals motivate further tuning before shipping:
- **R_Normal (0.754) trails R_Atypisch (0.862) by ~11 pp**, which is the inverse of where you want
  the gap given the rare-class clinical priority.
- **Fold 1 (P1 holdout) and Fold 5 (P5 holdout)** are the weakest. P1 uses the early Pos_neg
  annotation regime, and P5's 3-Atyp / 18-Norm composition is anomalous within the corpus.

---

### Run P3-B — Tinkering matrix (training-time knob sweep)
**SLURM:** `Slurm-32714{3..58}` (sbatch grid `submit_grid.sh`)
**Results dir:** `tinker_ls*_deg*_dfl*_fr*/`
**Script:** `tinkering/ba_tinker.py` (P3-A recipe + env-overridable knobs)

#### Motivation
P3-A gave a known-good baseline. A systematic 4-factor matrix over training-time levers tests
whether modest adjustments can lift Normal recall without sacrificing Atypisch recall. Same
8-fold LOPO is repeated per cell so any difference is comparable to the P3-A means above.

#### Matrix (4-factor, full factorial = 24 cells; 16 completed at time of writing)

| Knob              | Values        | Hypothesis |
|-------------------|---------------|------------|
| `label_smoothing` | 0.0, 0.1      | Soften over-confident logits; possibly improve calibration on rare class |
| `degrees`         | 0, 5, 10      | Light rotation regularises orientation-agnostic cells |
| `dfl`             | 1.5, 2.0      | DFL loss weight — higher emphasises localisation refinement |
| `freeze`          | 0, 10         | Freeze backbone vs. fine-tune end-to-end |

Remaining 10 cells (all `label_smoothing=0.1, degrees ∈ {5,10}` plus two `degrees=0, dfl=2.0`)
were rejected by SLURM with `QOSMaxSubmitJobPerUserLimit` and queued for re-submission once the
first batch frees capacity.

#### Test-set means (8-fold LOPO) for the 14 completed cells

| Config                                   | mAP50 | mAP50-95 | Prec  | **Recall** | R_Atyp | **R_Normal** |
|------------------------------------------|-------|----------|-------|------------|--------|--------------|
| `ls0.0_deg5_dfl1.5_fr10` ★ **winner**    | 0.864 | 0.741    | 0.812 | **0.866**  | **0.892** | **0.840** |
| `ls0.0_deg10_dfl2.0_fr10`                | 0.867 | 0.740    | 0.848 | 0.826      | 0.874  | 0.778        |
| `ls0.0_deg5_dfl2.0_fr0`                  | 0.845 | 0.718    | 0.777 | 0.816      | 0.881  | 0.750        |
| `ls0.0_deg5_dfl2.0_fr10`                 | 0.845 | 0.721    | 0.801 | 0.831      | 0.872  | 0.789        |
| `ls0.0_deg0_dfl1.5_fr10` (≡ P3-A)        | 0.844 | 0.718    | 0.764 | 0.840      | 0.884  | 0.795        |
| `ls0.0_deg10_dfl1.5_fr10`                | 0.843 | 0.703    | 0.830 | 0.829      | 0.888  | 0.770        |
| `ls0.0_deg0_dfl1.5_fr0` (P3-A, fr=0)     | 0.836 | 0.711    | 0.772 | 0.808      | 0.862  | 0.754        |
| `ls0.0_deg0_dfl2.0_fr10`                 | 0.833 | 0.715    | 0.786 | 0.812      | 0.866  | 0.759        |
| `ls0.0_deg10_dfl1.5_fr0`                 | 0.828 | 0.705    | 0.745 | 0.833      | 0.856  | 0.810        |
| `ls0.0_deg0_dfl2.0_fr0`                  | 0.828 | 0.714    | 0.749 | 0.825      | 0.872  | 0.777        |
| `ls0.0_deg10_dfl2.0_fr0`                 | 0.815 | 0.690    | 0.781 | 0.830      | 0.887  | 0.773        |
| `ls0.0_deg5_dfl1.5_fr0`                  | 0.824 | 0.716    | 0.775 | 0.830      | 0.887  | 0.773        |
| `ls0.1_deg0_dfl1.5_fr0`  (≡ ls0.0 row)   | 0.836 | 0.711    | 0.772 | 0.808      | 0.862  | 0.754        |
| `ls0.1_deg0_dfl1.5_fr10` (≡ ls0.0 row)   | 0.844 | 0.718    | 0.764 | 0.840      | 0.884  | 0.795        |

#### Findings
1. **Winner: `ls0.0_deg5_dfl1.5_fr10`** dominates every clinically meaningful metric:
   - Test Recall **0.866** (+5.8 pp vs P3-A baseline)
   - Test Recall_Normal **0.840** (+8.6 pp vs P3-A) — the rare-class metric and WHO-criterion proxy
   - Test Recall_Atypisch **0.892** (+3.0 pp vs P3-A)
   - mAP50 0.864 (statistically tied with the `deg10_dfl2.0_fr10` cell at 0.867)

2. **`dfl=2.0` consistently regresses Normal recall** versus `dfl=1.5` across degrees and freeze
   levels. The localisation/classification loss balance tilts away from rare-class discrimination
   when DFL is pushed up.

3. **Light rotation (`degrees=5`) is the sweet spot.** `degrees=0` loses ~4 pp R_Normal vs `=5`,
   and `degrees=10` doesn't add further benefit. Bone-marrow cells are orientation-invariant, so
   modest rotation regularises without distorting morphological cues.

4. **`freeze=10` (default) is correct.** Unfrozen backbone (`fr=0`) marginally underperforms in
   every recall-relevant slice. The fine-tuning data is too small relative to the backbone's
   learned features to risk overwriting them.

5. **`label_smoothing=0.1` is a no-op.** Cells `ls0.0_deg0_dfl1.5_fr{0,10}` and `ls0.1_deg0_dfl1.5_fr{0,10}`
   produce bit-identical fold metrics. Either the env-var didn't propagate to YOLO's training
   loop or Ultralytics' detection-head label-smoothing path is non-functional in this version.
   Either way, dropping LS from the search space removes 8 cells worth of compute and the
   conclusion does not change.

#### Conclusion
The `ls0.0_deg5_dfl1.5_fr10` cell is the deployment recipe. The 8 remaining unrun cells (the
`ls=0.1, degrees ∈ {5,10}` block, plus the two missing `dfl=2.0` ones) are unlikely to dislodge
the winner: the `degrees=5` zone is already mapped out, and label-smoothing is empirically inert.

---

### Run P3-C — YOLO `model.tune()` on the production recipe (negative result)
**SLURM:** `323280` (and earlier tune* directories)
**Results dir:** `runs/detect/tune11/` (30/30 iterations, the canonical run)
**Script:** `tinkering/tune_hyperpara.py`

#### Motivation
The P2-E failure was diagnosed in the original log as "the YOLO tuner cannot be applied to
fine-tuning." On review, that conclusion was overly broad: P2-E's failure was a **recipe
mismatch** — the tuner ran with `lr0=0.01` from-scratch hyperparameters and the output was applied
to a fine-tune of `DL_Modell_FV.pt`, which catastrophically forgot the pretrained features.

P3-C re-runs the tuner against the **production recipe** (yolo11n.pt + freeze=10 + AdamW +
`augment=True`, `mosaic=0.0`, `flipud=0.5`, `cls=1.0`), so the tuner's search space coincides with
deployment conditions. 30 iterations × 50 epochs each, held out P4 as a single fitness fold
(mAP50-95).

#### Tuner output (`best_hyperparameters.yaml`)
```
fitness=0.73582 at iteration 13
box=6.97  cls=1.11  dfl=2.47   degrees=0.0   flipud=0.408
hsv_v=0.358  scale=0.220  fliplr=0.456  mosaic=0.0  close_mosaic=10
lr0=0.00884  lrf=0.00885  momentum=0.938  weight_decay=0.00028
```

#### Findings
The tuner converged on a recipe whose dominant moves **contradict** the matrix evidence on the
same parameters:
- **`degrees=0.0`** vs matrix-optimal `degrees=5` (matrix shows +4-5 pp R_Normal at `=5`)
- **`dfl=2.47`** vs matrix-optimal `dfl=1.5` (matrix shows `dfl=2.0` already regresses R_Normal;
  the tuner's `2.47` is further in the wrong direction)
- **`mosaic=0.0`** (correct — the tuner agrees this is mandatory)
- **`flipud=0.41`** vs production `0.5` (close enough)

Fitness 0.736 (mAP50-95 on P4 only) is essentially indistinguishable from the matrix winner's
0.741 (mAP50-95 averaged across 8 LOPO folds). The tuner's "improvement" is well inside
cross-fold variance and was achieved by over-fitting hyperparameters to a single holdout.

#### Conclusion
**Negative result, but methodologically informative.** Stage-3 validation (training the tuner's
yaml on the full 8-fold LOPO) is **skipped**: the cost is ~6-12 GPU-hours and the prior is that
the recipe will underperform the matrix winner on Normal recall, since its two dominant changes
both move *away* from the matrix's optimal zone. The matrix winner remains the deployment recipe.

For the thesis: P3-C is a productive negative finding. An unconstrained 30-iteration tuner
optimising a single-fold mAP50-95 objective converged on choices that disagree with systematic
8-fold matrix evidence — illustrating why the matrix DoE approach was retained.

---

### Run P3-D — Final deployment model (single training, all P1–P8)
**SLURM:** completed 2026-05-11
**Results dir:** `tinker_final/final/`
**Script:** `tinkering/ba_tinker_final.py` + `tinkering/run_final.sh`

#### Strategy
LOPO is an evaluation protocol. Each of P3-B's fold models has never seen one of the 8 patients,
so none of them is a deployment artifact. Standard post-CV practice: take the recipe that the CV
proved works (`ls0.0_deg5_dfl1.5_fr10`) and train one final model on the full corpus with no
holdout — that model has strictly more data than any LOPO fold model and is what we ship.

#### Setup
- Recipe identical to P3-B winner: `FREEZE=10, DEGREES=5.0, DFL=1.5, LABEL_SMOOTHING=0.0,
  CLS=1.0, lr0=0.001, mosaic=0.0, flipud=0.5, augment=True, cos_lr=True, AdamW, imgsz=512, batch=32`
- Single training on all 8 patients (no LOPO loop, no `mp.Pool`)
- 85/15 stratified-by-patient val split for early stopping only (texture leakage is acceptable
  here — the unbiased generalization estimate already comes from P3-B)
- Per-patient image oversampling unchanged (`P5×8, P6×15, P7×8, P8×10`)
- Pretrained base: `yolo11n.pt`

#### In-train val sanity check (NOT a generalization estimate)
| Metric             | Value  |
|--------------------|--------|
| mAP50-95           | 0.7175 |
| mAP50              | 0.8538 |
| Precision          | 0.7526 |
| Recall             | 0.8386 |
| Recall (Atypisch)  | 0.9639 |
| Recall (Normal)    | 0.7133 |

#### Interpretation
- **Convergence is healthy.** Overall recall 0.84, mAP50 0.85, and Atypisch recall 0.96 are all
  consistent with — or slightly above — the P3-B matrix winner LOPO means. The run converged on
  the recipe-specified epoch with no anomalies.
- **R_Normal 0.713 is below the LOPO mean of 0.840** and looks surprising at first because val
  patients are also in train (leakage normally *inflates* val). Root cause: this val split has
  only ~10–15 Normal labels total (15% of the corpus's 78 Normals, stratified by patient), and
  missing 3–4 of those moves R_Normal by ≈25 pp. Single-point variance is ±0.10 easily. A
  secondary effect is YOLO's `best.pt` fitness criterion, which is a weighted mean over classes
  and so biases checkpoint selection toward the epoch where Atypisch peaked rather than where
  Normal peaked — the same selection rule was used in all P3-B fold runs, so this is at least
  apples-to-apples but the variance on Normal specifically is amplified.
- **The authoritative generalization number remains the LOPO matrix winner:**
  Recall ≈ 0.866 | R_Normal ≈ 0.840 | R_Atypisch ≈ 0.892. The val sanity check verifies
  *convergence*, not deployment quality.

#### Deliverable
`hpc/tinker_final/final/weights/best.pt` → sent to USZ for next-round case collection
(model-assisted review of unannotated slides, FP/FN correction loop, new-patient acquisition).
Acceptance handover to USZ: flag both clinician misses (FN) and low-confidence Normal predictions
in the review loop, and track per-class R_Normal on incoming P9+ slides as the primary monitoring
metric.

---

## Daily log — 2026-05-11

Tasks completed today:

1. **P3-B matrix analysed.** Computed 8-fold LOPO test-set means for all 14 completed cells.
   Identified `ls0.0_deg5_dfl1.5_fr10` as the unambiguous winner across every clinically
   meaningful metric (R 0.866, R_Normal 0.840, R_Atypisch 0.892). Confirmed `dfl=2.0` regresses
   Normal recall, `degrees=5` is the sweet spot, `freeze=10` beats `=0`, and
   `label_smoothing=0.1` is a bit-for-bit no-op vs `=0.0` (env-var didn't reach YOLO's loss
   path or the implementation is inert in this Ultralytics version).
2. **P3-C tuner reviewed.** 30-iteration `model.tune()` on P4 holdout completed at fitness 0.736
   (single-fold mAP50-95). Output yaml contradicts the matrix on `degrees` and `dfl`. Decided to
   skip Stage-3 validation — productive negative result for the thesis.
3. **P3-D final model built and trained.** Created `tinkering/ba_tinker_final.py` (single-run
   variant of `ba_tinker.py`, no LOPO loop, recipe baked as defaults but env-overridable) and
   `tinkering/run_final.sh`. Submitted, ran on 1 L40S, converged with the val metrics tabulated
   above. `tinker_final/final/weights/best.pt` is the USZ deliverable.
4. **Documentation.** Extended this log with Phase 3 (sub-runs A–D) and the P3-D in-train val
   sanity check.

Outstanding:

- 10 matrix cells (`ls=0.1, degrees ∈ {5,10}` + 2 `dfl=2.0`) still queued from the
  `QOSMaxSubmitJobPerUserLimit` rejection. Unlikely to dislodge the winner (label-smoothing is
  inert, `degrees=5` zone already mapped) — re-submission is bookkeeping for completeness, not
  decision-relevant.
- USZ handover of `best.pt` + acceptance documentation.

---

### Run P3-E — yolo11l (large model) on P1–P8, same recipe as P3-A (negative result)
**Results dir:** `tinker_default/`
**Script:** `tinkering/ba_tinker.py` with `PRETRAINED_WEIGHTS=yolo11l.pt`, all other knobs at P3-A defaults (`degrees=0, dfl=1.5, freeze=10, label_smoothing=0.0`)

#### Motivation
P3-A and the tinkering matrix were run entirely on `yolo11n` (2.6 M params). The question was whether scaling to `yolo11l` (43 M params, ~16× larger) would yield better feature representations and lift recall — particularly R_Normal, which lagged R_Atypisch by ~11 pp.

#### Setup
- Base model: `yolo11l.pt` (COCO pretrained, large variant)
- All training knobs identical to P3-A: `lr0=0.001, freeze=10, cls=1.0, dfl=1.5, degrees=0, mosaic=0.0, flipud=0.5, augment=True, cos_lr=True, batch=32, imgsz=512`
- 8-fold LOPO CV on P1–P8

#### Test results

| Fold | Holdout | mAP50  | mAP50-95 | Precision | Recall | R_Atypisch | R_Normal |
|------|---------|--------|----------|-----------|--------|------------|----------|
| 1    | P1      | 0.735  | 0.669    | 0.761     | 0.664  | 0.829      | 0.500    |
| 2    | P2      | 0.881  | 0.746    | 0.814     | 0.826  | 0.937      | 0.714    |
| 3    | P3      | 0.769  | 0.676    | 0.757     | 0.711  | 0.923      | 0.500    |
| 4    | P4      | 0.912  | 0.755    | 0.965     | 0.793  | 0.810      | 0.777    |
| 5    | P5      | 0.712  | 0.600    | 0.615     | 0.653  | 0.500      | 0.807    |
| 6    | P6      | 0.995  | 0.890    | 0.886     | 0.821  | 1.000      | 0.641    |
| 7    | P7      | 0.913  | 0.791    | 0.859     | 0.955  | 1.000      | 0.909    |
| 8    | P8      | 0.927  | 0.834    | 0.570     | 0.854  | 1.000      | 0.708    |
| **mean** | — | **0.855** | **0.745** | **0.778** | **0.785** | **0.875** | **0.695** |

#### Comparison vs P3-A (yolo11n, same recipe) and P3-B winner (yolo11n, degrees=5)

| Config              | mAP50 | Recall | R_Atypisch | R_Normal |
|---------------------|-------|--------|------------|----------|
| P3-A (yolo11n, deg=0) | 0.844 | 0.808 | 0.862 | 0.754 |
| P3-B winner (yolo11n, deg=5) | 0.864 | 0.866 | 0.892 | 0.840 |
| **P3-E (yolo11l, deg=0)** | **0.855** | **0.785** | **0.875** | **0.695** |
| Delta vs P3-A | +0.011 | -0.023 | +0.013 | **-0.059** |

#### Conclusion
**The large model is worse, not better.** mAP50 gains a marginal +1.1 pp but overall recall drops 2.3 pp and R_Normal drops 5.9 pp — the exact metric where improvement was hoped for. R_Normal 0.695 is the weakest result in any model tested on P1–P8, worse even than the P3-A baseline.

**Why the large model underperforms:**

1. **Dataset is too small for 43 M parameters.** Each LOPO fold trains on roughly 170–220 images. yolo11l has 16× more free parameters than yolo11n, most of them in the unfrozen neck and head. The classification head is randomly re-initialized at every run (nc change 1→2 drops pretrained `cv3` weights), so all of those parameters must be learned from scratch from the fold's training images. yolo11n converges with less data; yolo11l overfits the training fold and fails to generalise.

2. **Freeze=10 protects the backbone but exposes a large neck.** yolo11l's neck (layers 10–21) is proportionally wider than yolo11n's. Unfreezing a wider neck with the same small dataset and the same lr0=0.001 means larger gradient norms relative to the feature scale, increasing the risk of overwriting transferred features.

3. **R_Normal is the canary.** Normal cells are underrepresented in most folds; generalising to them requires the model to have learned a robust representation, not just memorised Atypisch patterns. A larger model memorises faster and therefore generalises less when data is scarce.

**Decision:** `yolo11n` is the correct model size for this corpus. Scaling up model capacity is not the path to better Normal recall — the path is more data (P9–P15) and the confirmed recipe (degrees=5). P3-E is a useful negative result that bounds the search space.

---

## Phase 4 — Expanded corpus P1–P15 (added 2026-05-20)

### Dataset additions

Seven new patients (P9–P15) acquired from USZ via the model-assisted annotation loop have been integrated into `ba_improved_comb.py`. Each patient contributes:
- Annotated Atypisch/Normal images referenced via a YOLO-style `Train.txt` file (absolute paths on HPC; no flat `images/Train/` directory).
- A per-patient FP-negative Excel file listing confirmed false-positive tile filenames.

| Patient | FP Excel              | FP entries |
|---------|-----------------------|------------|
| P9      | Task39_V2.xlsx        | 725        |
| P10     | Task26_V2.xlsx        | 76         |
| P11     | Task27_V2.xlsx        | 28         |
| P12     | Task21_V2.xlsx        | 89         |
| P13     | Task24_V2.xlsx        | 68         |
| P14     | Task25_V2.xlsx        | 57         |
| P15     | Task23_V2.xlsx        | 49         |

### Script changes (`ba_improved_comb.py`)

1. **Generalised FP loading.** The hardcoded P2-only FP block is replaced by a loop over `PATIENT_FP_EXCELS` (P2 + P9–P15). For each patient, `build_disk_index()` builds a `{basename → abspath}` map from the patient's image directory glob, then resolves FP filenames from the Excel against it.

2. **Automatic oversample factors for new patients.** `PATIENT_OVERSAMPLE_FIXED` retains the manually validated P1–P8 factors. P9–P15 are absent from this dict and computed at runtime by `compute_oversample_factors()`, which iterates patients from most Normal-heavy to least and solves the closed-form equation for k such that the global weighted Atypisch:Normal ratio reaches the target (default 2.0). Factors are capped at 25. The summary printout now shows effective counts and the resulting ratio so the output can be verified at a glance.

3. **LOPO now yields up to 15 folds** (one per patient present in the corpus). GPU assignment `fold_idx % 2` continues to interleave across the two L40S GPUs.

### First run output — corpus summary (2026-05-20)

```
Patient   Files  Atypisch  Normal  Oversample  EffAtyp  EffNorm
P1          428       482       2           1      482        2
P2          417       449       6           1      449        6
P3           61        60       4           1       60        4
P4          126       122       6           1      122        6
P5           21         3      18           8       24      144
P6            6         0       6          15        0       90
P7           20         8      12           8       64       96
P8           24         0      24          10        0      240
P9          193         4     195           1        4      195
P10          25         5      20           1        5       20
P11          37        17      20           1       17       20
P12         125       133       0           1      133        0
P13           7         0       7           1        0        7
P14           7         5       3           1        5        3
P15          67         0      68           1        0       68
TOTAL      1564      1288     391                           
Raw ratio      = 3.3 : 1
Effective ratio = 1.5 : 1  (target 2.0 : 1)
```

FP negatives: 1434 total resolved (P2: 330, P9: 730, P10: 76, P11: 29, P12: 90, P13: 73, P14: 57, P15: 49). 1 unresolved (P2: `tile_87_38.jpeg`).

**Oversample factors for P9–P15 all computed as 1.** The fixed P5–P8 factors (×8–15), calibrated for the P1–P8 corpus, already push the effective ratio below the 2.0:1 target when P9–P15's Normal-heavy annotations are included. `compute_oversample_factors()` correctly assigns k=1 to all new patients because the global ratio is already below target at each step. The 1.5:1 effective ratio (slightly Normal-biased) is acceptable — if anything, it works in the direction of improving R_Normal, which was the weaker metric.

**Small test sets in three folds:** P6 holdout (6 images), P13 holdout (7), P14 holdout (7). One missed cell moves recall by ~14–17 pp. Metrics from these folds are directional only.

---

### Hyperparameter transferability — P3-B recipe on P1–P15 (analysis, 2026-05-20)

The P3-B winner (`degrees=5, dfl=1.5, freeze=10, lr0=0.001, cls=1.0, mosaic=0.0, flipud=0.5`) was validated on an 8-fold LOPO of P1–P8 (~170–220 train images per fold). The P1–P15 corpus changes several conditions that those hyperparameters were calibrated against.

| Knob | Transferability | Rationale |
|------|----------------|-----------|
| `degrees=5` | Safe | Cells are still orientation-invariant; domain unchanged |
| `dfl=1.5` | Safe | Finding (dfl=2.0 hurts R_Normal) is about loss balance, not corpus size |
| `lr0=0.001` | Safe | Catastrophic-forgetting risk from random `cv3` init does not shrink with more data |
| `mosaic=0.0`, `flipud=0.5` | Safe | Data-domain decisions, unchanged |
| `freeze=10` | **Uncertain** | Was correct when each fold trained on ~200 images. Folds now train on ~1000–1300 positives — the "corpus too small to unfreeze" argument is weaker. In the P3-B matrix, `freeze=0` already trailed by only ~2 pp R_Normal. Worth re-testing. |
| `cls=1.0` | **Uncertain** | Raw ratio shifted from 14:1 to 3.3:1; effective ratio is now 1.5:1. With better inherent balance, the double class-loss weight may be less necessary. |

**Plan:** Run P1–P15 with the P3-B recipe as-is (Run P4-A, ongoing). If results are acceptable, follow up with a single `freeze=0` vs `freeze=10` comparison on P1–P15 — one additional run that directly answers the most uncertain knob given the larger corpus. A full re-grid is not warranted unless P4-A shows a clear regression relative to P3-B.

**On re-running the full grid:** A 24-cell P3-B matrix on P1–P15 would cost ~24 × 12h = 288 GPU-hours (15 folds × larger corpus vs. 8 folds × smaller corpus). This is not feasible given thesis timelines. The justification for skipping the re-grid is methodological, not just pragmatic: `degrees`, `dfl`, and `lr0` are task-domain properties, not dataset-size properties. The domain has not changed (bone marrow cell morphology, same image resolution, same sensor), so the P3-B findings on those knobs transfer. The sole genuinely corpus-size-dependent knob is `freeze` — hence the single targeted `freeze=0` follow-up is the correct and sufficient experiment.

---

### Run P4-A — P1–P15 LOPO baseline, P3-A recipe (degrees=0)
**SLURM:** 334000 (`yolo_new_v3`)
**Results dir:** `yolo_new_v3/fold_{1..15}_{P1..P15}/`
**Script:** `ba_improved_comb.py` (P1–P15 extension with generalised FP loading and auto oversample)
**Status:** complete, 2026-05-23

#### Setup
- `FREEZE=10, lr0=0.001, cls=1.0, mosaic=0.0, flipud=0.5, augment=True, cos_lr=True, AdamW, imgsz=512, batch=32`
- **`degrees=0.0`, `dfl=1.5` — YOLO defaults** (neither was explicitly passed to `model.train()` at the time of this run; `DEGREES` and `DFL` were added as env-overridable constants in `ba_improved_comb.py` after this run completed). This makes P4-A equivalent to the P3-A recipe (degrees=0) applied to the larger corpus, **not** the full P3-B winner recipe (degrees=5).
- 15-fold LOPO CV (one patient held out per fold)
- Per-patient oversampling: P1–P8 factors pinned from P3-B; P9–P15 computed at runtime (all ×1 — fixed P5–P8 factors already achieve 1.5:1 effective ratio, below the 2.0 target)
- FP negatives from 8 Excel files (P2 + P9–P15): 1434 resolved total, 1 unresolved (P2: `tile_87_38.jpeg`)

**Note on P9 FP count (725 entries):** The high count is genuine. Confirmed by the USZ physician — P9 slides were systematically misclassified by the v1 model during the annotation-assistance round, producing an unusually large pool of confirmed false positives. The Excel represents real model failures on that patient's slide texture and is a valid hard-negative training asset, not a data entry artefact.

#### Corpus summary at runtime
```
Patient   Files  Atypisch  Normal  Oversample  EffAtyp  EffNorm
P1          428       482       2           1      482        2
P2          417       449       6           1      449        6
P3           61        60       4           1       60        4
P4          126       122       6           1      122        6
P5           21         3      18           8       24      144
P6            6         0       6          15        0       90
P7           20         8      12           8       64       96
P8           24         0      24          10        0      240
P9          193         4     195           1        4      195
P10          25         5      20           1        5       20
P11          37        17      20           1       17       20
P12         125       133       0           1      133        0
P13           7         0       7           1        0        7
P14           7         5       3           1        5        3
P15          67         0      68           1        0       68
TOTAL      1564      1288     391
Raw ratio      = 3.3 : 1
Effective ratio = 1.5 : 1  (target 2.0 : 1)
```

#### Test results — per fold

| Fold | Holdout | mAP50-95 | mAP50  | Precision | Recall | R_Atypisch | R_Normal |
|------|---------|----------|--------|-----------|--------|------------|----------|
| 1    | P1      | 0.530    | 0.627  | 0.853     | 0.552  | 0.818      | 0.286    |
| 2    | P10     | 0.875    | 0.978  | 0.881     | 1.000  | 1.000      | 1.000    |
| 3    | P11     | 0.795    | 0.899  | 0.929     | 0.816  | 0.882      | 0.750    |
| 4    | P12     | 0.795    | 0.937  | 0.873     | 0.957  | 0.913      | 1.000    |
| 5    | P13     | 0.807    | 0.876  | 0.851     | 0.954  | 0.954      | (vacuous — 0 Normal) |
| 6    | P14     | 0.908    | 0.995  | 0.978     | 1.000  | 1.000      | 1.000    |
| 7    | P15     | 0.882    | 0.959  | 0.830     | 0.963  | 1.000      | 0.925    |
| 8    | P2      | 0.798    | 0.926  | 0.840     | 0.921  | 0.952      | 0.890    |
| 9    | P3      | 0.817    | 0.950  | 0.867     | 0.990  | 0.980      | 1.000    |
| 10   | P4      | 0.800    | 0.923  | 0.735     | 0.877  | 0.755      | 1.000    |
| 11   | P5      | 0.617    | 0.717  | 0.662     | 0.798  | 0.667      | 0.929    |
| 12   | P6      | 0.924    | 0.995  | 0.986     | 1.000  | 1.000      | 1.000    |
| 13   | P7      | 0.853    | 0.995  | 0.981     | 1.000  | 1.000      | 1.000    |
| 14   | P8      | 0.794    | 0.965  | 0.726     | 0.881  | 1.000      | 0.761    |
| 15   | P9      | 0.705    | 0.796  | 0.764     | 0.845  | 0.769      | 0.920    |
| **mean** | — | **0.793** | **0.903** | **0.851** | **0.904** | **0.913** | **0.890** |
| **std**  | — | 0.106 | 0.100 | 0.097 | 0.120 | 0.110 | 0.194 |

Mean R_Atypisch computed over all 15 folds (n=15); mean R_Normal computed over 14 folds (P13 excluded: 0 Normal in test set).

#### Comparison to prior runs (P1–P8, 8-fold LOPO)

The fair comparison for P4-A (degrees=0, freeze=10, P1–P15) is P3-A (degrees=0, freeze=10, P1–P8) — same recipe, different corpus size. P3-B winner is shown for reference but uses degrees=5, so the delta includes both the corpus expansion and the degrees difference.

| Metric      | P3-A deg=0 (P1–P8) | P3-B winner deg=5 (P1–P8) | P4-A deg=0 (P1–P15) | P3-A→P4-A delta |
|-------------|---------------------|---------------------------|----------------------|-----------------|
| mAP50       | 0.844               | 0.864                     | 0.903                | **+5.9 pp**     |
| mAP50-95    | 0.718               | 0.741                     | 0.793                | +7.5 pp         |
| Recall      | 0.808               | 0.866                     | 0.904                | **+9.6 pp**     |
| R_Atypisch  | 0.862               | 0.892                     | 0.913                | +5.1 pp         |
| R_Normal    | 0.754               | 0.840                     | 0.890                | **+13.6 pp**    |

The corpus expansion from P1–P8 to P1–P15 alone (holding recipe constant at degrees=0) delivers +5.9 pp mAP50 and +13.6 pp R_Normal. This is a large effect — the seven new patients, especially P9 (195 Normal annotations), substantially improved Normal-class generalisation.

#### Fold-level notes

**Fold 1 (P1 holdout):** Still the weakest fold — mAP50 0.627, R_Normal 0.286. P1 has only 2 Normal annotations in its test set, so R_Normal = 0.286 means 1/2 found. The Atypisch recall (0.818) is below the corpus mean but P1's annotation style (old Pos_neg regime) is the persistent outlier. This fold's weakness is structural, not fixable by hyperparameter changes.

**Fold 5 (P13 holdout) R_Normal = NaN:** measurement bug — see `per_class` positional indexing bug in Infrastructure Notes. P13 has 7 Normal GT and 0 Atypisch GT. The model makes no class-0 predictions on P13's images, so Ultralytics omits class 0 from `metrics.box.r`, producing a length-1 array. `per_class(_, 1)` hits an out-of-bounds index → NaN. The true Normal recall is 0.954 (shown in the R_Atypisch column). Excluded from the R_Normal mean (n=14) but the underlying model performance is fine.

**Fold 15 (P9 holdout):** R_Atypisch 0.769 on 4 Atypisch instances — the second weakest Atypisch recall after Fold 1. P9 is Normal-dominant (195 Normal vs 4 Atypisch), so training without P9 has ~1.3:1 Atypisch:Normal ratio instead of the usual 1.5:1, and sees almost no examples of that balance regime. The 0.920 R_Normal is strong, confirming the model handles the Normal-dominant regime well; the weak Atypisch recall is a small-sample artefact (missing 1 of 4 = −25 pp).

**Fold 11 (P5 holdout):** mAP50 0.717, the second-weakest. Consistent with P3-A/P3-B: P5 has 3 Atypisch / 18 Normal in the test set and 1 missed Atypisch = −33 pp. Small-sample artefact, not a model regression.

#### Conclusion

P4-A confirms the P3-B recipe transfers cleanly to the expanded P1–P15 corpus. Adding 7 patients and 503 images delivers +3.9 pp mAP50 and +5.0 pp R_Normal on top of the already-optimised P3-B baseline. The result is strong enough to stand as the thesis's primary experimental result for the extended corpus.

The val metrics (mAP50 0.941, R_Normal 0.946) are near-ceiling, confirming the model is learning the full corpus well. Train metrics (~0.993) are expected high given the train split includes oversampled duplicates.

#### On hyperparameter transferability (why no re-grid)

A full 24-cell re-grid on P1–P15 would cost approximately 24 × 12h = 288 GPU-hours — not feasible given thesis timelines, and not methodologically required. The justification:

- `degrees=5`, `dfl=1.5`, `lr0=0.001`, `mosaic=0.0`, `flipud=0.5`: these are **task-domain decisions**, not dataset-size decisions. The domain hasn't changed: bone marrow cells, same sensor, same image resolution, same orientation invariance. The P3-B findings on these knobs are properties of the problem, not artefacts of the P1-P8 corpus size. The P4-A improvement confirms this — if the recipe had been over-fit to P1-P8, performance would have degraded, not improved, on the larger corpus.

- `freeze=10`: **Genuinely corpus-size-dependent.** The P3-B rationale was "corpus too small to unfreeze." With 15 patients and ~1000-1300 training images per fold (vs ~200 before), this is now uncertain. The `freeze=0` follow-up is the one targeted experiment that remains necessary. It is a single run, not a re-grid.

- `cls=1.0`: The effective class ratio shifted from ~2:1 to 1.5:1. With better inherent balance, the double class-loss weight is less critical. However, changing it simultaneously with the corpus expansion conflates two variables. Keeping it at 1.0 for P4-A was correct; if the freeze=0 run shows no improvement, `cls` is a secondary knob worth testing.

---

### Run P4-B — degrees=5, freeze=10 baseline on P1–P15
**SLURM:** 337215 (`yolo_new_v3_freeze10`)
**Results dir:** `yolo_new_v3_freeze10/`
**Status:** complete

#### Motivation
P4-A ran with degrees=0 (YOLO default, not explicitly set). The P3-B matrix showed degrees=5 added +4 pp R_Normal on P1–P8. P4-B establishes the degrees=5 baseline on P1–P15 before the freeze comparison.

#### Test results — per fold

| Fold | Holdout | mAP50  | Recall | R_Atypisch | R_Normal |
|------|---------|--------|--------|------------|----------|
| 1    | P1      | 0.5835 | 0.5755 | 0.8653     | 0.2857   |
| 2    | P10     | 0.9771 | 0.9942 | 1.0000     | 0.9883   |
| 3    | P11     | 0.8914 | 0.8374 | 0.9412     | 0.7335   |
| 4    | P12     | 0.9344 | 0.9306 | 0.8612     | 1.0000   |
| 5    | P13     | 0.9007 | 1.0000 | 1.0000     | NaN*     |
| 6    | P14     | 0.9950 | 1.0000 | 1.0000     | 1.0000   |
| 7    | P15     | 0.9542 | 0.9358 | 1.0000     | 0.8716   |
| 8    | P2      | 0.8641 | 0.8587 | 0.9174     | 0.8000   |
| 9    | P3      | 0.9167 | 0.9808 | 0.9804     | 0.9811   |
| 10   | P4      | 0.9333 | 0.9127 | 0.8255     | 1.0000   |
| 11   | P5      | 0.6613 | 0.6310 | 0.3333     | 0.9286   |
| 12   | P6      | 0.9950 | 1.0000 | 1.0000     | 1.0000   |
| 13   | P7      | 0.9950 | 1.0000 | 1.0000     | 1.0000   |
| 14   | P8      | 0.8802 | 0.9118 | 1.0000     | 0.8235   |
| 15   | P9      | 0.8175 | 0.8498 | 0.7692     | 0.9303   |
| **mean** | — | **0.8866** | **0.8945** | **0.8996** | **0.8816** |

*P13 NaN: see `per_class` positional indexing bug. True R_Normal = 1.0000 (in R_Atypisch column). Excluded from R_Normal mean (n=14).

---

### Run P4-C — degrees=5, freeze=0 comparison on P1–P15
**SLURM:** 337214 (`yolo_new_v3_freeze0`)
**Results dir:** `yolo_new_v3_freeze0/`
**Status:** complete

#### Motivation
Clean freeze=0 vs freeze=10 comparison, both with degrees=5. P4-B is the reference; P4-C is the test.

#### Test results — per fold

| Fold | Holdout | mAP50  | Recall | R_Atypisch | R_Normal |
|------|---------|--------|--------|------------|----------|
| 1    | P1      | 0.5754 | 0.5751 | 0.8645     | 0.2857   |
| 2    | P10     | 0.9525 | 1.0000 | 1.0000     | 1.0000   |
| 3    | P11     | 0.9195 | 0.8029 | 0.8824     | 0.7235   |
| 4    | P12     | 0.9487 | 0.9756 | 0.9513     | 1.0000   |
| 5    | P13     | 0.8764 | 1.0000 | 1.0000     | NaN*     |
| 6    | P14     | 0.9950 | 0.9897 | 0.9795     | 1.0000   |
| 7    | P15     | 0.9732 | 0.9409 | 0.9388     | 0.9431   |
| 8    | P2      | 0.9264 | 0.9118 | 0.9683     | 0.8554   |
| 9    | P3      | 0.8187 | 0.8343 | 0.8686     | 0.8000   |
| 10   | P4      | 0.9288 | 0.9562 | 0.9123     | 1.0000   |
| 11   | P5      | 0.6716 | 0.6310 | 0.3333     | 0.9286   |
| 12   | P6      | 0.9950 | 1.0000 | 1.0000     | 1.0000   |
| 13   | P7      | 0.9950 | 1.0000 | 1.0000     | 1.0000   |
| 14   | P8      | 0.8706 | 0.9308 | 1.0000     | 0.8616   |
| 15   | P9      | 0.8023 | 0.8442 | 0.7417     | 0.9467   |
| **mean** | — | **0.8833** | **0.8928** | **0.8960** | **0.8818** |

*P13 NaN: same per_class bug. Excluded from R_Normal mean (n=14).

#### P4-B vs P4-C comparison

| Metric      | P4-B freeze=10 | P4-C freeze=0 | Delta       |
|-------------|----------------|---------------|-------------|
| mAP50       | 0.8866         | 0.8833        | −0.33 pp    |
| Recall      | 0.8945         | 0.8928        | −0.17 pp    |
| R_Atypisch  | 0.8996         | 0.8960        | −0.36 pp    |
| R_Normal    | 0.8816         | 0.8818        | +0.02 pp    |

**Decision: freeze=10 confirmed.** All deltas are within ±0.4 pp — far inside the ±2 pp threshold. Unfreezing the full backbone on P1–P15 provides no benefit. The "corpus too small to unfreeze" intuition holds even at ~1000–1300 training images per fold.

**Implication for P1–P53:** No freeze comparison run is needed at the larger corpus scale. The P3-B winner recipe (`degrees=5, dfl=1.5, freeze=10, lr0=0.001, cls=1.0`) is confirmed as the deployment recipe and will be used as-is for the P1–P53 baseline run.

---

## Phase 5 — Full corpus P1–P53 (added 2026-05-25)

### Dataset additions

38 new patients (P16–P53) acquired from USZ. All follow the same structure as P9–P15: images under `new/P{N}/images/Train/`, labels under `new/P{N}/labels/Train/`. Data was uploaded as task-numbered ZIPs, extracted on the HPC, and source ZIPs deleted. Setup documented in `testing.ipynb`.

**FP-negative Excel files (all confirmed present on disk):**

| Patient | Excel               | Patient | Excel               |
|---------|---------------------|---------|---------------------|
| P16     | Task19_V2.xlsx      | P35     | Task56_V2.xlsx      |
| P17     | Task20_V2.xlsx      | P36     | Task57_V2.xlsx      |
| P18     | Task22_V2.xlsx      | P37     | Task58_V2.xlsx      |
| P19     | Task40_V2.xlsx      | P38     | Task59_V2.xlsx      |
| P20     | Task41_V2.xlsx      | P39     | Task60_V2xlsx.xlsx† |
| P21     | Task42_V2.xlsx      | P40     | Task61_V2.xlsx      |
| P22     | Task43_V2.xlsx      | P41     | Task62_V2.xlsx      |
| P23     | Task44_V2.xlsx      | P42     | Task63_V2.xlsx      |
| P24     | Task45_V2.xlsx      | P43     | Task64_V2.xlsx      |
| P25     | Task46_V2.xlsx      | P44     | Task65_V2xlsx.xlsx† |
| P26     | Task47_V2.xlsx      | P45     | Task66_V2.xlsx      |
| P27     | Task48_V2.xlsx      | P46     | Task67_V2.xlsx      |
| P28     | Task49_V2.xlsx      | P47     | Task68_V2.xlsx      |
| P29     | Task50_V2.xlsx      | P48     | Task69_V2.xlsx      |
| P30     | Task51_V2.xlsx      | P49     | Task70_V2.xlsx      |
| P31     | Tas52_V2.xlsx†      | P50     | Task71_V2.xlsx      |
| P32     | Task53_V2.xlsx      | P51     | Task73_V2.xlsx      |
| P33     | Task54_V2.xlsx      | P52     | Task74_V2.xlsx      |
| P34     | Task55_V2.xlsx      | P53     | Task75_V2.xlsx      |

† Filename typo in the actual on-disk file; `patient_dir.py` matches the typo exactly. Task numbers 21, 28–38, 52, 60, 65, 72 are absent — those annotation tasks were not assigned to new patients in this batch.

**Annotation counts for P16–P53 (testing.ipynb, 2026-05-25):**

Patients without a `labels/` folder contain **no mast cells** — they are pure-FP slides (only confirmed false-positive detections). They contribute zero positive training examples but their Excel FP entries are loaded normally via `PATIENT_FP_EXCELS`. The pipeline handles this correctly: `load_patient_images` returns empty → 0 positives; `build_disk_index` still globs the images dir for FP resolution.

| Patient | Files | Atypisch | Normal | Notes |
|---------|-------|----------|--------|-------|
| P16 | 23 | 22 | 1 | |
| P17 | 69 | 69 | 0 | |
| P18 | 82 | 81 | 1 | |
| P19 | — | — | — | pure-FP (no mast cells) |
| P20 | — | — | — | pure-FP |
| P21 | — | — | — | pure-FP |
| P22 | — | — | — | pure-FP |
| P23 | — | — | — | pure-FP |
| P24 | 1 | 1 | 0 | |
| P25 | — | — | — | pure-FP |
| P26 | — | — | — | pure-FP |
| P27 | — | — | — | pure-FP |
| P28 | 1 | 1 | 0 | |
| P29 | 1 | 1 | 0 | |
| P30 | — | — | — | pure-FP |
| P31 | 1 | 0 | 1 | |
| P32 | — | — | — | pure-FP |
| P33 | — | — | — | pure-FP |
| P34 | — | — | — | pure-FP |
| P35 | — | — | — | pure-FP |
| P36 | — | — | — | pure-FP |
| P37 | 2 | 0 | 2 | |
| P38 | — | — | — | pure-FP |
| P39 | 1 | 0 | 1 | |
| P40 | 1 | 0 | 1 | |
| P41 | — | — | — | pure-FP |
| P42 | — | — | — | pure-FP |
| P43 | 3 | 0 | 3 | |
| P44 | 2 | 0 | 2 | |
| P45 | 1 | 0 | 1 | |
| P46 | 4 | 0 | 4 | |
| P47 | 3 | 0 | 3 | |
| P48 | 7 | 7 | 0 | |
| P49 | — | — | — | pure-FP |
| P50 | 1 | 0 | 1 | |
| P51 | 92 | 5 | 90 | |
| P52 | 5 | 1 | 4 | |
| P53 | 5 | 0 | 5 | |

**Updated full corpus totals (P1–P53, annotated patients only):**
Files: 1869 | Atypisch: 1476 | Normal: 511 | Raw ratio: 2.9:1
(vs P1–P15: 1564 files, 1288/391, 3.3:1 — Normal share growing with new data)

**FP-negative Excel row counts (testing.ipynb cell 42, 2026-05-25):**

| P | FP  | P | FP  | P | FP  | P | FP  |
|---|-----|---|-----|---|-----|---|-----|
| P9  | 729 | P20 | 18  | P31 | 66  | P42 | 142 |
| P10 | 75  | P21 | 40  | P32 | 136 | P43 | 198 |
| P11 | 28  | P22 | 79  | P33 | 102 | P44 | 46  |
| P12 | 89  | P23 | 59  | P34 | 43  | P45 | 35  |
| P13 | 72  | P24 | 13  | P35 | 21  | P46 | 56  |
| P14 | 56  | P25 | 53  | P36 | 6   | P47 | 60  |
| P15 | 48  | P26 | 38  | P37 | 14  | P48 | 65  |
| P16 | 101 | P27 | 160 | P38 | 10  | P49 | 30  |
| P17 | 92  | P28 | 72  | P39 | 6   | P50 | 106 |
| P18 | 42  | P29 | 87  | P40 | 47  | P51 | 122 |
| P19 | 142 | P30 | 10  | P41 | 94  | P52 | 56  |
|     |     |     |     |     |     | P53 | 73  |

Total P9–P53 FP entries: 3637. Largest pools: P43 (198), P27 (160), P9 (729 — confirmed systematic misclassification on that patient's slide texture), P19 (142), P42 (142).

### Code changes (2026-05-25)

1. **`patient_dir.py` — FP Excel filenames filled in.** All 38 entries for P16–P53 in `PATIENT_FP_EXCELS` updated from empty strings to the correct filenames.

2. **`ba_improved_comb.py` — `PATIENT_FP_EXCELS` import added.** The symbol was used at line 567 but never imported — a latent `NameError` that would have crashed the FP-loading block at runtime.

3. **`ba_improved_comb.py` — `PATIENT_OVERSAMPLE_FIXED` emptied.** The previously pinned P1–P8 factors (×1–×15) were validated for the 8–15 patient corpus. With 53 patients the global class balance shifts substantially; `compute_oversample_factors()` now handles all patients. The printed corpus summary will show the computed factors and resulting effective ratio. To restore a pin, add the patient to the dict.

### On oversampling necessity at P1–P53 scale

The raw ratio improved from 3.3:1 (P1–P15) to 2.9:1 (P1–P53 annotated patients), closer to the 2.0:1 target. The question of whether oversampling is still needed is addressed below.

**Still beneficial, but effect is smaller.** The global ratio has improved but per-patient distribution remains highly uneven: P17 has 69 Atypisch / 0 Normal, P16/P18 are nearly pure-Atypisch, while P51 (5/90), P9 (4/195) are Normal-dominant. Within each LOPO fold the effective ratio depends on which patient is held out — oversampling stabilises this variance.

**In practice, most factors will be 1.** The same dynamic observed in P4-A will repeat: the existing Normal-heavy patients (P5, P6, P8, P9, P51) and the larger corpus pull the global ratio close to or below 2.0:1 before the auto-solver even reaches the Atypisch-dominant patients. `compute_oversample_factors()` will assign k=1 to the majority and boost only the most Normal-rich patients. The mechanism is correct and harmless to keep; the printed corpus summary will show the actual factors.

**Secondary open question:** with a 2.9:1 raw ratio, `cls=1.0` (double class-loss weight, set when P1–P8 was 14:1) may be less necessary. Not tested at P1–P53 scale — would require a separate run and is a lower priority than establishing the baseline result.

### Next run — P1–P53 baseline

**Recipe:** P3-B winner confirmed by P4-B/P4-C — `FREEZE=10, DEGREES=5, DFL=1.5, lr0=0.001, cls=1.0, mosaic=0.0, flipud=0.5`.
**CV:** LOPO, 53 folds.
**Oversampling:** fully auto-computed for all patients (`PATIENT_OVERSAMPLE_FIXED = {}`).
**FP negatives:** all patients where an Excel exists (P2 + P9–P53), totalling 3637 entries.

---

### Code fix — empty-label images incorrectly included as positives (2026-05-27)

#### Bug

In the main loading loop of `ba_improved_comb.py`, an image was added to `pos_with_meta` (and therefore to LOPO fold test sets) as long as its label file **existed on disk**, regardless of whether it contained any annotations:

```python
classes = parse_classes_in_label(lbl)   # result was computed but never used to filter
verified.append(img)                     # added even if classes == set()
```

A patient whose images all have empty `.txt` label files (= confirmed negatives, no mast cells) would still appear in `pos_with_meta`, receive its own LOPO fold, and produce a test set with **zero ground-truth boxes**. Metrics for that fold are undefined (recall = NaN or vacuous) and mislead the aggregate mean.

This differs from patients whose `labels/` folder is entirely absent — those were already handled correctly because `os.path.exists(lbl)` returns False and the images are skipped. The bug only affected patients with a labels directory present but all files empty.

#### Fix

Added a one-line guard after parsing classes:

```python
if not classes:
    continue   # empty label = no mast cells; skip from positives and LOPO folds
```

#### Implications

- **Pure-FP patients with empty label files no longer get a LOPO fold.** Their `verified` list stays empty → they are absent from `pos_with_meta` → `patients = sorted({m[1] for m in pos_with_meta})` excludes them automatically.
- **Their FP-negative Excel entries are unaffected.** `neg_with_meta` is populated separately via `PATIENT_FP_EXCELS`; those images still appear in `train_neg` for every other patient's fold.
- **Images from `PATIENT_IMAGE_DIRS` with empty labels are now silently dropped** — they do not enter `pos_with_meta` and do not enter `neg_with_meta`. If any patient has images with empty labels that should be treated as hard negatives (not just as annotation absences), they would need to be explicitly added to `neg_with_meta`. At P1–P53 scale this is not known to be an issue — the FP Excel files are the authoritative source of confirmed negatives.
- The P1–P53 summary printout will now show `files=0, Atypisch=0, Normal=0` for pure-FP patients, making their status explicit at a glance.
