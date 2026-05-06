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

### Grid search infrastructure (added 2026-05-06)

`BG_RATIO` and `FP_NEG_OVERSAMPLE` are now read from environment variables with fallback defaults. `submit_grid.sh` submits all 12 combinations (fp∈{1,2,3} × bgr∈{0,1,2,3}) as independent SLURM jobs in one command. `PROJECT_DIR` auto-names from the job name, so results land in `train_p2_fp1_bgr0/`, etc.

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

### Run P2-H — Lower LR + Backbone Freeze, No Background (planned)
**SLURM:** next submission  
**Status:** pending

#### Motivation
Root cause analysis across all P2 runs points to `lr0=0.01` (YOLO default, calibrated for
training from scratch) being too high when fine-tuning from `DL_Modell_FV.pt`. A 10× reduction
(`lr0=0.001`) keeps the pretrained feature representations intact while only the detection heads
adapt significantly. Freezing the backbone (first 10 layers) additionally prevents the feature
extractor from being overwritten during the early aggressive phase of training.

BG_RATIO is set to 0 in this run to isolate the effect of the LR and freeze changes. If the
folds become stable, backgrounds can be reintroduced at ratio 1 in a subsequent run.

#### Changes from P2-G
| Parameter      | P2-G          | P2-H              |
|----------------|---------------|-------------------|
| `lr0`          | 0.01 (default)| **0.001**         |
| `lrf`          | 0.01 (default)| **0.01** (final lr = 1e-5) |
| `freeze`       | —             | **10** (first 10 layers frozen) |
| `BG_RATIO`     | 2             | **0** (removed to isolate effect) |
| `FP_NEG_OVERSAMPLE` | 1       | 1 (unchanged)     |
| `cfg`          | not used      | not used          |

#### Expected outcome
All 5 folds should converge rather than peak and collapse, bringing std below ~0.06.
Mean test mAP50 expected to remain ≥0.59 (matching P2-D) with more consistent recall.

#### Results
*(to be filled in after SLURM job completes)*

---

## Summary Table — Test Set Results (all P2 runs)

| Run   | SLURM     | Base Model    | BG_RATIO | lr0   | freeze | mAP50 (mean±std)   | Recall (mean) |
|-------|-----------|---------------|----------|-------|--------|--------------------|---------------|
| P2-A  | 319399    | yolo11n       | 0        | 0.01  | —      | 0.580 ± 0.172      | 0.617         |
| P2-B  | 319400    | yolo11n (mod) | 0        | 0.01  | —      | 0.614 ± 0.155      | 0.667         |
| P2-C  | 319401    | DL_Modell_FV  | 0        | 0.01  | —      | 0.469 ± 0.035      | 0.397         |
| P2-D  | 319416    | DL_Modell_FV (mod) | 0   | 0.01  | —      | 0.590 ± 0.090      | 0.655         |
| P2-E  | 319536    | DL_Modell_FV + tuner cfg | 0 | 0.01 | —   | 0.448 ± 0.007 ❌  | 0.413         |
| P2-F  | 319538    | DL_Modell_FV + mod + tuner cfg | 0 | 0.01 | — | 0.508 ± 0.057 | 0.571      |
| P2-G  | 319934    | DL_Modell_FV  | 2        | 0.01  | —      | 0.530 ± 0.090      | 0.484         |
| **P2-H** | pending | DL_Modell_FV | 0       | **0.001** | **10** | *pending*      | *pending*     |

---

## Planned: Run P2-I — Reintroduce Background at BG_RATIO=1
*(contingent on P2-H being stable)*

If P2-H confirms that `lr0=0.001` + `freeze=10` eliminates fold-collapse, backgrounds will be
reintroduced at 1:1 ratio (one background tile per annotated image) to test whether they add
precision benefit without destabilising training.

If test-set precision is still a concern after P2-H, `FP_NEG_OVERSAMPLE` will be increased from
1 to 2–3. The FP negatives are high-quality hard negatives (the model's known failure cases) and
increasing their weight is a targeted way to reduce false positives without adding random noise.
