# Mast Cell Detection — Bachelor Thesis

Automated detection and morphological classification of mast cells (Atypical / Normal) in bone marrow aspirate images using YOLO11n (Ultralytics).

**Author:** Florian Vollmer (ZHAW Applied Digital Life Science)  
**Clinical partner:** University Hospital Zurich (USZ)  
**Supervisor:** Dr. Stefan Glüge, Cross-corrector: Prof. Dr. Robert Vorburger  
**Submission:** 2026-07-02

[**MastCellDetector**](https://github.com/kekslol24/MastCellDetector) is the tool produced within this thesis.

## What this repo is

This is the full project repository for a bachelor thesis that builds a deep learning pipeline to help clinicians count atypical vs. normal mast cells in bone marrow aspirate microscopy images. A WHO criterion for systemic mastocytosis requires >=25% atypical mast cells, which is currently assessed by laborious manual counting.

The pipeline covers: data management (53 patients, YOLO-format annotations), HPC training (SLURM, YOLO11n fine-tuning, 5-fold grouped cross-validation), post-hoc evaluation, mass inference, and a Gradio web application for clinical use. The thesis is written in Quarto and lives in the same repo.

---

## Repository Structure

```
BA/
├── app/
├── data/
├── hpc/
│   ├── Data/
│   ├── jobs/
│   │   ├── slurm_jobs&projects/
│   │   ├── slurm_tinkering_jobs&projects/
│   │   └── failed_jobs/
│   ├── mass_inference_results/
│   └── python_files/
├── model_weights/
├── Tuning_Dataset/
├── Writing/
│   └── extensions/
│       └── figures/
├── environment.yaml
├── requirements.txt
├── testing.ipynb
└── notizen.md
```

---

## `app/`

Gradio web UI for clinical use. Start it with `gradio app.py` (opens on `http://127.0.0.1:7860`). Contains two tabs:

- **Analysis tab:** batch upload images, `conf=0.58`, draws bounding boxes with class labels, shows confidence statistics
- **Research tab:** lower threshold `conf=0.25` for exploratory inspection

`run_gui.sh` is a convenience shell wrapper to launch the app on the hpc.

---

## `data/`

Local mirror of the patient dataset. On the HPC the canonical copy lives at `/cfs/earth/scratch/vollmflo/BA/data/`. The local `data/` directory has the same layout but images are not stored online because of storage capacity reasons.

### `data/old/`

Contains the two earliest patients in their original CVAT export format, before the standardised layout was established.

```
data/old/
├── P1/
│   ├── Negativ 12241515/          # Negative tiles only (background, no mast cells)
│   │   └── data.yaml
│   └── Pos_neg 12241515/          # Mixed positive + negative tiles (used for training)
│       └── data.yaml
└── P2/
    ├── 1224151atypisch_normal/    # Atypisch + Normal annotated tiles
    │   └── data.yaml
    ├── Task 6_1224151_negative.xlsx   # FP-negative tile list
    └── Task6_re-edit_1224151/     # Re-annotated version (used in pipeline)
        └── data.yaml
```

P1 uses the `Pos_neg 12241515/` subfolder; `patient_dir.py` points directly to its `images/Train/` path. P2's re-edited version is the one referenced in `patient_dir.py`.

### `data/new/`

All patients P1–P53 in the standardised per-patient layout. Despite the folder being named `new/`, it also contains a re-organised copy of P1.

```
data/new/
├── data.yaml              # Global dataset descriptor
├── P1/
│   └── data.yaml          # Per-patient dataset descriptor
├── P2/
│   └── data.yaml
├── P9/
│   ├── Task39/            # Original CVAT export folder (early annotation pass)
│   │   ├── Train.txt
│   │   ├── data.yaml
│   │   └── labels/
│   ├── Task39_V2.xlsx     # FP-negative tile list (V2 pass)
│   └── data.yaml
├── P10/
│   ├── Task26_V2.xlsx     # FP-negative tile list
│   └── data.yaml
...
└── P53/
    ├── Task75_V2.xlsx
    └── data.yaml
```

Each patient folder contains:
- `data.yaml`: Ultralytics dataset descriptor pointing to the `images/` and `labels/` subdirectories (these are on the HPC; only metadata is tracked locally)
- `TaskXX_V2.xlsx` (most patients): spreadsheet of false-positive negative tiles exported from CVAT, used by `patient_dir.py` via `PATIENT_FP_EXCELS` to build hard-negative training samples

Patients without an xlsx file (P1, P2, P3, P4, P5, P6, P7, P8) either have no FP-negative tiles or their negatives are embedded in the annotation export.

---

## `hpc/`

All training-related code, job archives, and outputs. The local `hpc/` directory is synced to `/cfs/earth/scratch/vollmflo/BA/hpc/` on the ZHAW HPC cluster before every training run.

### `hpc/python_files/`

All Python scripts for training, evaluation, and utilities.

| Script | Status | Purpose |
|--------|--------|---------|
| `ba_improved_comb.py` | **Active (main)** | Full P1–P53 training. Implements grouped CV, LOPO, and legacy k-fold via `CV_MODE` env var. Also imported as a library by eval and retrain scripts. |
| `ba_improved_comb_CV.py` | Active (variant) | Grouped-CV-only variant with minor config differences. |
| `patient_dir.py` | **Active (registry)** | Central registry: `PATIENT_IMAGE_DIRS` and `PATIENT_FP_EXCELS` for all 53 patients. This is the single source of truth for patient paths. Update here when adding a new patient. |
| `eval_lopo_folds.py` | Active | Post-hoc evaluation: rebuilds fold splits from `ba_improved_comb` logic and runs `model.val()` on saved `best.pt` weights. Used when training finished but `fold_results.csv` was not written (e.g., wall-time kill). Controlled by `SLURM_JOB_NAME` env var. |
| `retrain_single_fold.py` | Active (caution) | Retrains exactly one fold in isolation. Use after wall-time kills. |
| `inference.py` | Active | Mass inference on a ZIP archive of slide images. Outputs `confidence_distribution.csv` and annotated images. Filters out known training images by filename. |
| `ba_g1g4_cv.py` | Experimental | CV variant restricted to groups G1 and G4 for rapid iteration. |
| `hyperpara_tune.py` | Archived | Ran `model.tune()` for hyperparameter search. Results were harmful when combined with domain-pretrained weights; do not re-enable. |
| `hyperpara_test.py` | Archived | Evaluated specific hyperparameter combinations from the tuner output. |
| `ba_improved_P2.py` | Archived | Main script for early phases (P1/P2 corpus, stratified k-fold). No longer used. |
| `ba_improved_P2_fixed.py` | Archived | Bug-fixed version of the above. |
| `ba_improved.py` | Archived | Earlier single-patient training variant. |
| `ba_testrun.py` / `ba_testrun_augment.py` | Archived | Quick smoke-test scripts from the early development phase. |
| `eval_folds.py` / `eval_folds_v4.py` | Archived | Older evaluation scripts before the per-class indexing bug was fixed. |
| `create_labels_for_neg.py` | Utility | Creates empty YOLO label files for FP-negative tiles (required format for background-only images). |
| `filter_tiles.py` | Utility | Filters tile images based on annotation criteria. |
| `model_testing.py` | Utility | Ad-hoc model testing and inspection. |
| `train_val_split.py` | Utility | Manual train/val split helper. |
| `yolo_test.py` / `yolo_test_multiprocessing.py` | Utility | Early YOLO inference test scripts. |

### `hpc/jobs/`

SLURM job scripts for submitting work to the cluster.

| File | Purpose |
|------|---------|
| `run_training.sh` | Primary submit script. Edit this to switch between training and evaluation modes (training line is commented out by default). Set `--job-name` to control the output directory. |
| `run_eval_lopo.sh` | Submit `eval_lopo_folds.py` as a SLURM job. Set `SLURM_JOB_NAME` to point at the project directory with saved weights. |
| `run_inference.sh` | Submit `inference.py` for mass inference. |
| `run_hyperpara_tune.sh` / `run_hyperpara_test.sh` | Archived: hyperparameter search jobs. |
| `submit_grid.sh` | Submits a grid of tinkering jobs in parallel (used for the P3-B tinkering matrix). |
| `remove_slurm.sh` | Cleans up old SLURM output files. |

#### `hpc/jobs/slurm_jobs&projects/`

Archive of all main training and evaluation SLURM jobs. Each subfolder is named `Slurm-XXXXXX (description)` and contains the SLURM script used, stdout/stderr logs, and (for some jobs) the training output symlinked or copied under a `yolo_runs_*/` subdirectory.

The `projects_dir/` subfolder mirrors the named project output directories from the HPC (e.g., `yolo_new_v4_freeze10_re_run_CV/`). These contain the `fold_N_<patient>/weights/best.pt` files and `fold_results.csv`.

Notable project directories:
- `yolo_new_v3_freeze10/` — Phase 4-B: 15-fold LOPO on P1–P15, the benchmark for per-fold breakdown in the thesis
- `yolo_new_v4_freeze10_re_run_CV/` — Phase 5-B: 5-group CV on all 35 annotated patients, the primary thesis result
- `yolo_new_v4_retrain_fold17/` — Single-fold retrain for P37 (Fold 17 wall-time casualty)

#### `hpc/jobs/slurm_tinkering_jobs&projects/`

Archive of the P3-B hyperparameter tinkering matrix: a full-factorial grid over label smoothing (ls), rotation degrees (deg), DFL weight, and backbone freeze. Each `Slurm-XXXXXX (tinker_*)` folder holds one combination. Corresponding output in `projects_dir/tinker_*/`. The winning configuration (deg=5, dfl=1.5, freeze=10, ls=0.0) became the production recipe.

#### `hpc/jobs/failed_jobs/`

Logs from SLURM jobs that failed outright (as opposed to hitting the wall-time limit). Kept for debugging reference.

### `hpc/Data/`

Local data artifacts from the earliest development phase.

```
hpc/Data/
└── Pos_neg 12241515/
    └── augmentation_check/        # Side-by-side comparison images verifying
                                   # that on-the-fly augmentation produces visually
                                   # distinct copies from the same source tile
```

### `hpc/mass_inference_results/`

Output of `inference.py` runs on the P1 slide image archive.

```
hpc/mass_inference_results/
├── run_1/
│   ├── confidence_distribution.csv   # Raw confidence scores for all detections
│   └── confidence_histogram.png      # Distribution plot
└── run_final/
    ├── confidence_distribution.csv
    ├── confidence_histogram.png
    └── confidence_histogram_no_axv.png   # Clean version used in the thesis figure
```

### `hpc/experiment_log.md`

The authoritative record of every SLURM run: configuration, per-fold test metrics, reasoning, conclusions, and known infrastructure issues (concurrent-job collisions, per-class indexing bug, inode exhaustion). Update this after every run. It is the primary source for the Methods/Results section of the thesis. Also rendered as `experiment_log.html` for easier reading.

### `hpc/THESIS_ANALYSIS.md`

Supplementary analysis document used during thesis writing.

---

## `model_weights/`

Pretrained and fine-tuned model weights.

| File | Description |
|------|-------------|
| `yolo11n.pt` | Base YOLO11n weights from Ultralytics (2.6M parameters) |
| `yolo26n.pt` | Alternative base (YOLO26n variant, explored but not used in production) |
| `DL_Modell_FV.pt` | v1 domain-pretrained weights: fine-tuned from `yolo11n.pt` on P1 data only. Used as starting point for all subsequent training. Changing the number of classes from 1 to 2 causes the detection head (cv3) to reinitialise. |
| `DL_Modell_FV_v2.pt` | v2: fine-tuned on P1–P15 corpus (15 patients). Used for annotating P16–P53 in CVAT. |
| `DL_Modell_FV_v3.pt` | v3: fine-tuned on the full P1–P53 corpus (35 annotated patients). Current production weights. |


---

## `Tuning_Dataset/`

A small curated dataset used during hyperparameter tuning runs (`hyperpara_tune.py`, `hyperpara_test.py`).

```
Tuning_Dataset/
├── data.yaml
└── labels/
    ├── train.cache
    └── val.cache
```

The image files are not tracked locally (stored on HPC). The cache files are Ultralytics label-index caches from previous tuning runs.

---

## `Writing/`

All thesis writing artifacts.

```
Writing/
└── extensions/
    ├── BA_FV.qmd              
    ├── BA_FV.pdf
    ├── BA_FV_v2.qmd           
    ├── BA_FV_v2.pdf
    ├── BA_FV_v3.qmd           # Active thesis file
    ├── BA_FV_v3.pdf           # Latest compiled PDF
    ├── references.bib         # Active bibliography (Zotero-managed)
    ├── ieee.csl               # IEEE citation style
    ├── cover.png              # Thesis cover image
    ├── for-devs.qmd           # Developer notes for the Quarto extension
    ├── _extensions/           # zhaw-lsfm-typst Quarto extension
    └── figures/               # All figures embedded in the thesis
```
