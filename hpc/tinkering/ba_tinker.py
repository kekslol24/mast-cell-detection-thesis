"""
ba_improved_comb.py — Full P1–P8 corpus training with class-imbalance handling.

Strategy (see hpc/experiment_log.md, "P1–P8 Full Dataset — Class Imbalance Strategy"):

  1. PER-PATIENT IMAGE OVERSAMPLING (the core fix).
     Total dataset is 1124 Atypisch vs 78 Normal — a ≈ 14.4 : 1 imbalance.
     Normal cells live in only 4 of 8 patients (P5–P8); P6 and P8 are *pure-Normal*
     slides. Image-level oversampling of P5–P8 brings the effective per-epoch class
     ratio from 14:1 down to ≈ 2:1. We duplicate symlinks (with `_dupK` suffix), not
     annotations, so on-the-fly augmentation makes each duplicate visually distinct.

  2. LEAVE-ONE-PATIENT-OUT (LOPO) CV.
     With only 8 patients and Normal cells concentrated in 4 of them, stratified
     random k-fold leaks patient-specific texture between train and val and inflates
     metrics relative to true generalisation. LOPO yields 8 folds, one patient held
     out per fold — variance reflects real biological heterogeneity, not split luck.
     When P6 / P8 (pure-Normal) is held out, only Normal recall is measurable, which
     is exactly the metric of interest for non-SM generalisation.

  3. ON-THE-FLY AUGMENTATION instead of pre-applied.
     The DoE concluded that pre-applied (per-image, baked-to-disk) augmentation
     exhausts variability after epoch 1. We pass the originals to YOLO and let
     `augment=True` re-sample transforms each epoch. `mosaic=0.0` is mandatory:
     mosaic composites four images, which scrambles single-cell crops at 512 px.
     `flipud=0.5` is valid because bone-marrow cells have no canonical orientation.

  4. CLASS LOSS WEIGHT (`cls=1.0`, double the YOLO default).
     Marginal effect compared to oversampling; investing more model capacity in
     class discrimination relative to localisation. Secondary lever.

  5. NO BACKGROUND TILES, MINIMAL FP NEGATIVES.
     `BG_RATIO=0`, `FP_NEG_OVERSAMPLE=1`. The DoE showed BG tiles uniformly hurt
     and FP oversampling plateaus above ×1. We isolate the oversampling effect.

  6. CASE-INSENSITIVE PATH RESOLUTION.
     'Train' vs 'train' inconsistencies on disk for both `images/` and `labels/`
     subdirs are resolved transparently by `case_insensitive_resolve`.

  7. PER-CLASS RECALL is the primary clinical metric, not aggregate mAP50.
     Atypisch recall and Normal recall are recorded separately. WHO uses a 25 %
     Atypisch threshold for SM; under-detection of *either* class produces wrong
     ratios and false flags.
"""

import os
import gc
import glob
import yaml
import numpy as np
import pandas as pd
import multiprocessing as mp
import torch
from ultralytics import YOLO
from sklearn.model_selection import StratifiedKFold, train_test_split

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# CV strategy — LOPO is the recommended setting for the full P1–P8 corpus.
# Set LOPO_CV=False to fall back to stratified random k-fold (legacy behaviour).
LOPO_CV            = os.environ.get("LOPO_CV", "1") == "1"
N_SPLITS_FALLBACK  = 5   # only used when LOPO_CV=False

# Pretrained weights — domain-specific FV model is the established baseline (P2-H).
PRETRAINED_WEIGHTS = "yolo11n.pt"

# Fine-tuning hyperparameters from P2-H baseline.
EPOCHS_PER_FOLD    = 5000
PATIENCE           = 100
BATCH_SIZE         = 32
IMGSZ              = 512
LR0                = 0.001    # critical: 0.01 causes catastrophic forgetting
MOSAIC             = 0.0      # mandatory off for cell-tile data
FLIPUD             = 0.5      # cells have no canonical orientation
COS_LR             = True

# ---- TINKERING KNOBS (env-overridable, defaults match P3-A baseline) ----
# Each knob can be set per SLURM job via `--export=ALL,KNOB=VALUE`. Defaults
# reproduce the P3-A run (yolo_new_v1) exactly when no env var is set.
#
# When TUNER_CFG points at a yaml from `model.tune()`, that yaml becomes the
# hyperparameter base and only env-vars *explicitly set by the caller* override
# its values. This is the "validate the tuner output" mode (Step 3) and the
# "combine tuner + matrix winner" mode (Step 4) of the tinkering workflow.
TUNER_CFG          = os.environ.get("TUNER_CFG", "")

def _env_float(name): v = os.environ.get(name); return None if v is None else float(v)
def _env_int(name):   v = os.environ.get(name); return None if v is None else int(v)

# Sentinel-aware reads: None means "caller did not set it"
_FREEZE_ENV          = _env_int(  "FREEZE")
_CLS_LOSS_WEIGHT_ENV = _env_float("CLS_LOSS_WEIGHT")
_LABEL_SMOOTHING_ENV = _env_float("LABEL_SMOOTHING")
_DEGREES_ENV         = _env_float("DEGREES")
_DFL_ENV             = _env_float("DFL")
_HSV_V_ENV           = _env_float("HSV_V")
_MIXUP_ENV           = _env_float("MIXUP")

# Effective values for the default (no TUNER_CFG) path
FREEZE             = _FREEZE_ENV          if _FREEZE_ENV          is not None else 10
CLS_LOSS_WEIGHT    = _CLS_LOSS_WEIGHT_ENV if _CLS_LOSS_WEIGHT_ENV is not None else 1.0
LABEL_SMOOTHING    = _LABEL_SMOOTHING_ENV if _LABEL_SMOOTHING_ENV is not None else 0.0
DEGREES            = _DEGREES_ENV         if _DEGREES_ENV         is not None else 0.0
DFL                = _DFL_ENV             if _DFL_ENV             is not None else 1.5
HSV_V              = _HSV_V_ENV           if _HSV_V_ENV           is not None else 0.4
MIXUP              = _MIXUP_ENV           if _MIXUP_ENV           is not None else 0.0

# Negatives — DoE conclusion: BG tiles uniformly hurt, FP plateaus above ×1.
BG_RATIO           = int(os.environ.get("BG_RATIO", "1"))
FP_NEG_OVERSAMPLE  = int(os.environ.get("FP_NEG_OVERSAMPLE", "1"))

# Per-patient oversample factors — see strategy note in experiment_log.md.
# Targets effective per-epoch class ratio of roughly 2 : 1 (Atypisch : Normal).
PATIENT_OVERSAMPLE = {
    "P1": 1,    # SM-dominant, already over-represented
    "P2": 1,
    "P3": 1,
    "P4": 1,
    "P5": 8,    # 3 atyp / 18 norm — Normal-rich
    "P6": 15,   # 0 atyp /  6 norm — pure Normal, scarcest
    "P7": 8,    # 8 atyp / 12 norm — mixed
    "P8": 10,   # 0 atyp / 24 norm — pure Normal
}

# Data roots
BASE_DATA_PATH     = "/cfs/earth/scratch/vollmflo/BA/data"

# Per-phase image directories. Resolution is case-insensitive at runtime, so
# 'Train' vs 'train' on disk both work.
PATIENT_IMAGE_DIRS = {
    "P1": os.path.join(BASE_DATA_PATH, "old", "P1", "Pos_neg 12241515", "images", "Train"),
    "P2": os.path.join(BASE_DATA_PATH, "new", "P2", "images", "Train"),
    "P3": os.path.join(BASE_DATA_PATH, "new", "P3", "images", "Train"),
    "P4": os.path.join(BASE_DATA_PATH, "new", "P4", "images", "Train"),
    "P5": os.path.join(BASE_DATA_PATH, "new", "P5", "images", "Train"),
    "P6": os.path.join(BASE_DATA_PATH, "new", "P6", "images", "Train"),
    "P7": os.path.join(BASE_DATA_PATH, "new", "P7", "images", "Train"),
    "P8": os.path.join(BASE_DATA_PATH, "new", "P8", "images", "Train"),
}

# Per-patient label directory override.
# Default behaviour: replace 'images' → 'labels' in the image's path. That
# fails for P1, where the images live under `old/` but the annotations were
# moved to `new/`. Add an entry here to point a patient's labels somewhere
# that is NOT a sibling of its images dir. Other patients can stay unset and
# use the default swap.
PATIENT_LABEL_DIRS = {
    "P1": os.path.join(BASE_DATA_PATH, "new", "P1", "labels", "Train"),
}

# Negatives belong to specific patients — important for LOPO (don't leak the
# held-out patient's data into train via FP/BG).
P1_BG_DIR          = os.path.join(BASE_DATA_PATH, "old", "P1", "Negativ 12241515", "images", "train")
P2_FP_EXCEL        = os.path.join(BASE_DATA_PATH, "old", "P2", "Task 6_1224151_negative.xlsx")
P2_ROOT            = os.path.join(BASE_DATA_PATH, "new", "P2", "images", "Train")

CLASS_NAMES        = ["Atypisch", "Normal"]
NC                 = len(CLASS_NAMES)

# Job-scoped output dirs (avoid collisions with concurrent SLURM jobs — see the
# 'Concurrent job interference' note in experiment_log.md).
PROJECT_DIR        = f"./{os.environ.get('SLURM_JOB_NAME', 'local_run_comb')}"
_JOB_ID            = os.environ.get("SLURM_JOB_ID", "local")
TEMP_DIR           = os.path.abspath(f".cv_temp_{_JOB_ID}")

os.makedirs(TEMP_DIR, exist_ok=True)


# ==============================================================================
# CASE-INSENSITIVE PATH RESOLUTION
# ==============================================================================
# Some patient folders use 'Train' (capital T), others 'train'. Same for
# 'labels/Train' vs 'labels/train'. We walk path components and match
# case-insensitively where the literal path doesn't exist.
def case_insensitive_resolve(abs_path: str) -> str:
    """Return the on-disk path with components matched case-insensitively."""
    if os.path.exists(abs_path):
        return abs_path
    parts = abs_path.split(os.sep)
    if abs_path.startswith(os.sep):
        current = os.sep
        parts = parts[1:]
    else:
        current = "."
    for part in parts:
        if not part:
            continue
        if os.path.isdir(current):
            entries = os.listdir(current)
            match = (next((e for e in entries if e == part), None)
                     or next((e for e in entries if e.lower() == part.lower()), None))
            if match is not None:
                current = os.path.join(current, match)
                continue
        current = os.path.join(current, part)
    return current


def image_to_label_path(img_path: str, patient: str | None = None) -> str:
    """
    Map an image path to its YOLO label path.

    If `patient` is in PATIENT_LABEL_DIRS (e.g. P1, where images live in `old/`
    but labels live in `new/`), use that override. Otherwise replace 'images'
    → 'labels' in the image's directory and case-insensitively resolve any
    'Train' vs 'train' difference.
    """
    label_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"

    # Per-patient override — used when labels and images are on different roots.
    if patient and patient in PATIENT_LABEL_DIRS:
        override_dir = case_insensitive_resolve(PATIENT_LABEL_DIRS[patient])
        return os.path.join(override_dir, label_name)

    img_dir = os.path.dirname(img_path)

    # Default: swap 'images' → 'labels' in the dir path (case-insensitive on
    # the component name).
    parts     = img_dir.split(os.sep)
    new_parts = ["labels" if p.lower() == "images" else p for p in parts]
    naive_label_dir = os.sep.join(new_parts)

    candidate = os.path.join(naive_label_dir, label_name)
    if os.path.exists(candidate):
        return candidate

    # Fallback: case-insensitive resolution of the labels/<Split> dir.
    resolved_dir = case_insensitive_resolve(naive_label_dir)
    return os.path.join(resolved_dir, label_name)


def glob_images(dir_path: str) -> list[str]:
    """Glob jpeg/jpg in a directory with case-insensitive parent resolution."""
    resolved = case_insensitive_resolve(dir_path)
    found    = []
    for pat in ("*.jpeg", "*.jpg", "*.JPEG", "*.JPG"):
        found.extend(glob.glob(os.path.join(resolved, pat)))
    return sorted(set(found))


def parse_classes_in_label(label_path: str) -> set[int]:
    """Return set of class IDs present in a YOLO .txt label file."""
    if not label_path or not os.path.exists(label_path):
        return set()
    classes = set()
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 5:
                try:
                    classes.add(int(parts[0]))
                except ValueError:
                    pass
    return classes


# ==============================================================================
# TRAIN-WORKSPACE LINKING
# ==============================================================================
# Each fold gets its own workspace where train/val/test images and labels are
# symlinked. Oversampling happens here — same source image, multiple symlinks
# with `_dupK` suffix. YOLO sees them as distinct files, so on-the-fly
# augmentation produces a different transform per epoch per copy.
def link_one(src_img: str, fold_img_dir: str, fold_lbl_dir: str,
             oversample: int = 1, is_negative: bool = False,
             patient: str | None = None) -> list[str]:
    """Symlink one source image into the fold workspace `oversample` times.

    `patient` is forwarded to image_to_label_path so per-patient label
    overrides (e.g. P1 labels living under `new/` while images live under
    `old/`) are honoured when resolving the source label.
    """
    linked = []
    name, ext = os.path.splitext(os.path.basename(src_img))

    if is_negative:
        src_lbl = None  # negatives have no annotations; we write an empty .txt
    else:
        src_lbl = image_to_label_path(src_img, patient=patient)
        if not os.path.exists(src_lbl):
            return linked  # missing label — skip silently (caller already filtered)

    for k in range(oversample):
        suffix  = f"_dup{k}" if k > 0 else ""
        img_dst = os.path.join(fold_img_dir, f"{name}{suffix}{ext}")
        lbl_dst = os.path.join(fold_lbl_dir, f"{name}{suffix}.txt")

        if not os.path.exists(img_dst):
            os.symlink(src_img, img_dst)

        if not os.path.exists(lbl_dst):
            if src_lbl is not None:
                os.symlink(src_lbl, lbl_dst)
            else:
                open(lbl_dst, "w").close()  # empty label = confirmed background

        linked.append(img_dst)
    return linked


# ==============================================================================
# PER-FOLD TRAINING
# ==============================================================================
def train_fold(fold_params):
    (fold_idx, fold_label,
     train_pos, val_pos, test_pos,           # each: list[(img_path, patient)]
     train_neg) = fold_params                  # list[(img_path, patient)] — train only

    fold_workspace = os.path.join(TEMP_DIR, f"fold_{fold_idx}_workspace")
    fold_img_dir   = os.path.join(fold_workspace, "images")
    fold_lbl_dir   = os.path.join(fold_workspace, "labels")
    os.makedirs(fold_img_dir, exist_ok=True)
    os.makedirs(fold_lbl_dir, exist_ok=True)

    # Train: positives oversampled per patient + negatives at FP_NEG_OVERSAMPLE.
    train_paths = []
    for src, patient in train_pos:
        n = PATIENT_OVERSAMPLE.get(patient, 1)
        train_paths.extend(link_one(src, fold_img_dir, fold_lbl_dir,
                                    oversample=n, is_negative=False,
                                    patient=patient))
    for src, patient in train_neg:
        train_paths.extend(link_one(src, fold_img_dir, fold_lbl_dir,
                                    oversample=FP_NEG_OVERSAMPLE, is_negative=True,
                                    patient=patient))

    # Val and test: never oversampled (must reflect true distribution).
    val_paths  = []
    for src, patient in val_pos:
        val_paths.extend(link_one(src, fold_img_dir, fold_lbl_dir,
                                  oversample=1, is_negative=False,
                                  patient=patient))
    test_paths = []
    for src, patient in test_pos:
        test_paths.extend(link_one(src, fold_img_dir, fold_lbl_dir,
                                   oversample=1, is_negative=False,
                                   patient=patient))

    np.savetxt(os.path.join(fold_workspace, 'train.txt'), train_paths, fmt='%s')
    np.savetxt(os.path.join(fold_workspace, 'val.txt'),   val_paths,   fmt='%s')
    np.savetxt(os.path.join(fold_workspace, 'test.txt'),  test_paths,  fmt='%s')

    yaml_path = os.path.join(fold_workspace, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump({
            'path':  fold_workspace,
            'train': 'train.txt',
            'val':   'val.txt',
            'test':  'test.txt',
            'nc':    NC,
            'names': CLASS_NAMES,
        }, f)

    gpu_id = fold_idx % 2


    # `cfg=...` is conditionally used only when `TUNER_CFG` points at a yaml
    # produced by `tune_hyperpara.py`. The earlier P2-E failure mode came from
    # layering a from-scratch tuner output onto a fine-tune; here the tuner is
    # run against the production recipe so the yaml's regime matches deployment.
    train_kwargs = dict(
        data     = yaml_path,
        epochs   = EPOCHS_PER_FOLD,
        patience = PATIENCE,
        batch    = BATCH_SIZE,
        device   = gpu_id,
        project  = PROJECT_DIR,
        imgsz    = IMGSZ,
        name     = f"fold_{fold_idx + 1}_{fold_label}",
        workers  = 0,
        cache    = False,
        exist_ok = True,
        verbose  = False,
        # Hard production constants — kept explicit even with cfg= (mosaic=0
        # is mandatory for cell tiles; flipud/cos_lr/augment are domain facts).
        augment  = True,
        mosaic   = MOSAIC,
        flipud   = FLIPUD,
        cos_lr   = COS_LR,
    )

    if TUNER_CFG and os.path.exists(TUNER_CFG):
        # Yaml is the hyperparameter base; only explicitly env-set knobs
        # override (Step 3 = no overrides; Step 4 = matrix winner overrides).
        train_kwargs['cfg'] = TUNER_CFG
        if _FREEZE_ENV          is not None: train_kwargs['freeze']          = _FREEZE_ENV
        if _CLS_LOSS_WEIGHT_ENV is not None: train_kwargs['cls']             = _CLS_LOSS_WEIGHT_ENV
        if _LABEL_SMOOTHING_ENV is not None: train_kwargs['label_smoothing'] = _LABEL_SMOOTHING_ENV
        if _DEGREES_ENV         is not None: train_kwargs['degrees']         = _DEGREES_ENV
        if _DFL_ENV             is not None: train_kwargs['dfl']             = _DFL_ENV
        if _HSV_V_ENV           is not None: train_kwargs['hsv_v']           = _HSV_V_ENV
        if _MIXUP_ENV           is not None: train_kwargs['mixup']           = _MIXUP_ENV
    else:
        # No tuner yaml — P3-A baseline + any env-set matrix knobs.
        train_kwargs.update(
            lr0             = LR0,
            freeze          = FREEZE,
            cls             = CLS_LOSS_WEIGHT,
            dfl             = DFL,
            label_smoothing = LABEL_SMOOTHING,
            degrees         = DEGREES,
            hsv_v           = HSV_V,
            mixup           = MIXUP,
        )

    model = YOLO(PRETRAINED_WEIGHTS)
    model.train(**train_kwargs)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    best_weights = os.path.join(PROJECT_DIR, f"fold_{fold_idx + 1}_{fold_label}",
                                'weights', 'best.pt')
    val_model    = YOLO(best_weights)

    metrics_test  = val_model.val(data=yaml_path, split='test',  verbose=False,
                                  workers=0, device=gpu_id, batch=BATCH_SIZE)
    metrics_val   = val_model.val(data=yaml_path, split='val',   verbose=False,
                                  workers=0, device=gpu_id, batch=BATCH_SIZE)
    metrics_train = val_model.val(data=yaml_path, split='train', verbose=False,
                                  workers=0, device=gpu_id, batch=BATCH_SIZE)

    # Per-class recall is the clinical metric. metrics.box.r is shape (nc,) but
    # may be empty/short when a class has no ground truth in the split (common
    # for LOPO P6/P8 — pure-Normal — where Atypisch GT is absent).
    def per_class(metrics, idx):
        try:
            arr = metrics.box.r
            return float(arr[idx]) if idx < len(arr) else float('nan')
        except (AttributeError, IndexError, TypeError):
            return float('nan')

    return {
        'Fold':                fold_idx + 1,
        'Holdout':             fold_label,
        'Train_mAP50-95':      metrics_train.box.map,
        'Train_mAP50':         metrics_train.box.map50,
        'Train_Precision':     metrics_train.box.mp,
        'Train_Recall':        metrics_train.box.mr,
        'Val_mAP50-95':        metrics_val.box.map,
        'Val_mAP50':           metrics_val.box.map50,
        'Val_Precision':       metrics_val.box.mp,
        'Val_Recall':          metrics_val.box.mr,
        'Test_mAP50-95':       metrics_test.box.map,
        'Test_mAP50':          metrics_test.box.map50,
        'Test_Precision':      metrics_test.box.mp,
        'Test_Recall':         metrics_test.box.mr,
        'Test_Recall_Atypisch': per_class(metrics_test, 0),
        'Test_Recall_Normal':   per_class(metrics_test, 1),
        'Val_Recall_Atypisch':  per_class(metrics_val, 0),
        'Val_Recall_Normal':    per_class(metrics_val, 1),
    }


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Load annotated positives from all 8 patients, with patient tag
    # ------------------------------------------------------------------
    pos_with_meta: list[tuple[str, str]] = []   # (img_path, patient)
    summary: dict[str, dict] = {}

    for patient, img_dir in PATIENT_IMAGE_DIRS.items():
        candidates = glob_images(img_dir)
        verified, atyp_count, norm_count = [], 0, 0
        for img in candidates:
            lbl = image_to_label_path(img, patient=patient)
            if not os.path.exists(lbl):
                continue
            classes = parse_classes_in_label(lbl)
            verified.append(img)
            atyp_count += sum(1 for _ in open(lbl) if _.split() and _.split()[0] == "0")
            norm_count += sum(1 for _ in open(lbl) if _.split() and _.split()[0] == "1")

        for img in verified:
            pos_with_meta.append((img, patient))

        summary[patient] = {
            "files":       len(verified),
            "atypisch":    atyp_count,
            "normal":      norm_count,
            "oversample":  PATIENT_OVERSAMPLE.get(patient, 1),
        }

    print("\n=== Per-patient annotation summary ===")
    print(f"{'Patient':<8}{'Files':>8}{'Atypisch':>10}{'Normal':>10}{'Oversample':>12}")
    total_files = total_atyp = total_norm = 0
    for p, s in summary.items():
        print(f"{p:<8}{s['files']:>8}{s['atypisch']:>10}{s['normal']:>10}{s['oversample']:>12}")
        total_files += s['files']
        total_atyp  += s['atypisch']
        total_norm  += s['normal']
    print(f"{'TOTAL':<8}{total_files:>8}{total_atyp:>10}{total_norm:>10}")
    if total_norm > 0:
        print(f"Atypisch:Normal ratio = {total_atyp / total_norm:.1f} : 1")

    # ------------------------------------------------------------------
    # 2. Load P1 backgrounds and P2 FP negatives (tagged by patient)
    # ------------------------------------------------------------------
    neg_with_meta: list[tuple[str, str]] = []

    if BG_RATIO > 0:
        bg_paths = glob_images(P1_BG_DIR)
        # BG_RATIO sets how many BG tiles per annotated positive (subsample).
        rng     = np.random.default_rng(seed=42)
        target  = BG_RATIO * len(pos_with_meta)
        if target < len(bg_paths):
            chosen = rng.choice(bg_paths, size=target, replace=False)
            bg_paths = list(chosen)
        for p in bg_paths:
            neg_with_meta.append((p, "P1"))
        print(f"\nP1 backgrounds (BG_RATIO={BG_RATIO}): {len(bg_paths)}")
    else:
        print("\nP1 backgrounds disabled (BG_RATIO=0) — DoE conclusion.")

    if FP_NEG_OVERSAMPLE > 0 and os.path.exists(P2_FP_EXCEL):
        print("Indexing P2 disk for FP resolution...")
        p2_disk_dir = case_insensitive_resolve(P2_ROOT)
        p2_disk     = glob.glob(os.path.join(p2_disk_dir, "**/*.jpeg"), recursive=True)
        p2_disk    += glob.glob(os.path.join(p2_disk_dir, "**/*.JPEG"), recursive=True)
        p2_index    = {os.path.basename(p): p for p in p2_disk}

        fp_df        = pd.read_excel(P2_FP_EXCEL, header=None)
        fp_filenames = fp_df[0].dropna().tolist()
        fp_paths, fp_missing = [], []
        for fn in fp_filenames:
            basename = os.path.basename(str(fn))
            if basename in p2_index:
                fp_paths.append(p2_index[basename])
            else:
                fp_missing.append(fn)

        for p in fp_paths:
            neg_with_meta.append((p, "P2"))

        print(f"P2 FP negatives (oversample={FP_NEG_OVERSAMPLE}): {len(fp_paths)} resolved")
        if fp_missing:
            print(f"  not found: {len(fp_missing)} (first 5: {fp_missing[:5]})")

    # ------------------------------------------------------------------
    # 3. Build CV folds
    # ------------------------------------------------------------------
    if LOPO_CV:
        # One fold per patient. The test set is everything from the held-out
        # patient. Train set is the union of the other patients (positives) plus
        # all negatives whose patient is also in train.
        patients = sorted({m[1] for m in pos_with_meta})
        fold_tasks = []
        print(f"\n=== Leave-One-Patient-Out CV — {len(patients)} folds ===")
        for fold_idx, holdout in enumerate(patients):
            train_pos_full = [m for m in pos_with_meta if m[1] != holdout]
            test_pos       = [m for m in pos_with_meta if m[1] == holdout]
            train_neg      = [m for m in neg_with_meta if m[1] != holdout]

            # Carve out a small in-train val set (15%) stratified by patient
            # so each train patient appears in val proportionally.
            X = np.array([m[0] for m in train_pos_full])
            y = np.array([m[1] for m in train_pos_full])
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=0.15, stratify=y, random_state=42,
            )
            train_pos = list(zip(X_tr.tolist(), y_tr.tolist()))
            val_pos   = list(zip(X_va.tolist(), y_va.tolist()))

            fold_tasks.append((fold_idx, holdout,
                               train_pos, val_pos, test_pos, train_neg))
            print(f"  Fold {fold_idx+1}: holdout={holdout}, "
                  f"train={len(train_pos)} pos / {len(train_neg)} neg, "
                  f"val={len(val_pos)}, test={len(test_pos)}")
    else:
        # Legacy stratified k-fold on positive/negative class. Patient
        # identities collapse into 'positive' (0) vs 'negative' (1) strata.
        all_paths = [m[0] for m in pos_with_meta] + [m[0] for m in neg_with_meta]
        all_pat   = [m[1] for m in pos_with_meta] + [m[1] for m in neg_with_meta]
        all_strat = ([0] * len(pos_with_meta) + [1] * len(neg_with_meta))
        X_all = np.array(all_paths)
        y_all = np.array(all_strat)
        skf = StratifiedKFold(n_splits=N_SPLITS_FALLBACK, shuffle=True, random_state=42)
        fold_tasks = []
        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X_all, y_all)):
            tr_pos = [(all_paths[i], all_pat[i]) for i in tr_idx if all_strat[i] == 0]
            te_pos = [(all_paths[i], all_pat[i]) for i in te_idx if all_strat[i] == 0]
            tr_neg = [(all_paths[i], all_pat[i]) for i in tr_idx if all_strat[i] == 1]

            X_tr, X_va, y_tr, y_va = train_test_split(
                np.array([m[0] for m in tr_pos]),
                np.array([m[1] for m in tr_pos]),
                test_size=0.15, random_state=42,
            )
            train_pos = list(zip(X_tr.tolist(), y_tr.tolist()))
            val_pos   = list(zip(X_va.tolist(), y_va.tolist()))
            fold_tasks.append((fold_idx, f"fold{fold_idx+1}",
                               train_pos, val_pos, te_pos, tr_neg))
        print(f"\n=== Stratified {N_SPLITS_FALLBACK}-fold CV (legacy) ===")

    # ------------------------------------------------------------------
    # 4. Run training in parallel — one fold per process, two GPUs
    #    interleaved. Note: many simultaneous folds can over-subscribe
    #    a single GPU. With LOPO (8 folds), 4 land on each GPU; if VRAM
    #    is tight, lower MAX_PARALLEL via the env var.
    # ------------------------------------------------------------------
    n_folds      = len(fold_tasks)
    max_parallel = int(os.environ.get("MAX_PARALLEL", str(n_folds)))
    print(f"\nLaunching {n_folds} folds, {max_parallel} in parallel.")
    print(f"Project dir: {PROJECT_DIR}\n")

    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=max_parallel) as pool:
        final_results = pool.map(train_fold, fold_tasks)

    # ------------------------------------------------------------------
    # 5. Results — aggregate and per-class recall
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(final_results)
    results_df.to_csv(os.path.join(PROJECT_DIR, "fold_results.csv"), index=False)

    for split in ['Train', 'Val', 'Test']:
        cols = ['Fold', 'Holdout'] + [c for c in results_df.columns if c.startswith(split)]
        df   = results_df[cols].copy()
        df.columns = (['Fold', 'Holdout']
                      + [c.replace(f'{split}_', '') for c in df.columns
                         if c not in ('Fold', 'Holdout')])
        print("\n" + "=" * 70)
        print(f"CV RESULTS — {split.upper()}")
        print("=" * 70)
        print(df.to_string(index=False))
        if 'mAP50-95' in df.columns:
            print(f"\nMean mAP50-95 : {df['mAP50-95'].mean():.4f} +/- {df['mAP50-95'].std():.4f}")
            print(f"Mean Precision: {df['Precision'].mean():.4f} +/- {df['Precision'].std():.4f}")
            print(f"Mean Recall   : {df['Recall'].mean():.4f} +/- {df['Recall'].std():.4f}")
        if split == 'Test' and 'Recall_Atypisch' in df.columns:
            ra = df['Recall_Atypisch'].dropna()
            rn = df['Recall_Normal'].dropna()
            if len(ra):
                print(f"Mean Recall Atypisch (folds with Atyp GT, n={len(ra)}): "
                      f"{ra.mean():.4f} +/- {ra.std():.4f}")
            if len(rn):
                print(f"Mean Recall Normal   (folds with Norm GT, n={len(rn)}): "
                      f"{rn.mean():.4f} +/- {rn.std():.4f}")
