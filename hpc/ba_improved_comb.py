"""
ba_improved_comb.py — Full P1–P15 corpus training with class-imbalance handling.

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
FREEZE             = 10       # freeze backbone, fine-tune neck + head only
CLS_LOSS_WEIGHT    = 1.0      # double default 0.5 to invest more in class discrimination
MOSAIC             = 0.0      # mandatory off for cell-tile data
FLIPUD             = 0.5      # cells have no canonical orientation
COS_LR             = True

# Negatives — DoE conclusion: BG tiles uniformly hurt, FP plateaus above ×1.
BG_RATIO           = int(os.environ.get("BG_RATIO", "1"))
FP_NEG_OVERSAMPLE  = int(os.environ.get("FP_NEG_OVERSAMPLE", "1"))

# Manually validated oversample factors for P1–P8 (see experiment_log.md).
# P9–P15 are NOT listed here — their factors are computed automatically at
# runtime by compute_oversample_factors() to target a ~2:1 Atypisch:Normal
# ratio. Add a patient here to pin its factor and skip auto-computation.
PATIENT_OVERSAMPLE_FIXED = {
    "P1":  1,    # SM-dominant, already over-represented
    "P2":  1,
    "P3":  1,
    "P4":  1,
    "P5":  8,    # 3 atyp / 18 norm — Normal-rich
    "P6":  15,   # 0 atyp /  6 norm — pure Normal, scarcest
    "P7":  8,    # 8 atyp / 12 norm — mixed
    "P8":  10,   # 0 atyp / 24 norm — pure Normal
}
OVERSAMPLE_TARGET_RATIO = 2.0   # global effective Atypisch:Normal goal
OVERSAMPLE_CAP          = 25    # hard ceiling to avoid extreme duplication

# Data roots
BASE_DATA_PATH     = "/cfs/earth/scratch/vollmflo/BA/data"

# Per-phase image directories for patients whose images live in a flat
# directory (P1–P8). Resolution is case-insensitive at runtime.
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

# P9–P15 use a YOLO-style Train.txt listing absolute image paths (no flat
# images/Train/ directory). Entries here are read line-by-line at runtime.
PATIENT_IMAGE_TXTS = {
    "P9":  os.path.join(BASE_DATA_PATH, "new", "P9",  "Train.txt"),
    "P10": os.path.join(BASE_DATA_PATH, "new", "P10", "Train.txt"),
    "P11": os.path.join(BASE_DATA_PATH, "new", "P11", "Train.txt"),
    "P12": os.path.join(BASE_DATA_PATH, "new", "P12", "Train.txt"),
    "P13": os.path.join(BASE_DATA_PATH, "new", "P13", "Train.txt"),
    "P14": os.path.join(BASE_DATA_PATH, "new", "P14", "Train.txt"),
    "P15": os.path.join(BASE_DATA_PATH, "new", "P15", "Train.txt"),
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

# FP-negative Excel files per patient. Each Excel contains filenames of tiles
# that the model incorrectly detected (confirmed false positives). They are
# resolved against the patient's image index at runtime and added to train-only
# splits as empty-label negatives.
PATIENT_FP_EXCELS = {
    "P2":  os.path.join(BASE_DATA_PATH, "old",  "P2",  "Task 6_1224151_negative.xlsx"),
    "P9":  os.path.join(BASE_DATA_PATH, "new",  "P9",  "Task39_V2.xlsx"),
    "P10": os.path.join(BASE_DATA_PATH, "new",  "P10", "Task26_V2.xlsx"),
    "P11": os.path.join(BASE_DATA_PATH, "new",  "P11", "Task27_V2.xlsx"),
    "P12": os.path.join(BASE_DATA_PATH, "new",  "P12", "Task21_V2.xlsx"),
    "P13": os.path.join(BASE_DATA_PATH, "new",  "P13", "Task24_V2.xlsx"),
    "P14": os.path.join(BASE_DATA_PATH, "new",  "P14", "Task25_V2.xlsx"),
    "P15": os.path.join(BASE_DATA_PATH, "new",  "P15", "Task23_V2.xlsx"),
}

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


def load_patient_images(patient: str) -> list[str]:
    """Return all annotated image paths for a patient.

    P1–P8 are resolved by globbing their images/Train directory. P9–P15 list
    their images in a Train.txt file (absolute paths, one per line).
    """
    if patient in PATIENT_IMAGE_DIRS:
        return glob_images(PATIENT_IMAGE_DIRS[patient])
    txt = PATIENT_IMAGE_TXTS.get(patient)
    if txt and os.path.exists(txt):
        with open(txt) as f:
            return [ln.strip() for ln in f if ln.strip()]
    return []


def build_disk_index(patient: str) -> dict[str, str]:
    """Return {basename → abspath} for all images of a patient.

    Used to resolve FP-negative filenames (from Excel) to full disk paths.
    For dir-based patients we glob; for txt-based patients we parse Train.txt
    (faster and avoids an additional directory walk).
    """
    paths = load_patient_images(patient)
    return {os.path.basename(p): p for p in paths}


def compute_oversample_factors(
    summary: dict[str, dict],
    fixed: dict[str, int],
    target_ratio: float = OVERSAMPLE_TARGET_RATIO,
    cap: int = OVERSAMPLE_CAP,
) -> dict[str, int]:
    """Compute per-patient oversample factors targeting a global Atypisch:Normal ratio.

    Patients in `fixed` keep their pinned value unchanged. All others are
    processed in descending Normal-heaviness order (most imbalanced first) so
    each factor is solved against the already-assigned state of prior patients.

    For each auto patient the closed-form solution to
        (tot_atyp + atyp_p*k) / (tot_norm + norm_p*k) = target_ratio
    is:
        k = (target_ratio*tot_norm - tot_atyp) / (atyp_p - target_ratio*norm_p)

    Atypisch-dominant or balanced patients (atyp >= norm) get k=1. Pure-Normal
    patients (atyp=0) reduce purely to k = (tot_atyp/tot_norm - target_ratio) *
    tot_norm / (target_ratio * norm_p), which is capped at `cap`.
    """
    factors: dict[str, int] = {}

    # Seed with fixed values (contributes their known weighted counts).
    for p, k in fixed.items():
        if p in summary:
            factors[p] = k

    # Patients that need auto-computation, sorted most Normal-heavy first.
    auto = [p for p in summary if p not in fixed]
    auto.sort(
        key=lambda p: summary[p]["normal"] / max(summary[p]["atypisch"], 1),
        reverse=True,
    )

    for p in auto:
        atyp_p = summary[p]["atypisch"]
        norm_p = summary[p]["normal"]

        if norm_p == 0 or atyp_p >= norm_p:
            factors[p] = 1
            continue

        # Weighted global totals from all patients assigned so far (excl. p).
        tot_atyp = sum(summary[q]["atypisch"] * factors.get(q, 1)
                       for q in summary if q != p)
        tot_norm = sum(summary[q]["normal"] * factors.get(q, 1)
                       for q in summary if q != p)

        denom = atyp_p - target_ratio * norm_p  # always negative for norm > atyp
        numer = target_ratio * tot_norm - tot_atyp

        if abs(denom) < 1e-9 or numer <= 0:
            # Already at or below target without boosting this patient.
            factors[p] = 1
        else:
            k_exact = numer / denom
            factors[p] = min(cap, max(1, round(k_exact)))

    return factors


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
     train_neg,                               # list[(img_path, patient)] — train only
     oversample_factors) = fold_params        # dict[patient, k] — computed in __main__

    fold_workspace = os.path.join(TEMP_DIR, f"fold_{fold_idx}_workspace")
    fold_img_dir   = os.path.join(fold_workspace, "images")
    fold_lbl_dir   = os.path.join(fold_workspace, "labels")
    os.makedirs(fold_img_dir, exist_ok=True)
    os.makedirs(fold_lbl_dir, exist_ok=True)

    # Train: positives oversampled per patient + negatives at FP_NEG_OVERSAMPLE.
    train_paths = []
    for src, patient in train_pos:
        n = oversample_factors.get(patient, 1)
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


    # `cfg=...` is intentionally NOT used — tuner-derived hyperparameters are
    # calibrated for scratch training and overwrite pretrained features.
    model = YOLO(PRETRAINED_WEIGHTS)
    model.train(
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
        # On-the-fly augmentation (replaces the disabled pre-applied pipeline).
        augment  = True,
        mosaic   = MOSAIC,
        flipud   = FLIPUD,
        # Fine-tuning hyperparameters
        lr0      = LR0,
        freeze   = FREEZE,
        cls      = CLS_LOSS_WEIGHT,
        cos_lr   = COS_LR,
    )

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
    # 1. Load annotated positives from all patients (P1–P15), with patient tag
    # ------------------------------------------------------------------
    pos_with_meta: list[tuple[str, str]] = []   # (img_path, patient)
    summary: dict[str, dict] = {}

    all_patients = sorted(PATIENT_IMAGE_DIRS.keys() | PATIENT_IMAGE_TXTS.keys(),
                          key=lambda p: int(p[1:]))
    for patient in all_patients:
        candidates = load_patient_images(patient)
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
            "files":    len(verified),
            "atypisch": atyp_count,
            "normal":   norm_count,
        }

    # Compute oversample factors: P1–P8 use pinned values, P9–P15 are solved
    # automatically to target OVERSAMPLE_TARGET_RATIO globally.
    oversample_factors = compute_oversample_factors(summary, PATIENT_OVERSAMPLE_FIXED)

    # Effective weighted counts after oversampling.
    eff_atyp = sum(summary[p]["atypisch"] * oversample_factors[p] for p in summary)
    eff_norm = sum(summary[p]["normal"]   * oversample_factors[p] for p in summary)

    print("\n=== Per-patient annotation summary ===")
    print(f"{'Patient':<8}{'Files':>8}{'Atypisch':>10}{'Normal':>10}{'Oversample':>12}"
          f"{'EffAtyp':>10}{'EffNorm':>10}")
    total_files = total_atyp = total_norm = 0
    for p in sorted(summary, key=lambda x: int(x[1:])):
        s  = summary[p]
        k  = oversample_factors[p]
        ea = s["atypisch"] * k
        en = s["normal"]   * k
        print(f"{p:<8}{s['files']:>8}{s['atypisch']:>10}{s['normal']:>10}{k:>12}{ea:>10}{en:>10}")
        total_files += s["files"]
        total_atyp  += s["atypisch"]
        total_norm  += s["normal"]
    print(f"{'TOTAL':<8}{total_files:>8}{total_atyp:>10}{total_norm:>10}")
    if total_norm > 0:
        print(f"Raw ratio      = {total_atyp / total_norm:.1f} : 1")
    if eff_norm > 0:
        print(f"Effective ratio = {eff_atyp / eff_norm:.1f} : 1  (target {OVERSAMPLE_TARGET_RATIO:.0f}:1)")

    # ------------------------------------------------------------------
    # 2. Load P1 backgrounds and per-patient FP negatives (tagged by patient)
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

    if FP_NEG_OVERSAMPLE > 0:
        print(f"\nLoading FP negatives (oversample={FP_NEG_OVERSAMPLE})...")
        total_fp_missing = []
        for fp_patient, excel_path in PATIENT_FP_EXCELS.items():
            if not os.path.exists(excel_path):
                print(f"  {fp_patient}: Excel not found, skipping ({excel_path})")
                continue

            disk_index   = build_disk_index(fp_patient)
            fp_df        = pd.read_excel(excel_path, header=None)
            fp_filenames = fp_df[0].dropna().tolist()
            fp_paths, fp_missing = [], []
            for fn in fp_filenames:
                bn = os.path.basename(str(fn))
                if bn in disk_index:
                    fp_paths.append(disk_index[bn])
                else:
                    fp_missing.append(fn)

            for p in fp_paths:
                neg_with_meta.append((p, fp_patient))

            print(f"  {fp_patient}: {len(fp_paths)} resolved, {len(fp_missing)} missing")
            if fp_missing:
                print(f"    first 5 missing: {fp_missing[:5]}")
            total_fp_missing.extend(fp_missing)

        print(f"Total FP negatives added: {sum(1 for _, p in neg_with_meta if p in PATIENT_FP_EXCELS)}"
              f"  ({len(total_fp_missing)} unresolved across all patients)")

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
                               train_pos, val_pos, test_pos, train_neg,
                               oversample_factors))
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
                               train_pos, val_pos, te_pos, tr_neg,
                               oversample_factors))
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
