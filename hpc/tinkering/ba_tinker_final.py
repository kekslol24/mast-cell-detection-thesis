"""
ba_tinker_final.py — Single training run on the full P1–P8 corpus.

Unlike ba_tinker.py, which runs 8 LOPO folds for *evaluation* of the recipe,
this script trains ONE model on all 8 patients (with a small in-train val
split for early-stopping) to produce the *deployment* artifact for USZ.

LOPO cross-validation gave the unbiased generalization estimate
(mean Test_Recall ≈ 0.866, Test_Recall_Normal ≈ 0.840 on the winning recipe).
The model trained here is what we actually ship — every patient is in train,
so it has strictly more data than any individual LOPO fold model.

Recipe defaults = LOPO matrix winner `tinker_ls0.0_deg5_dfl1.5_fr10`:
    FREEZE          = 10
    DEGREES         = 5.0
    DFL             = 1.5
    LABEL_SMOOTHING = 0.0
    + production constants (mosaic=0.0, flipud=0.5, augment=True, AdamW,
      lr0=0.001, cos_lr=True, imgsz=512, batch=32)

All knobs are env-overridable so a different matrix cell can be reused
without code changes. Per-patient image oversampling and negatives behave
identically to ba_tinker.py.

Deliverable: `<PROJECT_DIR>/final/weights/best.pt`
"""

import os
import gc
import glob as _glob
import yaml
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

# Reuse all data-loading + workspace helpers from the LOPO script so the
# preprocessing path is bit-identical to the run that produced the matrix.
from ba_tinker import (
    PATIENT_IMAGE_DIRS, PATIENT_OVERSAMPLE,
    P1_BG_DIR, P2_FP_EXCEL, P2_ROOT,
    CLASS_NAMES, NC,
    BATCH_SIZE, IMGSZ, MOSAIC, FLIPUD, COS_LR, LR0,
    EPOCHS_PER_FOLD, PATIENCE,
    glob_images, image_to_label_path, parse_classes_in_label,
    case_insensitive_resolve, link_one,
)

# ==============================================================================
# RECIPE — defaults reproduce the LOPO matrix winner; env-overridable
# ==============================================================================
FREEZE             = int(  os.environ.get("FREEZE",            "10"))
CLS_LOSS_WEIGHT    = float(os.environ.get("CLS_LOSS_WEIGHT",   "1.0"))
LABEL_SMOOTHING    = float(os.environ.get("LABEL_SMOOTHING",   "0.0"))
DEGREES            = float(os.environ.get("DEGREES",           "5.0"))
DFL                = float(os.environ.get("DFL",               "1.5"))
HSV_V              = float(os.environ.get("HSV_V",             "0.4"))
MIXUP              = float(os.environ.get("MIXUP",             "0.0"))

BG_RATIO           = int(os.environ.get("BG_RATIO",            "1"))
FP_NEG_OVERSAMPLE  = int(os.environ.get("FP_NEG_OVERSAMPLE",   "1"))
VAL_FRACTION       = float(os.environ.get("VAL_FRACTION",      "0.15"))
PRETRAINED_WEIGHTS = os.environ.get("PRETRAINED_WEIGHTS",      "yolo11n.pt")

PROJECT_DIR = f"./{os.environ.get('SLURM_JOB_NAME', 'tinker_final')}"
_JOB_ID     = os.environ.get("SLURM_JOB_ID", "local")
TEMP_DIR    = os.path.abspath(f".tinker_final_temp_{_JOB_ID}")
os.makedirs(TEMP_DIR, exist_ok=True)


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Load annotated positives from all 8 patients
    # ------------------------------------------------------------------
    pos_with_meta: list[tuple[str, str]] = []
    summary: dict[str, dict] = {}

    for patient, img_dir in PATIENT_IMAGE_DIRS.items():
        verified, atyp_count, norm_count = [], 0, 0
        for img in glob_images(img_dir):
            lbl = image_to_label_path(img, patient=patient)
            if not os.path.exists(lbl):
                continue
            verified.append(img)
            with open(lbl) as f:
                for line in f:
                    parts = line.split()
                    if parts and parts[0] == "0":
                        atyp_count += 1
                    elif parts and parts[0] == "1":
                        norm_count += 1
        for img in verified:
            pos_with_meta.append((img, patient))
        summary[patient] = {
            "files":      len(verified),
            "atypisch":   atyp_count,
            "normal":     norm_count,
            "oversample": PATIENT_OVERSAMPLE.get(patient, 1),
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
    # 2. Load P1 backgrounds and P2 FP negatives (same rules as ba_tinker)
    # ------------------------------------------------------------------
    neg_with_meta: list[tuple[str, str]] = []

    if BG_RATIO > 0:
        bg_paths = glob_images(P1_BG_DIR)
        rng    = np.random.default_rng(seed=42)
        target = BG_RATIO * len(pos_with_meta)
        if target < len(bg_paths):
            bg_paths = list(rng.choice(bg_paths, size=target, replace=False))
        for p in bg_paths:
            neg_with_meta.append((p, "P1"))
        print(f"\nP1 backgrounds (BG_RATIO={BG_RATIO}): {len(bg_paths)}")
    else:
        print("\nP1 backgrounds disabled (BG_RATIO=0).")

    if FP_NEG_OVERSAMPLE > 0 and os.path.exists(P2_FP_EXCEL):
        p2_disk_dir = case_insensitive_resolve(P2_ROOT)
        p2_disk     = _glob.glob(os.path.join(p2_disk_dir, "**/*.jpeg"), recursive=True)
        p2_disk    += _glob.glob(os.path.join(p2_disk_dir, "**/*.JPEG"), recursive=True)
        p2_index    = {os.path.basename(p): p for p in p2_disk}

        fp_df        = pd.read_excel(P2_FP_EXCEL, header=None)
        fp_filenames = fp_df[0].dropna().tolist()
        fp_paths, fp_missing = [], []
        for fn in fp_filenames:
            base = os.path.basename(str(fn))
            if base in p2_index:
                fp_paths.append(p2_index[base])
            else:
                fp_missing.append(fn)
        for p in fp_paths:
            neg_with_meta.append((p, "P2"))
        print(f"P2 FP negatives (oversample={FP_NEG_OVERSAMPLE}): {len(fp_paths)} resolved")
        if fp_missing:
            print(f"  not found: {len(fp_missing)} (first 5: {fp_missing[:5]})")

    # ------------------------------------------------------------------
    # 3. Train/val split — 85/15 stratified by patient.
    #    Texture leakage into val is acceptable here because LOPO already
    #    supplied the unbiased generalization estimate; this val set only
    #    serves as an early-stopping signal during the final training.
    # ------------------------------------------------------------------
    X = np.array([m[0] for m in pos_with_meta])
    y = np.array([m[1] for m in pos_with_meta])
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=VAL_FRACTION, stratify=y, random_state=42,
    )
    train_pos = list(zip(X_tr.tolist(), y_tr.tolist()))
    val_pos   = list(zip(X_va.tolist(), y_va.tolist()))

    print(f"\nTrain positives: {len(train_pos)} | Val positives: {len(val_pos)} "
          f"({100 * VAL_FRACTION:.0f}% stratified by patient)")

    # ------------------------------------------------------------------
    # 4. Build workspace with per-patient oversampling on train only
    # ------------------------------------------------------------------
    workspace = os.path.join(TEMP_DIR, "workspace")
    img_dir   = os.path.join(workspace, "images")
    lbl_dir   = os.path.join(workspace, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    train_paths = []
    for src, patient in train_pos:
        n = PATIENT_OVERSAMPLE.get(patient, 1)
        train_paths.extend(link_one(src, img_dir, lbl_dir,
                                    oversample=n, is_negative=False,
                                    patient=patient))
    for src, patient in neg_with_meta:
        train_paths.extend(link_one(src, img_dir, lbl_dir,
                                    oversample=FP_NEG_OVERSAMPLE,
                                    is_negative=True, patient=patient))

    val_paths = []
    for src, patient in val_pos:
        val_paths.extend(link_one(src, img_dir, lbl_dir,
                                  oversample=1, is_negative=False,
                                  patient=patient))

    np.savetxt(os.path.join(workspace, 'train.txt'), train_paths, fmt='%s')
    np.savetxt(os.path.join(workspace, 'val.txt'),   val_paths,   fmt='%s')

    yaml_path = os.path.join(workspace, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump({
            'path':  workspace,
            'train': 'train.txt',
            'val':   'val.txt',
            'nc':    NC,
            'names': CLASS_NAMES,
        }, f)

    print(f"Train symlinks (after oversample + negatives): {len(train_paths)}")
    print(f"Val symlinks (no oversample): {len(val_paths)}")

    # ------------------------------------------------------------------
    # 5. Train — single GPU, no multiprocessing
    # ------------------------------------------------------------------
    print(f"\nRecipe: FREEZE={FREEZE} DEGREES={DEGREES} DFL={DFL} "
          f"CLS={CLS_LOSS_WEIGHT} LABEL_SMOOTHING={LABEL_SMOOTHING} "
          f"HSV_V={HSV_V} MIXUP={MIXUP}")
    print(f"Project dir: {PROJECT_DIR}\n")

    model = YOLO(PRETRAINED_WEIGHTS)
    model.train(
        data            = yaml_path,
        epochs          = EPOCHS_PER_FOLD,
        patience        = PATIENCE,
        batch           = BATCH_SIZE,
        device          = 0,
        project         = PROJECT_DIR,
        name            = "final",
        imgsz           = IMGSZ,
        workers         = 0,
        cache           = False,
        exist_ok        = True,
        verbose         = False,
        augment         = True,
        mosaic          = MOSAIC,
        flipud          = FLIPUD,
        cos_lr          = COS_LR,
        lr0             = LR0,
        freeze          = FREEZE,
        cls             = CLS_LOSS_WEIGHT,
        dfl             = DFL,
        label_smoothing = LABEL_SMOOTHING,
        degrees         = DEGREES,
        hsv_v           = HSV_V,
        mixup           = MIXUP,
    )
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------------------------------
    # 6. Sanity-check the final best.pt on the in-train val split.
    #    This is NOT a generalization estimate (val patients are also in
    #    train); the LOPO numbers in fold_results.csv are. We only report
    #    these to confirm the run converged sensibly.
    # ------------------------------------------------------------------
    best_pt = os.path.join(PROJECT_DIR, "final", "weights", "best.pt")
    last_pt = os.path.join(PROJECT_DIR, "final", "weights", "last.pt")
    print(f"\nBest weights: {best_pt}")

    val_model = YOLO(best_pt)
    metrics   = val_model.val(data=yaml_path, split='val', workers=0,
                              device=0, batch=BATCH_SIZE, verbose=False)

    def per_class(idx):
        try:
            arr = metrics.box.r
            return float(arr[idx]) if idx < len(arr) else float('nan')
        except (AttributeError, IndexError, TypeError):
            return float('nan')

    print("\nIn-train val sanity check (NOT a generalization estimate):")
    print(f"  mAP50-95         : {metrics.box.map:.4f}")
    print(f"  mAP50            : {metrics.box.map50:.4f}")
    print(f"  Precision        : {metrics.box.mp:.4f}")
    print(f"  Recall           : {metrics.box.mr:.4f}")
    print(f"  Recall (Atypisch): {per_class(0):.4f}")
    print(f"  Recall (Normal)  : {per_class(1):.4f}")
    print(f"\nGeneralization estimate (from LOPO matrix winner):")
    print(f"  Recall ≈ 0.866 | Recall_Normal ≈ 0.840 | Recall_Atypisch ≈ 0.892")
    print(f"\nShip this file to USZ:")
    print(f"  {best_pt}")
    print(f"  (last.pt also at {last_pt} for resume / debugging)")
