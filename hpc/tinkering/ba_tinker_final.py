"""
ba_tinker_final.py — Single training run on the full P1–P53 corpus.

Unlike ba_tinker.py, which runs folds for *evaluation* of the recipe,
this script trains ONE model on all patients except P11 (with a small
in-train val split for early-stopping) to produce the *deployment*
artifact for USZ.

P11 (17 Atypisch / 20 Normal, 37 annotated files) is withheld as the
out-of-sample calibration patient. After training, run inference.py on
P11's full tile set (not just the 37 annotated files) to obtain the
TP/FP confidence distribution for USZ threshold selection.

Grouped 5-fold CV (P5-B) gave the unbiased generalization estimate.
The model trained here is what we actually ship — 34 of 35 annotated
patients are in train, so it has strictly more data than any individual
CV fold model.

Recipe defaults = P5-B grouped CV winner:
    FREEZE          = 10
    DEGREES         = 5.0
    DFL             = 1.5
    LABEL_SMOOTHING = 0.0
    + production constants (mosaic=0.0, flipud=0.5, augment=True, AdamW,
      lr0=0.001, cos_lr=True, imgsz=512, batch=32)

All knobs are env-overridable. Per-patient image oversampling and negatives
behave identically to ba_improved_comb.py.

Deliverable: `<PROJECT_DIR>/final/weights/best.pt`
"""

import os
import gc
import yaml
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

# Reuse all data-loading + workspace helpers from the combined script so the
# preprocessing path is bit-identical to the run that produced the matrix.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ba_improved_comb import (
    PATIENT_IMAGE_DIRS, PATIENT_OVERSAMPLE_FIXED,
    PATIENT_FP_EXCELS, P1_BG_DIR,
    CLASS_NAMES, NC,
    BATCH_SIZE, IMGSZ, MOSAIC, FLIPUD, COS_LR, LR0,
    EPOCHS_PER_FOLD, PATIENCE,
    OVERSAMPLE_TARGET_RATIO, OVERSAMPLE_CAP,
    glob_images, image_to_label_path, parse_classes_in_label,
    case_insensitive_resolve, link_one,
    compute_oversample_factors, build_disk_index, load_patient_images,
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

CALIBRATION_PATIENT = "P11"   # withheld permanently — do not add back

PROJECT_DIR = f"./{os.environ.get('SLURM_JOB_NAME', 'tinker_final')}"
_JOB_ID     = os.environ.get("SLURM_JOB_ID", "local")
TEMP_DIR    = os.path.abspath(f".tinker_final_temp_{_JOB_ID}")
os.makedirs(TEMP_DIR, exist_ok=True)


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Load annotated positives — all patients except P11 (calibration)
    # ------------------------------------------------------------------
    pos_with_meta: list[tuple[str, str]] = []
    summary: dict[str, dict] = {}

    all_patients = sorted(PATIENT_IMAGE_DIRS.keys(), key=lambda p: int(p[1:]))
    for patient in all_patients:
        if patient == CALIBRATION_PATIENT:
            continue
        candidates = load_patient_images(patient)
        verified, atyp_count, norm_count = [], 0, 0
        for img in candidates:
            lbl = image_to_label_path(img, patient=patient)
            if not os.path.exists(lbl):
                continue
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

    oversample_factors = compute_oversample_factors(summary, PATIENT_OVERSAMPLE_FIXED)

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
        print(f"Raw ratio       = {total_atyp / total_norm:.1f} : 1")
    if eff_norm > 0:
        print(f"Effective ratio = {eff_atyp / eff_norm:.1f} : 1  (target {OVERSAMPLE_TARGET_RATIO:.0f}:1)")

    # ------------------------------------------------------------------
    # 2. Load P1 backgrounds and per-patient FP negatives (skip P11)
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

    if FP_NEG_OVERSAMPLE > 0:
        print(f"\nLoading FP negatives (oversample={FP_NEG_OVERSAMPLE})...")
        total_fp_missing = []
        for fp_patient, excel_path in PATIENT_FP_EXCELS.items():
            if fp_patient == CALIBRATION_PATIENT:
                continue
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
        n = oversample_factors.get(patient, 1)
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
    print(f"\nGeneralization estimate (P5-B grouped 5-fold CV, G1–G5):")
    print(f"  Recall ≈ 0.853 | R_Atypisch ≈ 0.885 | R_Normal ≈ 0.821 (0.861 ex-G1)")
    print(f"\nShip this file to USZ:")
    print(f"  {best_pt}")
    print(f"  (last.pt also at {last_pt} for resume / debugging)")
