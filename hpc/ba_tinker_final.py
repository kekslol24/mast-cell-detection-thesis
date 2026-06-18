"""
ba_tinker_final.py — Final deployment model training.

Trains on the full P1–P53 corpus with one patient permanently withheld as
the out-of-sample calibration set:

  CALIBRATION_PATIENT = "P11"  (17 Atypisch / 20 Normal, 37 annotated files)

P11 is excluded from every split (train, val, negatives). After training,
run inference.py on P11's full tile set (not just the 37 annotated files)
to obtain the TP/FP confidence distribution for USZ threshold selection.

No cross-validation — single training run, train/val split only.
Production recipe identical to ba_improved_comb.py (P5-B winner):
  degrees=5, dfl=1.5, freeze=10, lr0=0.001, cls=1.0,
  mosaic=0.0, flipud=0.5, augment=True, epochs=700, patience=50
"""

import os
import gc
import yaml
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from patient_dir import PATIENT_IMAGE_DIRS, PATIENT_FP_EXCELS
from ba_improved_comb import (
    case_insensitive_resolve,
    image_to_label_path,
    glob_images,
    load_patient_images,
    build_disk_index,
    compute_oversample_factors,
    parse_classes_in_label,
    link_one,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CALIBRATION_PATIENT = "P11"   # withheld permanently for threshold calibration

PRETRAINED_WEIGHTS  = "yolo11n.pt"
EPOCHS              = 700
PATIENCE            = 50
BATCH_SIZE          = 32
BATCH_SIZE_VAL      = 16
IMGSZ               = 512
LR0                 = 0.001
FREEZE              = int(os.environ.get("FREEZE", "10"))
DEGREES             = float(os.environ.get("DEGREES", "5.0"))
DFL                 = float(os.environ.get("DFL", "1.5"))
CLS_LOSS_WEIGHT     = 1.0
MOSAIC              = 0.0
FLIPUD              = 0.5
COS_LR              = True

# DoE conclusion: BG tiles hurt; default off for final model.
BG_RATIO          = int(os.environ.get("BG_RATIO", "0"))
FP_NEG_OVERSAMPLE = int(os.environ.get("FP_NEG_OVERSAMPLE", "1"))

PATIENT_OVERSAMPLE_FIXED = {}
OVERSAMPLE_TARGET_RATIO  = 2.0
OVERSAMPLE_CAP           = 25

BASE_DATA_PATH = "/cfs/earth/scratch/vollmflo/BA/data"
PATIENT_LABEL_DIRS = {
    "P1": os.path.join(BASE_DATA_PATH, "new", "P1", "labels", "Train"),
}
P1_BG_DIR   = os.path.join(BASE_DATA_PATH, "old", "P1", "Negativ 12241515", "images", "train")
CLASS_NAMES = ["Atypisch", "Normal"]
NC          = len(CLASS_NAMES)

PROJECT_DIR = f"./{os.environ.get('SLURM_JOB_NAME', 'local_run_final')}"
_JOB_ID     = os.environ.get("SLURM_JOB_ID", "local")
TEMP_DIR    = os.path.abspath(f".cv_temp_{_JOB_ID}")

os.makedirs(TEMP_DIR, exist_ok=True)


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print(f"Calibration patient withheld: {CALIBRATION_PATIENT} (excluded from all splits)")
    print(f"Project dir: {PROJECT_DIR}\n")

    # ------------------------------------------------------------------
    # 1. Load annotated positives — skip P11
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
            classes = parse_classes_in_label(lbl)
            if not classes:
                continue
            verified.append(img)
            atyp_count += sum(1 for ln in open(lbl) if ln.split() and ln.split()[0] == "0")
            norm_count  += sum(1 for ln in open(lbl) if ln.split() and ln.split()[0] == "1")
        for img in verified:
            pos_with_meta.append((img, patient))
        summary[patient] = {"files": len(verified), "atypisch": atyp_count, "normal": norm_count}

    oversample_factors = compute_oversample_factors(summary, PATIENT_OVERSAMPLE_FIXED)

    eff_atyp = sum(summary[p]["atypisch"] * oversample_factors[p] for p in summary)
    eff_norm = sum(summary[p]["normal"]   * oversample_factors[p] for p in summary)

    print("=== Per-patient annotation summary (P11 excluded) ===")
    print(f"{'Patient':<8}{'Files':>8}{'Atypisch':>10}{'Normal':>10}{'Oversample':>12}"
          f"{'EffAtyp':>10}{'EffNorm':>10}")
    total_files = total_atyp = total_norm = 0
    for p in sorted(summary, key=lambda x: int(x[1:])):
        s  = summary[p]
        k  = oversample_factors[p]
        ea = s["atypisch"] * k
        en = s["normal"]   * k
        print(f"{p:<8}{s['files']:>8}{s['atypisch']:>10}{s['normal']:>10}"
              f"{k:>12}{ea:>10}{en:>10}")
        total_files += s["files"]
        total_atyp  += s["atypisch"]
        total_norm  += s["normal"]
    print(f"{'TOTAL':<8}{total_files:>8}{total_atyp:>10}{total_norm:>10}")
    if total_norm > 0:
        print(f"Raw ratio       = {total_atyp / total_norm:.1f} : 1")
    if eff_norm > 0:
        print(f"Effective ratio = {eff_atyp / eff_norm:.1f} : 1  (target {OVERSAMPLE_TARGET_RATIO:.0f}:1)")

    # ------------------------------------------------------------------
    # 2. Load negatives — skip P11's FP entries
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
        for fp_patient, excel_path in PATIENT_FP_EXCELS.items():
            if fp_patient == CALIBRATION_PATIENT:
                continue
            if not os.path.exists(excel_path):
                print(f"  {fp_patient}: Excel not found, skipping")
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
        total_fp = sum(1 for _, p in neg_with_meta if p in PATIENT_FP_EXCELS)
        print(f"Total FP negatives: {total_fp}")

    # ------------------------------------------------------------------
    # 3. Train / val split (15 % val, patient-stratified)
    # ------------------------------------------------------------------
    X = np.array([m[0] for m in pos_with_meta])
    y = np.array([m[1] for m in pos_with_meta])

    unique_pats, pat_counts = np.unique(y, return_counts=True)
    singleton_mask = np.isin(y, unique_pats[pat_counts < 2])
    X_single, y_single = X[singleton_mask],  y[singleton_mask]
    X_multi,  y_multi  = X[~singleton_mask], y[~singleton_mask]

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_multi, y_multi, test_size=0.15, stratify=y_multi, random_state=42,
    )
    X_tr = np.concatenate([X_tr, X_single])
    y_tr = np.concatenate([y_tr, y_single])

    train_pos = list(zip(X_tr.tolist(), y_tr.tolist()))
    val_pos   = list(zip(X_va.tolist(), y_va.tolist()))
    print(f"\nTrain: {len(train_pos)} pos + {len(neg_with_meta)} neg | Val: {len(val_pos)}")

    # ------------------------------------------------------------------
    # 4. Build workspace and symlinks
    # ------------------------------------------------------------------
    workspace = os.path.join(TEMP_DIR, "final_workspace")
    img_dir   = os.path.join(workspace, "images")
    lbl_dir   = os.path.join(workspace, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    train_paths: list[str] = []
    for src, patient in train_pos:
        n = oversample_factors.get(patient, 1)
        train_paths.extend(link_one(src, img_dir, lbl_dir,
                                    oversample=n, is_negative=False,
                                    patient=patient))
    for src, patient in neg_with_meta:
        train_paths.extend(link_one(src, img_dir, lbl_dir,
                                    oversample=FP_NEG_OVERSAMPLE, is_negative=True,
                                    patient=patient))

    val_paths: list[str] = []
    for src, patient in val_pos:
        val_paths.extend(link_one(src, img_dir, lbl_dir,
                                  oversample=1, is_negative=False,
                                  patient=patient))

    np.savetxt(os.path.join(workspace, "train.txt"), train_paths, fmt="%s")
    np.savetxt(os.path.join(workspace, "val.txt"),   val_paths,   fmt="%s")

    yaml_path = os.path.join(workspace, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump({
            "path":  workspace,
            "train": "train.txt",
            "val":   "val.txt",
            "nc":    NC,
            "names": CLASS_NAMES,
        }, f)

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    model = YOLO(PRETRAINED_WEIGHTS)
    model.train(
        data     = yaml_path,
        epochs   = EPOCHS,
        patience = PATIENCE,
        batch    = BATCH_SIZE,
        device   = 0,
        project  = PROJECT_DIR,
        imgsz    = IMGSZ,
        name     = "final_model",
        workers  = 0,
        cache    = False,
        exist_ok = True,
        verbose  = False,
        augment  = True,
        mosaic   = MOSAIC,
        flipud   = FLIPUD,
        lr0      = LR0,
        freeze   = FREEZE,
        cls      = CLS_LOSS_WEIGHT,
        cos_lr   = COS_LR,
        degrees  = DEGREES,
        dfl      = DFL,
    )

    del model
    torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------------------------------
    # 6. Val metrics on best.pt
    # ------------------------------------------------------------------
    best_weights = os.path.join(PROJECT_DIR, "final_model", "weights", "best.pt")
    val_model    = YOLO(best_weights)
    metrics      = val_model.val(data=yaml_path, split="val", verbose=False,
                                 workers=0, device=0, batch=BATCH_SIZE_VAL)

    def per_class(idx):
        try:
            arr = metrics.box.r
            return float(arr[idx]) if idx < len(arr) else float("nan")
        except (AttributeError, IndexError, TypeError):
            return float("nan")

    print("\n=== Final model — val metrics ===")
    print(f"mAP50      : {metrics.box.map50:.4f}")
    print(f"mAP50-95   : {metrics.box.map:.4f}")
    print(f"Precision  : {metrics.box.mp:.4f}")
    print(f"Recall     : {metrics.box.mr:.4f}")
    print(f"R_Atypisch : {per_class(0):.4f}")
    print(f"R_Normal   : {per_class(1):.4f}")
    print(f"\nWeights    : {best_weights}")
    print(f"\nNext step  : run inference.py on P11's full tile set (conf=0.276)"
          f" to obtain the TP/FP confidence distribution for USZ calibration.")
