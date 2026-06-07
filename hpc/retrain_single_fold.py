#!/usr/bin/env python3
"""
retrain_single_fold.py — Retrain exactly one LOPO fold.

Rebuilds the fold split for the specified holdout patient and runs train_fold,
overwriting the existing fold directory in RESULTS_DIR. Intended for recovering
folds that were killed by the HPC wall-time limit.

Usage (from hpc/):
    HOLDOUT_PATIENT=P37 python retrain_single_fold.py

Env overrides:
    HOLDOUT_PATIENT  patient to hold out as test set    (required, e.g. P37)
    RESULTS_DIR      project dir to write output into   (default: yolo_new_v4_freeze10)
    N_GPUS           GPUs available                     (default: 2)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from patient_dir import PATIENT_IMAGE_DIRS, PATIENT_FP_EXCELS
from ba_improved_comb import (
    image_to_label_path, load_patient_images, build_disk_index,
    compute_oversample_factors, parse_classes_in_label, glob_images,
    BASE_DATA_PATH, CLASS_NAMES, NC, PATIENT_LABEL_DIRS,
    PATIENT_OVERSAMPLE_FIXED, OVERSAMPLE_TARGET_RATIO, OVERSAMPLE_CAP,
    P1_BG_DIR, BG_RATIO, FP_NEG_OVERSAMPLE,
    train_fold,
    PROJECT_DIR, TEMP_DIR,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
HOLDOUT_PATIENT = os.environ.get("HOLDOUT_PATIENT", "")
RESULTS_DIR     = os.environ.get("RESULTS_DIR",     "yolo_new_v4_freeze10")
N_GPUS          = int(os.environ.get("N_GPUS",      "2"))

if not HOLDOUT_PATIENT:
    print("ERROR: set HOLDOUT_PATIENT env var (e.g. HOLDOUT_PATIENT=P37)")
    sys.exit(1)


def _carve_val(train_pos_full):
    X = np.array([m[0] for m in train_pos_full])
    y = np.array([m[1] for m in train_pos_full])
    unique_pats, pat_counts = np.unique(y, return_counts=True)
    singleton_mask = np.isin(y, unique_pats[pat_counts < 2])
    X_single, y_single = X[singleton_mask],  y[singleton_mask]
    X_multi,  y_multi  = X[~singleton_mask], y[~singleton_mask]
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_multi, y_multi, test_size=0.15, stratify=y_multi, random_state=42,
    )
    X_tr = np.concatenate([X_tr, X_single])
    y_tr = np.concatenate([y_tr, y_single])
    return (list(zip(X_tr.tolist(), y_tr.tolist())),
            list(zip(X_va.tolist(), y_va.tolist())))


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Load all annotated positives
    # ------------------------------------------------------------------
    pos_with_meta: list[tuple[str, str]] = []
    summary: dict[str, dict] = {}

    for patient in sorted(PATIENT_IMAGE_DIRS, key=lambda p: int(p[1:])):
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
        summary[patient] = {"files": len(verified),
                            "atypisch": atyp_count, "normal": norm_count}

    oversample_factors = compute_oversample_factors(summary, PATIENT_OVERSAMPLE_FIXED)

    # ------------------------------------------------------------------
    # 2. Load FP negatives
    # ------------------------------------------------------------------
    neg_with_meta: list[tuple[str, str]] = []
    for fp_patient, excel_path in PATIENT_FP_EXCELS.items():
        if not os.path.exists(excel_path):
            continue
        disk_index = build_disk_index(fp_patient)
        fp_df = pd.read_excel(excel_path, header=None)
        for fn in fp_df[0].dropna().tolist():
            bn = os.path.basename(str(fn))
            if bn in disk_index:
                neg_with_meta.append((disk_index[bn], fp_patient))

    if BG_RATIO > 0:
        bg_paths = glob_images(P1_BG_DIR)
        rng = np.random.default_rng(seed=42)
        target = BG_RATIO * len(pos_with_meta)
        if target < len(bg_paths):
            bg_paths = list(rng.choice(bg_paths, size=target, replace=False))
        for p in bg_paths:
            neg_with_meta.append((p, "P1"))

    # ------------------------------------------------------------------
    # 3. Build the single fold for HOLDOUT_PATIENT
    # ------------------------------------------------------------------
    patients = sorted({m[1] for m in pos_with_meta})
    if HOLDOUT_PATIENT not in patients:
        print(f"ERROR: '{HOLDOUT_PATIENT}' has no annotated positives — nothing to hold out.")
        sys.exit(1)

    fold_idx = patients.index(HOLDOUT_PATIENT)

    train_pos_full = [m for m in pos_with_meta if m[1] != HOLDOUT_PATIENT]
    test_pos       = [m for m in pos_with_meta if m[1] == HOLDOUT_PATIENT]
    train_neg      = [m for m in neg_with_meta if m[1] != HOLDOUT_PATIENT]
    train_pos, val_pos = _carve_val(train_pos_full)

    print(f"\nRetraining fold {fold_idx + 1}: holdout={HOLDOUT_PATIENT}")
    print(f"  train={len(train_pos)} pos / {len(train_neg)} neg, "
          f"val={len(val_pos)}, test={len(test_pos)}")
    print(f"  Output → {RESULTS_DIR}/fold_{fold_idx + 1}_{HOLDOUT_PATIENT}/\n")

    fold_params = (fold_idx, HOLDOUT_PATIENT,
                   train_pos, val_pos, test_pos, train_neg,
                   oversample_factors)

    # train_fold writes directly to PROJECT_DIR which is set in ba_improved_comb.
    # Override PROJECT_DIR to point at RESULTS_DIR so the output lands in the
    # right place alongside the other completed folds.
    import ba_improved_comb as _comb
    _comb.PROJECT_DIR = RESULTS_DIR

    result = train_fold(fold_params)

    if result:
        print(f"\nFold {fold_idx + 1} ({HOLDOUT_PATIENT}) complete:")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
