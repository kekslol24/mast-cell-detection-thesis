#!/usr/bin/env python3
"""
eval_folds.py — Re-evaluate trained fold models without retraining.

Rebuilds fold splits deterministically (random_state=42, same as training),
runs val_model.val() on held-out test and val splits, and writes
fold_results.csv to RESULTS_DIR. No training happens — inference only.

Usage (from hpc/):
    python eval_folds.py

Env overrides:
    RESULTS_DIR    folder containing fold_N_* subdirs  (default: yolo_new_v4_freeze10)
    CV_MODE        lopo | grouped                       (default: lopo)
    N_GPUS         GPUs to interleave across            (default: 2)
    MAX_PARALLEL   concurrent eval processes            (default: 4)
    BATCH_SIZE_VAL batch size for val() calls           (default: 16)
"""

import os
import gc
import yaml
import tempfile
import numpy as np
import pandas as pd
import multiprocessing as mp
import torch
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

from patient_dir import PATIENT_IMAGE_DIRS, PATIENT_FP_EXCELS
from ba_improved_comb import (
    image_to_label_path, load_patient_images, build_disk_index,
    compute_oversample_factors, parse_classes_in_label,
    BASE_DATA_PATH, CLASS_NAMES, NC,
    PATIENT_GROUPS, PATIENT_OVERSAMPLE_FIXED,
    OVERSAMPLE_TARGET_RATIO, OVERSAMPLE_CAP,
    FP_NEG_OVERSAMPLE,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RESULTS_DIR    = os.environ.get("RESULTS_DIR",    "yolo_new_v4_freeze10")
CV_MODE        = os.environ.get("CV_MODE",        "lopo")
N_GPUS         = int(os.environ.get("N_GPUS",         "2"))
MAX_PARALLEL   = int(os.environ.get("MAX_PARALLEL",    "4"))
BATCH_SIZE_VAL = int(os.environ.get("BATCH_SIZE_VAL",  "16"))


# ==============================================================================
# HELPERS
# ==============================================================================
def _carve_val(train_pos_full):
    """85/15 stratified val split by patient, singleton-safe (mirrors training)."""
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


def per_class(metrics, class_id):
    """Per-class recall via ap_class_index (fixes positional-indexing bug)."""
    try:
        idx_map = list(metrics.box.ap_class_index)
        if class_id not in idx_map:
            return float("nan")
        return float(metrics.box.r[idx_map.index(class_id)])
    except (AttributeError, IndexError, TypeError):
        return float("nan")


# ==============================================================================
# PER-FOLD EVALUATION
# ==============================================================================
def eval_fold(fold_params):
    fold_idx, fold_label, val_pos, test_pos = fold_params

    fold_name    = f"fold_{fold_idx + 1}_{fold_label}"
    best_weights = os.path.join(RESULTS_DIR, fold_name, "weights", "best.pt")

    if not os.path.exists(best_weights):
        print(f"  SKIP {fold_name}: best.pt not found")
        return None

    gpu_id = fold_idx % N_GPUS

    # Minimal workspace: txt files with absolute paths + data.yaml.
    # No symlinks needed — YOLO reads absolute paths from txt files directly.
    # train.txt is a required yaml key but is not evaluated; val_paths used as placeholder.
    with tempfile.TemporaryDirectory(prefix=f"eval_{fold_idx}_") as tmp:
        val_paths  = [m[0] for m in val_pos]
        test_paths = [m[0] for m in test_pos]

        np.savetxt(os.path.join(tmp, "train.txt"), val_paths,  fmt="%s")
        np.savetxt(os.path.join(tmp, "val.txt"),   val_paths,  fmt="%s")
        np.savetxt(os.path.join(tmp, "test.txt"),  test_paths, fmt="%s")

        with open(os.path.join(tmp, "data.yaml"), "w") as f:
            yaml.dump({"path": tmp, "train": "train.txt", "val": "val.txt",
                       "test": "test.txt", "nc": NC, "names": CLASS_NAMES}, f)

        yaml_path = os.path.join(tmp, "data.yaml")
        val_model = YOLO(best_weights)

        m_test = val_model.val(data=yaml_path, split="test", verbose=False,
                               workers=0, device=gpu_id, batch=BATCH_SIZE_VAL)
        m_val  = val_model.val(data=yaml_path, split="val",  verbose=False,
                               workers=0, device=gpu_id, batch=BATCH_SIZE_VAL)

        del val_model
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "Fold":                 fold_idx + 1,
            "Holdout":              fold_label,
            "Val_mAP50-95":         m_val.box.map,
            "Val_mAP50":            m_val.box.map50,
            "Val_Precision":        m_val.box.mp,
            "Val_Recall":           m_val.box.mr,
            "Val_Recall_Atypisch":  per_class(m_val, 0),
            "Val_Recall_Normal":    per_class(m_val, 1),
            "Test_mAP50-95":        m_test.box.map,
            "Test_mAP50":           m_test.box.map50,
            "Test_Precision":       m_test.box.mp,
            "Test_Recall":          m_test.box.mr,
            "Test_Recall_Atypisch": per_class(m_test, 0),
            "Test_Recall_Normal":   per_class(m_test, 1),
        }


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Load annotated positives
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

    print(f"Loaded {len(pos_with_meta)} annotated images, "
          f"{sum(v['files'] > 0 for v in summary.values())} patients with positives")

    # ------------------------------------------------------------------
    # 2. Build fold tasks (val_pos + test_pos only — no training needed)
    # ------------------------------------------------------------------
    fold_tasks: list[tuple] = []

    if CV_MODE == "grouped":
        print(f"\n=== Grouped {len(PATIENT_GROUPS)}-Fold CV — eval only ===")
        for fold_idx, (group_label, holdout_patients) in enumerate(PATIENT_GROUPS.items()):
            holdout_set    = set(holdout_patients)
            train_pos_full = [m for m in pos_with_meta if m[1] not in holdout_set]
            test_pos       = [m for m in pos_with_meta if m[1] in holdout_set]
            _, val_pos     = _carve_val(train_pos_full)
            fold_tasks.append((fold_idx, group_label, val_pos, test_pos))
            print(f"  Fold {fold_idx+1} ({group_label}): val={len(val_pos)}, test={len(test_pos)}")

    elif CV_MODE == "lopo":
        patients = sorted({m[1] for m in pos_with_meta})
        print(f"\n=== LOPO — {len(patients)} folds — eval only ===")
        for fold_idx, holdout in enumerate(patients):
            train_pos_full = [m for m in pos_with_meta if m[1] != holdout]
            test_pos       = [m for m in pos_with_meta if m[1] == holdout]
            _, val_pos     = _carve_val(train_pos_full)
            fold_tasks.append((fold_idx, holdout, val_pos, test_pos))
            print(f"  Fold {fold_idx+1}: holdout={holdout}, "
                  f"val={len(val_pos)}, test={len(test_pos)}")
    else:
        raise ValueError(f"CV_MODE='{CV_MODE}' not supported. Use 'lopo' or 'grouped'.")

    # ------------------------------------------------------------------
    # 3. Run evaluations in parallel
    # ------------------------------------------------------------------
    n_folds      = len(fold_tasks)
    max_parallel = min(MAX_PARALLEL, n_folds)
    print(f"\nEvaluating {n_folds} folds, {max_parallel} in parallel — {RESULTS_DIR}\n")

    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=max_parallel) as pool:
        raw_results = pool.map(eval_fold, fold_tasks)

    final_results = [r for r in raw_results if r is not None]
    if not final_results:
        raise SystemExit("No fold results returned — check best.pt paths.")

    # ------------------------------------------------------------------
    # 4. Save and print
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(final_results)
    out_csv    = os.path.join(RESULTS_DIR, "fold_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    for split in ("Val", "Test"):
        cols = ["Fold", "Holdout"] + [c for c in results_df.columns if c.startswith(split)]
        df   = results_df[cols].copy()
        df.columns = (["Fold", "Holdout"]
                      + [c.replace(f"{split}_", "") for c in df.columns
                         if c not in ("Fold", "Holdout")])
        print("\n" + "=" * 72)
        print(f"CV RESULTS — {split.upper()}")
        print("=" * 72)
        print(df.to_string(index=False))
        if "mAP50" in df.columns:
            print(f"\nMean mAP50    : {df['mAP50'].mean():.4f} +/- {df['mAP50'].std():.4f}")
            print(f"Mean mAP50-95 : {df['mAP50-95'].mean():.4f} +/- {df['mAP50-95'].std():.4f}")
            print(f"Mean Precision: {df['Precision'].mean():.4f} +/- {df['Precision'].std():.4f}")
            print(f"Mean Recall   : {df['Recall'].mean():.4f} +/- {df['Recall'].std():.4f}")
        if split == "Test":
            ra = df["Recall_Atypisch"].dropna()
            rn = df["Recall_Normal"].dropna()
            if len(ra):
                print(f"Mean R_Atypisch (n={len(ra)}): {ra.mean():.4f} +/- {ra.std():.4f}")
            if len(rn):
                print(f"Mean R_Normal   (n={len(rn)}): {rn.mean():.4f} +/- {rn.std():.4f}")
