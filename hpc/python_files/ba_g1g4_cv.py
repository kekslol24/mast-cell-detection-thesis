"""
ba_g1g4_cv.py — Grouped CV on G1–G4 only (G5 withheld as calibration set).

Ablation run to prove that removing G5 from training does not degrade
performance on G1–G4. Compare the four G1–G4 fold results here against the
matching folds from the original 5-group run (yolo_new_v4_freeze10_re_run_CV)
to confirm G5 data was not load-bearing for the other groups.

Identical recipe to ba_improved_comb.py (production settings). The only
difference: all G5 patients are removed from pos_with_meta, neg_with_meta,
and PATIENT_GROUPS before fold construction. G5 patients never appear in
any fold's train, val, or test split.
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
from patient_dir import PATIENT_IMAGE_DIRS, PATIENT_FP_EXCELS

# Import all utility functions from the main script — no duplication.
from ba_improved_comb import (
    case_insensitive_resolve,
    image_to_label_path,
    glob_images,
    load_patient_images,
    build_disk_index,
    compute_oversample_factors,
    parse_classes_in_label,
    link_one,
    train_fold,
)

# ==============================================================================
# CONFIGURATION — identical to ba_improved_comb.py except PATIENT_GROUPS
# ==============================================================================
CV_MODE           = os.environ.get("CV_MODE", "grouped")
N_SPLITS_FALLBACK = 5

# G5 withheld entirely — used as the held-out calibration set for threshold
# selection before final deployment.
PATIENT_GROUPS = {
    "G1": ["P1",  "P13", "P31"],
    "G2": ["P2",  "P8",  "P37"],
    "G3": ["P9",  "P12", "P53"],
    "G4": ["P3",  "P6",  "P15", "P16", "P17", "P18", "P48"],
}

G5_PATIENTS = [
    "P4",  "P5",  "P7",  "P10", "P11", "P14", "P24", "P28", "P29",
    "P39", "P40", "P43", "P44", "P45", "P46", "P47", "P50", "P51", "P52",
]
EXCLUDED_PATIENTS = set(G5_PATIENTS)

PRETRAINED_WEIGHTS  = "yolo11n.pt"
EPOCHS_PER_FOLD     = 700
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

BG_RATIO            = int(os.environ.get("BG_RATIO", "1"))
FP_NEG_OVERSAMPLE   = int(os.environ.get("FP_NEG_OVERSAMPLE", "1"))

PATIENT_OVERSAMPLE_FIXED = {}
OVERSAMPLE_TARGET_RATIO  = 2.0
OVERSAMPLE_CAP           = 25

BASE_DATA_PATH = "/cfs/earth/scratch/vollmflo/BA/data"
PATIENT_LABEL_DIRS = {
    "P1": os.path.join(BASE_DATA_PATH, "new", "P1", "labels", "Train"),
}
P1_BG_DIR  = os.path.join(BASE_DATA_PATH, "old", "P1", "Negativ 12241515", "images", "train")
CLASS_NAMES = ["Atypisch", "Normal"]
NC          = len(CLASS_NAMES)

PROJECT_DIR = f"./{os.environ.get('SLURM_JOB_NAME', 'local_run_g1g4')}"
_JOB_ID     = os.environ.get("SLURM_JOB_ID", "local")
TEMP_DIR    = os.path.abspath(f".cv_temp_{_JOB_ID}")

os.makedirs(TEMP_DIR, exist_ok=True)


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Load annotated positives — exclude G5 patients entirely
    # ------------------------------------------------------------------
    pos_with_meta: list[tuple[str, str]] = []
    summary: dict[str, dict] = {}

    all_patients = sorted(PATIENT_IMAGE_DIRS.keys(), key=lambda p: int(p[1:]))
    for patient in all_patients:
        if patient in EXCLUDED_PATIENTS:
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
            atyp_count += sum(1 for _ in open(lbl) if _.split() and _.split()[0] == "0")
            norm_count  += sum(1 for _ in open(lbl) if _.split() and _.split()[0] == "1")

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

    print("\n=== Per-patient annotation summary (G1–G4 only; G5 excluded) ===")
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
    # 2. Load negatives — skip FP entries belonging to G5 patients
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
            if fp_patient in EXCLUDED_PATIENTS:
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
              f"  ({len(total_fp_missing)} unresolved)")

    # ------------------------------------------------------------------
    # 3. Build CV folds (grouped only — G1–G4)
    # ------------------------------------------------------------------
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

    fold_tasks = []
    print(f"\n=== Grouped {len(PATIENT_GROUPS)}-Fold CV (G1–G4; G5 withheld) ===")
    for fold_idx, (group_label, holdout_patients) in enumerate(PATIENT_GROUPS.items()):
        holdout_set    = set(holdout_patients)
        train_pos_full = [m for m in pos_with_meta if m[1] not in holdout_set]
        test_pos       = [m for m in pos_with_meta if m[1] in holdout_set]
        train_neg      = [m for m in neg_with_meta if m[1] not in holdout_set]
        train_pos, val_pos = _carve_val(train_pos_full)
        fold_tasks.append((fold_idx, group_label,
                           train_pos, val_pos, test_pos, train_neg,
                           oversample_factors))
        test_atyp = sum(summary[p]["atypisch"] for p in holdout_patients if p in summary)
        test_norm = sum(summary[p]["normal"]   for p in holdout_patients if p in summary)
        print(f"  Fold {fold_idx+1} ({group_label}): holdout={holdout_patients}, "
              f"train={len(train_pos)} pos / {len(train_neg)} neg, "
              f"val={len(val_pos)}, test={len(test_pos)} "
              f"[Atyp={test_atyp} Norm={test_norm}]")

    # ------------------------------------------------------------------
    # 4. Run training
    # ------------------------------------------------------------------
    n_folds      = len(fold_tasks)
    max_parallel = int(os.environ.get("MAX_PARALLEL", str(n_folds)))
    print(f"\nLaunching {n_folds} folds, {max_parallel} in parallel.")
    print(f"Project dir: {PROJECT_DIR}\n")

    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=max_parallel) as pool:
        final_results = pool.map(train_fold, fold_tasks)

    # ------------------------------------------------------------------
    # 5. Results
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
