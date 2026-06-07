"""
eval_lopo_folds.py — Evaluate saved LOPO fold weights on their holdout test sets.

Use this when training completed (or OOM'd) with best.pt saved per fold but
fold_results.csv was never written. Reconstructs each fold's test workspace
from the same data sources as ba_improved_comb.py and runs model.val().

Fold-to-patient mapping is identical to ba_improved_comb (string-sorted patients,
fold_1 = first in sort order). Missing weights are skipped with a warning.

Usage (from hpc/):
    SLURM_JOB_NAME=<project_dir_name> python eval_lopo_folds.py
    MAX_PARALLEL=4 SLURM_JOB_NAME=local_run_comb python eval_lopo_folds.py
"""

import os
import gc
import yaml
import numpy as np
import pandas as pd
import torch
import multiprocessing as mp
from ultralytics import YOLO

from ba_improved_comb import (
    PATIENT_IMAGE_DIRS,
    CLASS_NAMES, NC,
    BATCH_SIZE, IMGSZ,
    image_to_label_path, load_patient_images, link_one,
)

PROJECT_DIR = f"./{os.environ.get('SLURM_JOB_NAME', 'local_run_comb')}"
_JOB_ID     = os.environ.get("SLURM_JOB_ID", "local")
TEMP_DIR    = os.path.abspath(f".eval_lopo_temp_{_JOB_ID}")
os.makedirs(TEMP_DIR, exist_ok=True)


def eval_fold(args):
    fold_idx, holdout, test_imgs = args

    best_pt = os.path.join(PROJECT_DIR, f"fold_{fold_idx + 1}_{holdout}",
                           "weights", "best.pt")
    if not os.path.exists(best_pt):
        print(f"  [SKIP] Fold {fold_idx + 1} ({holdout}): weights not found — {best_pt}")
        return None

    # Build a minimal test-only workspace so YOLO can resolve labels correctly.
    # (P1 images live under old/ but labels under new/ — link_one handles this.)
    fold_workspace = os.path.join(TEMP_DIR, f"fold_{fold_idx}_eval")
    fold_img_dir   = os.path.join(fold_workspace, "images")
    fold_lbl_dir   = os.path.join(fold_workspace, "labels")
    os.makedirs(fold_img_dir, exist_ok=True)
    os.makedirs(fold_lbl_dir, exist_ok=True)

    test_paths = []
    for src in test_imgs:
        test_paths.extend(link_one(src, fold_img_dir, fold_lbl_dir,
                                   oversample=1, is_negative=False,
                                   patient=holdout))

    np.savetxt(os.path.join(fold_workspace, "test.txt"), test_paths, fmt="%s")

    # Count GT instances per class so we can correctly mask per-class recall.
    # Ultralytics returns ap_class_index=[0] and r of length 1 when only class 1
    # has GT — both are wrong. Counting GT ourselves is the only reliable fix.
    gt_counts = {c: 0 for c in range(NC)}
    for src in test_imgs:
        lbl = image_to_label_path(src, patient=holdout)
        if os.path.exists(lbl):
            for ln in open(lbl):
                parts = ln.strip().split()
                if parts:
                    c = int(parts[0])
                    if c in gt_counts:
                        gt_counts[c] += 1

    yaml_path = os.path.join(fold_workspace, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump({
            "path":  fold_workspace,
            "train": "test.txt",
            "val":   "test.txt",
            "test":  "test.txt",
            "nc":    NC,
            "names": CLASS_NAMES,
        }, f)

    gpu_id = fold_idx % 2
    model  = YOLO(best_pt)
    metrics = model.val(data=yaml_path, split="test", verbose=False,
                        workers=0, device=gpu_id, batch=BATCH_SIZE)
    del model
    torch.cuda.empty_cache()
    gc.collect()

    def per_class(idx):
        if gt_counts.get(idx, 0) == 0:
            return float("nan")
        try:
            idx_map = list(metrics.box.ap_class_index)
            if idx in idx_map:
                pos = idx_map.index(idx)
            else:
                # ap_class_index is wrong; derive position from sorted GT classes
                gt_classes = sorted(c for c, n in gt_counts.items() if n > 0)
                pos = gt_classes.index(idx)
            arr = metrics.box.r
            return float(arr[pos]) if pos < len(arr) else float("nan")
        except (AttributeError, IndexError, TypeError):
            return float("nan")

    return {
        "Fold":                 fold_idx + 1,
        "Holdout":              holdout,
        "Test_mAP50-95":        metrics.box.map,
        "Test_mAP50":           metrics.box.map50,
        "Test_Precision":       metrics.box.mp,
        "Test_Recall":          metrics.box.mr,
        "Test_Recall_Atypisch": per_class(0),
        "Test_Recall_Normal":   per_class(1),
    }


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Build LOPO fold list — identical patient sort as ba_improved_comb.py
    # ------------------------------------------------------------------
    pos_with_meta: list[tuple[str, str]] = []
    for patient in sorted(PATIENT_IMAGE_DIRS.keys(), key=lambda p: int(p[1:])):
        for img in load_patient_images(patient):
            lbl = image_to_label_path(img, patient=patient)
            if os.path.exists(lbl):
                pos_with_meta.append((img, patient))

    # String sort (no key) — must match ba_improved_comb fold assignment.
    patients = sorted({m[1] for m in pos_with_meta})

    fold_tasks = []
    print(f"\n=== LOPO eval plan — {len(patients)} folds ===")
    for fold_idx, holdout in enumerate(patients):
        test_imgs = [m[0] for m in pos_with_meta if m[1] == holdout]
        best_pt   = os.path.join(PROJECT_DIR, f"fold_{fold_idx + 1}_{holdout}",
                                 "weights", "best.pt")
        status    = "OK" if os.path.exists(best_pt) else "MISSING"
        print(f"  Fold {fold_idx + 1:>2}: holdout={holdout:<4}  test={len(test_imgs):>4} imgs  weights={status}")
        fold_tasks.append((fold_idx, holdout, test_imgs))

    n_folds      = len(fold_tasks)
    max_parallel = int(os.environ.get("MAX_PARALLEL", str(n_folds)))
    print(f"\nEvaluating {n_folds} folds, up to {max_parallel} in parallel.")
    print(f"Project dir: {PROJECT_DIR}\n")

    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=max_parallel) as pool:
        raw_results = pool.map(eval_fold, fold_tasks)

    results = [r for r in raw_results if r is not None]
    if not results:
        print("No results — no weights found. Set SLURM_JOB_NAME to your project dir.")
        raise SystemExit(1)

    results_df = pd.DataFrame(results)
    csv_path   = os.path.join(PROJECT_DIR, "fold_results_eval.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}\n")

    # ------------------------------------------------------------------
    # Print results — same format as ba_improved_comb.py
    # ------------------------------------------------------------------
    cols = ["Fold", "Holdout"] + [c for c in results_df.columns if c.startswith("Test")]
    df   = results_df[cols].copy()
    df.columns = (["Fold", "Holdout"]
                  + [c.replace("Test_", "") for c in df.columns
                     if c not in ("Fold", "Holdout")])

    print("=" * 70)
    print("CV RESULTS — TEST")
    print("=" * 70)
    print(df.to_string(index=False))

    if "mAP50-95" in df.columns:
        print(f"\nMean mAP50-95 : {df['mAP50-95'].mean():.4f} +/- {df['mAP50-95'].std():.4f}")
        print(f"Mean Precision: {df['Precision'].mean():.4f} +/- {df['Precision'].std():.4f}")
        print(f"Mean Recall   : {df['Recall'].mean():.4f} +/- {df['Recall'].std():.4f}")

    if "Recall_Atypisch" in df.columns:
        ra = df["Recall_Atypisch"].dropna()
        rn = df["Recall_Normal"].dropna()
        if len(ra):
            print(f"Mean Recall Atypisch (folds with Atyp GT, n={len(ra)}): "
                  f"{ra.mean():.4f} +/- {ra.std():.4f}")
        if len(rn):
            print(f"Mean Recall Normal   (folds with Norm GT, n={len(rn)}): "
                  f"{rn.mean():.4f} +/- {rn.std():.4f}")
