"""
tune_hyperpara.py — Hyperparameter tuning on the production LOPO recipe, single fold.

Tunes from `yolo11n.pt` on Fold 4 (P4 holdout) — the average-difficulty fold from
P3-A. P4 is in train for every other fold, so tuning here generalises better than
tuning on a corner-case patient.

CRITICAL: this script tunes the *same recipe* that ba_improved_comb.py / ba_tinker.py
deploy — same anchor, same freeze, same augmentation flags. Tuning a different recipe
than what gets deployed is the P2-E failure mode (see experiment_log.md): the tuner
collapses to lr0=0.01 because it's silently optimising for from-scratch training, then
that yaml gets layered onto a fine-tuning recipe and overwrites pretrained features.

Usage (from hpc/ as cwd, set by run_tune.sh):
    python tinkering/tune_hyperpara.py

Env-var overrides:
    TUNE_HOLDOUT      patient to hold out as val/test (default: P4)
    TUNE_ITERATIONS   number of search iterations (default: 30)
    TUNE_EPOCHS       epochs per iteration (default: 50)

Output:
    runs/detect/tune/best_hyperparameters.yaml — apply via cfg=... in a follow-up
    LOPO run, but only after sanity-checking on one fold first.
"""

import os
import glob
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# Helpers + constants from the production script. cwd is hpc/ when this runs
# (run_tune.sh does `cd ..` from hpc/tinkering/), so the import resolves.
from ba_improved_comb import (
    PATIENT_IMAGE_DIRS, PATIENT_OVERSAMPLE,
    P1_BG_DIR, P2_FP_EXCEL, P2_ROOT,
    CLASS_NAMES, NC,
    glob_images, image_to_label_path, link_one, case_insensitive_resolve,
)

# ==============================================================================
# CONFIG
# ==============================================================================
HOLDOUT     = os.environ.get("TUNE_HOLDOUT", "P4")
ITERATIONS  = int(os.environ.get("TUNE_ITERATIONS", "100"))
EPOCHS      = int(os.environ.get("TUNE_EPOCHS", "300"))
BATCH_SIZE  = 32
IMGSZ       = 512
FREEZE      = 10
ANCHOR      = "yolo11n.pt"

WORKSPACE = os.path.abspath(
    f"./.tune_workspace_{HOLDOUT}_{os.environ.get('SLURM_JOB_ID', 'local')}"
)


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # 1. Load all positives across P1–P8 with patient tags
    # --------------------------------------------------------------------------
    pos_with_meta: list[tuple[str, str]] = []
    for patient, img_dir in PATIENT_IMAGE_DIRS.items():
        for img in glob_images(img_dir):
            lbl = image_to_label_path(img, patient=patient)
            if os.path.exists(lbl):
                pos_with_meta.append((img, patient))
    print(f"Positives loaded: {len(pos_with_meta)} from "
          f"{len(set(m[1] for m in pos_with_meta))} patients")

    # --------------------------------------------------------------------------
    # 2. Load negatives (P1 gold backgrounds + P2 hard FPs), tagged by patient
    # --------------------------------------------------------------------------
    neg_with_meta: list[tuple[str, str]] = []
    for p in glob_images(P1_BG_DIR):
        neg_with_meta.append((p, "P1"))
    if os.path.exists(P2_FP_EXCEL):
        p2_disk_dir = case_insensitive_resolve(P2_ROOT)
        p2_disk = (glob.glob(os.path.join(p2_disk_dir, "**/*.jpeg"), recursive=True)
                 + glob.glob(os.path.join(p2_disk_dir, "**/*.JPEG"), recursive=True))
        p2_index = {os.path.basename(p): p for p in p2_disk}
        fp_df = pd.read_excel(P2_FP_EXCEL, header=None)
        for fn in fp_df[0].dropna().tolist():
            base = os.path.basename(str(fn))
            if base in p2_index:
                neg_with_meta.append((p2_index[base], "P2"))
    print(f"Negatives loaded : {len(neg_with_meta)} "
          f"(P1 gold + P2 FP, tagged by patient)")

    # --------------------------------------------------------------------------
    # 3. Build the chosen LOPO fold (HOLDOUT held out as val+test)
    # --------------------------------------------------------------------------
    train_pos_full = [m for m in pos_with_meta if m[1] != HOLDOUT]
    test_pos       = [m for m in pos_with_meta if m[1] == HOLDOUT]
    train_neg      = [m for m in neg_with_meta if m[1] != HOLDOUT]

    if not train_pos_full or not test_pos:
        raise RuntimeError(
            f"Holdout '{HOLDOUT}' produced an empty split — check TUNE_HOLDOUT."
        )

    # In-train val carve-out (15%, stratified by patient)
    X = np.array([m[0] for m in train_pos_full])
    y = np.array([m[1] for m in train_pos_full])
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42,
    )
    train_pos = list(zip(X_tr.tolist(), y_tr.tolist()))
    val_pos   = list(zip(X_va.tolist(), y_va.tolist()))

    print(f"\nFold (holdout={HOLDOUT}):")
    print(f"  train: {len(train_pos)} pos / {len(train_neg)} neg")
    print(f"  val  : {len(val_pos)}")
    print(f"  test : {len(test_pos)}")

    # --------------------------------------------------------------------------
    # 4. Materialise the workspace via the production link_one helper.
    #    This applies the same per-patient oversample factors as ba_improved_comb.
    # --------------------------------------------------------------------------
    img_dir = os.path.join(WORKSPACE, "images")
    lbl_dir = os.path.join(WORKSPACE, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    train_paths: list[str] = []
    for src, patient in train_pos:
        n = PATIENT_OVERSAMPLE.get(patient, 1)
        train_paths.extend(link_one(src, img_dir, lbl_dir,
                                    oversample=n, is_negative=False, patient=patient))
    for src, patient in train_neg:
        train_paths.extend(link_one(src, img_dir, lbl_dir,
                                    oversample=1, is_negative=True, patient=patient))
    val_paths = []
    for src, patient in val_pos:
        val_paths.extend(link_one(src, img_dir, lbl_dir,
                                  oversample=1, is_negative=False, patient=patient))
    test_paths = []
    for src, patient in test_pos:
        test_paths.extend(link_one(src, img_dir, lbl_dir,
                                   oversample=1, is_negative=False, patient=patient))

    np.savetxt(os.path.join(WORKSPACE, "train.txt"), train_paths, fmt="%s")
    np.savetxt(os.path.join(WORKSPACE, "val.txt"),   val_paths,   fmt="%s")
    np.savetxt(os.path.join(WORKSPACE, "test.txt"),  test_paths,  fmt="%s")

    yaml_path = os.path.join(WORKSPACE, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump({
            "path":  WORKSPACE,
            "train": "train.txt",
            "val":   "val.txt",
            "test":  "test.txt",
            "nc":    NC,
            "names": CLASS_NAMES,
        }, f)
    print(f"\nWorkspace: {yaml_path}")

    # --------------------------------------------------------------------------
    # 5. Tune. Recipe MUST match production deployment (see docstring).
    # --------------------------------------------------------------------------
    print(f"\nTuning {ANCHOR}: {ITERATIONS} iterations × {EPOCHS} epochs, "
          f"freeze={FREEZE}, optimizer=AdamW")
    model = YOLO(ANCHOR)
    model.tune(
        data       = yaml_path,
        epochs     = EPOCHS,
        iterations = ITERATIONS,
        optimizer  = "AdamW",
        batch      = BATCH_SIZE,
        imgsz      = IMGSZ,
        freeze     = FREEZE,
        # Match production augmentation so the search space reflects deployment.
        augment    = True,
        mosaic     = 0.0,
        flipud     = 0.5,
        cls        = 1.0,
        plots      = False,
        val        = True,
        verbose    = True,
    )
