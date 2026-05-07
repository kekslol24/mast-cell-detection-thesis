import os
import gc
import glob
import yaml
import numpy as np
import cv2 as cv
import pandas as pd
import multiprocessing as mp
import albumentations as A
import torch
from ultralytics import YOLO
from sklearn.model_selection import StratifiedKFold, train_test_split

# ==============================================================================
# CONFIGURATION
# ==============================================================================
N_SPLITS           = 5
MODEL_SIZE         = "yolo11n.pt"
EPOCHS_PER_FOLD    = 5000
PATIENCE           = 50
BATCH_SIZE         = 32
IMGSZ              = 512

BASE_DATA_PATH     = "/cfs/earth/scratch/vollmflo/BA/data"

# P2-only paths (HPC)
P2_POS_DIR         = os.path.join(BASE_DATA_PATH, "P2", "1224151atypisch_normal", "images", "Train")
P2_FP_EXCEL        = os.path.join(BASE_DATA_PATH, "P2", "Task 6_1224151_negative.xlsx")
P2_ROOT            = os.path.join(BASE_DATA_PATH, "P2", "1224151atypisch_normal")

CLASS_NAMES        = ["Atypisch", "Normal"]
NC                 = len(CLASS_NAMES)

# Job-scoped output dirs (avoid collisions with concurrent SLURM jobs)
PROJECT_DIR        = f"./{os.environ.get('SLURM_JOB_NAME', 'local_run_p2_fixed')}"
_JOB_ID            = os.environ.get("SLURM_JOB_ID", "local")
TEMP_DIR           = os.path.abspath(f".cv_temp_{_JOB_ID}")
PROCESSED_DIR      = os.path.abspath(f".processed_data_{_JOB_ID}")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ==============================================================================
# AUGMENTATION (annotated images only)
# ==============================================================================
# augmenter = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.2),
#     A.Affine(rotate=(-15, 15), p=0.3, mode=cv.BORDER_CONSTANT, cval=0),
# ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# ==============================================================================
# PREPROCESSING — augment positives, copy negatives unchanged, all to PROCESSED_DIR
# ==============================================================================
def preprocess_image(img_path):
    img = cv.imread(img_path)
    if img is None:
        return None

    ext        = os.path.splitext(img_path)[1]
    label_path = (img_path.replace(os.sep + "images" + os.sep,
                                   os.sep + "labels" + os.sep)
                          .replace(ext, ".txt"))

    bboxes, class_labels = [], []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:]])

    # # Only augment if there are bboxes — negatives stay as-is
    # if bboxes:
    #     try:
    #         transformed  = augmenter(image=img, bboxes=bboxes, class_labels=class_labels)
    #         img          = transformed['image']
    #         bboxes       = transformed['bboxes']
    #         class_labels = transformed['class_labels']
    #     except Exception:
    #         pass

    base_name    = os.path.basename(img_path)
    new_img_path = os.path.join(PROCESSED_DIR, base_name)
    cv.imwrite(new_img_path, img)

    new_label_path = os.path.splitext(new_img_path)[0] + ".txt"
    with open(new_label_path, 'w') as f:
        for cls, box in zip(class_labels, bboxes):
            f.write(f"{cls} {' '.join([f'{x:.6f}' for x in box])}\n")
    # Negatives get an empty .txt — confirms to YOLO that nothing should be detected

    return new_img_path

# ==============================================================================
# TRAINING (one fold per process, two GPUs interleaved)
# ==============================================================================
def train_fold(fold_params):
    fold_idx, train_idx, test_idx, X_all, y_all = fold_params

    fold_workspace = os.path.join(TEMP_DIR, f"fold_{fold_idx}_workspace")
    fold_img_dir   = os.path.join(fold_workspace, "images")
    fold_lbl_dir   = os.path.join(fold_workspace, "labels")
    os.makedirs(fold_img_dir, exist_ok=True)
    os.makedirs(fold_lbl_dir, exist_ok=True)

    def link_data(paths):
        linked = []
        for src in paths:
            base    = os.path.basename(src)
            img_dst = os.path.join(fold_img_dir, base)
            if not os.path.exists(img_dst):
                os.symlink(src, img_dst)

            src_lbl = os.path.splitext(src)[0] + ".txt"
            lbl_dst = os.path.join(fold_lbl_dir, os.path.splitext(base)[0] + ".txt")
            if os.path.exists(src_lbl) and not os.path.exists(lbl_dst):
                os.symlink(src_lbl, lbl_dst)

            linked.append(img_dst)
        return linked

    X_train_val = X_all[train_idx]
    y_train_val = y_all[train_idx]
    X_test      = X_all[test_idx]

    X_train, X_val = train_test_split(
        X_train_val, test_size=0.15, stratify=y_train_val, random_state=42
    )

    train_paths = link_data(X_train)
    val_paths   = link_data(X_val)
    test_paths  = link_data(X_test)

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

    model = YOLO(MODEL_SIZE)
    model.train(
        data     = yaml_path,
        epochs   = EPOCHS_PER_FOLD,
        patience = PATIENCE,
        batch    = BATCH_SIZE,
        device   = gpu_id,
        project  = PROJECT_DIR,
        imgsz    = IMGSZ,
        name     = f"fold_{fold_idx + 1}",
        workers  = 0,
        cache    = False,
        exist_ok = True,
        verbose  = False,
        augment  = False,
        cos_lr   = True,
    )

    del model
    torch.cuda.empty_cache()
    gc.collect()

    best_weights = os.path.join(PROJECT_DIR, f"fold_{fold_idx + 1}", 'weights', 'best.pt')
    val_model    = YOLO(best_weights)

    metrics_test  = val_model.val(data=yaml_path, split='test',  verbose=False, workers=0, device=gpu_id, batch=BATCH_SIZE)
    metrics_val   = val_model.val(data=yaml_path, split='val',   verbose=False, workers=0, device=gpu_id, batch=BATCH_SIZE)
    metrics_train = val_model.val(data=yaml_path, split='train', verbose=False, workers=0, device=gpu_id, batch=BATCH_SIZE)

    return {
        'Fold':            fold_idx + 1,
        'Train_mAP50-95':  metrics_train.box.map,
        'Train_mAP50':     metrics_train.box.map50,
        'Train_Precision': metrics_train.box.mp,
        'Train_Recall':    metrics_train.box.mr,
        'Val_mAP50-95':    metrics_val.box.map,
        'Val_mAP50':       metrics_val.box.map50,
        'Val_Precision':   metrics_val.box.mp,
        'Val_Recall':      metrics_val.box.mr,
        'Test_mAP50-95':   metrics_test.box.map,
        'Test_mAP50':      metrics_test.box.map50,
        'Test_Precision':  metrics_test.box.mp,
        'Test_Recall':     metrics_test.box.mr,
    }

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Load P2 annotated positives
    # ------------------------------------------------------------------
    raw_positives = glob.glob(os.path.join(P2_POS_DIR, "*.jpeg"))
    print(f"P2 candidate images: {len(raw_positives)} in {P2_POS_DIR}")

    verified_pos, missing_labels = [], []
    for img_path in raw_positives:
        ext      = os.path.splitext(img_path)[1]
        lbl_path = (img_path.replace(os.sep + "images" + os.sep,
                                     os.sep + "labels" + os.sep)
                            .replace(ext, ".txt"))
        if os.path.exists(lbl_path):
            verified_pos.append(img_path)
        else:
            missing_labels.append(img_path)

    print(f"Positives verified : {len(verified_pos)} with labels confirmed")
    if missing_labels:
        print(f"Skipped (no label) : {len(missing_labels)}  (first 3: {missing_labels[:3]})")

    # ------------------------------------------------------------------
    # 2. Load P2 FP negatives (Excel-listed hard negatives)
    # ------------------------------------------------------------------
    print("\nIndexing P2 disk for FP resolution...")
    p2_disk  = glob.glob(os.path.join(P2_ROOT, "**/*.jpeg"), recursive=True)
    p2_index = {os.path.basename(p): p for p in p2_disk}

    fp_df        = pd.read_excel(P2_FP_EXCEL, header=None)
    fp_filenames = fp_df[0].dropna().tolist()

    fp_paths, fp_missing = [], []
    for fn in fp_filenames:
        basename = os.path.basename(str(fn))
        if basename in p2_index:
            fp_paths.append(p2_index[basename])
        else:
            fp_missing.append(fn)

    print(f"P2 FP negatives    : {len(fp_paths)} resolved")
    if fp_missing:
        print(f"FP not found       : {len(fp_missing)}  (first 5: {fp_missing[:5]})")

    # ------------------------------------------------------------------
    # 3. Combine sources and assign stratification labels
    #    0 = positive (annotated)
    #    1 = negative (FP — confirmed empty)
    # ------------------------------------------------------------------
    all_images = verified_pos + fp_paths
    all_strata = [0] * len(verified_pos) + [1] * len(fp_paths)

    print(f"\nP2 dataset:")
    print(f"  Positives   : {len(verified_pos)}")
    print(f"  FP negatives: {len(fp_paths)}")
    print(f"  TOTAL       : {len(all_images)}")
    print(f"  Pos:Neg     : {len(verified_pos)}:{len(fp_paths)}")

    # ------------------------------------------------------------------
    # 4. Preprocess in parallel (augment positives, copy negatives)
    # ------------------------------------------------------------------
    print(f"\nPreprocessing {len(all_images)} images...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        processed = pool.map(preprocess_image, all_images)

    X_all, y_all = [], []
    for img_p, stratum in zip(processed, all_strata):
        if img_p is not None:
            X_all.append(img_p)
            y_all.append(stratum)
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    print(f"After preprocessing: {len(X_all)} images")

    # ------------------------------------------------------------------
    # 5. Clear stale YOLO label cache
    # ------------------------------------------------------------------
    cache = os.path.join(PROCESSED_DIR, "labels.cache")
    if os.path.exists(cache):
        os.remove(cache)

    # ------------------------------------------------------------------
    # 6. Stratified K-Fold — folds run in parallel across GPUs
    #    Stratification keeps the positive:negative ratio constant per fold
    # ------------------------------------------------------------------
    skf        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_tasks = [
        (i, train_idx, test_idx, X_all, y_all)
        for i, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all))
    ]

    print(f"\nStarting {N_SPLITS}-fold CV training from: {MODEL_SIZE}")
    print(f"Project dir: {PROJECT_DIR}\n")

    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=N_SPLITS) as pool:
        final_results = pool.map(train_fold, fold_tasks)

    # ------------------------------------------------------------------
    # 7. Results
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(final_results)

    for split in ['Train', 'Val', 'Test']:
        cols  = ['Fold'] + [c for c in results_df.columns if c.startswith(split)]
        df    = results_df[cols].copy()
        df.columns = ['Fold'] + [c.replace(f'{split}_', '') for c in df.columns if c != 'Fold']
        print("\n" + "=" * 55)
        print(f"STRATIFIED CV RESULTS - {split.upper()}")
        print("=" * 55)
        print(df.to_string(index=False))
        print(f"\nMean mAP50-95 : {df['mAP50-95'].mean():.4f} +/- {df['mAP50-95'].std():.4f}")
        print(f"Mean Precision: {df['Precision'].mean():.4f} +/- {df['Precision'].std():.4f}")
        print(f"Mean Recall   : {df['Recall'].mean():.4f} +/- {df['Recall'].std():.4f}")
