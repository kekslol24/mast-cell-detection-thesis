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
# PRETRAINED_WEIGHTS = "DL_Modell_FV.pt"
PRETRAINED_WEIGHTS = "yolo11n.pt"
EPOCHS_PER_FOLD    = 5000
PATIENCE           = 50
BATCH_SIZE         = 32
IMGSZ              = 512
FP_NEG_OVERSAMPLE  = 3                  #First run with base nano yolo and no FP oversampling

BASE_DATA_PATH     = "/cfs/earth/scratch/vollmflo/BA/data/P2/1224151atypisch_normal"
FP_NEG_EXCEL_PATH  = "/cfs/earth/scratch/vollmflo/BA/data/P2/Task 6_1224151_negative.xlsx"
CFG_PATH           = "jobs/Slurm-269569 (tune 300x100)/runs/detect/tune/best_hyperparameters.yaml"
TEMP_DIR           = os.path.abspath("./cv_temp_isolated/")
PROCESSED_DIR      = os.path.abspath("./processed_data")
PROJECT_DIR        = "./yolo_runs_hpc_final"
CLASS_NAMES        = ["Atypisch", "Normal"]
NC                 = len(CLASS_NAMES)

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ==============================================================================
# AUGMENTATION PIPELINE (annotated images only)
# ==============================================================================
augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Affine(rotate=(-15, 15), p=0.3, mode=cv.BORDER_CONSTANT, cval=0),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# ==============================================================================
# PREPROCESSING: ANNOTATED IMAGES (augmentation + copy to PROCESSED_DIR)
# ==============================================================================
def preprocess_gold_image(img_path):
    img = cv.imread(img_path)
    if img is None:
        return None

    label_path = (img_path
                  .replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
                  .replace(".jpeg", ".txt"))
    bboxes, class_labels = [], []

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:]])

    try:
        transformed  = augmenter(image=img, bboxes=bboxes, class_labels=class_labels)
        img          = transformed['image']
        bboxes       = transformed['bboxes']
        class_labels = transformed['class_labels']
    except Exception:
        pass

    base_name    = os.path.basename(img_path)
    new_img_path = os.path.join(PROCESSED_DIR, base_name)
    cv.imwrite(new_img_path, img)

    new_label_path = new_img_path.replace(".jpeg", ".txt")
    with open(new_label_path, 'w') as f:
        for cls, box in zip(class_labels, bboxes):
            f.write(f"{cls} {' '.join([f'{x:.6f}' for x in box])}\n")

    return new_img_path

# ==============================================================================
# TRAINING 
# ==============================================================================
def train_fold(fold_params):
    fold_idx, train_idx, test_idx, X_gold, y_gold, fp_neg_paths = fold_params

    fold_workspace = os.path.join(TEMP_DIR, f"fold_{fold_idx}_workspace")
    fold_img_dir   = os.path.join(fold_workspace, "images")
    fold_lbl_dir   = os.path.join(fold_workspace, "labels")
    os.makedirs(fold_img_dir, exist_ok=True)
    os.makedirs(fold_lbl_dir, exist_ok=True)

    def link_gold(paths):
        """Symlink annotated images and their real label files."""
        linked = []
        for src in paths:
            base    = os.path.basename(src)
            img_dst = os.path.join(fold_img_dir, base)
            lbl_dst = os.path.join(fold_lbl_dir, base.replace(".jpeg", ".txt"))

            if not os.path.exists(img_dst):
                os.symlink(src, img_dst)

            src_lbl = src.replace(".jpeg", ".txt")  # already in PROCESSED_DIR flat structure
            if os.path.exists(src_lbl) and not os.path.exists(lbl_dst):
                os.symlink(src_lbl, lbl_dst)

            linked.append(img_dst)
        return linked

    def link_fp_negatives(paths):
        """Symlink FP negative images and write empty label files."""
        linked = []
        for src in paths:
            base    = os.path.basename(src)
            img_dst = os.path.join(fold_img_dir, base)
            lbl_dst = os.path.join(fold_lbl_dir, base.replace(".jpeg", ".txt"))

            if not os.path.exists(img_dst):
                os.symlink(src, img_dst)

            # Empty label = confirmed negative, no bounding boxes
            open(lbl_dst, 'w').close()

            linked.append(img_dst)
        return linked

    X_train_val = X_gold[train_idx]
    y_train_val = y_gold[train_idx]
    X_test      = X_gold[test_idx]

    X_train, X_val = train_test_split(
        X_train_val, test_size=0.15, stratify=y_train_val, random_state=42
    )

    train_paths = link_gold(X_train)
    val_paths   = link_gold(X_val)
    test_paths  = link_gold(X_test)

    # FP negatives: train split only, oversampled
    fp_linked           = link_fp_negatives(fp_neg_paths)
    train_paths_with_fp = train_paths + fp_linked * FP_NEG_OVERSAMPLE

    np.savetxt(os.path.join(fold_workspace, 'train.txt'), train_paths_with_fp, fmt='%s')
    np.savetxt(os.path.join(fold_workspace, 'val.txt'),   val_paths,           fmt='%s')
    np.savetxt(os.path.join(fold_workspace, 'test.txt'),  test_paths,          fmt='%s')

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

    # Fine-tune from existing weights
    model = YOLO(PRETRAINED_WEIGHTS)
    model.train(
        data     = yaml_path,
        # cfg      = CFG_PATH,
        epochs   = EPOCHS_PER_FOLD,
        patience = PATIENCE,
        batch    = BATCH_SIZE,
        device   = 0,
        project  = PROJECT_DIR,
        imgsz    = IMGSZ,
        name     = f"fold_{fold_idx + 1}",
        workers  = 0,
        device = gpu_id,
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
    # 1. Load annotated images and verify label files exist
    # ------------------------------------------------------------------
    all_raw = glob.glob(os.path.join(BASE_DATA_PATH, "images", "Train", "*.jpeg"))
 
    verified, missing_labels = [], []
    for img_path in all_raw:
        lbl_path = img_path.replace(
            os.sep + "images" + os.sep,
            os.sep + "labels" + os.sep
        ).replace(".jpeg", ".txt")
        if os.path.exists(lbl_path):
            verified.append(img_path)
        else:
            missing_labels.append(img_path)
 
    print(f"Annotated images : {len(verified)} with labels confirmed")
    if missing_labels:
        print(f"Skipped (no label): {len(missing_labels)}  (first 3: {missing_labels[:3]})")
 
    # ------------------------------------------------------------------
    # 2. Load FP negatives from Excel, resolve via filename index
    # ------------------------------------------------------------------
    print("\nIndexing all images in BASE_DATA_PATH...")
    all_disk = glob.glob(os.path.join(BASE_DATA_PATH, "**/*.jpeg"), recursive=True)
    filename_index = {os.path.basename(p): p for p in all_disk}
    print(f"Indexed {len(filename_index)} images on disk")
 
    fp_df        = pd.read_excel(FP_NEG_EXCEL_PATH, header=None)
    fp_filenames = fp_df[0].dropna().tolist()
 
    fp_neg_paths, fp_missing = [], []
    for fn in fp_filenames:
        basename = os.path.basename(str(fn))
        if basename in filename_index:
            fp_neg_paths.append(filename_index[basename])
        else:
            fp_missing.append(fn)
 
    print(f"FP negatives resolved : {len(fp_neg_paths)}")
    if fp_missing:
        print(f"Not found on disk     : {len(fp_missing)}  (first 5: {fp_missing[:5]})")
 
    # ------------------------------------------------------------------
    # 3. Preprocess annotated images (augmentation, copy to PROCESSED_DIR)
    #    FP negatives are used directly from disk -- no preprocessing needed
    # ------------------------------------------------------------------
    print(f"\nPreprocessing {len(verified)} annotated images...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        gold_processed = pool.map(preprocess_gold_image, verified)
 
    X_all, y_all = [], []
    for img_p in gold_processed:
        if img_p is not None:
            X_all.append(img_p)
            y_all.append(1)   # 1 = has mast cell annotations
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    print(f"Annotated images after preprocessing: {len(X_all)}")
 
    # ------------------------------------------------------------------
    # 4. Clear stale YOLO label cache
    # ------------------------------------------------------------------
    cache = os.path.join(PROCESSED_DIR, "labels.cache")
    if os.path.exists(cache):
        os.remove(cache)
 
    # ------------------------------------------------------------------
    # 5. Stratified K-Fold CV -- folds run in parallel across GPUs
    # ------------------------------------------------------------------
    skf        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_tasks = [
        (i, train_idx, test_idx, X_all, y_all, fp_neg_paths)
        for i, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all))
    ]
 
    print(f"\nStarting {N_SPLITS}-fold CV fine-tuning from: {PRETRAINED_WEIGHTS}")
    print(f"FP negatives added to each train split (x{FP_NEG_OVERSAMPLE} oversample)\n")
 
    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=N_SPLITS) as pool:
        final_results = pool.map(train_fold, fold_tasks)
 
    # ------------------------------------------------------------------
    # 6. Results
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
 