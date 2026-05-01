from ultralytics import YOLO
import os
import glob
import yaml
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DATA_PATH    = "/cfs/earth/scratch/vollmflo/BA/data/P2/1224151atypisch_normal"
FP_NEG_EXCEL_PATH = "/cfs/earth/scratch/vollmflo/BA/data/P2/Task 6_1224151_negative.xlsx"

FP_NEG_OVERSAMPLE = 3                        # repeat FP negatives N times in train.txt

VAL_SIZE          = 0.15
TEST_SIZE         = 0.15                     # of the original total

TUNE_WORKSPACE    = os.path.abspath("./tuning_dataset")
CLASS_NAMES       = ["Atypisch", "Normal"]

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
 
    # --------------------------------------------------------------------------
    # 1. Load annotated images and verify their labels exist
    # --------------------------------------------------------------------------
    annotated_images = glob.glob(os.path.join(BASE_DATA_PATH, "images/Train/*.jpeg"))

    missing_labels = []
    verified_images = []
    for img_path in annotated_images:
        lbl_path = img_path.replace(
            os.sep + "images" + os.sep,
            os.sep + "labels" + os.sep
        ).replace(".jpeg", ".txt")
        if os.path.exists(lbl_path):
            verified_images.append(img_path)
        else:
            missing_labels.append(img_path)

    annotated_images = verified_images
    print(f"Annotated images : {len(annotated_images)} with labels confirmed")
    if missing_labels:
        print(f"Skipped (no label): {len(missing_labels)}  (first 3: {missing_labels[:3]})")
    # --------------------------------------------------------------------------
    # 2. Build a filename -> full path index of everything in BASE_DATA_PATH
    #    so we can look up FP negatives by filename alone
    # --------------------------------------------------------------------------
    print("Indexing all images in BASE_DATA_PATH...")
    all_disk_images = glob.glob(os.path.join(BASE_DATA_PATH, "**/*.jpeg"), recursive=True)
    filename_index  = {os.path.basename(p): p for p in all_disk_images}
    print(f"Indexed {len(filename_index)} images on disk")
 
    # --------------------------------------------------------------------------
    # 3. Load FP negatives from Excel and resolve to full paths
    # --------------------------------------------------------------------------
    fp_df        = pd.read_excel(FP_NEG_EXCEL_PATH, header=None)
    fp_filenames = fp_df[0].dropna().tolist()
    print(f"First 3 entries: {fp_filenames[:3]}")
 
    neg_images, missing = [], []
    for fn in fp_filenames:
        basename = os.path.basename(fn)  # handle cases where Excel stores full or partial paths
        if basename in filename_index:
            neg_images.append(filename_index[basename])
        else:
            missing.append(fn)
 
    print(f"\nNegatives from Excel : {len(neg_images)} resolved")
    if missing:
        print(f"Not found on disk    : {len(missing)}  (first 5: {missing[:5]})")
 
    # --------------------------------------------------------------------------
    # 4. Combine with stratification labels
    #    1 = annotated (mast cells present)
    #    0 = confirmed negative (no mast cells)
    # --------------------------------------------------------------------------
    all_images = annotated_images + neg_images
    all_labels = [1] * len(annotated_images) + [0] * len(neg_images)
 
    X = np.array(all_images)
    y = np.array(all_labels)
 
    print(f"\nTotal: {len(X)} images  ({sum(y==1)} annotated / {sum(y==0)} negatives)")
 
    # --------------------------------------------------------------------------
    # 5. Stratified train / val / test split
    # --------------------------------------------------------------------------
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=y_trainval,
        random_state=42
    )
 
    print(f"\nSplit summary:")
    print(f"  Train : {len(X_train):>4}  ({sum(y_train==1)} annotated / {sum(y_train==0)} negatives)")
    print(f"  Val   : {len(X_val):>4}  ({sum(y_val==1)} annotated / {sum(y_val==0)} negatives)")
    print(f"  Test  : {len(X_test):>4}  ({sum(y_test==1)} annotated / {sum(y_test==0)} negatives)")
    print(f"\nNegatives oversampled x{FP_NEG_OVERSAMPLE} in train only")
    print(f"Effective train size: {len(X_train) + sum(y_train==0) * (FP_NEG_OVERSAMPLE - 1)}")
 
    # --------------------------------------------------------------------------
    # 6. Build workspace with symlinks and label files
    # --------------------------------------------------------------------------
    img_dir = os.path.join(TUNE_WORKSPACE, "images")
    lbl_dir = os.path.join(TUNE_WORKSPACE, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
 
    def link_split(paths, labels):
        linked = []
        for src, lbl in zip(paths, labels):
            base    = os.path.basename(src)
            img_dst = os.path.join(img_dir, base)
            lbl_dst = os.path.join(lbl_dir, base.replace(".jpeg", ".txt"))
 
            if not os.path.exists(img_dst):
                os.symlink(src, img_dst)
 
            if lbl == 0:
                # Confirmed negative -- empty label file
                open(lbl_dst, 'w').close()
            else:
                # Annotated -- symlink the real label file
                src_lbl = src.replace(
                    os.sep + "images" + os.sep,
                    os.sep + "labels" + os.sep
                ).replace(".jpeg", ".txt")
                if os.path.exists(src_lbl) and not os.path.exists(lbl_dst):
                    os.symlink(src_lbl, lbl_dst)
 
            linked.append(img_dst)
        return linked
 
    train_linked = link_split(X_train, y_train)
    val_linked   = link_split(X_val,   y_val)
    test_linked  = link_split(X_test,  y_test)
 
    # Oversample negatives in train only
    train_neg_paths = [p for p, lbl in zip(train_linked, y_train) if lbl == 0]
    train_final     = train_linked + train_neg_paths * (FP_NEG_OVERSAMPLE - 1)
 
    np.savetxt(os.path.join(TUNE_WORKSPACE, "train.txt"), train_final, fmt="%s")
    np.savetxt(os.path.join(TUNE_WORKSPACE, "val.txt"),   val_linked,  fmt="%s")
    np.savetxt(os.path.join(TUNE_WORKSPACE, "test.txt"),  test_linked, fmt="%s")
 
    # --------------------------------------------------------------------------
    # 7. Write data.yaml
    # --------------------------------------------------------------------------
    data_yaml_path = os.path.join(TUNE_WORKSPACE, "data.yaml")
    with open(data_yaml_path, "w") as f:
        yaml.dump({
            "path":  TUNE_WORKSPACE,
            "train": "train.txt",
            "val":   "val.txt",
            "test":  "test.txt",
            "nc":    len(CLASS_NAMES),
            "names": CLASS_NAMES,
        }, f)
 
    print(f"\nWorkspace ready: {data_yaml_path}")
 
    with open("jobs/Slurm-269569 (tune 300x100)/runs/detect/tune/best_hyperparameters.yaml", "r") as f:
        best_hps = yaml.safe_load(f)


    # --------------------------------------------------------------------------
    # 8. Tune from existing weights
    # --------------------------------------------------------------------------
    model = YOLO("DL_Modell_FV.pt")
 
    model.tune(
        batch=32,
        data       = data_yaml_path,
        epochs     = 300,
        iterations = 100,
        optimizer  = "auto",
        plots      = True,
        save       = True,
        val        = True,
        verbose    = True,
        **best_hps,
    )
 
