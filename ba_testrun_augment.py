import os
import glob
import yaml
import shutil
import numpy as np
import cv2 as cv
import pandas as pd
import multiprocessing as mp
import albumentations as A
import torch
from ultralytics import YOLO
from sklearn.model_selection import StratifiedKFold, train_test_split

# ==============================================================================
# KONFIGURATION
# ==============================================================================
N_SPLITS = 5
MODEL_SIZE = "yolo11l.pt"
EPOCHS_PER_FOLD = 200
BATCH_SIZE = 16 # Reduziert für Large Modell
BASE_DATA_PATH = "/cfs/earth/scratch/vollmflo/BA/data" 
TEMP_DIR = os.path.abspath("./cv_temp_isolated/")
PROCESSED_DIR = os.path.abspath("./processed_data")
PROJECT_DIR = "./yolo_runs_hpc_final"
CLASS_NAMES = ["Atypisch", "Normal"]
NC = len(CLASS_NAMES)

# --- AUGMENTATION PIPELINE ---
augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(rotate_limit=15, p=0.3, border_mode=cv.BORDER_CONSTANT, value=0),
    A.GaussNoise(p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ==============================================================================
# PREPROCESSING FUNKTIONEN (INKL. AUGMENTATION)
# ==============================================================================
def apply_smart_filter(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask = (gray < 100).astype(np.uint8) * 255
    kernel = np.ones((30, 30), np.uint8)
    cleaned_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    img[cleaned_mask == 255] = [0, 0, 0]
    return img

def preprocess_image(img_path):
    img = cv.imread(img_path)
    if img is None: return None
    
    # 1. Filter
    img = apply_smart_filter(img)
    
    # 2. Labels laden
    label_path = img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).replace(".jpg", ".txt")
    bboxes = []
    class_labels = []
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:]])

    # 3. Augmentierung anwenden (Synchronisiert mit BBoxes)
    if bboxes:
        try:
            transformed = augmenter(image=img, bboxes=bboxes, class_labels=class_labels)
            img = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        except Exception:
            pass # Falls Augmentation fehlschlägt, Originalbild nutzen

    # 4. Speichern
    base_name = os.path.basename(img_path)
    new_img_path = os.path.join(PROCESSED_DIR, base_name)
    cv.imwrite(new_img_path, img)
    
    # 5. Labels im neuen Ordner speichern
    new_label_path = new_img_path.replace(".jpg", ".txt")
    with open(new_label_path, 'w') as f:
        for cls, box in zip(class_labels, bboxes):
            f.write(f"{cls} {' '.join([f'{x:.6f}' for x in box])}\n")
    
    return new_img_path

# ==============================================================================
# PARALLELES TRAINING (TRAIN FOLD)
# ==============================================================================
def train_fold(fold_params):
    fold_idx, train_idx, test_idx, X_all, y_all = fold_params
    
    fold_workspace = os.path.join(TEMP_DIR, f"fold_{fold_idx}_workspace")
    fold_img_dir = os.path.join(fold_workspace, "images")
    fold_lbl_dir = os.path.join(fold_workspace, "labels")
    
    os.makedirs(fold_img_dir, exist_ok=True)
    os.makedirs(fold_lbl_dir, exist_ok=True)

    def link_data(paths):
        linked_paths = []
        for src in paths:
            base = os.path.basename(src)
            img_dst = os.path.join(fold_img_dir, base)
            if not os.path.exists(img_dst):
                os.symlink(src, img_dst)
            
            src_lbl = src.replace(".jpg", ".txt")
            lbl_dst = os.path.join(fold_lbl_dir, base.replace(".jpg", ".txt"))
            if os.path.exists(src_lbl) and not os.path.exists(lbl_dst):
                os.symlink(src_lbl, lbl_dst)
                
            linked_paths.append(img_dst)
        return linked_paths

    X_train_val, X_test = X_all[train_idx], X_all[test_idx]
    y_train_val = y_all[train_idx]
    X_train, X_val = train_test_split(X_train_val, test_size=0.15, stratify=y_train_val, random_state=42)

    train_paths = link_data(X_train)
    val_paths   = link_data(X_val)
    test_paths  = link_data(X_test)

    txt_paths = {
        'train': os.path.join(fold_workspace, 'train.txt'),
        'val': os.path.join(fold_workspace, 'val.txt'),
        'test': os.path.join(fold_workspace, 'test.txt')
    }
    np.savetxt(txt_paths['train'], train_paths, fmt='%s')
    np.savetxt(txt_paths['val'], val_paths, fmt='%s')
    np.savetxt(txt_paths['test'], test_paths, fmt='%s')

    yaml_path = os.path.join(fold_workspace, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump({
            'path': fold_workspace, 
            'train': 'train.txt', 'val': 'val.txt', 'test': 'test.txt',
            'nc': NC, 'names': CLASS_NAMES
        }, f)

    gpu_id = fold_idx % 2
    
    # --- TRAINING ---
    model = YOLO(MODEL_SIZE)
    model.train(
        data=yaml_path, epochs=EPOCHS_PER_FOLD, batch=BATCH_SIZE,
        device=gpu_id, project=PROJECT_DIR, name=f"fold_{fold_idx + 1}",
        workers=0, cache=False, exist_ok=True, verbose=False,
        augment=False # Wir haben bereits Albumentations genutzt
    )
    
    # Speicher freigeben vor Validierung (Wichtig für Large Modell)
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # --- EVALUATION ---
    best_weights = os.path.join(PROJECT_DIR, f"fold_{fold_idx + 1}", 'weights', 'best.pt')
    val_model = YOLO(best_weights)
    metrics = val_model.val(
        data=yaml_path, split='test', verbose=False, 
        workers=0, device=gpu_id, batch=BATCH_SIZE
    )
    
    return {
        'Fold': fold_idx + 1,
        'mAP50-95': metrics.box.map,
        'Precision': metrics.box.mp,
        'Recall': metrics.box.mr
    }

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    pos_images = glob.glob(os.path.join(BASE_DATA_PATH, "Pos_neg*/images/Train/*.jpg"))
    neg_images = glob.glob(os.path.join(BASE_DATA_PATH, "Negativ*/images/train/*.jpg"))
    
    all_raw_images = pos_images + neg_images
    raw_labels = [0] * len(pos_images) + [1] * len(neg_images)

    print(f"Starte Preprocessing & Augmentation für {len(all_raw_images)} Bilder...")
    # Preprocessing ist CPU-lastig, processes=2 oder höher je nach Node
    with mp.Pool(processes=8) as p:
        processed_results = p.map(preprocess_image, all_raw_images)

    X_all, y_all = [], []
    for img_p, lbl in zip(processed_results, raw_labels):
        if img_p is not None:
            X_all.append(img_p)
            y_all.append(lbl)
            
    X_all = np.array(X_all)
    y_all = np.array(y_all)

    if len(X_all) > 0:
        global_cache = os.path.join(PROCESSED_DIR, "labels.cache")
        if os.path.exists(global_cache):
            os.remove(global_cache)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_tasks = []
    for i, (t_idx, v_idx) in enumerate(skf.split(X_all, y_all)):
        fold_tasks.append((i, t_idx, v_idx, X_all, y_all))

    mp.set_start_method('spawn', force=True)
    
    # HINWEIS: Bei 2 GPUs und Large Modell maximal processes=2
    with mp.Pool(processes=2) as pool:
        final_results = pool.map(train_fold, fold_tasks)

    results_df = pd.DataFrame(final_results)
    print("\n" + "="*50)
    print("STRATIFIED CV ERGEBNISSE (WITH ALBUMENTATIONS)")
    print("="*50)
    print(results_df.to_string(index=False))
    print(f"\nØ mAP50-95: {results_df['mAP50-95'].mean():.4f}")