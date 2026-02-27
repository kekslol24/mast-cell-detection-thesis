import os
import glob
import yaml
import shutil
import numpy as np
import cv2 as cv
import pandas as pd
import multiprocessing as mp
from ultralytics import YOLO
from sklearn.model_selection import StratifiedKFold, train_test_split

# ==============================================================================
# KONFIGURATION
# ==============================================================================
N_SPLITS = 5
MODEL_SIZE = "yolo11l.pt"
EPOCHS_PER_FOLD = 20
BATCH_SIZE = 16           
BASE_DATA_PATH = "/cfs/earth/scratch/vollmflo/BA/data" 
TEMP_DIR = os.path.abspath("./cv_temp_isolated/")
PROCESSED_DIR = os.path.abspath("./processed_data")
PROJECT_DIR = "./yolo_runs_hpc_final"
CLASS_NAMES = ["Atypisch", "Normal"]
NC = len(CLASS_NAMES)

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ==============================================================================
# PREPROCESSING FUNKTIONEN
# ==============================================================================
def apply_smart_filter(img):
    """Morphologischer Filter gegen Hintergrundrauschen"""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask = (gray < 100).astype(np.uint8) * 255
    kernel = np.ones((30, 30), np.uint8)
    cleaned_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    img[cleaned_mask == 255] = [0, 0, 0]
    return img

def preprocess_image(img_path):
    """Filtert Bild, kopiert Labels und gibt neuen Pfad zurück"""
    img = cv.imread(img_path)
    if img is None: return None
    
    filtered_img = apply_smart_filter(img)
    base_name = os.path.basename(img_path)
    new_path = os.path.join(PROCESSED_DIR, base_name)
    cv.imwrite(new_path, filtered_img)
    
    # Label-Logik
    label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
    if os.path.exists(label_path):
        shutil.copy(label_path, new_path.replace(".jpg", ".txt"))
    
    return new_path

# ==============================================================================
# PARALLELES TRAINING (TRAIN FOLD)
# ==============================================================================
def train_fold(fold_params):
    fold_idx, train_idx, test_idx, X_all, y_all = fold_params
    
    # 1. ABSOLUTE ISOLATION: Eigener Workspace pro Fold
    fold_workspace = os.path.join(TEMP_DIR, f"fold_{fold_idx}_workspace")
    fold_img_dir = os.path.join(fold_workspace, "images")
    fold_lbl_dir = os.path.join(fold_workspace, "labels") # NEU: Der Labels-Ordner
    
    os.makedirs(fold_img_dir, exist_ok=True)
    os.makedirs(fold_lbl_dir, exist_ok=True)

    # 2. SYMLINKS ERSTELLEN
    def link_data(paths):
        linked_paths = []
        for src in paths:
            base = os.path.basename(src)
            img_dst = os.path.join(fold_img_dir, base)
            
            if not os.path.exists(img_dst):
                os.symlink(src, img_dst)
            
            # Label verlinken
            # Im PROCESSED_DIR liegen die Labels direkt neben den Bildern
            src_lbl = src.replace(".jpg", ".txt")
            lbl_dst = os.path.join(fold_lbl_dir, base.replace(".jpg", ".txt"))
            
            if os.path.exists(src_lbl) and not os.path.exists(lbl_dst):
                os.symlink(src_lbl, lbl_dst)
                
            linked_paths.append(img_dst)
        return linked_paths

    X_train_val, X_test = X_all[train_idx], X_all[test_idx]
    y_train_val = y_all[train_idx]
    X_train, X_val = train_test_split(X_train_val, test_size=0.15, stratify=y_train_val, random_state=42)

    # Verlinken und lokale Pfadlisten erstellen
    train_paths = link_data(X_train)
    val_paths   = link_data(X_val)
    test_paths  = link_data(X_test)

    # 3. YAML & TXT
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
    # 4. TRAINING (workers=0 ist Pflicht!)
    model = YOLO(MODEL_SIZE)
    model.train(
        data=yaml_path, epochs=EPOCHS_PER_FOLD, batch=BATCH_SIZE,
        device=gpu_id, project=PROJECT_DIR, name=f"fold_{fold_idx + 1}",
        workers=0, cache=False, exist_ok=True, verbose=False
    )
    
    # 5. EVALUATION (workers=0 fixiert den Daemon-Fehler)
    best_weights = os.path.join(PROJECT_DIR, f"fold_{fold_idx + 1}", 'weights', 'best.pt')
    val_model = YOLO(best_weights)
    metrics = val_model.val(data=yaml_path, split='test', verbose=False, workers=0, device=gpu_id)
    
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
    # 1. Daten sammeln & Labels für Stratifizierung bestimmen
    pos_images = glob.glob(os.path.join(BASE_DATA_PATH, "Pos_neg*/images/Train/*.jpg"))
    neg_images = glob.glob(os.path.join(BASE_DATA_PATH, "Negativ*/images/train/*.jpg"))
    
    all_raw_images = pos_images + neg_images
    raw_labels = [0] * len(pos_images) + [1] * len(neg_images)

    print(f"Starte Preprocessing für {len(all_raw_images)} Bilder...")
    with mp.Pool(processes=2) as p:
        processed_results = p.map(preprocess_image, all_raw_images)

    # Synchrones Filtern von X und y
    X_all, y_all = [], []
    for img_p, lbl in zip(processed_results, raw_labels):
        if img_p is not None:
            X_all.append(img_p)
            y_all.append(lbl)
            
    X_all = np.array(X_all)
    y_all = np.array(y_all)

    # 2. Cache-Reste löschen
    if len(X_all) > 0:
        global_cache = os.path.join(os.path.dirname(X_all[0]), "labels.cache")
        if os.path.exists(global_cache):
            os.remove(global_cache)
            print("Alte globale Cache-Datei gelöscht.")

    # 3. Stratified K-Fold Setup
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_tasks = []
    for i, (t_idx, v_idx) in enumerate(skf.split(X_all, y_all)):
        fold_tasks.append((i, t_idx, v_idx, X_all, y_all))

    # 4. Paralleles Training
    mp.set_start_method('spawn', force=True)
    print(f"Starte {N_SPLITS} Folds parallel auf GPU 0...")
    
    # HINWEIS: Bei OOM-Fehler auf mp.Pool(processes=2) reduzieren
    with mp.Pool(processes=N_SPLITS) as pool:
        final_results = pool.map(train_fold, fold_tasks)

    # 5. Auswertung
    results_df = pd.DataFrame(final_results)
    print("\n" + "="*50)
    print("STRATIFIED CV ERGEBNISSE")
    print("="*50)
    print(results_df.to_string(index=False))
    print(f"\nØ mAP50-95: {results_df['mAP50-95'].mean():.4f}")