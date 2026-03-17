import os
import glob
import yaml
import numpy as np
import multiprocessing as mp
from ultralytics import YOLO
from sklearn.model_selection import KFold, train_test_split

# ==============================================================================
# KONFIGURATION
# ==============================================================================
N_SPLITS = 5
MODEL_SIZE = "yolo11n.pt"
EPOCHS_PER_FOLD = 100
BATCH_SIZE = 64 #Etwas kleiner, damit 5 Folds sicher parallel in den VRAM passen
BASE_DATA_PATH = "archive/Glass Defect Detection.v3i.yolov11"
TEMP_DIR = os.path.abspath("./cv_temp_parallel/")
PROJECT_DIR = "./yolo_runs_parallel"
CLASS_NAMES = ["defect", "glass"]
NC = len(CLASS_NAMES)

os.makedirs(TEMP_DIR, exist_ok=True)

def train_fold(fold_params):
    """Funktion für einen einzelnen Fold auf der geteilten GPU"""
    fold_idx, train_idx, val_idx, X_pool = fold_params
    
    # Alle nutzen dieselbe GPU (0)
    gpu_id = 0 
    
    X_train_val = X_pool[train_idx]
    X_train, X_val = train_test_split(X_train_val, test_size=0.15, random_state=42)
    
    # Dateien für diesen Fold
    train_txt = os.path.join(TEMP_DIR, f'fold_{fold_idx}_train.txt')
    val_txt   = os.path.join(TEMP_DIR, f'fold_{fold_idx}_val.txt')
    np.savetxt(train_txt, X_train, fmt='%s')
    np.savetxt(val_txt, X_val, fmt='%s')
    
    yaml_data = {
        'path': '', 
        'train': train_txt,
        'val': val_txt,
        'nc': NC,
        'names': CLASS_NAMES
    }
    yaml_path = os.path.join(TEMP_DIR, f'fold_{fold_idx}.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)

    print(f"--- START FOLD {fold_idx + 1} (Parallel auf GPU 0) ---")
    model = YOLO(MODEL_SIZE)
    model.train(
        data=yaml_path,
        epochs=EPOCHS_PER_FOLD,
        batch=BATCH_SIZE,
        device=gpu_id,
        project=PROJECT_DIR,
        name=f"fold_{fold_idx + 1}",
        workers=0,      # WICHTIG: Wenige Worker pro Prozess, da sie sich die CPUs teilen
        cache=False,    # Kein RAM-Caching, um OOM zu vermeiden
        exist_ok=True
    )
    return f"Fold {fold_idx + 1} fertig."

if __name__ == "__main__":
    IMAGE_FOLDER = os.path.join(BASE_DATA_PATH, "images")
    X = np.array(sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))))
    
    y_dummy = np.zeros(len(X))
    X_pool, _, _, _ = train_test_split(X, y_dummy, test_size=0.2, random_state=42)
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_tasks = []
    for i, (t_idx, v_idx) in enumerate(kf.split(X_pool)):
        fold_tasks.append((i, t_idx, v_idx, X_pool))

    # 'spawn' Methode für CUDA
    mp.set_start_method('spawn', force=True)
    
    print(f"Starte {N_SPLITS} Folds gleichzeitig auf einer GPU...")
    # Wir setzen processes=5, damit alle gleichzeitig starten
    with mp.Pool(processes=N_SPLITS) as pool:
        results = pool.map(train_fold, fold_tasks)

    for r in results:
        print(r)