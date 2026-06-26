import os
import glob
from zipfile import ZipFile
from ultralytics import YOLO
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from collections import defaultdict
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, required=True)
args = parser.parse_args()
fold = args.fold

path = os.getcwd()
print(path)

model = YOLO("yolo11n.pt")

# ==============================================================================
# 0. KONFIGURATION & INITIALISIERUNG
# ==============================================================================

# --- ANPASSEN SIE DIESE VARIABLEN AN IHRE UMGEBUNG ---
N_SPLITS = 5
MODEL_SIZE = "yolo11n.pt"
EPOCHS_PER_FOLD = 100
TEST_SIZE = 0.2

# Absolute Basis-Pfade (WICHTIG!)
BASE_DATA_PATH = "archive/Glass Defect Detection.v3i.yolov11" # Z.B. /home/user/yolo_data/
TEMP_DIR = os.path.abspath("./cv_temp/") # Temporäre Dateien werden hier gespeichert

# YOLO-Klassendefinition
### Dataset B
# CLASS_NAMES = ["defect"] # Ihre tatsächlichen Klassen
# NC = len(CLASS_NAMES)

# Dataset A
CLASS_NAMES = ["defect", "glass"] # Ihre tatsächlichen Klassen
NC = len(CLASS_NAMES)
# --------------------------------------------------------

# Ordner für temporäre Dateien erstellen
os.makedirs(TEMP_DIR, exist_ok=True)
all_fold_metrics = defaultdict(list)
results_df = None

IMAGE_FOLDER = os.path.join(BASE_DATA_PATH, "images")

# Lists all images and saves them to X
X = np.array(sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))))

total_samples = len(X)
print(f"Loaded Images: {total_samples}")

# Dummy-y for KFold
y = np.zeros(total_samples)

# Wir splitten die Daten in einen Trainings-Pool (X_pool) und ein finales Test-Set (X_test).
# Da nur eine Klasse existiert, nutzen wir train_test_split ohne 'stratify'.
X_pool, X_test, _, _ = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=42,
    shuffle=True
)

print(f"Gesamtdaten: {len(X)}")
print(f"Test-Set (20%): {len(X_test)} Samples. Diese werden f.d. CV-Lauf IGNORIERT.")
print(f"Trainings-Pool (80%): {len(X_pool)} Samples.")

np.sum(y)

# ==============================================================================
#  CV OHNE HOLDOUT (TRAIN / VAL / TEST pro Fold)
# ==============================================================================

# KFold initialisieren
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
splits = list(kf.split(X, y))
train_val_index, test_index = splits[fold]

# Listen für Ergebnisse
fold_test_metrics = []

for fold, (train_val_index, test_index) in enumerate(kf.split(X, y)): ## Für non stratified -> kf.split
    print(f"\n==================================================")
    print(f"--- FOLD {fold+1}/{N_SPLITS} ---")
    print(f"==================================================")

    # 1. Erster Split: Trennung in (Train+Val) und (Test)
    # Das Test-Set ist hier der "Test"-Teil vom KFold (z.B. 20%)
    X_train_val = X[train_val_index]
    y_train_val = y[train_val_index]
    
    X_test = X[test_index]
    y_test = y[test_index] # Wird für split nicht gebraucht, aber der Vollständigkeit halber

    # 2. Zweiter Split (Nested): Trennung von (Train+Val) in Train und Val
    # Wir nehmen z.B. 10% bis 20% der Train+Val Daten für die Validierung
    X_train, X_val = train_test_split(
        X_train_val, 
        test_size=0.15,  # 15% von den verbleibenden 80% sind ca. 12% der Gesamtdaten
        # stratify=y_train_val, 
        random_state=42
    )

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 3. Pfade für temporäre Textdateien
    train_file_path = os.path.join(TEMP_DIR, f'fold_{fold}_train.txt')
    val_file_path   = os.path.join(TEMP_DIR, f'fold_{fold}_val.txt')
    test_file_path  = os.path.join(TEMP_DIR, f'fold_{fold}_test.txt')

    np.savetxt(train_file_path, X_train, fmt='%s')
    np.savetxt(val_file_path, X_val, fmt='%s')
    np.savetxt(test_file_path, X_test, fmt='%s')

    # 4. YAML erstellen (jetzt inklusive 'test' Pfad)
    temp_yaml_data = {
        'path': "",
        'train': train_file_path,
        'val': val_file_path,
        'test': test_file_path,  # Wichtig für model.val(split='test')
        'nc': NC,
        'names': CLASS_NAMES
    }
    
    temp_yaml_path = os.path.join(TEMP_DIR, f'fold_{fold}.yaml')
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(temp_yaml_data, f)

    # 5. Training
    model = YOLO(MODEL_SIZE)
    project_dir = "project_runs_array"
    import shutil

# Pfad zum Ordner, in dem die Bilder liegen
    cache_file = os.path.join(BASE_DATA_PATH, "labels.cache")
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print("Alte Cache-Datei gelöscht. Scanne Labels neu...")

    run_name = f"cv_fold:{fold}"

    model.train(
        data=temp_yaml_path,
        epochs=EPOCHS_PER_FOLD,
        name=run_name,
        project=project_dir,
        workers=8,
        batch=64,
        cache=False,
        val=True,   # Validiert auf 'val_file_path' während Training
        lr0=0.0001,
        verbose=False # Weniger Output um Notebook sauber zu halten
    )

    # 6. Evaluation auf dem TEST-Set dieses Folds
    print(f"-> Starte Evaluation auf Test-Set für Fold {fold+1}...")
    
    # Wir laden explizit die besten Gewichte für die Test-Prüfung
    best_weights = os.path.join(project_dir, run_name, 'weights', 'best.pt')
    val_model = YOLO(best_weights)
    
    metrics = val_model.val(
        data=temp_yaml_path,
        split='test',  # Nutzt 'test_file_path'
        project=project_dir,
        name=f"{run_name}_test_eval"
    )


    results_file = f"results_fold_{fold}.txt"
    
    # Ergebnisse speichern
    fold_result = {
        'fold': fold + 1,
        'mAP50': metrics.box.map50,
        'mAP50-95': metrics.box.map,
        'precision': metrics.box.p, # Beachten: ist oft eine Liste pro Klasse oder Mean
        'recall': metrics.box.r
    }
    
    # Falls Precision/Recall Arrays sind (pro Klasse), nehmen wir den Durchschnitt
    if isinstance(fold_result['precision'], list):
         fold_result['precision'] = np.mean(fold_result['precision'])
    if isinstance(fold_result['recall'], list):
         fold_result['recall'] = np.mean(fold_result['recall'])

    fold_test_metrics.append(fold_result)
    print(f"-> Fold {fold+1} Test mAP50-95: {fold_result['mAP50-95']:.4f}")

    # 7. Cleanup
    # Optional: Gewichte löschen, um Speicher zu sparen, wenn nur Metriken wichtig sind
    # shutil.rmtree(os.path.join(project_dir, run_name)) 

# --- Auswertung am Ende ---
results_df = pd.DataFrame(fold_test_metrics)
print("\n##################################################")
print("## DURCHSCHNITTLICHE TEST-ERGEBNISSE ##")
print("##################################################")
print(results_df)
print("\nMean mAP50-95:", results_df['mAP50-95'].mean())