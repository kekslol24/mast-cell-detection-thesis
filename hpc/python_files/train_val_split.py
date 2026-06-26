import os
import shutil
import random
from pathlib import Path

# ==============================================================================
# KONFIGURATION (Pfade anpassen, falls das Skript woanders liegt)
# ==============================================================================
# Der Hauptordner, in dem deine beiden Datensätze liegen
BASE_DATA_DIR = Path("./data") 

# Die beiden Quell-Ordner aus deinem Screenshot
POS_NEG_DIR = BASE_DATA_DIR / "Pos_neg 12241515"
NEGATIV_DIR = BASE_DATA_DIR / "Negativ 12241515"

# Der neue Ziel-Ordner für das Tuning
TUNING_DIR = Path("./Tuning_Dataset")

# YOLO Klassen (aus deinem vorherigen Skript übernommen)
CLASSES = ["Atypisch", "Normal"]

# ==============================================================================
# ORDNERSTRUKTUR ERSTELLEN
# ==============================================================================
# Löscht den Tuning-Ordner falls er schon existiert, um sauberen Start zu haben
if TUNING_DIR.exists():
    shutil.rmtree(TUNING_DIR)

# Erstellt die exakte YOLO-Struktur
for split in ['train', 'val']:
    (TUNING_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
    (TUNING_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

# ==============================================================================
# BILDER UND LABELS SUCHEN
# ==============================================================================
def get_image_label_pairs(dataset_dir):
    pairs = []
    # Sucht alle Bilder in allen Unterordnern von 'images'
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    
    if not images_dir.exists():
        return pairs

    # Alle gängigen Bildformate berücksichtigen
    for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG']:
        for img_path in images_dir.rglob(ext):
            
            # Suche das passende Label im labels-Ordner
            label_name = img_path.stem + ".txt"
            label_matches = list(labels_dir.rglob(label_name))
            
            if label_matches:
                label_path = label_matches[0]
            else:
                label_path = None # Wird später als leere Datei generiert
                
            pairs.append((img_path, label_path))
            
    return pairs

print("Sammle Daten aus den Ordnern...")
all_pairs = []
all_pairs.extend(get_image_label_pairs(POS_NEG_DIR))
all_pairs.extend(get_image_label_pairs(NEGATIV_DIR))

# WICHTIG: Die Liste gut durchmischen, damit Positiv- und Negativbilder gleichmässig verteilt sind
random.seed(42)
random.shuffle(all_pairs)

# ==============================================================================
# 80/20 SPLIT UND KOPIEREN
# ==============================================================================
total_files = len(all_pairs)
train_split_idx = int(total_files * 0.8)

train_pairs = all_pairs[:train_split_idx]
val_pairs = all_pairs[train_split_idx:]

def copy_data(pairs, split_name):
    print(f"Kopiere {len(pairs)} Dateien für den '{split_name}' Split...")
    
    for img_path, label_path in pairs:
        # 1. Bild kopieren
        dest_img_path = TUNING_DIR / 'images' / split_name / img_path.name
        shutil.copy2(img_path, dest_img_path)
        
        # 2. Label kopieren oder leeres Label erstellen
        dest_label_path = TUNING_DIR / 'labels' / split_name / (img_path.stem + ".txt")
        
        if label_path and label_path.exists():
            shutil.copy2(label_path, dest_label_path)
        else:
            # Erstellt eine leere Textdatei für negative Bilder
            dest_label_path.touch()

copy_data(train_pairs, 'train')
copy_data(val_pairs, 'val')

# ==============================================================================
# DATA.YAML GENERIEREN
# ==============================================================================
yaml_content = f"""path: {TUNING_DIR.absolute()}
train: images/train
val: images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""

yaml_path = TUNING_DIR / "data.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print("\n" + "="*50)
print(f"Erfolgreich abgeschlossen!")
print(f"Trainingsbilder: {len(train_pairs)}")
print(f"Validierungsbilder: {len(val_pairs)}")
print(f"Die Datei für das Tuning liegt hier: {yaml_path}")
print("="*50)