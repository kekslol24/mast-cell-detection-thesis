import os

# 1. Pfade anpassen (ersetze dies mit deinen exakten Pfaden zum Negativ Ordner)
# WICHTIG: Nutze r"..." für Windows-Pfade
images_dir = r"/cfs/earth/scratch/vollmflo/BA/data/Negativ 12241515/images/train"
labels_dir = r"/cfs/earth/scratch/vollmflo/BA/data/Negativ 12241515/labels/train"

# Stelle sicher, dass der labels Ordner überhaupt existiert
os.makedirs(labels_dir, exist_ok=True)

created_count = 0

# 2. Alle Bilder durchgehen
for image_name in os.listdir(images_dir):
    # Nur Bilddateien berücksichtigen
    if image_name.lower().endswith((".jpg")):
        
        # Den Dateinamen ohne Endung holen (z.B. "bild_01")
        base_name = os.path.splitext(image_name)[0]
        
        # Den Pfad für die zugehörige Textdatei bauen
        label_path = os.path.join(labels_dir, base_name + ".txt")
        
        # 3. Wenn noch keine Textdatei existiert -> leere Datei erstellen
        if not os.path.exists(label_path):
            with open(label_path, 'w') as f:
                pass # 'pass' macht nichts, erstellt also eine 0-Byte Datei
            created_count += 1

print(f"Fertig! Es wurden {created_count} leere Label-Dateien für negative Bilder erstellt.")