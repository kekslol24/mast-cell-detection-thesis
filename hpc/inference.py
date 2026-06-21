import os
import glob
from ultralytics import YOLO
from zipfile import ZipFile
import csv
import matplotlib.pyplot as plt

from patient_dir import PATIENT_IMAGE_DIRS


# ==========================================
# KONFIGURATION
# ==========================================
MODEL_PATH = "/cfs/earth/scratch/vollmflo/BA/hpc/tinker_final_holdoutP11/final/weights/best.pt"
INPUT_DIR = "/cfs/earth/scratch/vollmflo/BA/data/"
UNZIP_PATH = "/cfs/earth/scratch/vollmflo/BA/data/new/"

CONFIDENCE_CUT = None       #Confidence cut to get variance distribution

# Set to a patient ID (e.g. "P11") to use that patient as the test set,
# skipping the original ZIP unpack and training-image filter.
# Set to None to use the original ZIP-based pipeline.
TEST_PATIENT = "P11"

# If TEST_PATIENT is set and the images come as a ZIP, put the path here.
# The ZIP will be extracted to UNZIP_PATH and inference runs on the extracted folder.
# Set to None to read directly from PATIENT_IMAGE_DIRS[TEST_PATIENT] instead.
TEST_PATIENT_ZIP = "/cfs/earth/scratch/vollmflo/BA/data/P11.zip"

# NEU: Pfad zu den Bildern, die ignoriert werden sollen
# IGNORE_DIR = "/cfs/earth/scratch/vollmflo/BA/data/Pos_neg 12241515/images/Train/"

OUTPUT_PROJECT = "./mass_inference_results/run_final"

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

def main():
    if TEST_PATIENT is not None:
        # --- Patient mode ---
        if TEST_PATIENT_ZIP is not None:
            # Unzip the patient ZIP, then read from the extracted folder
            extract_folder = os.path.join(UNZIP_PATH, os.path.splitext(os.path.basename(TEST_PATIENT_ZIP))[0])
            print(f"Entpacke {TEST_PATIENT_ZIP} nach {extract_folder} ...")
            with ZipFile(TEST_PATIENT_ZIP, "r") as myzip:
                myzip.extractall(UNZIP_PATH)
            train_subdir = os.path.join(extract_folder, "images", "Train")
            image_source_dir=train_subdir if os.path.isdir(train_subdir) else extract_folder
        else:
            image_source_dir = PATIENT_IMAGE_DIRS[TEST_PATIENT]

        print(f"Verwende Patient {TEST_PATIENT}: {image_source_dir}")
        images_to_process = [
            os.path.join(image_source_dir, f)
            for f in os.listdir(image_source_dir)
            if f.lower().endswith(IMAGE_EXTS)
        ]
        print(f"Gefunden: {len(images_to_process)} Bilder für Patient {TEST_PATIENT}.")
    else:
        # --- Original ZIP pipeline ---
        # 1. ZIP entpacken
        zip_file_path = os.path.join(INPUT_DIR, "20241217_HAD_12241515.zip")
        extract_folder = os.path.join(UNZIP_PATH, "20241217_HAD_12241515")

        print("Entpacke ZIP-Datei...")
        with ZipFile(zip_file_path, "r") as myzip:
            myzip.extractall(UNZIP_PATH)

        # 2. Kugelsichere Blacklist erstellen (nur der Dateiname ohne Endung!)
        ignore_stems = set()
        for f in os.listdir(IGNORE_DIR):
            stem = os.path.splitext(f)[0]
            ignore_stems.add(stem)

        print(f"Gefunden: {len(ignore_stems)} eindeutige Bildnamen im Trainingsordner, die ignoriert werden.")

        # 3. Liste der neuen, zu verarbeitenden Bilder erstellen
        all_extracted_files = os.listdir(extract_folder)
        images_to_process = []

        for filename in all_extracted_files:
            file_stem = os.path.splitext(filename)[0]
            if file_stem not in ignore_stems and filename.lower().endswith(IMAGE_EXTS):
                full_path = os.path.join(extract_folder, filename)
                images_to_process.append(full_path)

        print(f"Gesamtanzahl im ZIP: {len(all_extracted_files)}")
        print(f"Übrig für die Inferenz nach Filterung: {len(images_to_process)}")

    if len(images_to_process) == 0:
        print("Es gibt keine neuen Bilder zum Verarbeiten. Skript wird beendet.")
        return

    # Zielordner für Bilder und Labels manuell erstellen
    os.makedirs(OUTPUT_PROJECT, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PROJECT, "labels"), exist_ok=True)

    print("Schreibe gefilterte Pfade in eine Textdatei für YOLO...")
    
    # NEU: Wir speichern die 22'000 Pfade in ein Textfile
    source_txt_path = "filtered_images_to_process.txt"
    with open(source_txt_path, "w") as f:
        for img_path in images_to_process:
            f.write(f"{img_path}\n")

    print("Lade Modell und starte Inferenz...")

    # 4. Modell laden
    model = YOLO(MODEL_PATH)

    # Inferenz starten - WICHTIG: source ist nun das Textfile!
    results = model.predict(
        source=source_txt_path, 
        # conf=0.276,
        iou=0.45,
        imgsz=512,
        batch=32,       
        stream=True,    
        save=False,      
        save_txt=False,  
        device=0,       
        verbose=False    
    )

    count = 0
    saved_count = 0
    
# NEU: CSV vorbereiten, um die Verteilung und Varianz im Nachhinein zu prüfen
    csv_path = os.path.join(OUTPUT_PROJECT, "confidence_distribution.csv")
    
    with open(csv_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["image_name", "class", "confidence"]) # Header
        
        # 5. Fliessband-Abfertigung
        for r in results:
            count += 1
            
            # Prüfen: Wurden Objekte (Bounding Boxes) gefunden, die den Cut überlebt haben?
            if len(r.boxes) > 0:
                saved_count += 1
                original_name = os.path.basename(r.path)
                
                # Jede gefundene Box ins CSV schreiben (für deine Verteilungsanalyse)
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf_val = float(box.conf[0].item())
                    csv_writer.writerow([original_name, cls_id, conf_val])
                
                # Bild mit Boxen speichern
                save_path = os.path.join(OUTPUT_PROJECT, original_name)
                r.save(filename=save_path)
                
                # Textdatei mit Koordinaten UND Konfidenzwerten speichern
                txt_name = os.path.splitext(original_name)[0] + ".txt"
                txt_path = os.path.join(OUTPUT_PROJECT, "labels", txt_name)
                r.save_txt(txt_file=txt_path, save_conf=True) # NEU: save_conf=True

            if count % 1000 == 0:
                print(f"{count} von {len(images_to_process)} Bildern verarbeitet... Davon {saved_count} Bilder mit gefundenen Objekten.")

    print(f"Fertig! Es wurden {saved_count} von {len(images_to_process)} analysierten Bildern gespeichert (Cut >= {CONFIDENCE_CUT}).")
    print(f"Die Bilder, Labels und die CSV-Datei zur Verteilungsanalyse liegen in: {OUTPUT_PROJECT}")


    # Read back the CSV and plot per-class confidence histograms
    confs_by_class = {}
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            cls = int(row["class"])
            conf = float(row["confidence"])
            confs_by_class.setdefault(cls, []).append(conf)

    if confs_by_class:
        class_names = {0: "Atypisch", 1: "Normal"}
        fig, ax = plt.subplots(figsize=(8, 5))
        for cls_id, confs in sorted(confs_by_class.items()):
            label = class_names.get(cls_id, f"Class {cls_id}")
            ax.hist(confs, bins=50, range=(0, 1), alpha=0.7, label=f"{label} (n={len(confs)})")
        ax.axvline(0.276, color="red", linestyle="--", linewidth=1, label="conf=0.276")
        ax.axvline(0.58,  color="orange", linestyle="--", linewidth=1, label="conf=0.58")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.set_title(f"Confidence distribution — {TEST_PATIENT or 'ZIP inference'}")
        ax.legend()
        hist_path = os.path.join(OUTPUT_PROJECT, "confidence_histogram.png")
        fig.savefig(hist_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Histogramm gespeichert: {hist_path}")
    else:
        print("Keine Detektionen — kein Histogramm erstellt.")



if __name__ == "__main__":
    main()