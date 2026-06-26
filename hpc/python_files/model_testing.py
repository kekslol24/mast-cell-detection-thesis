import os
from ultralytics import YOLO

def main():
    # ==========================================
    # 1. PFADE DEFINIEREN
    # ==========================================
    # Bitte hier die Pfade zu den eigenen Dateien anpassen
    model_path = "./DL_Modell_FV.pt"             # Pfad zu den trainierten Gewichten
    image_path = "./beispielbild.jpg"    # Pfad zum Bild, das analysiert werden soll
    output_name = "ergebnis.jpg"       # Name der Ausgabedatei

    # Prüfen, ob die Dateien wirklich existieren, um Fehler zu vermeiden
    if not os.path.exists(model_path):
        print(f"FEHLER: Modell-Datei '{model_path}' nicht gefunden.")
        return
    if not os.path.exists(image_path):
        print(f"FEHLER: Bild-Datei '{image_path}' nicht gefunden.")
        return

    # ==========================================
    # 2. MODELL LADEN
    # ==========================================
    print("Lade YOLO Modell...")
    model = YOLO(model_path)

    # ==========================================
    # 3. BILD ANALYSIEREN (INFERENZ)
    # ==========================================
    print(f"Analysiere Bild '{image_path}'...")
    
    # WICHTIG: Die Parameter conf=0.58 und iou=0.45 sind die 
    # evaluierten Optima aus der Bachelorarbeit für beste Resultate.

    results = model.predict(
        source=image_path, 
        conf=0.58, 
        iou=0.45,
        verbose=False # Verhindert zu viel Text in der Konsole
    )

    # ==========================================
    # 4. ERGEBNISSE SPEICHERN & AUSGEBEN
    # ==========================================
    # Da wir nur ein Bild übergeben, betrachten wir das erste Ergebnis
    result = results[0]
    
    # Anzahl der gefundenen Mastzellen auslesen
    anzahl_zellen = len(result.boxes)
    print(f"Analyse abgeschlossen! Es wurden {anzahl_zellen} Mastzellen gefunden.")

    # Das fertige Bild mit den gezeichneten Boxen speichern
    result.save(filename=output_name)
    print(f"Das Ergebnisbild wurde gespeichert unter: {os.path.abspath(output_name)}")

if __name__ == "__main__":
    main()