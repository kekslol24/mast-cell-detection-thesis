import gradio as gr
from ultralytics import YOLO
import PIL.Image as Image

# CSS, um die Galerie und den Hauptcontainer zu strecken
custom_css = """
.gradio-container { height: 100vh !important; }
"""

# 1. Modell laden (ersetze 'yolo11n.pt' durch deinen Pfad zu den Weights)
# model = YOLO(r'BA\jobs\Slurm-264235 (large 3rd 200epoch + aug)\yolo_runs_hpc_final\fold_1\weights\best.pt')
model = YOLO(r"..\jobs\Slurm-264235 (large 3rd 200epoch + aug)\yolo_runs_hpc_final\fold_1\weights\best.pt")

def process_batch(files):
    if not files:
        return [], "Bitte Bilder hochladen."

    processed_images = []
    stats = {"high": 0, "medium": 0, "attention": 0}
    
    results = model.predict(source=files, conf=0.25)

    for res in results:
        res_plotted = res.plot()
        processed_images.append(Image.fromarray(res_plotted[:, :, ::-1]))

        confidences = res.boxes.conf.cpu().numpy()
        
        for conf in confidences:
            if conf >= 0.80:
                stats["high"] += 1
            elif conf >= 0.40:
                stats["medium"] += 1
            else:
                stats["attention"] += 1

    total = sum(stats.values())
    report = f"""
## Analyse-Ergebnis
| Kategorie | Anzahl | Beschreibung |
| :--- | :--- | :--- |
| 🟢 **Hohe Sicherheit** | {stats['high']} | Konfidenz > 80% |
| 🟡 **Mittlere Sicherheit** | {stats['medium']} | Konfidenz 40% - 80% |
| 🔴 **Achtung (Unsicher)** | {stats['attention']} | Konfidenz < 40% |

**Gesamtanzahl gefundener Objekte:** {total}
    """
    
    return processed_images, report

# Gradio Interface Aufbau
with gr.Blocks(title="Mastcell Detector") as demo:
    gr.Markdown("# Automated Mastcell Detector")
    
    with gr.Sidebar(width=500):
        with gr.Tabs():
            with gr.TabItem("Upload & Visualisierung"):
                file_input = gr.File(
                    file_count="multiple", 
                    file_types=["image"], 
                    label="Bilder hier ablegen"
                )
                
                # Buttons nebeneinander anordnen
                with gr.Row():
                    run_btn = gr.Button("Analyse starten", variant="primary")
                    stop_btn = gr.Button("Stoppen", variant="stop")
                
                # ClearButton leert die zugeordneten Komponenten, ist nicht mehr im with drin, damit der Button darunter erscheint
                clear_btn = gr.ClearButton(
                    components=[file_input], 
                    value="Bilder und Resultate löschen"
                )

            with gr.TabItem("Statistik-Dashboard"):
                stats_output = gr.Markdown("Noch keine Daten vorhanden. Bitte Analyse im ersten Tab starten.")

    gallery_output = gr.Gallery(
        label="Ergebnisse", 
        columns=3,
        height="calc(100vh - 120px)", 
        preview=True, 
        object_fit="contain"
    )
    
    # Wir fügen die Ausgabefelder zum ClearButton hinzu, damit auch diese geleert werden
    clear_btn.add([gallery_output, stats_output])
            
    # 1. Das Event für den Start in einer Variable speichern
    run_event = run_btn.click(
        fn=process_batch,
        inputs=file_input,
        outputs=[gallery_output, stats_output]
    )
    
    # 2. Den Stopp-Button so konfigurieren, dass er dieses Event abbricht
    stop_btn.click(
        fn=None, 
        inputs=None, 
        outputs=None, 
        cancels=[run_event]
    )

if __name__ == "__main__":
    demo.launch(css=custom_css, share=True)