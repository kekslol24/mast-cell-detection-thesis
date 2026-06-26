import gradio as gr
from ultralytics import YOLO
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# CSS, um die Galerie zu strecken UND die Upload-Bereiche scrollbar zu machen
custom_css = """
.gradio-container { height: 100vh !important; }

/* Begrenzt die Höhe der Dateiliste und fügt einen Scrollbalken hinzu */
#scrollable-upload { max-height: 25vh !important; overflow-y: auto !important; }
"""

# 1. Modell laden (ersetze 'yolo11n.pt' durch deinen Pfad zu den Weights)
# model = YOLO(r'BA\jobs\Slurm-264235 (large 3rd 200epoch + aug)\yolo_runs_hpc_final\fold_1\weights\best.pt')
# model = YOLO(r"..\hpc\jobs\Slurm-264235 (large 3rd 200epoch + aug)\yolo_runs_hpc_final\fold_1\weights\best.pt")
model = YOLO(r"D:\Studium\6. Semester\BA\model_weights\DL_Modell_FV_v3.pt")

# --- HILFSFUNKTIONEN ---
def calculate_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# --- FUNKTION: TAB 1 (ANALYSIS) ---
def process_batch(files):
    if not files:
        return [], "Bitte Bilder hochladen."

    processed_images = []
    stats = {"high": 0, "medium": 0, "attention": 0}
    
    results = model.predict(source=files, conf=0.58, iou=0.3, batch=16, stream=True) # evtl. anpassen

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

# --- FUNKTION: TAB 2 (RESEARCH) ---
def evaluate_model(image_files, txt_files, iou_threshold=0.5):
    if not image_files or not txt_files:
        return [], "Bitte Bilder UND Textdateien hochladen.", None

    # Wörterbücher für einfaches Zuordnen via Dateiname (ohne Endung)
    img_dict = {os.path.splitext(os.path.basename(f.name))[0]: f.name for f in image_files}
    txt_dict = {os.path.splitext(os.path.basename(f.name))[0]: f.name for f in txt_files}
    
    stitched_images = []
    tp, fp, fn, tn = 0, 0, 0, 0

    for base_name, img_path in img_dict.items():
        if base_name not in txt_dict:
            continue 

        txt_path = txt_dict[base_name]
        
        # Originalbild laden
        img_pil = Image.open(img_path).convert("RGB")
        img_width, img_height = img_pil.size
        
        # 1. Ground Truth verarbeiten (Linkes Bild)
        img_gt = img_pil.copy()
        draw_gt = ImageDraw.Draw(img_gt)
        gt_boxes = []
        
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id, x_c, y_c, w, h = map(float, parts[:5])
                    abs_w, abs_h = w * img_width, h * img_height
                    abs_x, abs_y = x_c * img_width, y_c * img_height
                    
                    x1, y1 = abs_x - abs_w/2, abs_y - abs_h/2
                    x2, y2 = abs_x + abs_w/2, abs_y + abs_h/2
                    
                    gt_boxes.append([x1, y1, x2, y2])
                    draw_gt.rectangle([x1, y1, x2, y2], outline="lime", width=3)
                    draw_gt.text((x1, y1-10), "GT", fill="lime")

        # 2. Prediction verarbeiten (Rechtes Bild)
        result = model.predict(source=img_path, conf=0.25, iou=0.3)[0]
        res_plotted = result.plot()
        img_pred = Image.fromarray(res_plotted[:, :, ::-1])
        
        pred_boxes = []
        if len(result.boxes) > 0:
            pred_boxes = result.boxes.xyxy.cpu().numpy().tolist() 

        # 3. Metriken berechnen
        matched_gt = set()
        matched_pred = set()
        
        for p_idx, p_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            for g_idx, g_box in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                iou = calculate_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(p_idx)
            else:
                fp += 1

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            tn += 1
            continue

        fn += len(gt_boxes) - len(matched_gt)

        # 4. Bilder zusammenfügen (GT links, Pred rechts)
        stitched = Image.new('RGB', (img_width * 2, img_height))
        stitched.paste(img_gt, (0, 0))
        stitched.paste(img_pred, (img_width, 0))
        stitched_images.append(stitched)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    report = f"""
### Gesamtauswertung (Threshold IoU: {iou_threshold})
* **True Positives (Korrekt):** {tp}
* **True Negatives (Korrekt Negativ):** {tn}
* **False Positives (Falscher Alarm):** {fp}
* **False Negatives (Verpasst):** {fn}
* **Precision:** {precision:.3f}
* **Recall:** {recall:.3f}
    """

    # 5. Konfusionsmatrix zeichnen
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = np.array([[tp, fn], [fp, 0]]) 
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax, 
                xticklabels=['Mastcell', 'Background'], 
                yticklabels=['Mastcell', 'Background'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title('Konfusionsmatrix')
    plt.tight_layout()

    return stitched_images, report, fig


# --- UI AUFBAU ---
with gr.Blocks(title="Mastcell Detector", css=custom_css) as demo:
    gr.Markdown("# Automated Mastcell Detector")

    with gr.Tabs():
        
        # ==========================================
        # TAB 1: ANALYSIS
        # ==========================================
        with gr.TabItem("Analysis"):
            with gr.Row():
                # Linke Spalte (Simulierte Sidebar)
                with gr.Column(scale=1, variant="panel"):
                    
                    with gr.Tabs():
                        with gr.TabItem("Upload & Visualisierung"):
                            # elem_id für Scroll-Verhalten hinzufügen
                            file_input_analysis = gr.File(
                                file_count="multiple", 
                                file_types=["image"], 
                                label="Bilder hier ablegen",
                                elem_id="scrollable-upload"
                            )
                            
                            with gr.Row():
                                run_btn_analysis = gr.Button("Analyse starten", variant="primary")
                                stop_btn_analysis = gr.Button("Stoppen", variant="stop")
                            
                            clear_btn_analysis = gr.ClearButton(
                                components=[file_input_analysis], 
                                value="Bilder und Resultate löschen"
                            )
                        
                        with gr.TabItem("Statistik-Dashboard"):
                            stats_output = gr.Markdown("Noch keine Daten vorhanden. Bitte Analyse starten.")

                # Rechte Spalte (Hauptbereich)
                with gr.Column(scale=3):
                    gallery_output_analysis = gr.Gallery(
                        label="Ergebnisse", 
                        columns=3,
                        height="calc(100vh - 120px)", 
                        preview=True, 
                        object_fit="contain"
                    )
            
            # Events für Tab 1
            clear_btn_analysis.add([gallery_output_analysis, stats_output])
            
            run_event_analysis = run_btn_analysis.click(
                fn=process_batch,
                inputs=file_input_analysis,
                outputs=[gallery_output_analysis, stats_output]
            )
            
            stop_btn_analysis.click(fn=None, inputs=None, outputs=None, cancels=[run_event_analysis])


        # ==========================================
        # TAB 2: RESEARCH
        # ==========================================
        with gr.TabItem("Research"):
            with gr.Row():
                # Linke Spalte (Simulierte Sidebar)
                with gr.Column(scale=1, variant="panel"):
                    
                    # NEU: Tabs auch hier für Upload und Metriken trennen
                    with gr.Tabs():
                        with gr.TabItem("Upload & Setup"):
                            # elem_id macht beide Upload-Boxen unabhängig voneinander scrollbar
                            file_input_research_img = gr.File(
                                file_count="multiple", file_types=["image"], label="1. Originalbilder", elem_id="scrollable-upload"
                            )
                            file_input_research_txt = gr.File(
                                file_count="multiple", file_types=[".txt"], label="2. Ground Truth (.txt)", elem_id="scrollable-upload"
                            )
                            
                            with gr.Row():
                                run_btn_research = gr.Button("Evaluation starten", variant="primary")
                                stop_btn_research = gr.Button("Stoppen", variant="stop")
                                
                            clear_btn_research = gr.ClearButton(
                                components=[file_input_research_img, file_input_research_txt], 
                                value="Eingaben zurücksetzen"
                            )
                            
                        with gr.TabItem("Metriken & Plot"):
                            metrics_output = gr.Markdown("Metriken werden nach der Analyse hier angezeigt.")
                            plot_output = gr.Plot(label="Konfusionsmatrix")

                # Rechte Spalte (Galerie für die Gegenüberstellung)
                with gr.Column(scale=3):
                    gallery_output_research = gr.Gallery(
                        label="Links: Ground Truth | Rechts: Prediction", 
                        columns=1,
                        height="calc(100vh - 120px)", 
                        preview=True, 
                        object_fit="contain"
                    )
            
            # Events für Tab 2
            clear_btn_research.add([gallery_output_research, metrics_output, plot_output])
            
            run_event_research = run_btn_research.click(
                fn=evaluate_model,
                inputs=[file_input_research_img, file_input_research_txt],
                outputs=[gallery_output_research, metrics_output, plot_output]
            )
            
            stop_btn_research.click(fn=None, inputs=None, outputs=None, cancels=[run_event_research])

if __name__ == "__main__":
    # demo.launch(css=custom_css, server_name="0.0.0.0", server_port=7860)
    demo.launch()
