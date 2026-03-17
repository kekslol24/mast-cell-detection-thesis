from ultralytics import YOLO
import os

if __name__ == "__main__":

    current_dir = os.getcwd()
    data_path = os.path.normpath(os.path.join(current_dir, "..", "Data", "Pos_neg 12241515", "data.yaml"))

    # search_space = {
    #     "lr0": (1e-5, 1e-1),
    #     "degrees": (0.0, 45.0),
    # }

    model.tune(
        data=path+"../Data/Pos_neg 12241515/data.yaml",
        epochs=30,
        iterations=50,
        optimizer="AdamW",
        plots=True,
        save=True,
        val=True,
    )



