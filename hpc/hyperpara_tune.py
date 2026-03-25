from ultralytics import YOLO
import os

if __name__ == "__main__":


    model = YOLO("yolo11n.pt")
    current_dir = os.getcwd()
    data_path = os.path.normpath(os.path.join(current_dir, "..", "Tuning_Dataset", "data.yaml"))

    # search_space = {
    #     "lr0": (1e-5, 1e-1),
    #     "degrees": (0.0, 45.0),
    # }

    model.tune(
        data=data_path,
        epochs=300,
        iterations=100,
        optimizer="auto",
        plots=True,
        save=True,
        val=True,
        verbose=True
    )


