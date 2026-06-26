from ultralytics import YOLO

# --------------------------------------------------------------------------
# 9. Evaluate best tuned weights on the held-out test set
# tune() saves its best weights to runs/detect/tune/weights/best.pt
# --------------------------------------------------------------------------
tune_best_weights = os.path.join("runs", "detect", "tune", "weights", "best.pt")

if os.path.exists(tune_best_weights):
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION (best tuned weights)")
    print("=" * 50)
    best_model   = YOLO(tune_best_weights)
    metrics_test = best_model.val(data=data_yaml_path, split="test", verbose=True)
    print(f"\n  mAP50-95  : {metrics_test.box.map:.4f}")
    print(f"  mAP50     : {metrics_test.box.map50:.4f}")
    print(f"  Precision : {metrics_test.box.mp:.4f}")
    print(f"  Recall    : {metrics_test.box.mr:.4f}")
else:
    print(f"\nCould not find best weights at: {tune_best_weights}")
    print("Check your YOLO save directory -- it may differ based on your project/name settings.")