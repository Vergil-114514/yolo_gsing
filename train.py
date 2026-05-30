"""
Train YOLOv8n on the Cube 4-class dataset.

Output: runs/cube_yolov8n_n100/baseline/weights/best.pt
"""

from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
DATA_YAML = BASE / "Cube.yolov8" / "data_split.yaml"

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=16,
        project="runs/cube_yolov8n_n100",
        name="baseline",
        exist_ok=True,
        seed=42,
        deterministic=True,
        patience=100,
        close_mosaic=10,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        cutmix=0.0,
        auto_augment="randaugment",
        erasing=0.4,
        amp=True,
        plots=True,
        val=True,
    )
    print(f"Training complete. Best model saved.")
