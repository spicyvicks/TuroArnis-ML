"""
Train YOLOv8n-pose for Arnis stick detection.
CPU-optimized settings for local training.
"""
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO

# Paths — use absolute to avoid double-folder issues
PROJECT_DIR = Path(__file__).parent.resolve()
DATA_YAML = str(PROJECT_DIR / "dataset_stick" / "data.yaml")
BASE_MODEL = str(PROJECT_DIR / "yolov8n-pose.pt")
PROJECT = str(PROJECT_DIR / "runs" / "pose")
NAME = f"stick_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# CPU-optimized hyperparameters
# Epochs reduced to 25 — epoch 15 already showed pose mAP50-95 = 0.88
CONFIG = {
    "data": DATA_YAML,
    "epochs": 25,
    "imgsz": 416,
    "batch": 16,
    "patience": 8,
    "workers": 2,
    "device": "cpu",
    "project": PROJECT,
    "name": NAME,
    "exist_ok": True,
    "verbose": True,
    "save": True,
    "plots": True,
    "amp": False,  # AMP slows down on CPU
    # Augmentation tuned for stick keypoints
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "close_mosaic": 5,  # Reduced since fewer epochs
}

if __name__ == "__main__":
    print(f"[{datetime.now()}] Starting YOLOv8n-pose training")
    print(f"  Data: {DATA_YAML}")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  Config: {CONFIG}")

    model = YOLO(BASE_MODEL)
    results = model.train(**CONFIG)

    print(f"\n[{datetime.now()}] Training complete!")
    print(f"  Best model: {PROJECT}/{NAME}/weights/best.pt")
    print(f"  Results saved to: {PROJECT}/{NAME}")
