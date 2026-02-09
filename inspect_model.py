from ultralytics import YOLO
import torch

STICK_DETECTOR_PATH = "runs/pose/arnis_stick_detector/weights/best.pt"

print(f"Loading model from {STICK_DETECTOR_PATH}")
model = YOLO(STICK_DETECTOR_PATH)

# Check model metadata
print(f"\nModel type: {type(model)}")
print(f"Model task: {model.task}")

# Load the checkpoint directly to inspect
ckpt = torch.load(STICK_DETECTOR_PATH, map_location='cpu')
print(f"\nCheckpoint keys: {ckpt.keys()}")

if 'model' in ckpt:
    print(f"Model in checkpoint: {type(ckpt['model'])}")
    
# Try to check the model architecture
if hasattr(model, 'model'):
    print(f"\nModel architecture: {model.model}")
    if hasattr(model.model, 'names'):
        print(f"Class names: {model.model.names}")
