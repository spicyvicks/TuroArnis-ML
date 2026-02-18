from ultralytics import YOLO
import torch
import numpy as np
import cv2

STICK_DETECTOR_PATH = "runs/pose/arnis_stick_detector/weights/best.pt"

print("=" * 60)
print("Inspecting Stick Detection Model")
print("=" * 60)

# Load model
model = YOLO(STICK_DETECTOR_PATH)
print(f"\nModel task: {model.task}")
print(f"Model type: {type(model)}")

# Load checkpoint to inspect architecture
ckpt = torch.load(STICK_DETECTOR_PATH, map_location='cpu')
print(f"\nCheckpoint keys: {list(ckpt.keys())}")

# Check if it's a pose model with keypoints
if hasattr(model.model, 'names'):
    print(f"\nClass names: {model.model.names}")

# Check for keypoint information
if 'model' in ckpt:
    model_state = ckpt['model']
    if hasattr(model_state, 'kpt_shape'):
        print(f"\nKeypoint shape: {model_state.kpt_shape}")
        print("This is a POSE model detecting keypoints!")
    else:
        print("\nNo kpt_shape found - checking model architecture...")

# Try inference on a dummy image
print("\n" + "=" * 60)
print("Testing Inference")
print("=" * 60)

img = np.zeros((640, 640, 3), dtype=np.uint8)
cv2.imwrite("temp_test.jpg", img)

try:
    results = model("temp_test.jpg", verbose=False)
    print(f"\nInference successful!")
    print(f"Results type: {type(results[0])}")
    
    # Check what's in results
    if hasattr(results[0], 'keypoints'):
        print(f"\n✓ Model has KEYPOINTS (Pose model)")
        print(f"Keypoints shape: {results[0].keypoints}")
        if results[0].keypoints is not None:
            print(f"Keypoints data shape: {results[0].keypoints.shape if hasattr(results[0].keypoints, 'shape') else 'N/A'}")
    
    if hasattr(results[0], 'boxes'):
        print(f"\n✓ Model has BOXES (Detection model)")
        print(f"Boxes: {results[0].boxes}")
        
except Exception as e:
    print(f"\nError during inference: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
