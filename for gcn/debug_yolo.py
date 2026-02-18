from ultralytics import YOLO
import cv2
import numpy as np

STICK_DETECTOR_PATH = "runs/pose/arnis_stick_detector/weights/best.pt"

print(f"Loading model from {STICK_DETECTOR_PATH}")
try:
    model = YOLO(STICK_DETECTOR_PATH)
    print(f"Model loaded successfully. Type: {type(model)}")
    if hasattr(model, 'task'):
        print(f"Model task: {model.task}")
    
    # Create dummy image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    print("Running inference...")
    results = model(img)
    print("Inference successful")
    print(f"Results type: {type(results)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
