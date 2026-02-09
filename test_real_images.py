from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

STICK_DETECTOR_PATH = "runs/pose/arnis_stick_detector/weights/best.pt"

# Load model
model = YOLO(STICK_DETECTOR_PATH)
print(f"Model task: {model.task}")

# Test on a real image from the dataset
test_images = list(Path("dataset_split/train/front/neutral_stance").glob("*.jpg"))[:3]

for img_path in test_images:
    print(f"\n{'='*60}")
    print(f"Testing: {img_path.name}")
    print(f"{'='*60}")
    
    results = model.predict(str(img_path), verbose=False)
    
    print(f"Results type: {type(results)}")
    print(f"Results[0] type: {type(results[0])}")
    
    # Check what attributes are available
    print(f"\nAvailable attributes:")
    for attr in dir(results[0]):
        if not attr.startswith('_'):
            print(f"  - {attr}")
    
    # Check keypoints
    print(f"\nKeypoints attribute: {hasattr(results[0], 'keypoints')}")
    if hasattr(results[0], 'keypoints'):
        kpts = results[0].keypoints
        print(f"Keypoints type: {type(kpts)}")
        print(f"Keypoints value: {kpts}")
        
        if kpts is not None:
            print(f"Keypoints.data: {kpts.data}")
            print(f"Keypoints.data shape: {kpts.data.shape if hasattr(kpts.data, 'shape') else 'N/A'}")
            print(f"Keypoints.data length: {len(kpts.data)}")
            
            if len(kpts.data) > 0:
                print(f"\nFirst detection keypoints:")
                print(kpts.data[0])
    
    # Check boxes
    print(f"\nBoxes attribute: {hasattr(results[0], 'boxes')}")
    if hasattr(results[0], 'boxes'):
        boxes = results[0].boxes
        print(f"Boxes: {boxes}")
        if boxes is not None and len(boxes) > 0:
            print(f"Number of boxes: {len(boxes)}")
