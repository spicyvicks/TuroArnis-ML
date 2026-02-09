"""
Test script to verify stick detector compatibility with create_graph_dataset.py
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Load the stick detector
STICK_DETECTOR_PATH = "runs/pose/arnis_stick_detector/weights/best.pt"
print(f"Loading stick detector from: {STICK_DETECTOR_PATH}")

try:
    stick_detector = YOLO(STICK_DETECTOR_PATH)
    print("✓ Stick detector loaded successfully")
except Exception as e:
    print(f"✗ Failed to load stick detector: {e}")
    exit(1)

# Test on a sample image
test_image_path = "dataset_split/train/front/neutral_stance/1.jpg"

if not Path(test_image_path).exists():
    print(f"✗ Test image not found: {test_image_path}")
    exit(1)

print(f"\nTesting on: {test_image_path}")

# Run detection
try:
    results = stick_detector.predict(test_image_path, verbose=False)
    print("✓ Prediction successful")
    
    # Check keypoints
    if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        kpts = results[0].keypoints.data[0].cpu().numpy()
        print(f"✓ Keypoints detected: {kpts.shape}")
        print(f"  Grip: x={kpts[0, 0]:.1f}, y={kpts[0, 1]:.1f}, conf={kpts[0, 2]:.3f}")
        print(f"  Tip:  x={kpts[1, 0]:.1f}, y={kpts[1, 1]:.1f}, conf={kpts[1, 2]:.3f}")
        
        # Normalize coordinates
        image = cv2.imread(test_image_path)
        h, w = image.shape[:2]
        
        stick_grip = [kpts[0, 0] / w, kpts[0, 1] / h, float(kpts[0, 2])]
        stick_tip = [kpts[1, 0] / w, kpts[1, 1] / h, float(kpts[1, 2])]
        
        print(f"\n✓ Normalized coordinates:")
        print(f"  Grip: x={stick_grip[0]:.3f}, y={stick_grip[1]:.3f}, conf={stick_grip[2]:.3f}")
        print(f"  Tip:  x={stick_tip[0]:.3f}, y={stick_tip[1]:.3f}, conf={stick_tip[2]:.3f}")
        
        stick_nodes = np.array([stick_grip, stick_tip], dtype=np.float32)
        print(f"\n✓ Stick nodes shape: {stick_nodes.shape}")
        print(f"✓ Stick nodes dtype: {stick_nodes.dtype}")
        
        print("\n" + "="*60)
        print("SUCCESS: Stick detector is compatible!")
        print("="*60)
        print("\nYou can now run:")
        print("  python training/create_graph_dataset.py --dataset_root dataset_split --augment")
        
    else:
        print("✗ No stick detected in image")
        print("  This might be normal for neutral_stance images")
        print("  Try testing on a strike/block image")
        
except Exception as e:
    print(f"✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
