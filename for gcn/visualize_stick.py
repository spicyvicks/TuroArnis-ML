
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt

# Load the stick detector
STICK_DETECTOR_PATH = "runs/pose/arnis_stick_detector/weights/best.pt"
TEST_IMAGE_PATH = "dataset_split/train/front/neutral_stance/1.jpg"
OUTPUT_PATH = "stick_detection_test.jpg"

def visualize_stick():
    if not Path(TEST_IMAGE_PATH).exists():
        print(f"Error: Test image not found at {TEST_IMAGE_PATH}")
        return

    # Load model
    print(f"Loading model from {STICK_DETECTOR_PATH}...")
    model = YOLO(STICK_DETECTOR_PATH)

    # Predict
    print(f"Processing {TEST_IMAGE_PATH}...")
    results = model.predict(TEST_IMAGE_PATH, verbose=False)[0]

    # Load image for drawing
    img = cv2.imread(TEST_IMAGE_PATH)
    
    if results.keypoints is not None and len(results.keypoints.data) > 0:
        kpts = results.keypoints.data[0].cpu().numpy() # [2, 3]
        
        # Get coordinates
        grip_x, grip_y, grip_conf = kpts[0]
        tip_x, tip_y, tip_conf = kpts[1]
        
        print(f"Grip: ({grip_x:.1f}, {grip_y:.1f}) Conf: {grip_conf:.2f}")
        print(f"Tip:  ({tip_x:.1f}, {tip_y:.1f}) Conf: {tip_conf:.2f}")
        
        # Draw points
        # Grip = Green, Tip = Red
        cv2.circle(img, (int(grip_x), int(grip_y)), 8, (0, 255, 0), -1) # Green filled circle
        cv2.circle(img, (int(tip_x), int(tip_y)), 8, (0, 0, 255), -1)   # Red filled circle
        
        # Draw line connecting them
        cv2.line(img, (int(grip_x), int(grip_y)), (int(tip_x), int(tip_y)), (255, 0, 0), 3) # Blue line
        
        # Add labels
        cv2.putText(img, f"Grip {grip_conf:.2f}", (int(grip_x)+10, int(grip_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"Tip {tip_conf:.2f}", (int(tip_x)+10, int(tip_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save result
        cv2.imwrite(OUTPUT_PATH, img)
        print(f"Visualization saved to {OUTPUT_PATH}")
        
        # Verify file creation
        if Path(OUTPUT_PATH).exists():
             print("SUCCESS: Image created.")
    else:
        print("No stick detected.")

if __name__ == "__main__":
    visualize_stick()
