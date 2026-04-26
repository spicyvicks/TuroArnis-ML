"""
Test Stick Detector
===================
Quick smoke-test script for the deployed YOLOv8n-pose stick detector.

Usage:
    python test_stick_detector.py --image <path_to_image>
    python test_stick_detector.py --image dataset_stick/test/images/some_image.jpg

Output:
    - Console: detection summary (box, grip, tip, confidence)
    - File: stick_test.jpg (overlay saved to project root)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = Path(__file__).parent / "arnis_stick_detector" / "weights" / "best.pt"
OUTPUT_PATH = Path(__file__).parent / "stick_test.jpg"


def draw_overlay(image, result):
    """Draw bounding box + grip/tip keypoints on the image."""
    img = image.copy()
    h, w = img.shape[:2]

    if result.boxes is None or len(result.boxes) == 0:
        cv2.putText(img, "No stick detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img

    # Use the first (best) detection
    box = result.boxes[0]
    keypoints = result.keypoints[0]  # [2, 3] => grip, tip

    # Bounding box (xyxy format)
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    conf = float(box.conf[0])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"stick {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Keypoints: 0 = grip, 1 = tip
    kpts = keypoints.data.cpu().numpy()
    # Handle both [N, 2, 3] and [2, 3] shapes
    if kpts.ndim == 3:
        kpts = kpts[0]

    colors = [(0, 165, 255), (255, 0, 255)]  # orange = grip, magenta = tip
    labels = ["grip", "tip"]

    for i, kp in enumerate(kpts):
        kx, ky = int(kp[0]), int(kp[1])
        kconf = float(kp[2]) if len(kp) > 2 else 1.0
        color = colors[i]
        label = labels[i]
        # Draw circle
        cv2.circle(img, (kx, ky), 6, color, -1)
        cv2.circle(img, (kx, ky), 8, (255, 255, 255), 2)
        # Label
        cv2.putText(img, f"{label} {kconf:.2f}", (kx + 10, ky - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img


def main():
    parser = argparse.ArgumentParser(description="Test stick detector on a single image")
    parser.add_argument("--image", required=True, help="Path to test image")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        sys.exit(1)

    print(f"Model : {MODEL_PATH}")
    print(f"Image : {image_path}")
    print(f"Output: {OUTPUT_PATH}")
    print("-" * 50)

    # Load model
    print("Loading model...")
    model = YOLO(str(MODEL_PATH))
    print("Model loaded.")

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: Could not read image: {image_path}")
        sys.exit(1)

    # Run inference
    print("Running inference...")
    results = model(img, verbose=False)
    result = results[0]

    # Print detection summary
    if result.boxes is None or len(result.boxes) == 0:
        print("\nNo stick detected in this image.")
        overlay = draw_overlay(img, result)
        cv2.imwrite(str(OUTPUT_PATH), overlay)
        print(f"Saved overlay to: {OUTPUT_PATH}")
        sys.exit(0)

    box = result.boxes[0]
    keypoints = result.keypoints[0]
    kpts = keypoints.data.cpu().numpy()

    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    bw, bh = x2 - x1, y2 - y1
    box_conf = float(box.conf[0])

    # Handle both [N, 2, 3] and [2, 3] shapes
    if kpts.ndim == 3:
        kpts = kpts[0]  # Take first detection

    grip = kpts[0]
    tip = kpts[1]
    grip_x, grip_y = grip[0], grip[1]
    tip_x, tip_y = tip[0], tip[1]
    grip_conf = float(grip[2]) if len(grip) > 2 else 1.0
    tip_conf = float(tip[2]) if len(tip) > 2 else 1.0
    stick_length_px = np.sqrt((tip_x - grip_x)**2 + (tip_y - grip_y)**2)

    print("\n=== Detection Result ===")
    print(f"  Box         : ({x1:.1f}, {y1:.1f}, {bw:.1f}, {bh:.1f})")
    print(f"  Box conf    : {box_conf:.3f}")
    print(f"  Grip        : ({grip_x:.1f}, {grip_y:.1f})  conf={grip_conf:.3f}")
    print(f"  Tip         : ({tip_x:.1f}, {tip_y:.1f})  conf={tip_conf:.3f}")
    print(f"  Stick length: {stick_length_px:.1f} px")
    print(f"  Detections  : {len(result.boxes)}")

    # Draw and save
    overlay = draw_overlay(img, result)
    cv2.imwrite(str(OUTPUT_PATH), overlay)
    print(f"\nSaved annotated image to: {OUTPUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
