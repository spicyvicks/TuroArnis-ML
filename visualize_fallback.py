"""
Visualize fallback stick estimations for front view classes 0-3
Helps verify that finger-based stick estimation is producing reasonable results
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from pathlib import Path
import json
import sys
import random
import importlib.util

# Load the module with numeric prefix
spec = importlib.util.spec_from_file_location("features", "hybrid_classifier/2b_generate_node_hybrid_features.py")
features_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features_module)

extract_raw_features = features_module.extract_raw_features
estimate_stick_from_fingers = features_module.estimate_stick_from_fingers
CLASS_NAMES = features_module.CLASS_NAMES

# Config
STICK_MODEL = "runs/pose/arnis_stick_detector/weights/best.pt"
OUTPUT_DIR = Path("visualization/fallback_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Classes to analyze (front view 0-3)
TARGET_CLASSES = [0, 1, 2, 3]  # crown, left_chest, left_elbow, left_eye
CLASS_NAMES_SHORT = ['crown', 'left_chest', 'left_elbow', 'left_eye']

def visualize_sample(image_path, class_idx, stick_detector, save_path):
    """Visualize a single sample with fallback stick estimation"""
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        return None, "Failed to load image"
    
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect pose
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=2)
    results = pose_detector.process(img_rgb)
    pose_detector.close()
    
    if not results.pose_landmarks:
        return None, "No pose detected"
    
    # Get keypoints
    kpts = []
    for lm in results.pose_landmarks.landmark:
        kpts.append([lm.x, lm.y, lm.z, lm.visibility])
    kpts = np.array(kpts)
    
    # Try YOLO stick detection
    stick_results = stick_detector(str(image_path), verbose=False)[0]
    yolo_detected = (stick_results.keypoints is not None and 
                     len(stick_results.keypoints.data) > 0)
    
    # Create visualization
    vis_img = img.copy()
    
    # Draw MediaPipe pose (simplified - just key joints)
    POSE_CONNECTIONS = [
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (11, 12),  # Shoulders
        (11, 23), (12, 24),  # Torso
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28),  # Right leg
    ]
    
    for start, end in POSE_CONNECTIONS:
        if kpts[start][3] > 0.5 and kpts[end][3] > 0.5:
            x1, y1 = int(kpts[start][0] * w), int(kpts[start][1] * h)
            x2, y2 = int(kpts[end][0] * w), int(kpts[end][1] * h)
            cv2.line(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw important landmarks
    IMPORTANT_LANDMARKS = {
        15: ("L_Wrist", (255, 0, 0)),
        16: ("R_Wrist", (255, 0, 0)),
        17: ("L_Pinky", (0, 0, 255)),  # Key for fallback
        18: ("R_Pinky", (0, 0, 255)),  # Key for fallback
        19: ("L_Index", (255, 255, 0)),
        20: ("R_Index", (255, 255, 0)),
    }
    
    for idx, (name, color) in IMPORTANT_LANDMARKS.items():
        if kpts[idx][3] > 0.5:
            x, y = int(kpts[idx][0] * w), int(kpts[idx][1] * h)
            cv2.circle(vis_img, (x, y), 8, color, -1)
            cv2.putText(vis_img, name, (x + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Get fallback stick estimation
    grip_norm, tip_norm = estimate_stick_from_fingers(
        kpts,
        results.pose_world_landmarks.landmark if results.pose_world_landmarks else results.pose_landmarks.landmark,
        w, h,
        viewpoint='front'
    )
    
    if grip_norm is not None:
        # Convert normalized to pixel coordinates
        grip_px = (int(grip_norm[0] * w), int(grip_norm[1] * h))
        tip_px = (int(tip_norm[0] * w), int(tip_norm[1] * h))
        
        # Draw estimated stick (GREEN = fallback estimation)
        cv2.line(vis_img, grip_px, tip_px, (0, 255, 0), 4)
        cv2.circle(vis_img, grip_px, 10, (0, 255, 0), -1)
        cv2.circle(vis_img, tip_px, 10, (0, 255, 255), -1)
        
        # Calculate stick length
        stick_len_px = np.sqrt((tip_px[0] - grip_px[0])**2 + (tip_px[1] - grip_px[1])**2)
        stick_len_norm = stick_len_px / np.sqrt(w**2 + h**2)
        
        estimation_status = f"FALLBACK: len={stick_len_norm:.3f}"
    else:
        estimation_status = "FALLBACK FAILED: no fingers visible"
    
    # If YOLO detected, also draw it for comparison (BLUE)
    if yolo_detected:
        stick_kpts = stick_results.keypoints.data[0].cpu().numpy()
        yolo_grip = (int(stick_kpts[0, 0]), int(stick_kpts[0, 1]))
        yolo_tip = (int(stick_kpts[1, 0]), int(stick_kpts[1, 1]))
        
        cv2.line(vis_img, yolo_grip, yolo_tip, (255, 0, 0), 2)
        cv2.circle(vis_img, yolo_grip, 8, (255, 0, 0), -1)
        cv2.circle(vis_img, yolo_tip, 8, (255, 255, 0), -1)
        
        yolo_status = "YOLO: detected"
    else:
        yolo_status = "YOLO: NO detection"
    
    # Add text overlay
    class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"class_{class_idx}"
    
    text_lines = [
        f"Class: {class_name}",
        f"File: {image_path.name[:30]}",
        yolo_status,
        estimation_status,
        "Green=Fallback, Blue=YOLO (if present)",
        "Red=Pinky, Yellow=Index, Blue circles=Wrists"
    ]
    
    y_offset = 30
    for line in text_lines:
        cv2.putText(vis_img, line, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    
    # Save
    cv2.imwrite(str(save_path), vis_img)
    
    return {
        'yolo_detected': yolo_detected,
        'fallback_success': grip_norm is not None,
        'estimation_status': estimation_status
    }, None


def main():
    print("="*60)
    print("Front View Classes 0-3 Fallback Visualization")
    print("="*60)
    
    # Load stick detector
    print("\nLoading YOLO stick detector...")
    stick_detector = YOLO(STICK_MODEL)
    
    # Find samples for each class
    samples_per_class = 5
    all_samples = []
    
    for class_idx in TARGET_CLASSES:
        class_name = CLASS_NAMES[class_idx]
        class_dir = Path(f'dataset_split/train/front/{class_name}')
        
        if not class_dir.exists():
            print(f"⚠️  Class {class_idx} ({class_name}): directory not found")
            continue
        
        images = list(class_dir.glob('*.jpg'))
        if len(images) > samples_per_class:
            images = random.sample(images, samples_per_class)
        
        print(f"\nClass {class_idx} ({class_name}): {len(images)} samples")
        
        for img_path in images:
            all_samples.append((img_path, class_idx, class_name))
    
    print(f"\n\nProcessing {len(all_samples)} total samples...")
    print("="*60)
    
    # Process each sample
    results = {idx: {'total': 0, 'yolo_found': 0, 'fallback_success': 0, 'fallback_failed': 0} 
               for idx in TARGET_CLASSES}
    
    for idx, (img_path, class_idx, class_name) in enumerate(all_samples):
        save_path = OUTPUT_DIR / f"class{class_idx}_{idx:03d}_{img_path.stem}.jpg"
        
        result, error = visualize_sample(img_path, class_idx, stick_detector, save_path)
        
        results[class_idx]['total'] += 1
        if result:
            if result['yolo_detected']:
                results[class_idx]['yolo_found'] += 1
            if result['fallback_success']:
                results[class_idx]['fallback_success'] += 1
            else:
                results[class_idx]['fallback_failed'] += 1
            status = "✓"
        else:
            status = f"✗ ({error})"
        
        print(f"  [{idx+1}/{len(all_samples)}] {img_path.name[:40]:40s} {status}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for class_idx in TARGET_CLASSES:
        if results[class_idx]['total'] > 0:
            r = results[class_idx]
            print(f"\nClass {class_idx} ({CLASS_NAMES[class_idx]}):")
            print(f"  Total processed: {r['total']}")
            print(f"  YOLO detected: {r['yolo_found']} ({100*r['yolo_found']/r['total']:.0f}%)")
            print(f"  Fallback succeeded: {r['fallback_success']} ({100*r['fallback_success']/r['total']:.0f}%)")
            print(f"  Fallback failed: {r['fallback_failed']}")
    
    print(f"\n\nVisualizations saved to: {OUTPUT_DIR}")
    print("Review these images to check if fallback stick positions look reasonable.")
    print("\nKey:")
    print("  - Green line/circle = Fallback stick (from fingers)")
    print("  - Blue line/circle = YOLO detection (when present)")
    print("  - Red dots = Pinky landmarks (used for fallback)")
    print("  - Yellow dots = Index fingers (fallback chain)")


if __name__ == "__main__":
    main()
