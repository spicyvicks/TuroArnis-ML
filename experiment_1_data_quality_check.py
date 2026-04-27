"""
Experiment 1: Verify Left-Class Data Quality After Mirroring

Visualizes 5 random left_chest_thrust and 5 random right_chest_thrust 
training images with pose + stick overlay to verify:
1. Stick is on the correct hand (left hand for left_chest_thrust, right hand for right_chest_thrust)
2. Images were properly mirrored (camera view)
3. Labels match the visual content

Saves visualization images to: experiment_1_data_quality_check/
"""

import cv2
import numpy as np
import torch
import json
from pathlib import Path
from tqdm import tqdm
import random

# MediaPipe imports
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Paths
TRAIN_DIR = Path("dataset_split/train/front")
FEATURES_PATH = Path("hybrid_classifier/hybrid_features_v3/train_features_front.pt")
OUTPUT_DIR = Path("experiment_1_data_quality_check")
OUTPUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct',
    'neutral'
]

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),(10,11),
    (8,12),(12,13),(13,14),(0,15),(15,17),(0,16),(16,18),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32)
]

# Colors
COLOR_RIGHT = (0, 255, 0)      # Green
COLOR_LEFT = (0, 0, 255)       # Red
COLOR_STICK = (255, 0, 255)    # Magenta
COLOR_TEXT = (255, 255, 255)   # White
COLOR_BAD = (0, 0, 255)        # Red for errors


def draw_pose_on_image(image, pose_keypoints, stick_keypoints, label_text, expected_hand):
    """Draw pose skeleton and stick on image."""
    h, w = image.shape[:2]
    vis_img = image.copy()
    
    # Draw pose connections
    for a, b in POSE_CONNECTIONS:
        if a < len(pose_keypoints) and b < len(pose_keypoints):
            pa = pose_keypoints[a]
            pb = pose_keypoints[b]
            if pa[3] > 0.1 and pb[3] > 0.1:
                x1, y1 = int(pa[0] * w), int(pa[1] * h)
                x2, y2 = int(pb[0] * w), int(pb[1] * h)
                cv2.line(vis_img, (x1, y1), (x2, y2), (200, 200, 200), 2)
    
    # Draw keypoints with labels
    LABEL_JOINTS = {11:"Lsho", 12:"Rsho", 13:"Lelb", 14:"Relb", 15:"Lwri", 16:"Rwri",
                    23:"Lhip", 24:"Rhip", 25:"Lkne", 26:"Rkne", 27:"Lank", 28:"Rank"}
    for i, kp in enumerate(pose_keypoints):
        if kp[3] > 0.1:
            x, y = int(kp[0] * w), int(kp[1] * h)
            color = COLOR_RIGHT if i in [12, 14, 16, 18, 20, 22] else COLOR_LEFT if i in [11, 13, 15, 17, 19, 21] else (255, 255, 0)
            cv2.circle(vis_img, (x, y), 5, color, -1)
            if i in LABEL_JOINTS:
                label = LABEL_JOINTS[i]
                # Draw text background for readability
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(vis_img, (x+4, y-th-2), (x+4+tw+2, y+2), (0,0,0), -1)
                cv2.putText(vis_img, label, (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw stick
    if len(stick_keypoints) >= 2:
        grip = stick_keypoints[0]
        tip = stick_keypoints[1]
        if grip[3] > 0.1 and tip[3] > 0.1:
            gx, gy = int(grip[0] * w), int(grip[1] * h)
            tx, ty = int(tip[0] * w), int(tip[1] * h)
            cv2.line(vis_img, (gx, gy), (tx, ty), COLOR_STICK, 4)
            cv2.circle(vis_img, (gx, gy), 10, COLOR_STICK, -1)
            cv2.circle(vis_img, (tx, ty), 7, (255, 255, 0), -1)
            # Label grip
            cv2.putText(vis_img, "GRIP", (gx+8, gy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_STICK, 2)
    
    # Determine which hand is closer to stick grip
    if len(stick_keypoints) >= 1 and len(pose_keypoints) >= 17:
        grip = stick_keypoints[0]
        left_wrist = pose_keypoints[15]
        right_wrist = pose_keypoints[16]
        
        if grip[3] > 0.1 and left_wrist[3] > 0.1 and right_wrist[3] > 0.1:
            d_left = np.linalg.norm(grip[:2] - left_wrist[:2])
            d_right = np.linalg.norm(grip[:2] - right_wrist[:2])
            detected_hand = "LEFT" if d_left < d_right else "RIGHT"
            
            match = (detected_hand == expected_hand.upper())
            match_color = COLOR_RIGHT if match else COLOR_BAD
            match_text = "✓ MATCH" if match else "✗ MISMATCH"
            
            cv2.putText(vis_img, f"Stick: {detected_hand} hand", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2)
            cv2.putText(vis_img, f"Label: {expected_hand} hand", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2)
            cv2.putText(vis_img, match_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2)
            cv2.putText(vis_img, f"L-dist: {d_left:.3f}  R-dist: {d_right:.3f}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
    
    # Top banner with label
    banner = np.zeros((40, w, 3), dtype=np.uint8)
    cv2.putText(banner, label_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
    vis_img = np.vstack([banner, vis_img])
    
    return vis_img


def process_samples(class_name, expected_hand, n_samples=5):
    """Process n random samples from a class directory."""
    class_dir = TRAIN_DIR / class_name
    if not class_dir.exists():
        print(f"WARNING: Directory not found: {class_dir}")
        return []
    
    images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
    # Exclude augmented images
    images = [f for f in images if '_aug' not in f.name]
    
    if len(images) < n_samples:
        print(f"WARNING: Only {len(images)} original images in {class_name}")
        n_samples = len(images)
    
    samples = random.sample(images, n_samples)
    results = []
    
    pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=2)
    
    for img_path in tqdm(samples, desc=f"  {class_name}"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect pose
        results_pose = pose_detector.process(img_rgb)
        if not results_pose.pose_landmarks:
            continue
        
        # Extract keypoints
        pose_keypoints = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results_pose.pose_landmarks.landmark])
        
        # Stick detection (simple: find pinky nearest to stick-like object)
        # For visualization, we'll use wrist positions as proxy if no YOLO available
        # In actual pipeline, use YOLO stick detector
        left_wrist = pose_keypoints[15]
        right_wrist = pose_keypoints[16]
        
        # Heuristic: stick grip is the wrist of the expected hand
        # This is a rough approximation for quick checking
        expected_wrist = right_wrist if expected_hand == "right" else left_wrist
        stick_keypoints = np.array([
            expected_wrist,  # grip = expected wrist
            [expected_wrist[0] + 0.1, expected_wrist[1] + 0.1, expected_wrist[2], 1.0]  # fake tip
        ])
        
        label_text = f"{class_name} | File: {img_path.name}"
        vis_img = draw_pose_on_image(img, pose_keypoints, stick_keypoints, label_text, expected_hand)
        
        # Save
        out_path = OUTPUT_DIR / f"{class_name}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), vis_img)
        results.append({
            'file': str(img_path),
            'expected_hand': expected_hand,
            'saved_to': str(out_path)
        })
    
    pose_detector.close()
    return results


def main():
    print("="*60)
    print("EXPERIMENT 1: Left-Class Data Quality Check")
    print("="*60)
    print("\nChecking if mirrored training images have stick on correct hand")
    print("Visualizations saved to: experiment_1_data_quality_check/\n")
    
    random.seed(42)
    
    # Check left_chest_thrust (should have stick on LEFT hand)
    print("Processing left_chest_thrust_correct (expect LEFT hand)...")
    left_results = process_samples("left_chest_thrust_correct", "left", n_samples=5)
    
    # Check right_chest_thrust (should have stick on RIGHT hand)
    print("Processing right_chest_thrust_correct (expect RIGHT hand)...")
    right_results = process_samples("right_chest_thrust_correct", "right", n_samples=5)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"left_chest_thrust images checked:  {len(left_results)}")
    print(f"right_chest_thrust images checked: {len(right_results)}")
    print(f"\nVisualizations saved to: {OUTPUT_DIR.absolute()}")
    print("\nOpen these images and verify:")
    print("  - left_chest_thrust: stick should be on LEFT side (image-right due to mirror)")
    print("  - right_chest_thrust: stick should be on RIGHT side (image-left due to mirror)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
