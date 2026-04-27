"""
Option B: Verify Actual Stick Positions from .pt Feature Tensors

Reads the REAL extracted features (train_features_front.pt) and visualizes:
1. Actual pose keypoints from node_features (nodes 0-32)
2. Actual stick position from node_features (nodes 33=grip, 34=tip)
3. Distance from stick grip to Lwri (node 15) and Rwri (node 16)
4. The stick_right_hand boolean metadata
5. Whether visual stick position matches the metadata

This reveals if YOLO + Method 4 actually attached the stick to the correct hand.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import random

# Paths
TRAIN_DIR = Path("dataset_split/train/front")
FEATURES_PATH = Path("hybrid_classifier/hybrid_features_v3/train_features_front.pt")
OUTPUT_DIR = Path("option_b_stick_verification")
OUTPUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct',
    'neutral'
]

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Colors
COLOR_LWRI = (0, 0, 255)       # Red = left wrist
COLOR_RWRI = (0, 255, 0)       # Green = right wrist
COLOR_STICK = (255, 0, 255)    # Magenta = stick
COLOR_MATCH = (0, 255, 0)      # Green = metadata matches visual
COLOR_MISMATCH = (0, 0, 255)   # Red = metadata does NOT match visual
COLOR_TEXT = (255, 255, 255)   # White


def get_image_files_for_class(class_name, n_samples):
    """Get sorted list of original (non-aug) images for a class."""
    class_dir = TRAIN_DIR / class_name
    if not class_dir.exists():
        return []
    images = sorted([f for f in class_dir.iterdir()
                     if f.suffix.lower() in ('.jpg', '.png') and '_aug' not in f.name])
    return images[:n_samples]


def visualize_sample(img_path, node_features, label_idx, stick_right_hand, sample_idx, class_name):
    """Draw the actual pose and stick from the feature tensor onto the image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    vis = img.copy()
    
    # Nodes 0-32 are pose landmarks (x,y,z,vis,dist_to_hip,angle_from_hip)
    pose = node_features[:33]  # [33, 6]
    
    # Nodes 33-34 are stick (grip, tip)
    stick_grip = node_features[33]  # [6]
    stick_tip = node_features[34]   # [6]
    
    # Draw pose skeleton connections
    connections = [
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (11, 12),            # shoulders
        (11, 23), (12, 24),  # torso
        (23, 25), (25, 27),  # left leg
        (24, 26), (26, 28),  # right leg
    ]
    for a, b in connections:
        if a < len(pose) and b < len(pose):
            pa = pose[a]
            pb = pose[b]
            if pa[3] > 0.1 and pb[3] > 0.1:
                x1, y1 = int(pa[0] * w), int(pa[1] * h)
                x2, y2 = int(pb[0] * w), int(pb[1] * h)
                cv2.line(vis, (x1, y1), (x2, y2), (180, 180, 180), 2)
    
    # Draw all pose keypoints
    for i in range(33):
        kp = pose[i]
        if kp[3] > 0.1:  # visibility check
            x, y = int(kp[0] * w), int(kp[1] * h)
            color = (255, 255, 0)
            cv2.circle(vis, (x, y), 4, color, -1)
    
    # Highlight wrists
    lwri = pose[15]
    rwri = pose[12]
    if lwri[3] > 0.1:
        x, y = int(lwri[0] * w), int(lwri[1] * h)
        cv2.circle(vis, (x, y), 10, COLOR_LWRI, 2)
        cv2.putText(vis, "Lwri", (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LWRI, 2)
    if rwri[3] > 0.1:
        x, y = int(rwri[0] * w), int(rwri[1] * h)
        cv2.circle(vis, (x, y), 10, COLOR_RWRI, 2)
        cv2.putText(vis, "Rwri", (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RWRI, 2)
    
    # Draw ACTUAL stick from tensor (nodes 33, 34)
    grip_vis = stick_grip[3]  # visibility
    tip_vis = stick_tip[3]
    
    if grip_vis > 0.1:
        gx, gy = int(stick_grip[0] * w), int(stick_grip[1] * h)
        cv2.circle(vis, (gx, gy), 12, COLOR_STICK, -1)
        cv2.putText(vis, "GRIP", (gx+10, gy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_STICK, 2)
        
        if tip_vis > 0.1:
            tx, ty = int(stick_tip[0] * w), int(stick_tip[1] * h)
            cv2.line(vis, (gx, gy), (tx, ty), COLOR_STICK, 4)
            cv2.circle(vis, (tx, ty), 8, (255, 255, 0), -1)
    
    # Calculate distance from grip to each wrist
    if grip_vis > 0.1 and lwri[3] > 0.1 and rwri[3] > 0.1:
        d_left = np.linalg.norm(stick_grip[:2] - lwri[:2])
        d_right = np.linalg.norm(stick_grip[:2] - rwri[:2])
        
        visual_right = d_right < d_left
        metadata_right = bool(stick_right_hand)
        
        match = (visual_right == metadata_right)
        match_color = COLOR_MATCH if match else COLOR_MISMATCH
        match_text = "MATCH" if match else "MISMATCH!"
        
        # Draw info box
        y_offset = 30
        lines = [
            f"Class: {class_name}",
            f"File: {img_path.name}",
            f"stick_right_hand (meta): {metadata_right}",
            f"Visual closest: {'RIGHT' if visual_right else 'LEFT'}",
            f"Dist L: {d_left:.4f}  Dist R: {d_right:.4f}",
            f"RESULT: {match_text}",
        ]
        
        for i, line in enumerate(lines):
            color = match_color if i == len(lines)-1 else COLOR_TEXT
            cv2.putText(vis, line, (10, y_offset + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        cv2.putText(vis, "Missing grip or wrist data", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Banner
    banner = np.zeros((50, w, 3), dtype=np.uint8)
    cv2.putText(banner, f"{class_name} | {img_path.name} | Tensor idx {sample_idx}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
    vis = np.vstack([banner, vis])
    
    return vis


def main():
    print("="*60)
    print("OPTION B: Verify Actual Stick Positions from .pt Tensors")
    print("="*60)
    print(f"\nLoading features from: {FEATURES_PATH}")
    
    if not FEATURES_PATH.exists():
        print(f"ERROR: Features file not found: {FEATURES_PATH}")
        return
    
    data = torch.load(FEATURES_PATH, map_location='cpu')
    node_features = data['node_features']  # [N, 35, 6]
    labels = data['labels']                 # [N]
    stick_right_hand = data.get('stick_right_hand', torch.ones(len(labels), dtype=torch.bool))
    
    print(f"Total samples: {len(labels)}")
    print(f"Node features shape: {node_features.shape}")
    print(f"stick_right_hand available: {stick_right_hand is not None}")
    
    # Process left_chest_thrust and right_chest_thrust
    target_classes = ['left_chest_thrust_correct', 'right_chest_thrust_correct']
    n_check = 5  # samples per class
    
    for class_name in target_classes:
        class_idx = CLASS_TO_IDX[class_name]
        print(f"\n{'='*60}")
        print(f"Checking {class_name} (class index {class_idx})")
        print(f"{'='*60}")
        
        # Find tensor indices for this class
        class_mask = (labels == class_idx).nonzero(as_tuple=True)[0]
        n_class = len(class_mask)
        print(f"  Found {n_class} samples in tensor")
        
        # Get image files (sorted, non-augmented)
        image_files = get_image_files_for_class(class_name, n_check)
        print(f"  Found {len(image_files)} original images")
        
        if n_class < len(image_files):
            print(f"  WARNING: More images ({len(image_files)}) than tensor samples ({n_class})")
            image_files = image_files[:n_class]
        
        # Check each sample
        mismatches = 0
        for i in range(min(n_check, len(image_files), n_class)):
            tensor_idx = int(class_mask[i])
            img_path = image_files[i]
            
            nf = node_features[tensor_idx].numpy()
            srh = bool(stick_right_hand[tensor_idx])
            
            vis = visualize_sample(img_path, nf, class_idx, srh, tensor_idx, class_name)
            if vis is not None:
                out_path = OUTPUT_DIR / f"{class_name}_sample{i}_idx{tensor_idx}.jpg"
                cv2.imwrite(str(out_path), vis)
                
                # Quick check for mismatch
                pose = nf[:33]
                lwri = pose[15]
                rwri = pose[12]
                grip = nf[33]
                if grip[3] > 0.1 and lwri[3] > 0.1 and rwri[3] > 0.1:
                    d_left = np.linalg.norm(grip[:2] - lwri[:2])
                    d_right = np.linalg.norm(grip[:2] - rwri[:2])
                    visual_right = d_right < d_left
                    match = (visual_right == srh)
                    if not match:
                        mismatches += 1
                        print(f"    SAMPLE {i}: MISMATCH! meta={srh}, visual={'RIGHT' if visual_right else 'LEFT'}")
                    else:
                        print(f"    SAMPLE {i}: OK meta={srh}, visual={'RIGHT' if visual_right else 'LEFT'}")
                else:
                    print(f"    SAMPLE {i}: Missing data (grip_vis={grip[3]:.2f}, lwri_vis={lwri[3]:.2f}, rwri_vis={rwri[3]:.2f})")
        
        print(f"\n  {class_name}: {mismatches}/{min(n_check, len(image_files))} mismatches")
    
    print(f"\n{'='*60}")
    print("DONE")
    print(f"Visualizations saved to: {OUTPUT_DIR.absolute()}")
    print("\nOpen images and verify:")
    print("  - GRIP dot should be on the correct hand")
    print("  - 'stick_right_hand' metadata should match visual position")
    print("  - Red 'MISMATCH!' means the feature extractor attached stick to wrong hand")
    print(f"{'='*60}")
    
    # After verification, remind user
    print("\n" + "="*60)
    print("REMINDER: After reviewing these images,")
    print("run Option A (Experiment 2 - Regularization Training)")
    print("="*60)


if __name__ == "__main__":
    main()
