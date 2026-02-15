"""
Step 2: Generate Hybrid Features for Full Dataset
Uses feature templates to convert poses into similarity-based features
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from pathlib import Path
import json
import torch
from tqdm import tqdm

# Config
FEATURE_TEMPLATES = "hybrid_classifier/feature_templates.json"
DATASET_ROOT = Path("dataset_augmented")
OUTPUT_DIR = Path("hybrid_classifier/hybrid_features")
STICK_MODEL = "runs/pose/arnis_stick_detector/weights/best.pt"

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]

# Initialize detectors
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=2)
stick_detector = YOLO(STICK_MODEL)


def calculate_angle(p1, p2, p3):
    """Calculate angle at p2 formed by p1-p2-p3"""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)


def calculate_distance(p1, p2):
    """Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def extract_geometric_features(image_path):
    """Extract all geometric features from a single image"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # Extract body pose
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(img_rgb)
    
    if not results.pose_landmarks:
        return None
    
    # Get keypoints (normalized)
    kpts = []
    for lm in results.pose_landmarks.landmark:
        kpts.append([lm.x, lm.y, lm.visibility])
    kpts = np.array(kpts)
    
    # Extract stick
    stick_results = stick_detector(str(image_path), verbose=False)[0]
    if stick_results.keypoints is not None and len(stick_results.keypoints.data) > 0:
        stick_kpts = stick_results.keypoints.data[0].cpu().numpy()
        stick_grip = [stick_kpts[0, 0] / w, stick_kpts[0, 1] / h]
        stick_tip = [stick_kpts[1, 0] / w, stick_kpts[1, 1] / h]
    else:
        stick_grip = [0.5, 0.5]
        stick_tip = [0.5, 0.5]
    
    # Compute features (same as reference extraction)
    features = {}
    
    features['left_elbow_angle'] = calculate_angle(kpts[11], kpts[13], kpts[15])
    features['right_elbow_angle'] = calculate_angle(kpts[12], kpts[14], kpts[16])
    features['left_shoulder_angle'] = calculate_angle(kpts[13], kpts[11], kpts[23])
    features['right_shoulder_angle'] = calculate_angle(kpts[14], kpts[12], kpts[24])
    features['left_knee_angle'] = calculate_angle(kpts[23], kpts[25], kpts[27])
    features['right_knee_angle'] = calculate_angle(kpts[24], kpts[26], kpts[28])
    
    hip_center_y = (kpts[23][1] + kpts[24][1]) / 2
    features['left_wrist_height'] = hip_center_y - kpts[15][1]
    features['right_wrist_height'] = hip_center_y - kpts[16][1]
    features['left_elbow_height'] = hip_center_y - kpts[13][1]
    features['right_elbow_height'] = hip_center_y - kpts[14][1]
    features['stick_tip_height'] = hip_center_y - stick_tip[1]
    features['stick_grip_height'] = hip_center_y - stick_grip[1]
    
    hip_center_x = (kpts[23][0] + kpts[24][0]) / 2
    features['left_wrist_x'] = kpts[15][0] - hip_center_x
    features['right_wrist_x'] = kpts[16][0] - hip_center_x
    features['stick_tip_x'] = stick_tip[0] - hip_center_x
    features['stick_grip_x'] = stick_grip[0] - hip_center_x
    
    # Stick angle (relative to horizontal)
    stick_vector = np.array([stick_tip[0] - stick_grip[0], stick_tip[1] - stick_grip[1]])
    features['stick_angle'] = np.degrees(np.arctan2(stick_vector[1], stick_vector[0]))
    
    # Stick direction components (Normalized)
    stick_len = np.linalg.norm(stick_vector) + 1e-6
    features['stick_dx'] = stick_vector[0] / stick_len
    features['stick_dy'] = stick_vector[1] / stick_len
    
    # --- Expert Features ---
    root_x = (kpts[23][0] + kpts[24][0]) / 2  # Hip center X
    root_y = (kpts[23][1] + kpts[24][1]) / 2  # Hip center Y
    shoulder_y = (kpts[11][1] + kpts[12][1]) / 2
    nose_y = kpts[0][1]
    
    # 1. Height Levels
    features['tip_vs_nose'] = stick_tip[1] - nose_y
    features['tip_vs_shoulder'] = stick_tip[1] - shoulder_y
    features['tip_vs_hip'] = stick_tip[1] - root_y
    
    # 2. Hand Levels
    features['r_hand_vs_nose'] = kpts[16][1] - nose_y
    features['r_hand_vs_shoulder'] = kpts[16][1] - shoulder_y
    features['r_hand_vs_hip'] = kpts[16][1] - root_y
    
    # 3. Horizontal Directions
    features['tip_side'] = stick_tip[0] - root_x
    features['grip_side'] = stick_grip[0] - root_x
    
    # 4. Foot Stance
    features['foot_stagger'] = kpts[27][1] - kpts[28][1]
    
    # Distances
    features['hands_distance'] = calculate_distance(kpts[15], kpts[16])
    features['stick_length'] = calculate_distance(stick_grip, stick_tip)
    
    return features


def gaussian_similarity(value, mean, std):
    """Compute similarity score using Gaussian distribution"""
    if std < 1e-6:
        std = 1e-6
    return np.exp(-0.5 * ((value - mean) / std) ** 2)


def compute_hybrid_features(raw_features, templates, viewpoint, class_name):
    """Convert raw geometric features to similarity scores"""
    key = f"{viewpoint}_{class_name}"
    
    if key not in templates:
        # Fallback: return zeros if no template
        return np.zeros(len(raw_features))
    
    template = templates[key]
    hybrid_features = []
    
    for feat_name, feat_value in raw_features.items():
        if feat_name in template:
            mean = template[feat_name]['mean']
            std = template[feat_name]['std']
            similarity = gaussian_similarity(feat_value, mean, std)
            hybrid_features.append(similarity)
        else:
            hybrid_features.append(0.0)
    
    return np.array(hybrid_features, dtype=np.float32)


def process_dataset(viewpoint_filter=None):
    """Process all images and generate hybrid features"""
    # Load templates
    with open(FEATURE_TEMPLATES, 'r') as f:
        templates = json.load(f)
    
    print(f"Loaded {len(templates)} feature templates")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    viewpoints = [viewpoint_filter] if viewpoint_filter else ['front', 'left', 'right']
    
    for split in ['train', 'test']:
        split_features = []
        split_labels = []
        split_viewpoints = []
        
        split_path = DATASET_ROOT / split
        
        for viewpoint in viewpoints:
            viewpoint_path = split_path / viewpoint
            
            if not viewpoint_path.exists():
                continue
            
            for class_idx, class_name in enumerate(CLASS_NAMES):
                class_dir = viewpoint_path / class_name
                
                if not class_dir.exists():
                    continue
                
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                
                for img_path in tqdm(images, desc=f"{split}/{viewpoint}/{class_name}", leave=False):
                    raw_features = extract_geometric_features(img_path)
                    
                    if raw_features is None:
                        continue
                    
                    hybrid_features = compute_hybrid_features(raw_features, templates, viewpoint, class_name)
                    
                    split_features.append(hybrid_features)
                    split_labels.append(class_idx)
                    split_viewpoints.append(viewpoint)
        
        # Save as PyTorch tensors
        data = {
            'features': torch.tensor(split_features, dtype=torch.float32),
            'labels': torch.tensor(split_labels, dtype=torch.long),
            'viewpoints': split_viewpoints
        }
        
        suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
        output_file = OUTPUT_DIR / f"{split}_features{suffix}.pt"
        torch.save(data, output_file)
        
        print(f"✓ Saved {split} features: {len(split_features)} samples to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewpoint', type=str, default=None,
                        choices=['front', 'left', 'right'],
                        help='Process only specific viewpoint (default: all)')
    args = parser.parse_args()
    
    process_dataset(args.viewpoint)
