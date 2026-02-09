"""
Step 2b: Generate Node-Specific + Hybrid Features (Option 2)
- Node features: Per-node geometric data (x, y, visibility, angles, distances)
- Global features: Hybrid similarity scores (existing approach)
- Uses multiprocessing for faster extraction
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from pathlib import Path
import json
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Config
FEATURE_TEMPLATES = "hybrid_classifier/feature_templates.json"
DATASET_ROOT = Path("dataset_augmented")
OUTPUT_DIR = Path("hybrid_classifier/hybrid_features_v2")
STICK_MODEL = "runs/pose/arnis_stick_detector/weights/best.pt"

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'neutral_stance', 'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]

# Skeleton edges
SKELETON_EDGES = [
    (11, 12), (12, 11), (11, 23), (23, 11), (12, 24), (24, 12),
    (23, 24), (24, 23), (11, 13), (13, 11), (13, 15), (15, 13),
    (12, 14), (14, 12), (14, 16), (16, 14), (23, 25), (25, 23),
    (25, 27), (27, 25), (24, 26), (26, 24), (26, 28), (28, 26),
    (15, 33), (33, 15), (16, 33), (33, 16), (33, 34), (34, 33)
]


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


def extract_raw_features(image_path, stick_detector):
    """
    Extract raw features from a single image.
    Returns:
        - pose_keypoints: [33, 3] array (x, y, visibility)
        - stick_keypoints: [2, 3] array (grip and tip)
        - global_geometric_features: dict of computed features
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # Extract body pose
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=2)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(img_rgb)
    pose_detector.close()
    
    if not results.pose_landmarks:
        return None
    
    # Get keypoints (normalized 3D world coordinates)
    kpts = []
    for lm in results.pose_landmarks.landmark:
        kpts.append([lm.x, lm.y, lm.z, lm.visibility])
    kpts = np.array(kpts)
    
    # Extract stick
    stick_results = stick_detector(str(image_path), verbose=False)[0]
    if stick_results.keypoints is not None and len(stick_results.keypoints.data) > 0:
        stick_kpts = stick_results.keypoints.data[0].cpu().numpy()
        # Add z=0 for stick (YOLO doesn't provide depth)
        stick_grip = [stick_kpts[0, 0] / w, stick_kpts[0, 1] / h, 0.0, stick_kpts[0, 2]]
        stick_tip = [stick_kpts[1, 0] / w, stick_kpts[1, 1] / h, 0.0, stick_kpts[1, 2]]
    else:
        stick_grip = [0.5, 0.5, 0.0, 0.0]
        stick_tip = [0.5, 0.5, 0.0, 0.0]
    
    stick_keypoints = np.array([stick_grip, stick_tip])
    
    # Compute global geometric features (same as before)
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
    
    # Stick angle
    stick_vector = np.array([stick_tip[0] - stick_grip[0], stick_tip[1] - stick_grip[1]])
    features['stick_angle'] = np.degrees(np.arctan2(stick_vector[1], stick_vector[0]))
    
    stick_len = np.linalg.norm(stick_vector) + 1e-6
    features['stick_dx'] = stick_vector[0] / stick_len
    features['stick_dy'] = stick_vector[1] / stick_len
    
    # Expert features
    root_x = (kpts[23][0] + kpts[24][0]) / 2
    root_y = (kpts[23][1] + kpts[24][1]) / 2
    shoulder_y = (kpts[11][1] + kpts[12][1]) / 2
    nose_y = kpts[0][1]
    
    features['tip_vs_nose'] = stick_tip[1] - nose_y
    features['tip_vs_shoulder'] = stick_tip[1] - shoulder_y
    features['tip_vs_hip'] = stick_tip[1] - root_y
    
    features['r_hand_vs_nose'] = kpts[16][1] - nose_y
    features['r_hand_vs_shoulder'] = kpts[16][1] - shoulder_y
    features['r_hand_vs_hip'] = kpts[16][1] - root_y
    
    features['tip_side'] = stick_tip[0] - root_x
    features['grip_side'] = stick_grip[0] - root_x
    
    features['foot_stagger'] = kpts[27][1] - kpts[28][1]
    
    features['hands_distance'] = calculate_distance(kpts[15], kpts[16])
    features['stick_length'] = calculate_distance(stick_grip, stick_tip)
    
    return {
        'pose_keypoints': kpts,
        'stick_keypoints': stick_keypoints,
        'global_features': features
    }


def gaussian_similarity(value, mean, std):
    """Compute similarity score using Gaussian distribution"""
    if std < 1e-6:
        std = 1e-6
    return np.exp(-0.5 * ((value - mean) / std) ** 2)


def compute_hybrid_features(raw_features, templates, viewpoint, class_name):
    """Convert raw geometric features to similarity scores"""
    key = f"{viewpoint}_{class_name}"
    
    if key not in templates:
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


def extract_node_features(pose_keypoints, stick_keypoints):
    """
    Extract per-node features with 3D coordinates.
    Each node gets: [x, y, z, visibility, distance_to_hip_3d, angle_from_hip]
    """
    # Combine all nodes (33 pose + 2 stick)
    all_keypoints = np.vstack([pose_keypoints, stick_keypoints])
    
    # Compute hip center for reference (3D)
    hip_center = (pose_keypoints[23, :3] + pose_keypoints[24, :3]) / 2  # [x, y, z]
    
    node_features = []
    for i, kpt in enumerate(all_keypoints):
        x, y, z, vis = kpt
        
        # 3D Distance to hip center
        dist_to_hip = np.sqrt((x - hip_center[0])**2 + (y - hip_center[1])**2 + (z - hip_center[2])**2)
        
        # Angle from hip center (2D projection for compatibility)
        angle_from_hip = np.degrees(np.arctan2(y - hip_center[1], x - hip_center[0]))
        
        # Node feature: [x, y, z, vis, dist_to_hip_3d, angle_from_hip]
        node_features.append([x, y, z, vis, dist_to_hip, angle_from_hip])
    
    return np.array(node_features, dtype=np.float32)


def process_single_image(args):
    """Process a single image (for multiprocessing)"""
    img_path, class_idx, viewpoint, templates, stick_detector = args
    
    try:
        # Extract raw features
        raw_data = extract_raw_features(img_path, stick_detector)
        if raw_data is None:
            return None
        
        # Extract node-specific features
        node_features = extract_node_features(
            raw_data['pose_keypoints'],
            raw_data['stick_keypoints']
        )
        
        # Compute global hybrid features
        class_name = CLASS_NAMES[class_idx]
        hybrid_features = compute_hybrid_features(
            raw_data['global_features'],
            templates,
            viewpoint,
            class_name
        )
        
        return {
            'node_features': node_features,
            'hybrid_features': hybrid_features,
            'label': class_idx,
            'viewpoint': viewpoint
        }
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def process_dataset(viewpoint_filter=None, num_workers=None):
    """Process all images and generate node + hybrid features"""
    # Load templates
    with open(FEATURE_TEMPLATES, 'r') as f:
        templates = json.load(f)
    
    print(f"Loaded {len(templates)} feature templates")
    
    # Load stick detector once
    stick_detector = YOLO(STICK_MODEL)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"Using {num_workers} workers for multiprocessing")
    
    viewpoints = [viewpoint_filter] if viewpoint_filter else ['front', 'left', 'right']
    
    for split in ['train', 'test']:
        all_tasks = []
        
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
                
                for img_path in images:
                    all_tasks.append((img_path, class_idx, viewpoint, templates, stick_detector))
        
        print(f"\nProcessing {split} set: {len(all_tasks)} images")
        
        # Process with multiprocessing
        results = []
        with Pool(num_workers) as pool:
            for result in tqdm(pool.imap(process_single_image, all_tasks), total=len(all_tasks)):
                if result is not None:
                    results.append(result)
        
        # Aggregate results
        node_features_list = []
        hybrid_features_list = []
        labels_list = []
        viewpoints_list = []
        
        for result in results:
            node_features_list.append(result['node_features'])
            hybrid_features_list.append(result['hybrid_features'])
            labels_list.append(result['label'])
            viewpoints_list.append(result['viewpoint'])
        
        # Save as PyTorch tensors
        data = {
            'node_features': torch.tensor(np.array(node_features_list), dtype=torch.float32),
            'hybrid_features': torch.tensor(np.array(hybrid_features_list), dtype=torch.float32),
            'labels': torch.tensor(labels_list, dtype=torch.long),
            'viewpoints': viewpoints_list
        }
        
        suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
        output_file = OUTPUT_DIR / f"{split}_features{suffix}.pt"
        torch.save(data, output_file)
        
        print(f"✓ Saved {split} features: {len(results)} samples to {output_file}")
        print(f"  - Node features shape: {data['node_features'].shape}")
        print(f"  - Hybrid features shape: {data['hybrid_features'].shape}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewpoint', type=str, default=None,
                        choices=['front', 'left', 'right'],
                        help='Process only specific viewpoint (default: all)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: CPU count - 1)')
    args = parser.parse_args()
    
    process_dataset(args.viewpoint, args.workers)
