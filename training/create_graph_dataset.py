"""
Graph Dataset Creation Script
Converts raw images from dataset_split/ into graph-structured .pt files
Integrates MediaPipe pose extraction + YOLO stick detection
"""

import os
import cv2
import torch
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
from pathlib import Path

# Add current directory to path to allow imports
sys.path.append(str(Path(__file__).parent))

from torch_geometric.data import Data
from ultralytics import YOLO
from graph_augmentation import augment_graph_conservative
from pose_features import extract_geometric_features


# MediaPipe setup
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,  # Full BlazePose
    min_detection_confidence=0.5
)

# Stick detector setup
STICK_DETECTOR_PATH = "runs/pose/arnis_stick_detector/weights/best.pt"
stick_detector = None

# Class mapping (13 classes)
CLASS_NAMES = [
    'crown_thrust_correct',
    'left_chest_thrust_correct',
    'left_elbow_block_correct',
    'left_eye_thrust_correct',
    'left_knee_block_correct',
    'left_temple_block_correct',
    'neutral_stance',
    'right_chest_thrust_correct',
    'right_elbow_block_correct',
    'right_eye_thrust_correct',
    'right_knee_block_correct',
    'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]

# Skeleton edge connections (MediaPipe BlazePose)
SKELETON_EDGES = [
    # Torso
    (11, 12), (12, 11),  # shoulders
    (11, 23), (23, 11),  # left shoulder → left hip
    (12, 24), (24, 12),  # right shoulder → right hip
    (23, 24), (24, 23),  # hips
    # Left arm
    (11, 13), (13, 11),  # left shoulder → left elbow
    (13, 15), (15, 13),  # left elbow → left wrist
    # Right arm
    (12, 14), (14, 12),  # right shoulder → right elbow
    (14, 16), (16, 14),  # right elbow → right wrist
    # Left leg
    (23, 25), (25, 23),  # left hip → left knee
    (25, 27), (27, 25),  # left knee → left ankle
    # Right leg
    (24, 26), (26, 24),  # right hip → right knee
    (26, 28), (28, 26),  # right knee → right ankle
    # Face (optional, for stability)
    (0, 1), (1, 0),
    (1, 2), (2, 1),
    (2, 3), (3, 2),
    (3, 7), (7, 3),
    # Stick connections (wrists → stick nodes)
    (15, 33), (33, 15),  # left wrist → stick grip
    (16, 33), (33, 16),  # right wrist → stick grip
    (33, 34), (34, 33),  # stick grip ↔ stick tip
]


def extract_pose_keypoints(image_path):
    """
    Extract 33 MediaPipe pose keypoints from image.
    
    Returns:
        np.array of shape [33, 3] (x, y, visibility)
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = pose_detector.process(image_rgb)
    
    if not results.pose_landmarks:
        return None
    
    # Extract keypoints
    keypoints = []
    for landmark in results.pose_landmarks.landmark:
        keypoints.append([landmark.x, landmark.y, landmark.visibility])
    
    keypoints = np.array(keypoints, dtype=np.float32)
    
    # --- NORMALIZATION START ---
    # 1. Center pose: Move hip center to (0,0)
    # Hips are indices 23 (left) and 24 (right)
    hip_center = (keypoints[23, :2] + keypoints[24, :2]) / 2.0
    keypoints[:, :2] -= hip_center
    
    # 2. Scale pose: Normalize by torso height (shoulder to hip)
    # Shoulders: 11, 12. Hips: 23, 24
    shoulder_center = (keypoints[11, :2] + keypoints[12, :2]) / 2.0
    # Current hip center is (0,0) after centering
    
    torso_size = np.linalg.norm(shoulder_center)  # dist from (0,0) to shoulder center
    if torso_size > 0:
        keypoints[:, :2] /= (torso_size * 2.0) # Scale so torso is 0.5 units
        
    # --- NORMALIZATION END ---
    
    return keypoints

def load_stick_detector():
    global stick_detector
    if stick_detector is None:
        try:
            print(f"Loading stick detector from {STICK_DETECTOR_PATH}...")
            stick_detector = YOLO(STICK_DETECTOR_PATH)
        except Exception as e:
            print(f"Warning: Failed to load stick detector: {e}")
            stick_detector = None

def detect_stick(image_path):
    """
    Detect stick using YOLOv8-Pose and extract 2 keypoints (grip and tip).
    """
    if stick_detector is None:
        # Return dummy nodes if detector not loaded or skipped
        return np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], dtype=np.float32)

    try:
        results = stick_detector.predict(str(image_path), verbose=False)
        
        # Check if stick detected (keypoints available)
        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            kpts = results[0].keypoints.data[0].cpu().numpy()  # Shape: [2, 3]
            
            image = cv2.imread(str(image_path))
            h, w = image.shape[:2]
            
            # Normalize x, y to [0, 1]
            stick_grip = [kpts[0, 0] / w, kpts[0, 1] / h, float(kpts[0, 2])]
            stick_tip = [kpts[1, 0] / w, kpts[1, 1] / h, float(kpts[1, 2])]
            
            return np.array([stick_grip, stick_tip], dtype=np.float32)
        else:
            return np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], dtype=np.float32)
            
    except Exception as e:
        # print(f"Warning: Stick detection failed for {image_path}: {e}")
        return np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], dtype=np.float32)


def build_graph(pose_keypoints, stick_nodes, label, viewpoint):
    """
    Build PyG graph from pose keypoints and stick nodes.
    Now includes geometric features (angles, distances).
    """
    # 1. Extract geometric features (angles, distances, symmetry)
    # Returns vector of shape [num_features]
    geo_features = extract_geometric_features(pose_keypoints)
    
    # 2. Augment node features
    # Each node gets: [x, y, vis, ...global_geometric_features...]
    # Current node shape: [35, 3] -> [35, 3 + num_geo_features]
    
    # Combine nodes (33 pose + 2 stick)
    all_nodes = np.vstack([pose_keypoints, stick_nodes])
    num_nodes = all_nodes.shape[0]
    
    # Repeat global features for each node
    node_geo_features = np.tile(geo_features, (num_nodes, 1))
    
    # Concatenate: [35, 3] + [35, num_geo] -> [35, 3+num_geo]
    combined_features = np.hstack([all_nodes, node_geo_features])
    
    x = torch.tensor(combined_features, dtype=torch.float)
    
    # Edge index
    edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t().contiguous()
    
    # Label
    y = torch.tensor([label], dtype=torch.long)
    
    # Create graph
    graph = Data(x=x, edge_index=edge_index, y=y)
    graph.viewpoint = viewpoint
    
    return graph


def process_dataset(dataset_root, output_root, augment=False, skip_stick=False):
    """
    Process entire dataset_split/ directory.
    """
    if not skip_stick:
        load_stick_detector()
    else:
        print("Skipping stick detection (using dummy nodes).")

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    
    stats = {
        'total_images': 0,
        'successful': 0,
        'failed_pose': 0,
        'failed_stick': 0,
        'total_graphs': 0
    }
    
    # Process train and test splits
    for split in ['train', 'test']:
        split_path = dataset_root / split
        
        # Process each viewpoint
        for viewpoint in ['front', 'left', 'right']:
            viewpoint_path = split_path / viewpoint
            
            if not viewpoint_path.exists():
                continue
            
            # Process each class
            for class_dir in viewpoint_path.iterdir():
                if not class_dir.is_dir():
                    continue
                
                class_name = class_dir.name
                if class_name not in CLASS_NAMES:
                    print(f"Warning: Unknown class '{class_name}', skipping...")
                    continue
                
                class_idx = CLASS_NAMES.index(class_name)
                
                # Create output directory
                output_dir = output_root / split / viewpoint / class_name
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Process each image
                image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
                
                for img_path in tqdm(image_files, desc=f"{split}/{viewpoint}/{class_name}"):
                    stats['total_images'] += 1
                    
                    # Extract pose
                    pose_keypoints = extract_pose_keypoints(img_path)
                    if pose_keypoints is None:
                        stats['failed_pose'] += 1
                        continue
                    
                    # Detect stick
                    stick_nodes = detect_stick(img_path)
                    if stick_nodes[0, 2] == 0.0:  # No detection
                        stats['failed_stick'] += 1
                    
                    # Build graph
                    graph = build_graph(pose_keypoints, stick_nodes, class_idx, viewpoint)
                    
                    # Save original graph
                    output_path = output_dir / f"{img_path.stem}.pt"
                    torch.save(graph, output_path)
                    stats['successful'] += 1
                    stats['total_graphs'] += 1
                    
                    # Apply augmentation (only for training set)
                    if augment and split == 'train':
                        augmented_graphs = augment_graph_conservative(graph)
                        
                        # Save augmented graphs (skip first one, it's the original)
                        for i, aug_graph in enumerate(augmented_graphs[1:], start=1):
                            aug_output_path = output_dir / f"{img_path.stem}_aug{i}.pt"
                            torch.save(aug_graph, aug_output_path)
                            stats['total_graphs'] += 1
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create graph dataset from images")
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='dataset_split',
        help='Path to dataset_split directory'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='dataset_graphs',
        help='Path to output directory for graphs'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Apply augmentation to training set (4x multiplier)'
    )
    parser.add_argument(
        '--skip_stick',
        action='store_true',
        help='Skip stick detection (use dummy nodes)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Graph Dataset Creation")
    print("=" * 60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Output root: {args.output_root}")
    print(f"Augmentation: {'Enabled (4x)' if args.augment else 'Disabled'}")
    print(f"Stick detection: {'Skipped' if args.skip_stick else 'Enabled'}")
    print("=" * 60)
    
    # Process dataset
    stats = process_dataset(args.dataset_root, args.output_root, args.augment, args.skip_stick)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Processing Complete")
    print("=" * 60)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed (pose detection): {stats['failed_pose']}")
    print(f"Failed (stick detection): {stats['failed_stick']}")
    print(f"Total graphs created: {stats['total_graphs']}")
    print(f"Stick detection rate: {(stats['successful'] - stats['failed_stick']) / stats['successful'] * 100:.1f}%")
    print("=" * 60)
