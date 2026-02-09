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
from torch_geometric.data import Data
from ultralytics import YOLO
from graph_augmentation import augment_graph_conservative


# MediaPipe setup
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,  # Full BlazePose
    min_detection_confidence=0.5
)

# Stick detector setup (YOLOv8-Pose model with 2 keypoints: grip and tip)
STICK_DETECTOR_PATH = "runs/pose/arnis_stick_detector/weights/best.pt"
stick_detector = YOLO(STICK_DETECTOR_PATH)

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
    
    return np.array(keypoints, dtype=np.float32)


def detect_stick(image_path):
    """
    Detect stick using YOLOv8-Pose and extract 2 keypoints (grip and tip).
    
    Returns:
        np.array of shape [2, 3] (grip and tip with x, y, confidence)
    """
    try:
        results = stick_detector.predict(str(image_path), verbose=False)
        
        # Check if stick detected (keypoints available)
        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            # Get first detection (highest confidence)
            # keypoints.data shape: [num_detections, num_keypoints, 3]
            # We expect [1, 2, 3] for one stick with 2 keypoints (grip, tip)
            kpts = results[0].keypoints.data[0].cpu().numpy()  # Shape: [2, 3]
            
            # kpts contains [grip, tip] each with [x, y, confidence]
            # Normalize coordinates (YOLO returns pixel coordinates)
            image = cv2.imread(str(image_path))
            h, w = image.shape[:2]
            
            # Normalize x, y to [0, 1]
            stick_grip = [kpts[0, 0] / w, kpts[0, 1] / h, float(kpts[0, 2])]
            stick_tip = [kpts[1, 0] / w, kpts[1, 1] / h, float(kpts[1, 2])]
            
            return np.array([stick_grip, stick_tip], dtype=np.float32)
        else:
            # No stick detected → dummy nodes at center with 0 confidence
            return np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], dtype=np.float32)
            
    except Exception as e:
        print(f"Warning: Stick detection failed for {image_path}: {e}")
        # Return dummy nodes on error
        return np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], dtype=np.float32)


# def detect_stick_heuristic(image_path, pose_keypoints):
#     """
#     FALLBACK: Heuristic-based stick detection (commented for now).
#     Uses wrist positions to estimate stick location.
#     """
#     # Get wrist keypoints (15=left, 16=right)
#     left_wrist = pose_keypoints[15]
#     right_wrist = pose_keypoints[16]
#     
#     # Use wrist with higher visibility
#     if left_wrist[2] > right_wrist[2]:
#         wrist = left_wrist
#     else:
#         wrist = right_wrist
#     
#     # Estimate stick endpoints (simple heuristic)
#     stick_length = 0.3  # Normalized length
#     stick_top = [wrist[0], wrist[1] - stick_length, wrist[2]]
#     stick_bottom = [wrist[0], wrist[1] + stick_length, wrist[2]]
#     
#     return np.array([stick_top, stick_bottom], dtype=np.float32)


def build_graph(pose_keypoints, stick_nodes, label, viewpoint):
    """
    Build PyG graph from pose keypoints and stick nodes.
    
    Args:
        pose_keypoints: [33, 3] array
        stick_nodes: [2, 3] array
        label: int (class index)
        viewpoint: str ('front', 'left', 'right')
    
    Returns:
        PyG Data object
    """
    # Combine nodes (33 pose + 2 stick = 35 total)
    node_features = np.vstack([pose_keypoints, stick_nodes])
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Edge index
    edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t().contiguous()
    
    # Label
    y = torch.tensor([label], dtype=torch.long)
    
    # Create graph
    graph = Data(x=x, edge_index=edge_index, y=y)
    graph.viewpoint = viewpoint
    
    return graph


def process_dataset(dataset_root, output_root, augment=False):
    """
    Process entire dataset_split/ directory.
    
    Args:
        dataset_root: Path to dataset_split/
        output_root: Path to dataset_graphs/
        augment: Whether to apply augmentation (only for train set)
    """
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
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Graph Dataset Creation")
    print("=" * 60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Output root: {args.output_root}")
    print(f"Augmentation: {'Enabled (4x)' if args.augment else 'Disabled'}")
    print(f"Stick detector: {STICK_DETECTOR_PATH}")
    print("=" * 60)
    
    # Process dataset
    stats = process_dataset(args.dataset_root, args.output_root, args.augment)
    
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
