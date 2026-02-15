"""
Step 1: Extract Geometric Features from Reference Images
Analyzes reference poses to compute mean/std for geometric features
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from pathlib import Path
import json
from tqdm import tqdm

# Config
REFERENCE_DIR = Path("reference_poses")
STICK_MODEL = "runs/pose/arnis_stick_detector/weights/best.pt"
OUTPUT_FILE = "hybrid_classifier/feature_templates.json"

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]

VIEWPOINTS = ['front', 'left', 'right']

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


def apply_stick_method4_correction(raw_grip_px, raw_tip_px, kpts, img_width, img_height, world_landmarks):
    """
    Apply Stick Detection Method 4: Adaptive Stick Correction
    
    Corrects raw YOLO stick detection using body proportions from MediaPipe.
    YOLO gets the DIRECTION right but LENGTH wrong. This method keeps the direction
    and fixes the length using body landmarks.
    
    Args:
        raw_grip_px: (x, y) in pixels - raw YOLO grip point
        raw_tip_px: (x, y) in pixels - raw YOLO tip point
        kpts: [33, 4] array - MediaPipe landmarks (x, y, z, visibility) in normalized coords
        img_width: image width in pixels
        img_height: image height in pixels
        world_landmarks: MediaPipe world landmarks (3D in meters)
    
    Returns:
        corrected_grip_px, corrected_tip_px: (x, y) tuples in pixels
    """
    STICK_LENGTH_M = 0.71  # Standard Arnis stick length in meters
    FRONT_VIEW_THRESHOLD = 0.45
    TORSO_MULTIPLIER = 1.5
    
    # Convert normalized landmarks to pixel coordinates
    def to_pixels(lm):
        return np.array([lm[0] * img_width, lm[1] * img_height])
    
    left_shoulder = to_pixels(kpts[11])
    right_shoulder = to_pixels(kpts[12])
    left_hip = to_pixels(kpts[23])
    right_hip = to_pixels(kpts[24])
    left_elbow = to_pixels(kpts[13])
    right_elbow = to_pixels(kpts[14])
    left_wrist = to_pixels(kpts[15])
    right_wrist = to_pixels(kpts[16])
    
    # STEP 3: Determine Viewpoint (Front vs Side)
    shoulder_width = calculate_distance(left_shoulder, right_shoulder)
    left_torso_len = calculate_distance(left_shoulder, left_hip)
    right_torso_len = calculate_distance(right_shoulder, right_hip)
    avg_torso_px = (left_torso_len + right_torso_len) / 2
    
    view_ratio = shoulder_width / (avg_torso_px + 1e-6)
    is_front_view = view_ratio > FRONT_VIEW_THRESHOLD
    
    # STEP 4: Calculate Corrected Stick Length
    if is_front_view:
        # Front view: Use torso scaling
        stick_px = avg_torso_px * TORSO_MULTIPLIER
    else:
        # Side view: Use forearm scaling with 3D world landmarks
        # Identify which arm holds the stick (closest wrist to grip)
        dist_left = calculate_distance(left_wrist, raw_grip_px)
        dist_right = calculate_distance(right_wrist, raw_grip_px)
        
        if dist_left < dist_right:
            # Left arm holds stick
            wrist_idx, elbow_idx = 15, 13
        else:
            # Right arm holds stick
            wrist_idx, elbow_idx = 16, 14
        
        # Get 3D forearm length from world landmarks
        wrist_3d = world_landmarks[wrist_idx]
        elbow_3d = world_landmarks[elbow_idx]
        forearm_m = np.sqrt(
            (wrist_3d.x - elbow_3d.x)**2 + 
            (wrist_3d.y - elbow_3d.y)**2 + 
            (wrist_3d.z - elbow_3d.z)**2
        )
        
        # Calculate scaling ratio
        len_ratio = STICK_LENGTH_M / (forearm_m + 1e-6)
        
        # Get 2D forearm length in pixels
        wrist_2d = to_pixels(kpts[wrist_idx])
        elbow_2d = to_pixels(kpts[elbow_idx])
        forearm_px = calculate_distance(wrist_2d, elbow_2d)
        
        # Final stick length in pixels
        stick_px = forearm_px * len_ratio
    
    # STEP 5: Project Corrected Tip
    # Get direction from raw YOLO (reliable)
    direction = np.array([raw_tip_px[0] - raw_grip_px[0], raw_tip_px[1] - raw_grip_px[1]])
    direction_len = np.linalg.norm(direction) + 1e-6
    direction_normalized = direction / direction_len
    
    # Project new tip using corrected length
    corrected_tip_px = (
        raw_grip_px[0] + direction_normalized[0] * stick_px,
        raw_grip_px[1] + direction_normalized[1] * stick_px
    )
    
    # Grip stays at raw YOLO position (anchored to wrist, reliable)
    return raw_grip_px, corrected_tip_px


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
    
    # Extract stick with Method 4 correction
    stick_results = stick_detector(str(image_path), verbose=False)[0]
    if stick_results.keypoints is not None and len(stick_results.keypoints.data) > 0:
        stick_kpts = stick_results.keypoints.data[0].cpu().numpy()
        raw_grip_px = np.array([stick_kpts[0, 0], stick_kpts[0, 1]])
        raw_tip_px = np.array([stick_kpts[1, 0], stick_kpts[1, 1]])
        
        # Apply Method 4 correction
        try:
            corrected_grip_px, corrected_tip_px = apply_stick_method4_correction(
                raw_grip_px, raw_tip_px, kpts, w, h, results.pose_world_landmarks.landmark
            )
            # Normalize to [0, 1]
            stick_grip = [corrected_grip_px[0] / w, corrected_grip_px[1] / h]
            stick_tip = [corrected_tip_px[0] / w, corrected_tip_px[1] / h]
        except Exception:
            # Fallback to raw YOLO if Method 4 fails
            stick_grip = [stick_kpts[0, 0] / w, stick_kpts[0, 1] / h]
            stick_tip = [stick_kpts[1, 0] / w, stick_kpts[1, 1] / h]
    else:
        stick_grip = [0.5, 0.5]
        stick_tip = [0.5, 0.5]
    
    # Compute features
    features = {}
    
    # Joint angles
    features['left_elbow_angle'] = calculate_angle(kpts[11], kpts[13], kpts[15])
    features['right_elbow_angle'] = calculate_angle(kpts[12], kpts[14], kpts[16])
    features['left_shoulder_angle'] = calculate_angle(kpts[13], kpts[11], kpts[23])
    features['right_shoulder_angle'] = calculate_angle(kpts[14], kpts[12], kpts[24])
    features['left_knee_angle'] = calculate_angle(kpts[23], kpts[25], kpts[27])
    features['right_knee_angle'] = calculate_angle(kpts[24], kpts[26], kpts[28])
    
    # Heights (relative to hip center)
    hip_center_y = (kpts[23][1] + kpts[24][1]) / 2
    features['left_wrist_height'] = hip_center_y - kpts[15][1]
    features['right_wrist_height'] = hip_center_y - kpts[16][1]
    features['left_elbow_height'] = hip_center_y - kpts[13][1]
    features['right_elbow_height'] = hip_center_y - kpts[14][1]
    features['stick_tip_height'] = hip_center_y - stick_tip[1]
    features['stick_grip_height'] = hip_center_y - stick_grip[1]
    
    # Horizontal positions (relative to hip center)
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
    
    # --- Expert Features (Based on definitions) ---
    root_x = (kpts[23][0] + kpts[24][0]) / 2  # Hip center X
    root_y = (kpts[23][1] + kpts[24][1]) / 2  # Hip center Y
    shoulder_y = (kpts[11][1] + kpts[12][1]) / 2
    nose_y = kpts[0][1]
    
    # 1. Height Levels (Relative to nose/shoulder/hip)
    # Using raw y-diff (positive = lower than landmark)
    features['tip_vs_nose'] = stick_tip[1] - nose_y       # <0 means above head
    features['tip_vs_shoulder'] = stick_tip[1] - shoulder_y # <0 means above shoulder
    features['tip_vs_hip'] = stick_tip[1] - root_y       # <0 means above hip
    
    # 2. Hand Levels
    features['r_hand_vs_nose'] = kpts[16][1] - nose_y
    features['r_hand_vs_shoulder'] = kpts[16][1] - shoulder_y
    features['r_hand_vs_hip'] = kpts[16][1] - root_y
    
    # 3. Horizontal Directions (Relative to center)
    features['tip_side'] = stick_tip[0] - root_x  # <0 left, >0 right
    features['grip_side'] = stick_grip[0] - root_x
    
    # 4. Foot Stance (Left vs Right forward)
    # In front view, lower Y often means closer to camera
    features['foot_stagger'] = kpts[27][1] - kpts[28][1] # >0 left forward, <0 right forward
    
    # Distances
    features['hands_distance'] = calculate_distance(kpts[15], kpts[16])
    features['stick_length'] = calculate_distance(stick_grip, stick_tip)
    
    return features


def analyze_reference_images(viewpoint_filter=None):
    """Analyze all reference images and compute feature statistics"""
    templates = {}
    
    viewpoints = [viewpoint_filter] if viewpoint_filter else VIEWPOINTS
    
    for viewpoint in viewpoints:
        for class_name in CLASS_NAMES:
            class_dir = REFERENCE_DIR / viewpoint / class_name
            
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue
            
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            if len(images) == 0:
                print(f"Warning: No images in {class_dir}")
                continue
            
            print(f"Processing {viewpoint}/{class_name}: {len(images)} images")
            
            all_features = []
            for img_path in tqdm(images, desc=f"{viewpoint}/{class_name}", leave=False):
                features = extract_geometric_features(img_path)
                if features:
                    all_features.append(features)
            
            if len(all_features) == 0:
                print(f"  Warning: No valid features extracted")
                continue
            
            # Compute mean and std for each feature
            feature_stats = {}
            feature_names = all_features[0].keys()
            
            for feat_name in feature_names:
                values = [f[feat_name] for f in all_features]
                feature_stats[feat_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            
            key = f"{viewpoint}_{class_name}"
            templates[key] = feature_stats
    
    # Save templates
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(templates, f, indent=2)
    
    print(f"\n✓ Feature templates saved to {OUTPUT_FILE}")
    print(f"  Total templates: {len(templates)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewpoint', type=str, default=None,
                        choices=['front', 'left', 'right'],
                        help='Process only specific viewpoint (default: all)')
    args = parser.parse_args()
    
    analyze_reference_images(args.viewpoint)
