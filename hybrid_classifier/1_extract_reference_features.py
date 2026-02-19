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
    'solar_plexus_thrust_correct',
    'neutral'
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


def apply_stick_method4_correction(raw_grip_px, raw_tip_px, kpts, img_width, img_height, world_landmarks, viewpoint=None):
    """
    Apply Stick Detection Method 4 (Updated): Adaptive Stick Correction
    - Length: shin-based (knee→ankle 3D ratio) for ALL viewpoints
    - Anchor: UNIFIED MediaPipe Pinky (Left/Right based on proximity)
    - Safety: Foreshortening check (skips correction if stick < 40px)
    """
    STICK_LENGTH_M = 0.71   # Standard Arnis stick length in meters
    
    def to_pixels(lm):
        return np.array([lm[0] * img_width, lm[1] * img_height])

    # Body landmarks needed for calculation
    left_wrist     = to_pixels(kpts[15])
    right_wrist    = to_pixels(kpts[16])
    
    # Calculate average torso pixel length for sanity check clamping
    left_shoulder  = to_pixels(kpts[11])
    right_shoulder = to_pixels(kpts[12])
    left_hip       = to_pixels(kpts[23])
    right_hip      = to_pixels(kpts[24])
    
    avg_torso_px = (np.linalg.norm(left_shoulder - left_hip) +
                    np.linalg.norm(right_shoulder - right_hip)) / 2.0

    # --- UNIFIED LOGIC: Hand Proximity & Pinky Snap ---
    
    # 0. Foreshortening Check (New from App)
    grip_px = np.array(raw_grip_px, dtype=float)
    tip_px  = np.array(raw_tip_px,  dtype=float)
    
    raw_length = np.linalg.norm(tip_px - grip_px)
    FORESHORTEN_THRESHOLD_PX = 40
    
    if raw_length < FORESHORTEN_THRESHOLD_PX:
        # Stick is foreshortened (end-on view) -> Trust raw YOLO, skip correction
        return tuple(grip_px), tuple(tip_px)

    # 1. Identify Hand: Compare YOLO grip distance to Left vs Right Wrist
    dist_r = np.linalg.norm(grip_px - right_wrist)
    dist_l = np.linalg.norm(grip_px - left_wrist)
    
    hand_label = "RIGHT" if dist_r < dist_l else "LEFT"
    
    # 2. Snap Anchor: Use MediaPipe Pinky of the identified hand
    if hand_label == "RIGHT":
        pinky_idx = 18 # RIGHT_PINKY
    else:
        pinky_idx = 17 # LEFT_PINKY
        
    anchor_px = to_pixels(kpts[pinky_idx])
    grip_px = anchor_px # Update grip to snapped anchor
    
    # --- STEP 4: Shin-based stick length (unified for all views) ---
    lk = to_pixels(kpts[25]);  la = to_pixels(kpts[27])
    rk = to_pixels(kpts[26]);  ra = to_pixels(kpts[28])

    # 3D shin length (meters)
    def world_pt(idx):
        lm = world_landmarks[idx]
        return np.array([lm.x, lm.y, lm.z])

    shin_m = (np.linalg.norm(world_pt(25) - world_pt(27)) +
              np.linalg.norm(world_pt(26) - world_pt(28))) / 2.0

    # 2D shin length (pixels)
    shin_px = (np.linalg.norm(lk - la) + np.linalg.norm(rk - ra)) / 2.0

    # Stick length in pixels via ratio
    stick_px = shin_px * (STICK_LENGTH_M / (shin_m + 1e-6))

    # Clamp to 2.5× torso (sanity check)
    stick_px = min(stick_px, avg_torso_px * 2.5)

    # --- STEP 5: Project corrected tip using pure YOLO direction ---
    direction = tip_px - grip_px
    direction_len = np.linalg.norm(direction) + 1e-6
    direction_unit = direction / direction_len

    corrected_tip_px = grip_px + direction_unit * stick_px

    return tuple(grip_px), tuple(corrected_tip_px)


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
