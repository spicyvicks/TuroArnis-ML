"""
1_extract_reference_features_enhanced.py
============================================
Extracts reference pose templates with ADDITIONAL geometric features:
1. stick_forearm_angle - angle between stick and forearm (thrust vs block)
2. tip_nose_dist_3d - 3D distance from stick tip to nose (target-specific)
3. tip_chest_dist_3d - 3D distance from stick tip to chest center
4. body_rotation - hip-shoulder twist in horizontal plane
5. stance_width - ankle separation (stance characteristic)
6. stick_forearm_ratio - stick length vs forearm (person-invariant stick size)

Output: hybrid_classifier/feature_templates_enhanced.json
(Does NOT overwrite feature_templates.json)
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
STICK_MODEL = "runs/pose/stick_detector_20260425_212025/weights/best.pt"
OUTPUT_FILE = "hybrid_classifier/feature_templates_enhanced.json"

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct',
    'neutral'
]

VIEWPOINTS = ['front', 'left', 'right']

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=2)
stick_detector = YOLO(STICK_MODEL)


def calculate_angle_3d(p1, p2, p3):
    """Calculate 3D angle at p2."""
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1], p3[2]-p2[2]])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def calculate_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)


def extract_geometric_features_enhanced(image_path):
    """Extract all geometric features INCLUDING enhanced ones."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(img_rgb)
    
    if not results.pose_landmarks:
        return None
    
    # Normalized keypoints [x, y, visibility]
    kpts = []
    for lm in results.pose_landmarks.landmark:
        kpts.append([lm.x, lm.y, lm.visibility])
    kpts = np.array(kpts)
    
    # World landmarks (3D in meters)
    world_landmarks = results.pose_world_landmarks.landmark if results.pose_world_landmarks else None
    
    def get_world_point(idx):
        if world_landmarks is None:
            lm = results.pose_landmarks.landmark[idx]
            return [lm.x, lm.y, 0.0]
        lm = world_landmarks[idx]
        return [lm.x, lm.y, lm.z]
    
    # Stick detection
    stick_results = stick_detector(str(image_path), verbose=False)[0]
    if stick_results.keypoints is not None and len(stick_results.keypoints.data) > 0:
        stick_kpts = stick_results.keypoints.data[0].cpu().numpy()
        stick_grip = [stick_kpts[0, 0] / w, stick_kpts[0, 1] / h, 0.0]
        stick_tip = [stick_kpts[1, 0] / w, stick_kpts[1, 1] / h, 0.0]
    else:
        return None  # Skip if no stick (except neutral handled separately)
    
    features = {}
    
    # === ORIGINAL FEATURES (must match 1_extract_reference_features.py) ===
    features['left_elbow_angle'] = calculate_angle_3d(get_world_point(11), get_world_point(13), get_world_point(15))
    features['right_elbow_angle'] = calculate_angle_3d(get_world_point(12), get_world_point(14), get_world_point(16))
    features['left_shoulder_angle'] = calculate_angle_3d(get_world_point(13), get_world_point(11), get_world_point(23))
    features['right_shoulder_angle'] = calculate_angle_3d(get_world_point(14), get_world_point(12), get_world_point(24))
    features['left_knee_angle'] = calculate_angle_3d(get_world_point(23), get_world_point(25), get_world_point(27))
    features['right_knee_angle'] = calculate_angle_3d(get_world_point(24), get_world_point(26), get_world_point(28))
    
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
    
    stick_vector = np.array([stick_tip[0] - stick_grip[0], stick_tip[1] - stick_grip[1]])
    features['stick_angle'] = np.degrees(np.arctan2(stick_vector[1], stick_vector[0]))
    stick_len = np.linalg.norm(stick_vector) + 1e-6
    features['stick_dx'] = stick_vector[0] / stick_len
    features['stick_dy'] = stick_vector[1] / stick_len
    
    root_x = hip_center_x
    root_y = hip_center_y
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
    
    features['hands_distance'] = np.sqrt((kpts[15][0]-kpts[16][0])**2 + (kpts[15][1]-kpts[16][1])**2)
    features['stick_length'] = np.sqrt((stick_tip[0]-stick_grip[0])**2 + (stick_tip[1]-stick_grip[1])**2)
    features['stick_grip_to_r_wrist'] = np.sqrt((stick_grip[0]-kpts[16][0])**2 + (stick_grip[1]-kpts[16][1])**2)
    features['stick_right_of_center'] = stick_tip[0] - root_x
    features['r_wrist_vs_l_wrist_x'] = kpts[16][0] - kpts[15][0]
    
    # === ENHANCED FEATURES (new) ===
    
    # 1. Stick angle vs forearm (tells thrust vs block)
    # Determine which hand holds stick by proximity
    dist_to_rwrist = np.linalg.norm(np.array(stick_grip[:2]) - kpts[16, :2])
    dist_to_lwrist = np.linalg.norm(np.array(stick_grip[:2]) - kpts[15, :2])
    if dist_to_rwrist < dist_to_lwrist:
        # Right hand holds stick
        forearm_vec = np.array(get_world_point(14)) - np.array(get_world_point(16))  # elbow -> wrist
    else:
        forearm_vec = np.array(get_world_point(13)) - np.array(get_world_point(15))  # left elbow -> wrist
    
    stick_vec_3d = np.array(stick_tip) - np.array(stick_grip)
    forearm_vec = forearm_vec / (np.linalg.norm(forearm_vec) + 1e-6)
    stick_vec_3d = stick_vec_3d / (np.linalg.norm(stick_vec_3d) + 1e-6)
    dot = np.dot(forearm_vec, stick_vec_3d)
    features['stick_forearm_angle'] = np.degrees(np.arccos(np.clip(abs(dot), 0, 1)))
    
    # 2. Tip-to-nose 3D distance (target-specific)
    nose_world = np.array(get_world_point(0))
    tip_world = np.array([stick_tip[0], stick_tip[1], 0.0])  # No Z for stick, use 0
    # Approximate: use normalized coords as proxy for 3D
    features['tip_nose_dist_3d'] = np.linalg.norm(tip_world - nose_world)
    
    # 3. Tip-to-chest 3D distance
    chest_center = (np.array(get_world_point(11)) + np.array(get_world_point(12))) / 2
    features['tip_chest_dist_3d'] = np.linalg.norm(tip_world - chest_center)
    
    # 4. Body rotation - hip-shoulder twist in horizontal plane
    # Angle between hip line and shoulder line projected on XY plane
    hip_vec = np.array([kpts[24][0] - kpts[23][0], kpts[24][1] - kpts[23][1]])
    shoulder_vec = np.array([kpts[12][0] - kpts[11][0], kpts[12][1] - kpts[11][1]])
    hip_vec = hip_vec / (np.linalg.norm(hip_vec) + 1e-6)
    shoulder_vec = shoulder_vec / (np.linalg.norm(shoulder_vec) + 1e-6)
    twist_dot = np.dot(hip_vec, shoulder_vec)
    features['body_rotation'] = np.degrees(np.arccos(np.clip(abs(twist_dot), 0, 1)))
    
    # 5. Stance width - ankle separation
    features['stance_width'] = np.sqrt((kpts[27][0]-kpts[28][0])**2 + (kpts[27][1]-kpts[28][1])**2)
    
    # 6. Stick-to-forearm ratio (person-invariant)
    if dist_to_rwrist < dist_to_lwrist:
        forearm_len = np.linalg.norm(np.array(get_world_point(14)) - np.array(get_world_point(16)))
    else:
        forearm_len = np.linalg.norm(np.array(get_world_point(13)) - np.array(get_world_point(15)))
    stick_len_3d = np.linalg.norm(np.array(stick_tip) - np.array(stick_grip))
    features['stick_forearm_ratio'] = stick_len_3d / (forearm_len + 1e-6)
    
    # Metadata for validation
    metadata = {
        'pose_landmarks': results.pose_landmarks,
        'stick_detected': stick_results.keypoints is not None,
        'stick_confidence': float(stick_kpts[0, 2]) if 'stick_kpts' in locals() else 0.0
    }
    
    return features, metadata


def validate_features(features, pose_landmarks, stick_detected, stick_confidence, class_name=None):
    """Validation gate - relaxed for reference poses."""
    wrist_vis_15 = pose_landmarks.landmark[15].visibility
    wrist_vis_16 = pose_landmarks.landmark[16].visibility
    if wrist_vis_15 < 0.1 and wrist_vis_16 < 0.1:
        return False, "Both wrists low visibility"
    
    if class_name != 'neutral' and not stick_detected:
        return False, "Stick not detected"
    
    if class_name != 'neutral' and stick_confidence < 0.15:
        return False, f"Low stick confidence ({stick_confidence:.2f})"
    
    stick_len = features.get('stick_length', 0)
    if class_name != 'neutral' and (stick_len == 0 or np.isnan(stick_len)):
        return False, "Zero stick length"
    
    for angle_name in ['left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 
                       'right_shoulder_angle', 'left_knee_angle', 'right_knee_angle']:
        angle_val = features.get(angle_name, 0)
        if angle_val < 0 or angle_val > 180 or np.isnan(angle_val):
            return False, f"Impossible {angle_name}: {angle_val:.1f}"
    
    return True, "Valid"


def analyze_reference_images(viewpoint_filter=None):
    templates = {}
    viewpoints = [viewpoint_filter] if viewpoint_filter else VIEWPOINTS
    
    for viewpoint in viewpoints:
        for class_name in CLASS_NAMES:
            class_dir = REFERENCE_DIR / viewpoint / class_name
            if not class_dir.exists():
                continue
            
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            if not images:
                continue
            
            print(f"Processing {viewpoint}/{class_name}: {len(images)} images")
            all_features = []
            
            for img_path in tqdm(images, desc=f"{viewpoint}/{class_name}", leave=False):
                result = extract_geometric_features_enhanced(img_path)
                if result is None:
                    continue
                
                features, metadata = result
                is_valid, reason = validate_features(
                    features, metadata['pose_landmarks'],
                    metadata['stick_detected'], metadata['stick_confidence'],
                    class_name=class_name
                )
                
                if not is_valid:
                    continue
                
                all_features.append(features)
            
            if not all_features:
                print(f"  Warning: No valid features")
                continue
            
            # Strip stick features from neutral
            if class_name == 'neutral':
                STICK_FEATURES = [
                    'stick_tip_height', 'stick_grip_height', 'stick_tip_x', 'stick_grip_x',
                    'stick_angle', 'stick_dx', 'stick_dy', 'tip_vs_nose', 'tip_vs_shoulder',
                    'tip_vs_hip', 'tip_side', 'grip_side', 'stick_length', 'stick_grip_to_r_wrist',
                    'stick_right_of_center', 'stick_forearm_angle', 'tip_nose_dist_3d',
                    'tip_chest_dist_3d', 'stick_forearm_ratio'
                ]
                for feat_dict in all_features:
                    for sf in STICK_FEATURES:
                        feat_dict.pop(sf, None)
            
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
            print(f"  Accepted: {len(all_features)} | Features: {len(feature_stats)}")
    
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(templates, f, indent=2)
    
    print(f"\nEnhanced templates saved: {OUTPUT_FILE}")
    print(f"Total templates: {len(templates)}")
    print(f"Features per template: ~{len(next(iter(templates.values()))) if templates else 0}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewpoint', type=str, default=None, choices=['front', 'left', 'right'])
    args = parser.parse_args()
    analyze_reference_images(args.viewpoint)
