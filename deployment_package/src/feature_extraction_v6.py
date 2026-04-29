"""
Feature Extraction for HybridGCN V6 Deployment
- 6-dim node features with TRUE zero fallback for invisible/missing nodes
- Zero-stick fallback: [0,0,0,0] for grip and tip when YOLO fails
- node_mask generation for masked pooling
- 49 hybrid features (33 base Gaussian + 15 signed + 1 has_stick)
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO


# Signed direction features + has_stick (v6)
DIRECTION_FEATURES = {
    'stick_tip_signed_x': 0.5,
    'grip_signed_x': 0.5,
    'wrist_spread': 0.5,
    'stick_reach': 0.5,
    'tip_height_vs_grip': 0.5,
    'stick_forearm_dot': 1.0,
    'tip_vs_nose_signed': 0.5,
    'tip_vs_shoulder_signed': 0.5,
    'left_elbow_angle_signed': 1.0,
    'right_elbow_angle_signed': 1.0,
    'stick_angle_signed': 1.0,
    'right_wrist_height_signed': 0.5,
    'left_wrist_height_signed': 0.5,
    'left_wrist_x_signed': 0.5,
    'right_wrist_x_signed': 0.5,
    'has_stick': 1.0,  # binary: 1.0 = YOLO detected, 0.0 = zero-stick fallback
}


def calculate_angle(p1, p2, p3):
    """Calculate 3D angle at p2 formed by p1-p2-p3."""
    if len(p1) == 2:
        p1 = [p1[0], p1[1], 0.0]
    if len(p2) == 2:
        p2 = [p2[0], p2[1], 0.0]
    if len(p3) == 2:
        p3 = [p3[0], p3[1], 0.0]
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1], p3[2]-p2[2]])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)


def calculate_distance(p1, p2):
    """Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def gaussian_similarity(value, mean, std):
    """Compute similarity score using Gaussian distribution"""
    if std < 1e-6:
        std = 1e-6
    return np.exp(-0.5 * ((value - mean) / std) ** 2)


def extract_raw_features(image, stick_detector):
    """
    Extract raw features from a single image.
    
    Args:
        image: numpy array (BGR format) or path to image
        stick_detector: YOLO model for stick detection
    
    Returns:
        dict with pose_keypoints, stick_keypoints, global_features, has_stick_detected
        or None if pose detection fails
    """
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
    
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # Extract body pose with MediaPipe
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=2)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(img_rgb)
    pose_detector.close()
    
    if not results.pose_landmarks:
        return None
    
    # Get keypoints (normalized 3D)
    kpts = []
    for lm in results.pose_landmarks.landmark:
        kpts.append([lm.x, lm.y, lm.z, lm.visibility])
    kpts = np.array(kpts)
    
    # Extract stick with YOLO
    stick_results = stick_detector(img, verbose=False)[0]
    stick_detected = False
    
    if stick_results.keypoints is not None and len(stick_results.keypoints.data) > 0:
        stick_kpts = stick_results.keypoints.data[0].cpu().numpy()
        stick_grip = [stick_kpts[0, 0] / w, stick_kpts[0, 1] / h, 0.0, stick_kpts[0, 2]]
        stick_tip = [stick_kpts[1, 0] / w, stick_kpts[1, 1] / h, 0.0, stick_kpts[1, 2]]
        stick_detected = True
    else:
        # V6: TRUE zero-stick fallback — all 4 dims = 0
        stick_grip = [0.0, 0.0, 0.0, 0.0]
        stick_tip = [0.0, 0.0, 0.0, 0.0]
    
    stick_keypoints = np.array([stick_grip, stick_tip])
    
    # Compute global geometric features
    features = {}
    
    # 3D angles
    world_landmarks = results.pose_world_landmarks.landmark if results.pose_world_landmarks else None
    
    def get_world_point(idx):
        if world_landmarks is None:
            lm = results.pose_landmarks.landmark[idx]
            return [lm.x, lm.y, 0.0]
        lm = world_landmarks[idx]
        return [lm.x, lm.y, lm.z]
    
    features['left_elbow_angle'] = calculate_angle(
        get_world_point(11), get_world_point(13), get_world_point(15)
    )
    features['right_elbow_angle'] = calculate_angle(
        get_world_point(12), get_world_point(14), get_world_point(16)
    )
    features['left_shoulder_angle'] = calculate_angle(
        get_world_point(13), get_world_point(11), get_world_point(23)
    )
    features['right_shoulder_angle'] = calculate_angle(
        get_world_point(14), get_world_point(12), get_world_point(24)
    )
    features['left_knee_angle'] = calculate_angle(
        get_world_point(23), get_world_point(25), get_world_point(27)
    )
    features['right_knee_angle'] = calculate_angle(
        get_world_point(24), get_world_point(26), get_world_point(28)
    )
    
    # Heights relative to hip center
    hip_center_y = (kpts[23][1] + kpts[24][1]) / 2
    features['left_wrist_height'] = hip_center_y - kpts[15][1]
    features['right_wrist_height'] = hip_center_y - kpts[16][1]
    features['left_elbow_height'] = hip_center_y - kpts[13][1]
    features['right_elbow_height'] = hip_center_y - kpts[14][1]
    features['stick_tip_height'] = hip_center_y - stick_tip[1]
    features['stick_grip_height'] = hip_center_y - stick_grip[1]
    
    # Horizontal positions relative to hip center
    hip_center_x = (kpts[23][0] + kpts[24][0]) / 2
    features['left_wrist_x'] = kpts[15][0] - hip_center_x
    features['right_wrist_x'] = kpts[16][0] - hip_center_x
    features['stick_tip_x'] = stick_tip[0] - hip_center_x
    features['stick_grip_x'] = stick_grip[0] - hip_center_x
    
    # Stick orientation
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
    
    features['stick_grip_to_r_wrist'] = calculate_distance(stick_grip, kpts[16])
    features['stick_right_of_center'] = stick_tip[0] - root_x
    features['r_wrist_vs_l_wrist_x'] = kpts[16][0] - kpts[15][0]
    
    # === SIGNED DIRECTION FEATURES (V6) ===
    shoulder_width = np.linalg.norm(kpts[11, :2] - kpts[12, :2]) + 1e-8
    
    features['stick_tip_signed_x'] = (stick_tip[0] - hip_center_x) / shoulder_width
    features['grip_signed_x'] = (stick_grip[0] - hip_center_x) / shoulder_width
    features['wrist_spread'] = (kpts[15][0] - kpts[16][0]) / shoulder_width
    features['stick_reach'] = abs(stick_tip[0] - stick_grip[0]) / shoulder_width
    features['tip_height_vs_grip'] = (stick_tip[1] - stick_grip[1]) / shoulder_width
    
    grip_px = np.array([stick_grip[0], stick_grip[1]])
    dist_to_rwrist = np.linalg.norm(grip_px - kpts[16, :2])
    dist_to_lwrist = np.linalg.norm(grip_px - kpts[15, :2])
    if dist_to_rwrist < dist_to_lwrist:
        forearm_vec = kpts[16, :2] - kpts[14, :2]
    else:
        forearm_vec = kpts[15, :2] - kpts[13, :2]
    stick_vec_2d = np.array([stick_tip[0] - stick_grip[0], stick_tip[1] - stick_grip[1]])
    forearm_len = np.linalg.norm(forearm_vec) + 1e-8
    stick_len_2d = np.linalg.norm(stick_vec_2d) + 1e-8
    if forearm_len > 0.01 and stick_len_2d > 0.01:
        dot = np.dot(forearm_vec / forearm_len, stick_vec_2d / stick_len_2d)
        features['stick_forearm_dot'] = dot
    else:
        features['stick_forearm_dot'] = 0.5
    
    features['tip_vs_nose_signed'] = (stick_tip[1] - nose_y) / shoulder_width
    features['tip_vs_shoulder_signed'] = (stick_tip[1] - shoulder_y) / shoulder_width
    features['left_elbow_angle_signed'] = features['left_elbow_angle'] / 180.0
    features['right_elbow_angle_signed'] = features['right_elbow_angle'] / 180.0
    features['stick_angle_signed'] = features['stick_angle'] / 180.0
    features['right_wrist_height_signed'] = features['right_wrist_height'] / shoulder_width
    features['left_wrist_height_signed'] = features['left_wrist_height'] / shoulder_width
    features['left_wrist_x_signed'] = features['left_wrist_x'] / shoulder_width
    features['right_wrist_x_signed'] = features['right_wrist_x'] / shoulder_width
    
    # V6: has_stick binary hybrid feature
    features['has_stick'] = 1.0 if stick_detected else 0.0
    
    return {
        'pose_keypoints': kpts,
        'stick_keypoints': stick_keypoints,
        'global_features': features,
        'has_stick_detected': stick_detected
    }


def compute_hybrid_features(raw_features, templates, viewpoint, class_name):
    """
    Convert raw geometric features to hybrid similarity scores.
    
    For standard features: Gaussian similarity against template statistics.
    For DIRECTION_FEATURES: pass through raw signed value (normalized to ~[-1, 1]).
    """
    key = f"{viewpoint}_{class_name}"
    
    if key not in templates:
        return np.zeros(len(raw_features), dtype=np.float32)
    
    template = templates[key]
    hybrid_features = []
    
    for feat_name, feat_value in raw_features.items():
        if feat_name in DIRECTION_FEATURES:
            normalized = feat_value / DIRECTION_FEATURES[feat_name]
            normalized = np.clip(normalized, -3.0, 3.0)
            hybrid_features.append(normalized)
        elif feat_name in template:
            mean = template[feat_name]['mean']
            std = template[feat_name]['std']
            similarity = gaussian_similarity(feat_value, mean, std)
            hybrid_features.append(similarity)
        else:
            hybrid_features.append(0.0)
    
    return np.array(hybrid_features, dtype=np.float32)


def extract_node_features(pose_keypoints, stick_keypoints, has_stick_detected=True):
    """
    Extract per-node features with 3D coordinates.
    Each node gets: [x, y, z, visibility, distance_to_hip_3d, angle_from_hip, has_stick]
    
    V6: TRUE zero for invisible/missing nodes — all 7 dims = 0.0 when vis < 1e-6
    7th dimension (has_stick): 1.0 for body nodes and detected stick nodes, 0.0 for zero-stick fallback
    This prevents global_mean_pool from being poisoned by origin-based offsets.
    
    Args:
        pose_keypoints: [33, 4] array from MediaPipe
        stick_keypoints: [2, 4] array (true zeros when not detected)
        has_stick_detected: bool, whether YOLO actually detected the stick
    
    Returns:
        [35, 7] array of node features
    """
    all_keypoints = np.vstack([pose_keypoints, stick_keypoints])
    
    hip_center = (pose_keypoints[23, :3] + pose_keypoints[24, :3]) / 2
    
    node_features = []
    for i, kpt in enumerate(all_keypoints):
        x, y, z, vis = kpt
        is_stick_node = i >= 33
        
        if vis < 1e-6:
            # TRUE zero for invisible/missing nodes
            node_features.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            dist_to_hip = np.sqrt((x - hip_center[0])**2 + (y - hip_center[1])**2 + (z - hip_center[2])**2)
            angle_from_hip = np.degrees(np.arctan2(y - hip_center[1], x - hip_center[0]))
            
            if is_stick_node and not has_stick_detected:
                has_stick = 0.0  # zero-stick fallback nodes
            else:
                has_stick = 1.0  # body nodes and YOLO-detected stick nodes
            
            node_features.append([x, y, z, vis, dist_to_hip, angle_from_hip, has_stick])
    
    return np.array(node_features, dtype=np.float32)


def create_node_mask(has_stick_detected):
    """
    Create node-level mask for v6 masked pooling.
    
    Args:
        has_stick_detected: bool, whether YOLO detected the stick
    
    Returns:
        [35] array: 1.0 for body nodes (0-32), 1.0 for stick nodes (33-34) if detected, else 0.0
    """
    mask = np.ones(35, dtype=np.float32)
    if not has_stick_detected:
        mask[33:] = 0.0  # zero out stick nodes 33 and 34
    return mask
