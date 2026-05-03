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
DATASET_ROOT = Path("dataset_split")
OUTPUT_DIR = Path("hybrid_classifier/hybrid_features_v5")
STICK_MODEL = "runs/pose/stick_detector_20260425_212025/weights/best.pt"

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct',
    'neutral'
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
    """Calculate 3D angle at p2 formed by p1-p2-p3. Matches TuroArnis app."""
    # Handle 2D input for backward compatibility
    if len(p1) == 2:
        p1 = [p1[0], p1[1], 0.0]
    if len(p2) == 2:
        p2 = [p2[0], p2[1], 0.0]
    if len(p3) == 2:
        p3 = [p3[0], p3[1], 0.0]
    
    # 3D vectors
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1], p3[2]-p2[2]])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)


def calculate_distance(p1, p2):
    """Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def apply_stick_method4_correction(raw_grip_px, raw_tip_px, kpts, img_width, img_height, world_landmarks, viewpoint=None):
    """
    Apply Stick Detection Method 4 (Updated): Adaptive Stick Correction

    Changes from original:
    - Length: shin-based (knee→ankle 3D ratio) for ALL viewpoints
    - Front anchor: MediaPipe RIGHT pinky (index 18) instead of raw YOLO grip
    - Side anchor: raw YOLO grip (unchanged)
    - Grip/tip swap: Disabled for all viewpoints (trusts YOLO)

    Args:
        raw_grip_px: (x, y) in pixels - raw YOLO grip point
        raw_tip_px:  (x, y) in pixels - raw YOLO tip point
        kpts:        [33, 4] array - MediaPipe landmarks (x, y, z, visibility) normalized
        img_width:   image width in pixels
        img_height:  image height in pixels
        world_landmarks: MediaPipe world landmarks (3D in meters)
        viewpoint:   'front' | 'left' | 'right' | None (auto-detect if None)

    Returns:
        corrected_grip_px, corrected_tip_px: (x, y) tuples in pixels
    """
    STICK_LENGTH_M = 0.71   # Standard Arnis stick length in meters
    FRONT_VIEW_THRESHOLD = 0.45

    def to_pixels(lm):
        return np.array([lm[0] * img_width, lm[1] * img_height])

    # Body landmarks in pixels
    left_shoulder  = to_pixels(kpts[11])
    right_shoulder = to_pixels(kpts[12])
    left_hip       = to_pixels(kpts[23])
    right_hip      = to_pixels(kpts[24])
    left_wrist     = to_pixels(kpts[15])
    right_wrist    = to_pixels(kpts[16])
    
    # Calculate average torso pixel length for sanity check clamping
    avg_torso_px = (np.linalg.norm(left_shoulder - left_hip) +
                    np.linalg.norm(right_shoulder - right_hip)) / 2.0

    # --- UNIFIED LOGIC: Hand Proximity & Pinky Snap ---
    
    # 0. Foreshortening Check (New from App)
    # If raw YOLO stick is very short (pointing at camera), skip correction to avoid wild snaps
    grip_px = np.array(raw_grip_px, dtype=float)
    tip_px  = np.array(raw_tip_px,  dtype=float)
    
    raw_length = np.linalg.norm(tip_px - grip_px)
    FORESHORTEN_THRESHOLD_PX = 40
    
    if raw_length < FORESHORTEN_THRESHOLD_PX:
        # Stick is foreshortened (end-on view) -> Trust raw YOLO, skip correction
        # This prevents the stick from "snapping" to a full length when it should look short
        return tuple(grip_px), tuple(tip_px)

    # 1. Identify Hand: Compare YOLO grip distance to Left vs Right Wrist
    dist_r = np.linalg.norm(grip_px - right_wrist)
    dist_l = np.linalg.norm(grip_px - left_wrist)
    
    hand_label = "RIGHT" if dist_r < dist_l else "LEFT"
    
    # 2. Snap Anchor: Use MediaPipe Pinky of the identified hand
    # This is anatomically stable and robust against occlusion
    if hand_label == "RIGHT":
        pinky_idx = 18 # RIGHT_PINKY
    else:
        pinky_idx = 17 # LEFT_PINKY
        
    anchor_px = to_pixels(kpts[pinky_idx])
    grip_px = anchor_px # Update grip to snapped anchor
    
    # --- STEP 4: Shin-based stick length (unified for all views) ---
    # Shin (knee→ankle) is more stable than torso or forearm
    lk = to_pixels(kpts[25]);  la = to_pixels(kpts[27])   # left knee, left ankle
    rk = to_pixels(kpts[26]);  ra = to_pixels(kpts[28])   # right knee, right ankle

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
    # Direction is always derived from raw YOLO grip->tip vector
    direction = tip_px - grip_px
    direction_len = np.linalg.norm(direction) + 1e-6
    direction_unit = direction / direction_len

    corrected_tip_px = grip_px + direction_unit * stick_px

    return tuple(grip_px), tuple(corrected_tip_px)


def extract_raw_features(image_path, stick_detector, viewpoint=None, class_idx=None):
    """
    Extract raw features from a single image.
    
    For front view classes 0-3, implements fallback stick estimation using finger 
    landmarks when YOLO stick detection fails (stick appears as dot in front view).
    
    Args:
        image_path: Path to image file
        stick_detector: YOLO model for stick detection
        viewpoint: 'front', 'left', 'right', or None
        class_idx: Class index (0-12), used for front 0-3 fallback logic
    
    Returns:
        - pose_keypoints: [33, 3] array (x, y, visibility)
        - stick_keypoints: [2, 3] array (grip and tip)
        - global_geometric_features: dict of computed features
        - None if pose detection fails or stick+fallback both fail
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
    
    # Extract stick with Method 4 correction
    stick_results = stick_detector(str(image_path), verbose=False)[0]
    if stick_results.keypoints is not None and len(stick_results.keypoints.data) > 0:
        stick_kpts = stick_results.keypoints.data[0].cpu().numpy()
        
        # Get raw YOLO endpoints in pixels
        raw_grip_px = (stick_kpts[0, 0], stick_kpts[0, 1])
        raw_tip_px = (stick_kpts[1, 0], stick_kpts[1, 1])
        
        # Apply Method 4 correction using body proportions
        try:
            corrected_grip_px, corrected_tip_px = apply_stick_method4_correction(
                raw_grip_px, raw_tip_px, kpts, w, h,
                results.pose_world_landmarks.landmark,
                viewpoint=viewpoint
            )
            
            # Normalize corrected endpoints
            stick_grip = [corrected_grip_px[0] / w, corrected_grip_px[1] / h, 0.0, stick_kpts[0, 2]]
            stick_tip = [corrected_tip_px[0] / w, corrected_tip_px[1] / h, 0.0, stick_kpts[1, 2]]
            
            # Determine which hand holds the stick (for dynamic graph edges)
            grip_px = np.array(corrected_grip_px, dtype=float)
            dist_to_r = np.linalg.norm(grip_px - np.array([kpts[16, 0] * w, kpts[16, 1] * h]))
            dist_to_l = np.linalg.norm(grip_px - np.array([kpts[15, 0] * w, kpts[15, 1] * h]))
            stick_right_hand = bool(dist_to_r < dist_to_l)
        except Exception as e:
            # Fallback to raw YOLO if correction fails
            print(f"Method 4 correction failed for {image_path}: {e}, using raw YOLO")
            stick_grip = [stick_kpts[0, 0] / w, stick_kpts[0, 1] / h, 0.0, stick_kpts[0, 2]]
            stick_tip = [stick_kpts[1, 0] / w, stick_kpts[1, 1] / h, 0.0, stick_kpts[1, 2]]
            
            # Determine which hand holds the stick (for dynamic graph edges)
            grip_px = np.array(raw_grip_px, dtype=float)
            dist_to_r = np.linalg.norm(grip_px - np.array([kpts[16, 0] * w, kpts[16, 1] * h]))
            dist_to_l = np.linalg.norm(grip_px - np.array([kpts[15, 0] * w, kpts[15, 1] * h]))
            stick_right_hand = bool(dist_to_r < dist_to_l)
    else:
        # No stick detected by YOLO
        # ALL classes now use ZERO stick coordinates instead of rejecting the image
        # This keeps all 35 nodes but sets stick to (0,0,0,0) so model can ignore them
        stick_grip = [0.0, 0.0, 0.0, 0.0]  # x, y, z, confidence = 0
        stick_tip = [0.0, 0.0, 0.0, 0.0]
        stick_right_hand = True  # Default: stick is held in right hand
        
        # Track zero-stick count
        if class_idx is not None and class_idx in _fallback_counts:
            _fallback_counts[class_idx] += 1
        
        # Log periodically (every 10 samples)
        if class_idx is not None and _fallback_counts[class_idx] % 10 == 1:
            print(f"[ZERO_STICK_FALLBACK] {image_path.name}: class={CLASS_NAMES[class_idx]}, count={_fallback_counts[class_idx]}")
    
    stick_keypoints = np.array([stick_grip, stick_tip])
    
    # Compute global geometric features
    features = {}
    
    # Issue #4: Use 3D world landmarks for angle calculation
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
    
    # Engineered right-handedness features
    # Stick is always held in right hand — these give the model a strong structural signal
    features['stick_grip_to_r_wrist'] = calculate_distance(stick_grip, kpts[16])
    features['stick_right_of_center'] = stick_tip[0] - root_x  # positive = right side
    features['r_wrist_vs_l_wrist_x'] = kpts[16][0] - kpts[15][0]  # positive = right wrist is to the right
    
    # === SIGNED DIRECTION FEATURES (Path A) ===
    # These preserve left/right sign for the hybrid feature vector.
    # Person-normalized by shoulder width for invariance.
    shoulder_width = np.linalg.norm(kpts[11, :2] - kpts[12, :2]) + 1e-8
    hip_center_x = (kpts[23][0] + kpts[24][0]) / 2
    
    # 1. Stick tip horizontal offset from body center (negative = left target, positive = right target)
    features['stick_tip_signed_x'] = (stick_tip[0] - hip_center_x) / shoulder_width
    
    # 2. Stick grip horizontal offset (which side of body the hand is on)
    features['grip_signed_x'] = (stick_grip[0] - hip_center_x) / shoulder_width
    
    # 3. Wrist spread (how far apart are wrists horizontally)
    # For left-target techniques, left wrist is often extended further left
    features['wrist_spread'] = (kpts[15][0] - kpts[16][0]) / shoulder_width
    
    # 4. Stick horizontal reach (absolute distance from grip to tip)
    # Thrusts have longer horizontal reach than blocks
    features['stick_reach'] = abs(stick_tip[0] - stick_grip[0]) / shoulder_width
    
    # 5. Tip height relative to grip (negative = tip below grip, positive = tip above)
    # Crown/eye thrusts: positive. Chest/solar plexus: near 0. Knee blocks: negative.
    features['tip_height_vs_grip'] = (stick_tip[1] - stick_grip[1]) / shoulder_width
    
    # 6. Stick alignment with forearm (dot product: 0 = perpendicular/block, 1 = aligned/thrust)
    grip_px = np.array([stick_grip[0], stick_grip[1]])
    dist_to_rwrist = np.linalg.norm(grip_px - kpts[16, :2])
    dist_to_lwrist = np.linalg.norm(grip_px - kpts[15, :2])
    if dist_to_rwrist < dist_to_lwrist:
        forearm_vec = kpts[16, :2] - kpts[14, :2]  # right wrist - right elbow
    else:
        forearm_vec = kpts[15, :2] - kpts[13, :2]  # left wrist - left elbow
    
    stick_vec_2d = np.array([stick_tip[0] - stick_grip[0], stick_tip[1] - stick_grip[1]])
    forearm_len = np.linalg.norm(forearm_vec) + 1e-8
    stick_len_2d = np.linalg.norm(stick_vec_2d) + 1e-8
    if forearm_len > 0.01 and stick_len_2d > 0.01:
        dot = np.dot(forearm_vec / forearm_len, stick_vec_2d / stick_len_2d)
        features['stick_forearm_dot'] = dot  # raw cosine, not angle
    else:
        features['stick_forearm_dot'] = 0.5  # default: ambiguous
    
    # === EXPANDED SIGNED DIRECTION FEATURES (Path A v5) ===
    # These explicitly separate chest/eye/crown/elbow-block clusters
    # by exposing tip height vs body landmarks and elbow angles as raw signed values.
    
    # 7. Tip height vs nose (positive = above nose, negative = below)
    # Eye thrusts: +, Chest thrusts: ~0, Crown: ++, Knee blocks: -
    features['tip_vs_nose_signed'] = (stick_tip[1] - nose_y) / shoulder_width
    
    # 8. Tip height vs shoulder (positive = above shoulder, negative = below)
    # Temple blocks: ~0, Crown: ++, Knee blocks: -
    features['tip_vs_shoulder_signed'] = (stick_tip[1] - shoulder_y) / shoulder_width
    
    # 9. Left elbow angle (normalized to [0, 1])
    # Thrusts: ~1.0 (straight), Blocks: ~0.5 (bent), Guard: ~0.3
    features['left_elbow_angle_signed'] = features['left_elbow_angle'] / 180.0
    
    # 10. Right elbow angle (normalized to [0, 1])
    # Same interpretation — useful when stick is in right hand
    features['right_elbow_angle_signed'] = features['right_elbow_angle'] / 180.0
    
    # 11. Stick angle (normalized to [-1, 1])
    # Overhead crown: ~-0.5 (upward), Forward thrust: ~0, Downward: ~+0.5
    features['stick_angle_signed'] = features['stick_angle'] / 180.0
    
    # 12. Right wrist height — person-normalized, preserves absolute height ordering
    # Separates crown (>0.5) vs eye (~0.4) vs chest (~0.3) vs elbow block (~0.2) vs neutral (~0.0)
    features['right_wrist_height_signed'] = features['right_wrist_height'] / shoulder_width
    
    # 13. Left wrist height — person-normalized, captures active hand for left-target techniques
    # Crown (>0.5) > Eye (~0.4) > Chest (~0.3) > Elbow block (~0.2) > Neutral (~0.0)
    features['left_wrist_height_signed'] = features['left_wrist_height'] / shoulder_width
    
    return {
        'pose_keypoints': kpts,
        'stick_keypoints': stick_keypoints,
        'global_features': features,
        'stick_right_hand': stick_right_hand
    }


def gaussian_similarity(value, mean, std):
    """Compute similarity score using Gaussian distribution"""
    if std < 1e-6:
        std = 1e-6
    return np.exp(-0.5 * ((value - mean) / std) ** 2)


# === SIGNED DIRECTION FEATURES (Path A v5) ===
# These preserve left/right sign — passed through as raw normalized values,
# NOT converted to Gaussian similarity scores.
DIRECTION_FEATURES = {
    'stick_tip_signed_x': 0.5,       # typical range [-0.5, 0.5]
    'grip_signed_x': 0.5,
    'wrist_spread': 0.5,
    'stick_reach': 0.5,
    'tip_height_vs_grip': 0.5,
    'stick_forearm_dot': 1.0,        # already in [0, 1] range
    'tip_vs_nose_signed': 0.5,       # eye thrust (+) vs chest (~0) vs crown (-)
    'tip_vs_shoulder_signed': 0.5,   # temple (~0) vs knee (-)
    'left_elbow_angle_signed': 1.0,  # thrust (~1.0) vs block (~0.5)
    'right_elbow_angle_signed': 1.0, # same for right arm
    'stick_angle_signed': 1.0,       # overhead (-0.5) vs forward (0) vs down (0.5)
    'right_wrist_height_signed': 0.5,  # crown (>0.5) vs eye (~0.4) vs chest (~0.3)
    'left_wrist_height_signed': 0.5,   # active hand for left-target techniques
}


def compute_hybrid_features(raw_features, templates, viewpoint, class_name):
    """Convert raw geometric features to similarity scores.
    
    For standard features: use Gaussian similarity against template statistics.
    For DIRECTION_FEATURES: pass through raw signed value (normalized to ~[-1, 1]).
    """
    key = f"{viewpoint}_{class_name}"
    
    if key not in templates:
        return np.zeros(len(raw_features))
    
    template = templates[key]
    hybrid_features = []
    
    for feat_name, feat_value in raw_features.items():
        if feat_name in DIRECTION_FEATURES:
            # Pass through raw signed value, normalized by typical range
            normalized = feat_value / DIRECTION_FEATURES[feat_name]
            # Clip to reasonable range [-3, 3] to avoid extreme outliers
            normalized = np.clip(normalized, -3.0, 3.0)
            hybrid_features.append(normalized)
        elif feat_name in template:
            mean = template[feat_name]['mean']
            std = template[feat_name]['std']
            similarity = gaussian_similarity(feat_value, mean, std)
            hybrid_features.append(similarity)
        else:
            # Feature not in template and not a direction feature — pad with 0.0
            # so every sample has the same-length hybrid vector
            hybrid_features.append(0.0)
    
    return np.array(hybrid_features, dtype=np.float32)


def extract_node_features(pose_keypoints, stick_keypoints, include_stick=True):
    """
    Extract per-node features with 3D coordinates.
    Each node gets: [x, y, z, visibility, distance_to_hip_3d, angle_from_hip]
    
    Args:
        pose_keypoints: [33, 4] array of pose landmarks
        stick_keypoints: [2, 4] array of stick grip and tip (or None)
        include_stick: Whether to include stick nodes (35 total) or just pose (33 total)
    
    Returns:
        node_features: [N, 6] array where N is 33 or 35 depending on include_stick
    """
    # Compute hip center for reference (3D)
    hip_center = (pose_keypoints[23, :3] + pose_keypoints[24, :3]) / 2  # [x, y, z]
    
    if include_stick and stick_keypoints is not None:
        # Combine all nodes (33 pose + 2 stick)
        all_keypoints = np.vstack([pose_keypoints, stick_keypoints])
    else:
        # Pose only (33 nodes)
        all_keypoints = pose_keypoints
    
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


def extract_node_features_normalized(pose_keypoints, stick_keypoints, include_stick=True):
    """
    Extract person-invariant per-node features (Plan A).
    Each node gets: [x, y, z, vis, rel_x_to_hip, rel_y_to_hip, rel_z_to_hip, shoulder_width]
    
    Coordinates are normalized by shoulder width so the model sees geometry ratios
    rather than absolute person scale / camera distance.
    
    Args:
        pose_keypoints: [33, 4] array of pose landmarks (x, y, z, visibility)
        stick_keypoints: [2, 4] array of stick grip and tip (or None)
        include_stick: Whether to include stick nodes (35 total) or just pose (33)
    
    Returns:
        node_features: [N, 8] array
    """
    # Torso scale proxy: shoulder width in 3D
    shoulder_width = np.linalg.norm(pose_keypoints[11, :3] - pose_keypoints[12, :3])
    torso_scale = shoulder_width + 1e-8
    
    # Hip center for relative coordinates
    hip_center = (pose_keypoints[23, :3] + pose_keypoints[24, :3]) / 2
    
    if include_stick and stick_keypoints is not None:
        all_keypoints = np.vstack([pose_keypoints, stick_keypoints])
    else:
        all_keypoints = pose_keypoints
    
    node_features = []
    for kpt in all_keypoints:
        x, y, z, vis = kpt
        # Person-invariant relative position w.r.t hip center, scaled by shoulder width
        rel = (np.array([x, y, z]) - hip_center) / torso_scale
        node_features.append([x, y, z, vis, rel[0], rel[1], rel[2], shoulder_width])
    
    return np.array(node_features, dtype=np.float32)


# Track zero-stick counts for front view classes 0-3 and 12 (neutral)
_fallback_counts = {
    0: 0,   # crown_thrust_correct
    1: 0,   # left_chest_thrust_correct
    2: 0,   # left_elbow_block_correct
    3: 0,   # left_eye_thrust_correct
    12: 0,  # neutral
}


def estimate_stick_from_pose(kpts, img_width, img_height, class_idx=None):
    """
    Estimate stick grip and tip from body pose when YOLO stick detection fails.
    POSE-DRIVEN estimation using arm angles and body dynamics.
    
    Key insight: For front view techniques, we can infer stick direction from:
    - Which hand is dominant (forward/extended)
    - Arm angle (elbow position relative to shoulder/wrist)
    - Body orientation
    - Class-specific technique patterns
    
    Args:
        kpts: MediaPipe pose landmarks [33, 4] array (x, y, z, visibility)
        img_width, img_height: Image dimensions
        class_idx: Class index (0-3 for front view fallback classes)
    
    Returns:
        (grip_x, grip_y), (tip_x, tip_y) in normalized coordinates, or (None, None) if can't estimate
    """
    STICK_LENGTH_M = 0.71  # Standard Arnis stick length in meters
    
    def to_pixels(idx):
        return np.array([kpts[idx][0] * img_width, kpts[idx][1] * img_height])
    
    # Get body landmarks
    left_shoulder = to_pixels(11)
    right_shoulder = to_pixels(12)
    left_elbow = to_pixels(13)
    right_elbow = to_pixels(14)
    left_wrist = to_pixels(15)
    right_wrist = to_pixels(16)
    left_hip = to_pixels(23)
    right_hip = to_pixels(24)
    nose = to_pixels(0)
    
    # Calculate body center and orientation
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    body_dir = shoulder_center - hip_center  # Body up direction
    body_dir_norm = np.linalg.norm(body_dir) + 1e-6
    body_unit = body_dir / body_dir_norm
    
    # Calculate torso for stick length reference
    avg_torso_px = body_dir_norm
    stick_px = min(avg_torso_px * 1.8, img_width * 0.4)  # Stick ~1.8x torso, max 40% image width
    
    # Determine which arm is extended (holding the stick)
    # For classes 0-3, typically left arm is doing the technique
    # But we check visibility and extension to be sure
    
    left_arm_visible = kpts[13][3] > 0.5 and kpts[15][3] > 0.5  # elbow and wrist
    right_arm_visible = kpts[14][3] > 0.5 and kpts[16][3] > 0.5
    
    # Calculate arm extension (distance from shoulder to wrist)
    left_arm_len = np.linalg.norm(left_wrist - left_shoulder)
    right_arm_len = np.linalg.norm(right_wrist - right_shoulder)
    
    # Default to left arm for classes 0-3, but check if right is more extended
    use_left_arm = True
    if left_arm_visible and right_arm_visible:
        # Both visible - use the more extended arm (holding stick is typically extended)
        if right_arm_len > left_arm_len * 1.2:  # Right significantly more extended
            use_left_arm = False
    elif not left_arm_visible and right_arm_visible:
        use_left_arm = False
    elif not left_arm_visible and not right_arm_visible:
        return None, None  # Can't estimate without arm visibility
    
    # Select arm landmarks
    if use_left_arm:
        shoulder, elbow, wrist = left_shoulder, left_elbow, left_wrist
        elbow_idx, wrist_idx = 13, 15
    else:
        shoulder, elbow, wrist = right_shoulder, right_elbow, right_wrist
        elbow_idx, wrist_idx = 14, 16
    
    # Use wrist as grip position (stick is held in hand)
    grip_px = wrist
    
    # Calculate arm direction (shoulder -> elbow -> wrist chain)
    upper_arm = elbow - shoulder
    forearm = wrist - elbow
    arm_direction = wrist - shoulder  # Overall arm direction
    arm_dir_norm = np.linalg.norm(arm_direction) + 1e-6
    arm_unit = arm_direction / arm_dir_norm
    
    # Calculate forearm direction (more precise for stick pointing)
    forearm_norm = np.linalg.norm(forearm) + 1e-6
    forearm_unit = forearm / forearm_norm
    
    # Calculate elbow angle to determine arm extension
    # For thrusts: elbow is extended (almost straight)
    # For blocks: elbow is bent (90+ degrees)
    upper_arm_norm = np.linalg.norm(upper_arm) + 1e-6
    upper_unit = upper_arm / upper_arm_norm
    
    # Cosine of angle between upper arm and forearm
    cos_elbow_angle = np.dot(upper_unit, forearm_unit)
    elbow_angle_deg = np.degrees(np.arccos(np.clip(cos_elbow_angle, -1, 1)))
    
    # EXTENDED vs BENT arm classification
    is_extended = elbow_angle_deg > 150  # Almost straight arm (thrust)
    is_bent = elbow_angle_deg < 120  # Clearly bent (block/guard)
    
    # POSE-DRIVEN DIRECTION ESTIMATION
    # Base direction is forearm extension (where hand is pointing)
    direction = forearm_unit.copy()
    
    # Apply class-specific adjustments based on arm pose
    if class_idx == 0:  # crown_thrust_correct - OVERHEAD STRIKE
        # Crown thrust: Stick comes from above, striking down
        # Arm should be extended upward
        # Direction: Continue forearm upward, slightly toward head center
        
        # Weight toward nose from grip
        head_vector = nose - grip_px
        head_dist = np.linalg.norm(head_vector) + 1e-6
        head_unit = head_vector / head_dist
        
        # Blend: 70% forearm direction + 30% toward head
        # But ensure strong upward component
        direction = 0.6 * forearm_unit + 0.4 * head_unit
        direction[1] = min(direction[1], -0.3)  # Ensure upward (negative Y)
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        
    elif class_idx == 1:  # left_chest_thrust_correct - FORWARD THRUST
        # Chest thrust: Direct forward thrust to center
        # Arm extended forward toward opponent's chest
        
        # Direction is primarily forearm extension
        # For front view chest thrust, stick points toward center of image (opponent)
        center_target = np.array([img_width * 0.5, img_height * 0.45])
        center_vector = center_target - grip_px
        center_dist = np.linalg.norm(center_vector) + 1e-6
        center_unit = center_vector / center_dist
        
        # If arm is extended, trust forearm direction more
        # If bent, blend toward center more
        if is_extended:
            direction = 0.7 * forearm_unit + 0.3 * center_unit
        else:
            direction = 0.5 * forearm_unit + 0.5 * center_unit
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        
    elif class_idx == 2:  # left_elbow_block_correct - HORIZONTAL BLOCK
        # Elbow block: Stick held across body, blocking
        # Arm is typically bent at elbow, stick perpendicular to arm
        
        # For blocks, stick is perpendicular to forearm (across the body)
        # Rotate forearm 90 degrees
        perp_direction = np.array([-forearm_unit[1], forearm_unit[0]])  # 90 degree rotation
        
        # Determine direction: for left arm block, stick typically extends to right
        # Check which side of body the hand is on
        hand_x_ratio = grip_px[0] / img_width
        if hand_x_ratio < 0.5:  # Left side - block extends right
            # Keep as is (perp_direction)
            pass
        else:  # Right side - block extends left
            perp_direction = -perp_direction
        
        # Blend with slight forward component
        forward_unit = np.array([0, -1])  # Up in image (toward opponent)
        direction = 0.8 * perp_direction + 0.2 * forward_unit
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        
    elif class_idx == 3:  # left_eye_thrust_correct - UPWARD THRUST
        # Eye thrust: Stick thrust upward toward face
        # Similar to crown but less overhead, more forward-up
        
        # Weight toward nose/eye level
        eye_target = nose + np.array([0, img_height * 0.05])  # Slightly below nose (eye level)
        eye_vector = eye_target - grip_px
        eye_dist = np.linalg.norm(eye_vector) + 1e-6
        eye_unit = eye_vector / eye_dist
        
        # Blend forearm with eye direction
        direction = 0.5 * forearm_unit + 0.5 * eye_unit
        # Ensure upward component
        direction[1] = min(direction[1], -0.1)
        direction = direction / (np.linalg.norm(direction) + 1e-6)
    
    else:
        # Unknown class - use simple forearm extension
        direction = forearm_unit
    
    # Estimate tip position
    tip_px = grip_px + direction * stick_px
    
    # Ensure tip is within image bounds (with margin)
    margin = 10  # pixels
    tip_px[0] = np.clip(tip_px[0], margin, img_width - margin)
    tip_px[1] = np.clip(tip_px[1], margin, img_height - margin)
    
    # Normalize back to [0,1]
    grip_norm = [grip_px[0] / img_width, grip_px[1] / img_height]
    tip_norm = [tip_px[0] / img_width, tip_px[1] / img_height]
    
    return grip_norm, tip_norm


def process_single_image(args):
    """Process a single image (for multiprocessing)"""
    img_path, class_idx, viewpoint, templates, stick_detector, pure_gcn = args
    
    try:
        # Extract raw features (pass class_idx for front 0-3 improved stick estimation)
        # All classes now include stick nodes with improved pose-based estimation for front 0-3
        raw_data = extract_raw_features(img_path, stick_detector, viewpoint=viewpoint, class_idx=class_idx)
        if raw_data is None:
            return None
        
        # Extract node-specific features
        if pure_gcn:
            # Plan A: Person-normalized 8-dim features for PureGCN
            node_features = extract_node_features_normalized(
                raw_data['pose_keypoints'],
                raw_data['stick_keypoints'],
                include_stick=True
            )
        else:
            # Standard 6-dim format for HybridGCN/4e
            node_features = extract_node_features(
                raw_data['pose_keypoints'],
                raw_data['stick_keypoints'],
                include_stick=True
            )
        
        # Compute global hybrid features (kept for backward compatibility)
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
            'viewpoint': viewpoint,
            'has_stick_nodes': True,  # All samples have stick nodes
            'stick_right_hand': raw_data.get('stick_right_hand', True)
        }
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def process_dataset(viewpoint_filter=None, num_workers=None, pure_gcn=False, templates_path=None, dataset_root=None, output_dir=None):
    """Process all images and generate node + hybrid features"""
    # Load templates
    tp = templates_path if templates_path else FEATURE_TEMPLATES
    with open(tp, 'r') as f:
        templates = json.load(f)
    print(f"[INFO] Using templates: {tp}")

    # Dataset root override
    ds_root = Path(dataset_root) if dataset_root else DATASET_ROOT
    print(f"[INFO] Using dataset root: {ds_root}")

    # Output dir override
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    print(f"[INFO] Using output dir: {out_dir}")

    print(f"Loaded {len(templates)} feature templates")
    if pure_gcn:
        print("[MODE] PureGCN: Using person-normalized 8-dim node features")

    # Load stick detector once
    stick_detector = YOLO(STICK_MODEL)

    out_dir.mkdir(parents=True, exist_ok=True)
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"Using {num_workers} workers for multiprocessing")
    
    viewpoints = [viewpoint_filter] if viewpoint_filter else ['front', 'left', 'right']
    
    for split in ['train', 'test']:
        all_tasks = []
        
        split_path = ds_root / split
        
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
                    all_tasks.append((img_path, class_idx, viewpoint, templates, stick_detector, pure_gcn))
        
        print(f"\nProcessing {split} set: {len(all_tasks)} images")
        
        # Process with multiprocessing
        results = []
        with Pool(num_workers) as pool:
            for result in tqdm(pool.imap(process_single_image, all_tasks), total=len(all_tasks)):
                if result is not None:
                    results.append(result)
        
        # Aggregate results
        # Handle variable node counts: some samples have 33 nodes (pose only), others have 35 (pose + stick)
        # Pad 33-node samples to 35 nodes with zeros, track with a mask
        node_features_list = []
        hybrid_features_list = []
        labels_list = []
        viewpoints_list = []
        has_stick_mask = []  # Track which samples have real stick nodes
        stick_right_hand_list = []  # Plan A: dynamic stick edges
        
        # First pass: find max nodes and pad if needed
        max_nodes = 0
        for result in results:
            max_nodes = max(max_nodes, result['node_features'].shape[0])
        
        print(f"  - Max nodes per sample: {max_nodes} (33 = pose only, 35 = with stick)")
        
        for result in results:
            node_feats = result['node_features']
            has_stick = result.get('has_stick_nodes', True)
            
            # Pad if needed (samples with 33 nodes get 2 rows of zeros added)
            if node_feats.shape[0] < max_nodes:
                padding = np.zeros((max_nodes - node_feats.shape[0], node_feats.shape[1]), dtype=np.float32)
                node_feats = np.vstack([node_feats, padding])
            
            node_features_list.append(node_feats)
            hybrid_features_list.append(result['hybrid_features'])
            labels_list.append(result['label'])
            viewpoints_list.append(result['viewpoint'])
            has_stick_mask.append(has_stick)
            stick_right_hand_list.append(result.get('stick_right_hand', True))
        
        # Print stats about node counts
        no_stick_count = sum(1 for h in has_stick_mask if not h)
        print(f"  - Samples without stick nodes: {no_stick_count}/{len(results)} ({100*no_stick_count/len(results):.1f}%)")
        
        # Print dynamic edge stats
        right_hand_count = sum(stick_right_hand_list)
        print(f"  - Stick in right hand: {right_hand_count}/{len(results)} ({100*right_hand_count/len(results):.1f}%)")
        
        # Save as PyTorch tensors
        data = {
            'node_features': torch.tensor(np.array(node_features_list), dtype=torch.float32),
            'hybrid_features': torch.tensor(np.array(hybrid_features_list), dtype=torch.float32),
            'labels': torch.tensor(labels_list, dtype=torch.long),
            'viewpoints': viewpoints_list,
            'has_stick_nodes': torch.tensor(has_stick_mask, dtype=torch.bool),  # Mask for model
            'stick_right_hand': torch.tensor(stick_right_hand_list, dtype=torch.bool),  # Plan A: dynamic edges
            'pure_gcn': pure_gcn  # Feature mode flag
        }
        
        suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
        suffix += "_pure_gcn" if pure_gcn else ""
        output_file = out_dir / f"{split}_features{suffix}.pt"
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
    parser.add_argument('--pure_gcn', action='store_true',
                        help='Use person-normalized node features (8-dim) for PureGCN instead of standard 6-dim')
    parser.add_argument('--templates', type=str, default=None,
                        help='Path to feature templates JSON (default: feature_templates.json)')
    parser.add_argument('--dataset-root', type=str, default=None,
                        help='Override dataset root directory (default: dataset_split)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory (default: hybrid_classifier/hybrid_features_v5)')
    args = parser.parse_args()

    # Set global flag for feature extraction mode
    global _PURE_GCN_MODE
    _PURE_GCN_MODE = args.pure_gcn

    process_dataset(args.viewpoint, args.workers, pure_gcn=args.pure_gcn, templates_path=args.templates, dataset_root=args.dataset_root, output_dir=args.output_dir)
