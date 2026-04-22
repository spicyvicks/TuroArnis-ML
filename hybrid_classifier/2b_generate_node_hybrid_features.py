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
OUTPUT_DIR = Path("hybrid_classifier/hybrid_features_v3")
STICK_MODEL = "runs/pose/arnis_stick_detector/weights/best.pt"

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
        except Exception as e:
            # Fallback to raw YOLO if correction fails
            print(f"Method 4 correction failed for {image_path}: {e}, using raw YOLO")
            stick_grip = [stick_kpts[0, 0] / w, stick_kpts[0, 1] / h, 0.0, stick_kpts[0, 2]]
            stick_tip = [stick_kpts[1, 0] / w, stick_kpts[1, 1] / h, 0.0, stick_kpts[1, 2]]
    else:
        # No stick detected by YOLO
        # For front view classes 0-3, try finger fallback
        FRONT_VIEW_0_3_CLASSES = [0, 1, 2, 3]  # crown, left_chest, left_elbow, left_eye
        
        if (viewpoint == 'front' and 
            class_idx is not None and 
            class_idx in FRONT_VIEW_0_3_CLASSES):
            
            # Try to estimate stick from finger positions
            grip_norm, tip_norm = estimate_stick_from_fingers(
                kpts, 
                results.pose_world_landmarks.landmark if results.pose_world_landmarks else results.pose_landmarks.landmark,
                w, h,
                viewpoint=viewpoint
            )
            
            if grip_norm is not None:
                # Successfully estimated from fingers
                stick_grip = [grip_norm[0], grip_norm[1], 0.0, 0.5]  # z=0, confidence=0.5
                stick_tip = [tip_norm[0], tip_norm[1], 0.0, 0.5]
                
                # Track fallback count
                if class_idx in _fallback_counts:
                    _fallback_counts[class_idx] += 1
                
                # Log periodically (every 10 samples)
                if _fallback_counts[class_idx] % 10 == 1:
                    print(f"[FRONT_0-3_FALLBACK] {image_path.name}: class={CLASS_NAMES[class_idx]}, count={_fallback_counts[class_idx]}")
            else:
                # No fingers visible either, skip this sample
                return None
        else:
            # Not front 0-3 or class_idx not provided, skip as before
            return None
    
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


# Track fallback counts for front view classes 0-3
_fallback_counts = {
    0: 0,  # crown_thrust_correct
    1: 0,  # left_chest_thrust_correct
    2: 0,  # left_elbow_block_correct
    3: 0,  # left_eye_thrust_correct
}


def estimate_stick_from_fingers(kpts, world_landmarks, img_width, img_height, viewpoint=None):
    """
    Estimate stick grip and tip from hand fingers when YOLO stick detection fails.
    Used for front view classes 0-3 where stick appears as dot/end-on.
    
    Fallback chain: pinky -> index -> middle -> ring (same hand)
    
    Args:
        kpts: MediaPipe pose landmarks [33, 4] array (x, y, z, visibility)
        world_landmarks: MediaPipe world landmarks for 3D calculations
        img_width, img_height: Image dimensions
        viewpoint: 'front', 'left', 'right', or None
    
    Returns:
        (grip_x, grip_y), (tip_x, tip_y) in normalized coordinates, or (None, None) if no fingers visible
    """
    STICK_LENGTH_M = 0.71  # Standard Arnis stick length in meters
    
    def to_pixels(idx):
        return np.array([kpts[idx][0] * img_width, kpts[idx][1] * img_height])
    
    def get_world_point(idx):
        lm = world_landmarks[idx]
        return np.array([lm.x, lm.y, lm.z])
    
    # Get wrist positions
    left_wrist_px = to_pixels(15)
    right_wrist_px = to_pixels(16)
    
    # Finger indices per hand (MediaPipe)
    # Left hand: 17=pinky, 19=index, 21=middle, 23=ring
    # Right hand: 18=pinky, 20=index, 22=middle, 24=ring
    LEFT_FINGERS = [17, 19, 21, 23]  # pinky, index, middle, ring
    RIGHT_FINGERS = [18, 20, 22, 24]  # pinky, index, middle, ring
    
    # Calculate torso for length reference
    left_shoulder = to_pixels(11)
    left_hip = to_pixels(23)
    right_shoulder = to_pixels(12)
    right_hip = to_pixels(24)
    avg_torso_px = (np.linalg.norm(left_shoulder - left_hip) + 
                    np.linalg.norm(right_shoulder - right_hip)) / 2.0
    
    # Calculate shin length for stick length reference
    left_knee = to_pixels(25)
    left_ankle = to_pixels(27)
    right_knee = to_pixels(26)
    right_ankle = to_pixels(28)
    
    shin_px = (np.linalg.norm(left_knee - left_ankle) + 
               np.linalg.norm(right_knee - right_ankle)) / 2.0
    
    # 3D shin length (meters)
    shin_m = (np.linalg.norm(get_world_point(25) - get_world_point(27)) +
              np.linalg.norm(get_world_point(26) - get_world_point(28))) / 2.0
    
    # Calculate stick length in pixels
    stick_px = shin_px * (STICK_LENGTH_M / (shin_m + 1e-6))
    stick_px = min(stick_px, avg_torso_px * 2.5)  # Sanity check clamp
    
    # Try to identify which hand is holding the stick
    # In classes 0-3, typically left hand for crown/left_*, right hand might be involved
    # Try left hand first (classes 0-3 are left-side techniques)
    selected_hand = 'LEFT'
    selected_grip_px = None
    selected_finger_name = None
    
    # Try fingers in priority order: pinky -> index -> middle -> ring
    FINGER_NAMES = ['pinky', 'index', 'middle', 'ring']
    
    for hand, fingers in [('LEFT', LEFT_FINGERS), ('RIGHT', RIGHT_FINGERS)]:
        for finger_idx, finger_name in zip(fingers, FINGER_NAMES):
            visibility = kpts[finger_idx][3]
            if visibility > 0.5:  # Finger is visible
                selected_grip_px = to_pixels(finger_idx)
                selected_hand = hand
                selected_finger_name = finger_name
                break
        if selected_grip_px is not None:
            break
    
    if selected_grip_px is None:
        # No fingers visible, can't estimate
        return None, None
    
    # Estimate stick direction
    # For front view end-on stick: direction is roughly perpendicular to camera
    # Use body orientation to estimate
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    
    # Body orientation vector (pointing "forward" roughly)
    body_dir = shoulder_center - hip_center
    body_dir_len = np.linalg.norm(body_dir) + 1e-6
    
    # For front view, stick often points somewhat toward camera or at an angle
    # Use a combination of body orientation and a slight random/estimate
    # In practice, for front view thrusts, stick is roughly perpendicular to body plane
    
    # Perpendicular to body in image plane (left/right direction)
    perp_dir = np.array([-body_dir[1], body_dir[0]])  # Rotate 90 degrees
    perp_dir = perp_dir / (np.linalg.norm(perp_dir) + 1e-6)
    
    # Determine direction based on which hand
    if selected_hand == 'LEFT':
        # Left hand typically holds stick pointing leftward (from body perspective)
        direction = -perp_dir
    else:
        direction = perp_dir
    
    # Estimate tip position
    tip_px = selected_grip_px + direction * stick_px
    
    # Normalize back to [0,1]
    grip_norm = [selected_grip_px[0] / img_width, selected_grip_px[1] / img_height]
    tip_norm = [tip_px[0] / img_width, tip_px[1] / img_height]
    
    return grip_norm, tip_norm


def process_single_image(args):
    """Process a single image (for multiprocessing)"""
    img_path, class_idx, viewpoint, templates, stick_detector = args
    
    try:
        # Extract raw features (pass class_idx for front 0-3 fallback)
        raw_data = extract_raw_features(img_path, stick_detector, viewpoint=viewpoint, class_idx=class_idx)
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
