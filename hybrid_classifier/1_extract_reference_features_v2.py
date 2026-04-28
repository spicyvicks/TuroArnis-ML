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
STICK_MODEL = "runs/pose/stick_detector_20260425_212025/weights/best.pt"
OUTPUT_FILE = "hybrid_classifier/feature_templates.json"
OUTPUT_FILE_MIRRORED = "hybrid_classifier/feature_templates_mirrored.json"

# Features whose sign flips under a horizontal (left-right) mirror
HORIZONTAL_FEATURES = {
    'left_wrist_x', 'right_wrist_x',
    'stick_tip_x', 'stick_grip_x',
    'tip_side', 'grip_side',
    'stick_dx', 'stick_angle',
    'foot_stagger',
    'stick_right_of_center', 'r_wrist_vs_l_wrist_x',
}

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct',
    'neutral'
]

VIEWPOINTS = ['front', 'left', 'right']

# Classes where front-view stick detection often fails (end-on stick).
# For these classes, when YOLO stick is missing we use zero-stick fallback
# so the image is still accepted for template statistics.
FRONT_ZERO_STICK_CLASSES = {
    'crown_thrust_correct',
    'left_chest_thrust_correct',
    'left_elbow_block_correct',
    'left_eye_thrust_correct',
    'neutral'
}

# Initialize detectors
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=2)
stick_detector = YOLO(STICK_MODEL)


def calculate_angle(p1, p2, p3):
    """
    Calculate 3D angle at p2 formed by p1-p2-p3.
    Uses 3D coordinates to distinguish thrusts (forward in Z) from blocks (sideways).
    Matches TuroArnis app implementation.
    """
    # Handle 2D input for backward compatibility
    if len(p1) == 2:
        p1 = [p1[0], p1[1], 0.0]
    if len(p2) == 2:
        p2 = [p2[0], p2[1], 0.0]
    if len(p3) == 2:
        p3 = [p3[0], p3[1], 0.0]
    
    # 3D vector construction
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)


def calculate_distance(p1, p2):
    """Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def mirror_features(features):
    """Negate horizontal features to simulate a horizontally-flipped image.
    Use this to align a mirrored pose against non-mirrored reference templates."""
    mirrored = dict(features)
    for feat in HORIZONTAL_FEATURES:
        if feat in mirrored:
            mirrored[feat] = -mirrored[feat]
    return mirrored


def validate_features(features, pose_landmarks, stick_detected, stick_confidence=1.0, viewpoint=None, class_name=None, is_zero_stick_fallback=False):
    """
    Validate extracted features before including in template statistics.
    Returns (is_valid: bool, reason: str)
    
    Issue #2: Quality validation gates to prevent corrupted templates.
    Viewpoint-aware: Adjusts critical joints based on expected occlusion patterns.
    
    v2 changes:
    - Allows zero-stick fallback for front classes where stick is end-on (crown,
      left_chest, left_elbow, left_eye, neutral).
    - Returns detailed rejection reason for _rejection_reasons tracking.
    """
    # Rule 1: MediaPipe critical joint visibility check - ULTRA RELAXED
    # For martial arts poses, only wrists [15, 16] are critical
    # NOTE: Lowered to 0.1 to handle poor lighting/occlusion in reference images
    wrist_vis_15 = pose_landmarks.landmark[15].visibility
    wrist_vis_16 = pose_landmarks.landmark[16].visibility
    MIN_VISIBILITY = 0.1
    if wrist_vis_15 < MIN_VISIBILITY and wrist_vis_16 < MIN_VISIBILITY:
        return False, f"both_wrists_low_vis(15:{wrist_vis_15:.2f},16:{wrist_vis_16:.2f})"
    
    # Rule 2: YOLO stick detection check
    # v2: Allow zero-stick fallback for designated front classes
    if not stick_detected and not is_zero_stick_fallback:
        return False, "stick_not_detected"
    
    # Rule 3: YOLO confidence check - ULTRA RELAXED (skip for zero-stick fallback)
    MIN_STICK_CONFIDENCE = 0.15
    if not is_zero_stick_fallback and stick_confidence < MIN_STICK_CONFIDENCE:
        return False, f"low_stick_conf({stick_confidence:.2f}<{MIN_STICK_CONFIDENCE})"
    
    # Rule 4: Physical plausibility - reject zero or impossible stick length
    # v2: Skip for zero-stick fallback — length is intentionally zero
    stick_len = features.get('stick_length', 0)
    if not is_zero_stick_fallback and (stick_len == 0 or np.isnan(stick_len)):
        return False, f"zero_or_nan_stick_len({stick_len})"
    
    # Rule 5: Sanity check on angles (reject impossible values)
    ANGLE_FEATURES = [
        'left_elbow_angle', 'right_elbow_angle',
        'left_shoulder_angle', 'right_shoulder_angle',
        'left_knee_angle', 'right_knee_angle'
    ]
    
    for angle_name in ANGLE_FEATURES:
        angle_val = features.get(angle_name, 0)
        if angle_val < 0 or angle_val > 180 or np.isnan(angle_val):
            return False, f"impossible_{angle_name}({angle_val:.1f}deg)"
    
    return True, "valid"


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


def extract_geometric_features(image_path, apply_mirror=False, class_name=None, viewpoint=None):
    """Extract all geometric features from a single image.
    If apply_mirror=True, negate horizontal features (simulates a flipped image).
    
    v2 changes:
    - Accepts class_name and viewpoint for zero-stick fallback logic.
    - For designated front classes (crown, left_chest, left_elbow, left_eye, neutral)
      where YOLO fails, uses zero-stick coordinates instead of NaN so the image
      is still accepted for template statistics.
    """
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
    
    # Determine if this class qualifies for zero-stick fallback
    is_zero_stick_fallback = (
        viewpoint == 'front' and
        class_name in FRONT_ZERO_STICK_CLASSES
    )
    
    # Extract stick with Method 4 correction
    stick_results = stick_detector(str(image_path), verbose=False)[0]
    stick_detected = False
    stick_confidence = 0.0
    
    if stick_results.keypoints is not None and len(stick_results.keypoints.data) > 0:
        stick_kpts = stick_results.keypoints.data[0].cpu().numpy()
        raw_grip_px = np.array([stick_kpts[0, 0], stick_kpts[0, 1]])
        raw_tip_px = np.array([stick_kpts[1, 0], stick_kpts[1, 1]])
        stick_detected = True
        stick_confidence = float(stick_kpts[0, 2])  # grip confidence
        
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
        # v2: Zero-stick fallback for designated front classes
        if is_zero_stick_fallback:
            stick_grip = [0.0, 0.0]
            stick_tip = [0.0, 0.0]
            stick_detected = True  # Mark as detected so validation accepts it
            stick_confidence = 0.0
        else:
            # Issue #5: Use NaN sentinel for non-fallback classes
            stick_grip = [float('nan'), float('nan')]
            stick_tip = [float('nan'), float('nan')]
            stick_confidence = 0.0
    
    # Compute features
    features = {}
    
    # Issue #4: Use 3D world landmarks for angle calculation
    # This distinguishes thrusts (forward in Z) from blocks (sideways in XY)
    world_landmarks = results.pose_world_landmarks.landmark if results.pose_world_landmarks else None
    
    def get_world_point(idx):
        """Get 3D world coordinate for landmark (meters)."""
        if world_landmarks is None:
            # Fallback to 2D with z=0
            lm = results.pose_landmarks.landmark[idx]
            return [lm.x, lm.y, 0.0]
        lm = world_landmarks[idx]
        return [lm.x, lm.y, lm.z]
    
    # Joint angles - using 3D world coordinates for physical accuracy
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
    
    # 2. Stick grip horizontal offset
    features['grip_signed_x'] = (stick_grip[0] - hip_center_x) / shoulder_width
    
    # 3. Wrist spread (how far apart are wrists horizontally)
    features['wrist_spread'] = (kpts[15][0] - kpts[16][0]) / shoulder_width
    
    # 4. Stick horizontal reach
    features['stick_reach'] = abs(stick_tip[0] - stick_grip[0]) / shoulder_width
    
    # 5. Tip height relative to grip
    features['tip_height_vs_grip'] = (stick_tip[1] - stick_grip[1]) / shoulder_width
    
    # 6. Stick alignment with forearm (dot product: 0 = perpendicular/block, 1 = aligned/thrust)
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
        features['stick_forearm_dot'] = np.dot(forearm_vec / forearm_len, stick_vec_2d / stick_len_2d)
    else:
        features['stick_forearm_dot'] = 0.5
    
    # === EXPANDED SIGNED DIRECTION FEATURES (Path A v5) ===
    # These explicitly separate chest/eye/crown/elbow-block clusters.
    
    # 7. Tip height vs nose (positive = above nose, negative = below)
    features['tip_vs_nose_signed'] = (stick_tip[1] - nose_y) / shoulder_width
    
    # 8. Tip height vs shoulder (positive = above shoulder, negative = below)
    features['tip_vs_shoulder_signed'] = (stick_tip[1] - shoulder_y) / shoulder_width
    
    # 9. Left elbow angle (normalized to [0, 1])
    features['left_elbow_angle_signed'] = features['left_elbow_angle'] / 180.0
    
    # 10. Right elbow angle (normalized to [0, 1])
    features['right_elbow_angle_signed'] = features['right_elbow_angle'] / 180.0
    
    # 11. Stick angle (normalized to [-1, 1])
    features['stick_angle_signed'] = features['stick_angle'] / 180.0
    
    # 12. Right wrist height — person-normalized, preserves absolute height ordering
    # Crown (>0.5) > Eye (~0.4) > Chest (~0.3) > Elbow block (~0.2) > Neutral (~0.0)
    features['right_wrist_height_signed'] = features['right_wrist_height'] / shoulder_width

    # Apply horizontal mirror correction if requested
    if apply_mirror:
        features = mirror_features(features)

    # Return features with metadata for validation gate
    # v2: Use the already-computed stick_detected (includes zero-stick fallback)
    metadata = {
        'pose_landmarks': results.pose_landmarks,
        'stick_detected': stick_detected,
        'stick_confidence': stick_confidence,
        'is_zero_stick_fallback': is_zero_stick_fallback
    }
    
    return features, metadata


def analyze_reference_images(viewpoint_filter=None, apply_mirror=False):
    """Analyze all reference images and compute feature statistics.
    If apply_mirror=True, negate horizontal features on all images (simulates
    a horizontally-flipped camera) and saves to feature_templates_mirrored.json.
    
    v2 changes:
    - Passes class_name and viewpoint to extract_geometric_features for zero-stick fallback.
    - Tracks rejection reasons per class.
    - Stores _count, _acceptance_rate, and _rejection_reasons in each template.
    """
    templates = {}
    from collections import Counter

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
            rejected_count = 0
            total_count = 0
            rejection_reasons = Counter()
            
            for img_path in tqdm(images, desc=f"{viewpoint}/{class_name}", leave=False):
                total_count += 1
                result = extract_geometric_features(
                    img_path,
                    apply_mirror=apply_mirror,
                    class_name=class_name,
                    viewpoint=viewpoint
                )
                
                if result is None:
                    rejected_count += 1
                    rejection_reasons['pose_detection_failed'] += 1
                    continue
                
                features, metadata = result
                
                # v2: Pass is_zero_stick_fallback flag from metadata
                is_zero_stick_fallback = metadata.get('is_zero_stick_fallback', False)
                
                is_valid, reason = validate_features(
                    features,
                    metadata['pose_landmarks'],
                    metadata['stick_detected'],
                    metadata['stick_confidence'],
                    viewpoint=viewpoint,
                    class_name=class_name,
                    is_zero_stick_fallback=is_zero_stick_fallback
                )
                
                if not is_valid:
                    rejected_count += 1
                    rejection_reasons[reason] += 1
                    print(f"  [REJECTED] {img_path.name}: {reason}")
                    continue
                
                all_features.append(features)
            
            acceptance_rate = len(all_features) / total_count if total_count > 0 else 0.0
            print(f"  Accepted: {len(all_features)}/{total_count} ({100*acceptance_rate:.1f}%)")
            print(f"  Rejected: {rejected_count}/{total_count}")
            if rejection_reasons:
                print(f"  Rejection breakdown: {dict(rejection_reasons)}")
            
            # v2: Warn if acceptance rate is too low (dropped to 30% for front zero-stick classes)
            min_acceptance = 0.30 if (viewpoint == 'front' and class_name in FRONT_ZERO_STICK_CLASSES) else 0.50
            if acceptance_rate < min_acceptance:
                print(f"  WARNING: Low acceptance rate! Expected >{min_acceptance*100:.0f}%, "
                      f"got {100*acceptance_rate:.1f}%")
            
            if len(all_features) == 0:
                print(f"  CRITICAL: No valid features extracted — template will be MISSING")
                # v2: Still write an empty marker so downstream can detect it
                key = f"{viewpoint}_{class_name}"
                templates[key] = {
                    "_count": 0,
                    "_acceptance_rate": 0.0,
                    "_rejection_reasons": dict(rejection_reasons),
                    "_empty": True
                }
                continue
            
            # Option C: For neutral class, strip stick-dependent features from templates
            if class_name == 'neutral':
                STICK_FEATURES = [
                    'stick_tip_height', 'stick_grip_height',
                    'stick_tip_x', 'stick_grip_x',
                    'stick_angle', 'stick_dx', 'stick_dy',
                    'tip_vs_nose', 'tip_vs_shoulder', 'tip_vs_hip',
                    'tip_side', 'grip_side',
                    'stick_length', 'stick_grip_to_r_wrist',
                    'stick_right_of_center'
                ]
                for feat_dict in all_features:
                    for sf in STICK_FEATURES:
                        feat_dict.pop(sf, None)
                print(f"  [INFO] Stripped {len(STICK_FEATURES)} stick features from neutral template")
            
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
            
            # v2: Inject metadata so downstream can detect weak templates
            feature_stats['_count'] = len(all_features)
            feature_stats['_acceptance_rate'] = float(acceptance_rate)
            feature_stats['_rejection_reasons'] = dict(rejection_reasons)
            
            key = f"{viewpoint}_{class_name}"
            templates[key] = feature_stats
    
    # Save templates
    output_path = OUTPUT_FILE_MIRRORED if apply_mirror else OUTPUT_FILE
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(templates, f, indent=2)

    label = "[MIRRORED] " if apply_mirror else ""
    print(f"\n✓ {label}Feature templates saved to {output_path}")
    print(f"  Total templates: {len(templates)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewpoint', type=str, default=None,
                        choices=['front', 'left', 'right'],
                        help='Process only specific viewpoint (default: all)')
    parser.add_argument('--mirrored', action='store_true',
                        help='Negate horizontal features (simulate a mirrored camera). '
                             'Saves to feature_templates_mirrored.json instead.')
    args = parser.parse_args()

    analyze_reference_images(args.viewpoint, apply_mirror=args.mirrored)
