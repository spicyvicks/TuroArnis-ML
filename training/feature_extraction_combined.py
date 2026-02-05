"""
Combined feature extraction: MediaPipe body landmarks + YOLO stick detector
Extracts 68 body features + 24 stick features = 92 features total (+ class label)

Body features include:
- Joint angles (15), cross-body angles (4)
- Relative positions (22), distances (8)
- Symmetry (5), proportions (4)
- Laterality (10) - explicit left/right indicators

Stick features include:
- Basic: detected, length, angle, positions
- Strike targets: distance to eyes, crown, chest, solar plexus, knees
- Direction: normalized stick direction vector
"""
import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from ultralytics import YOLO
import warnings

# confidence thresholds
# NOTE: Lower threshold (0.5) increases detection rate but may include false positives
# Higher threshold (0.75) is more reliable but misses many valid sticks
STICK_CONFIDENCE_THRESHOLD = 0.5  # Lowered from 0.75 to improve detection rate (was 27%)
KEYPOINT_CONFIDENCE_THRESHOLD = 0.3  # Lowered from 0.5

# paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
STICK_MODEL_PATH = os.path.join(project_root, 'runs', 'pose', 'arnis_stick_detector', 'weights', 'best.pt')

# global instances for workers
worker_pose_instance = None
worker_stick_model = None

def init_worker():
    """initialize mediapipe pose and yolo stick detector for worker process"""
    global worker_pose_instance, worker_stick_model
    # Import MediaPipe inside worker to avoid multiprocessing issues on Windows
    import mediapipe as mp
    worker_pose_instance = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    )
    if os.path.exists(STICK_MODEL_PATH):
        worker_stick_model = YOLO(STICK_MODEL_PATH)

def calculate_angle_3d(a, b, c):
    """calculate angle at point b given 3 points"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    dot_product = np.dot(ba, bc)
    magnitude = np.linalg.norm(ba) * np.linalg.norm(bc)
    if magnitude == 0:
        return 0
    return np.degrees(np.arccos(np.clip(dot_product / magnitude, -1.0, 1.0)))

def extract_body_features(lm):
    """extract 58 body features from mediapipe landmarks (world coords)"""
    # ===== JOINT ANGLES (15 angles) =====
    angles = [
        # elbows
        calculate_angle_3d([lm[11].x, lm[11].y, lm[11].z], [lm[13].x, lm[13].y, lm[13].z], [lm[15].x, lm[15].y, lm[15].z]),
        calculate_angle_3d([lm[12].x, lm[12].y, lm[12].z], [lm[14].x, lm[14].y, lm[14].z], [lm[16].x, lm[16].y, lm[16].z]),
        # shoulders
        calculate_angle_3d([lm[23].x, lm[23].y, lm[23].z], [lm[11].x, lm[11].y, lm[11].z], [lm[13].x, lm[13].y, lm[13].z]),
        calculate_angle_3d([lm[24].x, lm[24].y, lm[24].z], [lm[12].x, lm[12].y, lm[12].z], [lm[14].x, lm[14].y, lm[14].z]),
        # wrists
        calculate_angle_3d([lm[13].x, lm[13].y, lm[13].z], [lm[15].x, lm[15].y, lm[15].z], [lm[19].x, lm[19].y, lm[19].z]),
        calculate_angle_3d([lm[14].x, lm[14].y, lm[14].z], [lm[16].x, lm[16].y, lm[16].z], [lm[20].x, lm[20].y, lm[20].z]),
        # hips
        calculate_angle_3d([lm[11].x, lm[11].y, lm[11].z], [lm[23].x, lm[23].y, lm[23].z], [lm[25].x, lm[25].y, lm[25].z]),
        calculate_angle_3d([lm[12].x, lm[12].y, lm[12].z], [lm[24].x, lm[24].y, lm[24].z], [lm[26].x, lm[26].y, lm[26].z]),
        # knees
        calculate_angle_3d([lm[23].x, lm[23].y, lm[23].z], [lm[25].x, lm[25].y, lm[25].z], [lm[27].x, lm[27].y, lm[27].z]),
        calculate_angle_3d([lm[24].x, lm[24].y, lm[24].z], [lm[26].x, lm[26].y, lm[26].z], [lm[28].x, lm[28].y, lm[28].z]),
        # ankles
        calculate_angle_3d([lm[25].x, lm[25].y, lm[25].z], [lm[27].x, lm[27].y, lm[27].z], [lm[31].x, lm[31].y, lm[31].z]),
        calculate_angle_3d([lm[26].x, lm[26].y, lm[26].z], [lm[28].x, lm[28].y, lm[28].z], [lm[32].x, lm[32].y, lm[32].z]),
        # arm-to-torso angles
        calculate_angle_3d([lm[12].x, lm[12].y, lm[12].z], [lm[11].x, lm[11].y, lm[11].z], [lm[13].x, lm[13].y, lm[13].z]),
        calculate_angle_3d([lm[11].x, lm[11].y, lm[11].z], [lm[12].x, lm[12].y, lm[12].z], [lm[14].x, lm[14].y, lm[14].z]),
        # torso angle
        calculate_angle_3d([(lm[11].x+lm[12].x)/2, (lm[11].y+lm[12].y)/2, (lm[11].z+lm[12].z)/2], 
                          [(lm[23].x+lm[24].x)/2, (lm[23].y+lm[24].y)/2, (lm[23].z+lm[24].z)/2],
                          [(lm[23].x+lm[24].x)/2, (lm[23].y+lm[24].y)/2 + 0.1, (lm[23].z+lm[24].z)/2]),
    ]
    
    # ===== CROSS-BODY ANGLES (4 new) =====
    cross_body_angles = [
        calculate_angle_3d([lm[12].x, lm[12].y, lm[12].z], [lm[11].x, lm[11].y, lm[11].z], [lm[15].x, lm[15].y, lm[15].z]),
        calculate_angle_3d([lm[11].x, lm[11].y, lm[11].z], [lm[12].x, lm[12].y, lm[12].z], [lm[16].x, lm[16].y, lm[16].z]),
        calculate_angle_3d([lm[11].x, lm[11].y, lm[11].z], [lm[24].x, lm[24].y, lm[24].z], [lm[26].x, lm[26].y, lm[26].z]),
        calculate_angle_3d([lm[12].x, lm[12].y, lm[12].z], [lm[23].x, lm[23].y, lm[23].z], [lm[25].x, lm[25].y, lm[25].z]),
    ]
    
    # ===== RELATIVE POSITIONS (22 features) =====
    left_wrist_rel = [lm[15].x - lm[11].x, lm[15].y - lm[11].y, lm[15].z - lm[11].z]
    right_wrist_rel = [lm[16].x - lm[12].x, lm[16].y - lm[12].y, lm[16].z - lm[12].z]
    
    hip_center = [(lm[23].x + lm[24].x)/2, (lm[23].y + lm[24].y)/2, (lm[23].z + lm[24].z)/2]
    left_hand_rel = [lm[19].x - hip_center[0], lm[19].y - hip_center[1], lm[19].z - hip_center[2]]
    right_hand_rel = [lm[20].x - hip_center[0], lm[20].y - hip_center[1], lm[20].z - hip_center[2]]
    
    left_foot_rel = [lm[31].x - hip_center[0], lm[31].y - hip_center[1], lm[31].z - hip_center[2]]
    right_foot_rel = [lm[32].x - hip_center[0], lm[32].y - hip_center[1], lm[32].z - hip_center[2]]
    
    shoulder_tilt = lm[11].y - lm[12].y
    hip_tilt = lm[23].y - lm[24].y
    stance_width = abs(lm[27].x - lm[28].x)
    facing_direction = lm[11].z - lm[12].z
    
    # ===== DISTANCE FEATURES (8) =====
    hand_distance = np.sqrt((lm[19].x - lm[20].x)**2 + (lm[19].y - lm[20].y)**2 + (lm[19].z - lm[20].z)**2)
    wrist_distance = np.sqrt((lm[15].x - lm[16].x)**2 + (lm[15].y - lm[16].y)**2 + (lm[15].z - lm[16].z)**2)
    left_arm_extension = np.sqrt((lm[15].x - lm[23].x)**2 + (lm[15].y - lm[23].y)**2)
    right_arm_extension = np.sqrt((lm[16].x - lm[24].x)**2 + (lm[16].y - lm[24].y)**2)
    left_elbow_dist = np.sqrt((lm[13].x - hip_center[0])**2 + (lm[13].y - hip_center[1])**2)
    right_elbow_dist = np.sqrt((lm[14].x - hip_center[0])**2 + (lm[14].y - hip_center[1])**2)
    knee_distance = np.sqrt((lm[25].x - lm[26].x)**2 + (lm[25].y - lm[26].y)**2)
    foot_distance = np.sqrt((lm[31].x - lm[32].x)**2 + (lm[31].y - lm[32].y)**2 + (lm[31].z - lm[32].z)**2)
    
    distances = [hand_distance, wrist_distance, left_arm_extension, right_arm_extension,
                left_elbow_dist, right_elbow_dist, knee_distance, foot_distance]
    
    # ===== SYMMETRY FEATURES (5) =====
    elbow_symmetry = angles[0] - angles[1]
    shoulder_symmetry = angles[2] - angles[3]
    knee_symmetry = angles[8] - angles[9]
    arm_raise_symmetry = angles[12] - angles[13]
    wrist_height_diff = lm[15].y - lm[16].y
    
    symmetry = [elbow_symmetry, shoulder_symmetry, knee_symmetry, arm_raise_symmetry, wrist_height_diff]
    
    # ===== BODY PROPORTIONS (4) =====
    arm_span = abs(lm[15].x - lm[16].x)
    body_height = abs(lm[0].y - (lm[27].y + lm[28].y)/2)
    arm_to_height_ratio = arm_span / (body_height + 0.001)
    stance_depth = abs(lm[27].z - lm[28].z)
    
    proportions = [arm_span, body_height, arm_to_height_ratio, stance_depth]
    
    # ===== LATERALITY FEATURES (10) - explicit left/right indicators =====
    # Which side is more extended/active? (positive = right side dominant)
    dominant_arm = 1.0 if right_arm_extension > left_arm_extension else 0.0
    dominant_elbow_height = 1.0 if lm[14].y < lm[13].y else 0.0  # Higher elbow (lower y = higher)
    dominant_wrist_forward = 1.0 if lm[16].z < lm[15].z else 0.0  # More forward (lower z)
    dominant_hand_height = 1.0 if lm[20].y < lm[19].y else 0.0
    
    # Absolute position indicators (which side of body center)
    body_center_x = (lm[11].x + lm[12].x + lm[23].x + lm[24].x) / 4
    left_wrist_side = lm[15].x - body_center_x  # Negative = left of center
    right_wrist_side = lm[16].x - body_center_x  # Positive = right of center
    active_wrist_x = right_wrist_side - left_wrist_side  # Which wrist is further from center
    
    # Elbow position relative to shoulder (tucked vs extended)
    left_elbow_tucked = abs(lm[13].x - lm[11].x)  # Small = tucked
    right_elbow_tucked = abs(lm[14].x - lm[12].x)
    
    laterality = [dominant_arm, dominant_elbow_height, dominant_wrist_forward, 
                  dominant_hand_height, active_wrist_x, left_wrist_side, right_wrist_side,
                  left_elbow_tucked, right_elbow_tucked, 
                  right_arm_extension - left_arm_extension]  # Signed difference
    
    # combine all 68 body features (was 58, added 10 laterality)
    features = (angles + cross_body_angles + 
               left_wrist_rel + right_wrist_rel +
               left_hand_rel + right_hand_rel +
               left_foot_rel + right_foot_rel +
               [shoulder_tilt, hip_tilt, stance_width, facing_direction] +
               distances + symmetry + proportions + laterality)
    
    return features

def extract_stick_features(image, lm_px, img_w, img_h, body_height):
    """Extract 24 stick features from YOLO detection (expanded from 14).
    
    New features focus on:
    - Stick position relative to strike targets (eyes, temples, chest, knees)
    - Stick trajectory/direction indicators
    - Quadrant-based positioning
    
    Returns zeros if no valid stick detected (maintains consistent feature vector).
    """
    global worker_stick_model
    
    # default zeros if no stick detected (24 features now)
    zero_features = [0.0] * 24
    
    if worker_stick_model is None:
        return zero_features
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = worker_stick_model(image, verbose=False)
        
        for result in results:
            if result.keypoints is None or len(result.boxes) == 0:
                continue
            
            # use highest confidence detection
            best_idx = result.boxes.conf.argmax().item()
            box = result.boxes[best_idx]
            kpts = result.keypoints[best_idx]
            
            stick_conf = box.conf.item()
            if stick_conf < STICK_CONFIDENCE_THRESHOLD:
                continue
            
            kpts_data = kpts.data[0].cpu().numpy()
            if len(kpts_data) < 2:
                continue
            
            grip_x, grip_y, grip_conf = kpts_data[0]
            tip_x, tip_y, tip_conf = kpts_data[1]
            
            if grip_conf < KEYPOINT_CONFIDENCE_THRESHOLD or tip_conf < KEYPOINT_CONFIDENCE_THRESHOLD:
                continue
            
            # ===== BASIC FEATURES (original) =====
            stick_detected = 1.0
            
            # stick length normalized to body height
            stick_length_px = np.sqrt((tip_x - grip_x)**2 + (tip_y - grip_y)**2)
            stick_length_norm = stick_length_px / (body_height * img_h + 0.001)
            
            # stick angle (-180 to 180)
            stick_angle_raw = np.degrees(np.arctan2(tip_y - grip_y, tip_x - grip_x))
            stick_angle = stick_angle_raw / 180.0  # Normalize to -1 to 1
            
            # normalized positions
            grip_x_norm = grip_x / img_w
            grip_y_norm = grip_y / img_h
            tip_x_norm = tip_x / img_w
            tip_y_norm = tip_y / img_h
            
            # ===== HOLDING HAND FEATURES =====
            left_wrist_px = (lm_px[15].x * img_w, lm_px[15].y * img_h)
            right_wrist_px = (lm_px[16].x * img_w, lm_px[16].y * img_h)
            
            dist_to_left = np.sqrt((grip_x - left_wrist_px[0])**2 + (grip_y - left_wrist_px[1])**2)
            dist_to_right = np.sqrt((grip_x - right_wrist_px[0])**2 + (grip_y - right_wrist_px[1])**2)
            
            img_diag = np.sqrt(img_w**2 + img_h**2)
            grip_to_holding_wrist = min(dist_to_left, dist_to_right) / img_diag
            holding_hand = 0.0 if dist_to_left < dist_to_right else 1.0
            
            # ===== TIP RELATIVE TO BODY LANDMARKS =====
            # Key body landmarks for strike targets
            left_eye = (lm_px[2].x, lm_px[2].y)
            right_eye = (lm_px[5].x, lm_px[5].y)
            nose = (lm_px[0].x, lm_px[0].y)
            left_shoulder = (lm_px[11].x, lm_px[11].y)
            right_shoulder = (lm_px[12].x, lm_px[12].y)
            left_hip = (lm_px[23].x, lm_px[23].y)
            right_hip = (lm_px[24].x, lm_px[24].y)
            left_knee = (lm_px[25].x, lm_px[25].y)
            right_knee = (lm_px[26].x, lm_px[26].y)
            
            # Center points
            shoulder_center = ((left_shoulder[0] + right_shoulder[0])/2, 
                              (left_shoulder[1] + right_shoulder[1])/2)
            hip_center = ((left_hip[0] + right_hip[0])/2, 
                         (left_hip[1] + right_hip[1])/2)
            chest_center = ((shoulder_center[0] + hip_center[0])/2,
                           (shoulder_center[1]*0.7 + hip_center[1]*0.3))  # Closer to shoulders
            
            # Original tip-relative features
            tip_vs_shoulder_y = tip_y_norm - shoulder_center[1]
            tip_vs_hip_x = tip_x_norm - hip_center[0]
            tip_vs_hip_y = tip_y_norm - hip_center[1]
            
            # ===== NEW: STRIKE TARGET DISTANCES =====
            # Distance from tip to each strike target (normalized)
            tip_to_left_eye = np.sqrt((tip_x_norm - left_eye[0])**2 + (tip_y_norm - left_eye[1])**2)
            tip_to_right_eye = np.sqrt((tip_x_norm - right_eye[0])**2 + (tip_y_norm - right_eye[1])**2)
            tip_to_crown = np.sqrt((tip_x_norm - nose[0])**2 + (tip_y_norm - (nose[1] - 0.1))**2)  # Above head
            tip_to_chest = np.sqrt((tip_x_norm - chest_center[0])**2 + (tip_y_norm - chest_center[1])**2)
            tip_to_solar = np.sqrt((tip_x_norm - hip_center[0])**2 + (tip_y_norm - (shoulder_center[1]*0.4 + hip_center[1]*0.6))**2)
            tip_to_left_knee = np.sqrt((tip_x_norm - left_knee[0])**2 + (tip_y_norm - left_knee[1])**2)
            tip_to_right_knee = np.sqrt((tip_x_norm - right_knee[0])**2 + (tip_y_norm - right_knee[1])**2)
            
            # ===== NEW: STICK DIRECTION RELATIVE TO BODY =====
            # Is stick pointing left, right, up, or down relative to body center?
            body_center_x = (shoulder_center[0] + hip_center[0]) / 2
            body_center_y = (shoulder_center[1] + hip_center[1]) / 2
            
            stick_dir_x = tip_x_norm - grip_x_norm  # Positive = pointing right
            stick_dir_y = tip_y_norm - grip_y_norm  # Positive = pointing down
            
            # Normalize direction to unit length
            dir_mag = np.sqrt(stick_dir_x**2 + stick_dir_y**2) + 0.001
            stick_dir_x_norm = stick_dir_x / dir_mag
            stick_dir_y_norm = stick_dir_y / dir_mag
            
            # Confidence values
            keypoint_conf = (grip_conf + tip_conf) / 2
            
            return [
                # Original 14 features
                stick_detected, stick_length_norm, stick_angle,
                grip_x_norm, grip_y_norm, tip_x_norm, tip_y_norm,
                grip_to_holding_wrist, holding_hand,
                tip_vs_shoulder_y, tip_vs_hip_x, tip_vs_hip_y,
                stick_conf, keypoint_conf,
                # NEW 10 features for strike targets
                tip_to_left_eye, tip_to_right_eye, tip_to_crown,
                tip_to_chest, tip_to_solar,
                tip_to_left_knee, tip_to_right_knee,
                stick_dir_x_norm, stick_dir_y_norm,
                body_center_y - tip_y_norm  # How high is tip relative to body center
            ]
        
        return zero_features
        
    except Exception as e:
        # Log error for debugging but don't crash extraction
        if os.environ.get('DEBUG_EXTRACTION'):
            print(f"[DEBUG] Stick extraction error: {e}")
        return zero_features

def extract_combined_features(image_path, verbose=False):
    """extract 72 combined features (58 body + 14 stick)"""
    global worker_pose_instance
    if worker_pose_instance is None:
        init_worker()
    
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = worker_pose_instance.process(image_rgb)
    
    if not results.pose_world_landmarks:
        return None
    
    try:
        # body features from world landmarks
        lm_world = results.pose_world_landmarks.landmark
        body_features = extract_body_features(lm_world)
        
        # get body height for normalization (from world coords)
        body_height = abs(lm_world[0].y - (lm_world[27].y + lm_world[28].y)/2)
        
        # stick features (needs pixel landmarks for wrist positions)
        lm_px = results.pose_landmarks.landmark
        stick_features = extract_stick_features(image, lm_px, w, h, body_height)
        
        return body_features + stick_features
        
    except Exception as e:
        if verbose:
            print(f"[WARN] Extraction failed for {image_path}: {e}")
        return None

def _extract_wrapper(args):
    """wrapper for multiprocessing"""
    image_path, class_name = args
    features = extract_combined_features(image_path)
    if features is not None:
        return [class_name] + list(features)
    return None

def get_feature_header():
    """get CSV header for combined features"""
    # 15 joint angles
    angle_names = ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
                  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                  'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                  'left_arm_raise', 'right_arm_raise', 'torso_lean']
    # 4 cross-body angles
    cross_body_names = ['cross_left_hand', 'cross_right_hand', 'diagonal_left', 'diagonal_right']
    # 22 relative positions
    position_names = ['lwrist_rel_x', 'lwrist_rel_y', 'lwrist_rel_z',
                     'rwrist_rel_x', 'rwrist_rel_y', 'rwrist_rel_z',
                     'lhand_rel_x', 'lhand_rel_y', 'lhand_rel_z',
                     'rhand_rel_x', 'rhand_rel_y', 'rhand_rel_z',
                     'lfoot_rel_x', 'lfoot_rel_y', 'lfoot_rel_z',
                     'rfoot_rel_x', 'rfoot_rel_y', 'rfoot_rel_z',
                     'shoulder_tilt', 'hip_tilt', 'stance_width', 'facing_direction']
    # 8 distance features
    distance_names = ['hand_distance', 'wrist_distance', 'left_arm_ext', 'right_arm_ext',
                     'left_elbow_dist', 'right_elbow_dist', 'knee_distance', 'foot_distance']
    # 5 symmetry features
    symmetry_names = ['elbow_symmetry', 'shoulder_symmetry', 'knee_symmetry', 
                     'arm_raise_symmetry', 'wrist_height_diff']
    # 4 body proportions
    proportion_names = ['arm_span', 'body_height', 'arm_to_height_ratio', 'stance_depth']
    # 10 laterality features (left/right differentiation)
    laterality_names = ['dominant_arm', 'dominant_elbow_height', 'dominant_wrist_forward',
                        'dominant_hand_height', 'active_wrist_x', 'left_wrist_side', 'right_wrist_side',
                        'left_elbow_tucked', 'right_elbow_tucked', 'arm_ext_diff']
    # 24 stick features (expanded from 14)
    stick_names = ['stick_detected', 'stick_length_norm', 'stick_angle',
                  'grip_x_norm', 'grip_y_norm', 'tip_x_norm', 'tip_y_norm',
                  'grip_to_holding_wrist', 'holding_hand',
                  'tip_vs_shoulder_y', 'tip_vs_hip_x', 'tip_vs_hip_y',
                  'stick_conf', 'keypoint_conf',
                  # NEW strike target features
                  'tip_to_left_eye', 'tip_to_right_eye', 'tip_to_crown',
                  'tip_to_chest', 'tip_to_solar',
                  'tip_to_left_knee', 'tip_to_right_knee',
                  'stick_dir_x', 'stick_dir_y', 'tip_height_vs_body']
    
    header = (['class'] + angle_names + cross_body_names + position_names + 
              distance_names + symmetry_names + proportion_names + laterality_names + stick_names)
    return header

def extract_features_from_dataset(dataset_path, csv_output_path):
    """extract combined features from all images in dataset and save to CSV"""
    print(f"\n[STAGE 1] Extracting COMBINED features (body + stick)...")
    
    header = get_feature_header()
    print(f"  Feature count: {len(header) - 1} (68 body + 24 stick)")
    
    # check stick model exists
    if not os.path.exists(STICK_MODEL_PATH):
        print(f"\n[WARN] Stick model not found: {STICK_MODEL_PATH}")
        print("       Stick features will be zeros.")
    
    # collect all image paths
    image_tasks = []
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    print(f"  Found {len(classes)} classes")
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_tasks.append((os.path.join(class_path, img_name), class_name))
    
    print(f"  Processing {len(image_tasks)} images...")
    
    # Extract features using multiprocessing
    # Note: YOLO has issues with multiprocessing, use fewer workers to avoid memory leaks
    # Consider setting n_workers=1 if you experience crashes
    n_workers = max(1, min(2, cpu_count() - 1))  # Reduced from 4 to 2 for YOLO stability
    print(f"  Using {n_workers} workers (reduced for YOLO stability)...")
    
    with Pool(n_workers, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(_extract_wrapper, image_tasks), total=len(image_tasks)))
    
    # filter out failed extractions
    valid_results = [r for r in results if r is not None]
    
    # count stick detections (stick_detected is 24th from end)
    stick_detected_count = sum(1 for r in valid_results if r[-24] == 1.0)
    
    print(f"  Extracted: {len(valid_results)}/{len(image_tasks)} images")
    print(f"  Stick detected: {stick_detected_count}/{len(valid_results)} ({100*stick_detected_count/max(1,len(valid_results)):.1f}%)")
    
    # save to CSV
    df = pd.DataFrame(valid_results, columns=header)
    df.to_csv(csv_output_path, index=False)
    
    print(f"  Saved to: {csv_output_path}")
    
    return df


if __name__ == "__main__":
    # test on single image
    import sys
    
    test_image = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        project_root, 'dataset', 'right_temple_block_correct', '5.jpg')
    
    print(f"Testing combined extraction on: {test_image}")
    init_worker()
    
    features = extract_combined_features(test_image, verbose=True)
    if features:
        header = get_feature_header()[1:]  # skip 'class'
        print(f"\nExtracted {len(features)} features:")
        for name, val in zip(header, features):
            print(f"  {name}: {val:.4f}" if isinstance(val, float) else f"  {name}: {val}")
    else:
        print("Extraction failed!")
