"""
Shared feature extraction functions for all training scripts
"""
import cv2
import numpy as np
import mediapipe as mp
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# mediapipe instance for worker processes
worker_pose_instance = None

def init_worker():
    """initialize mediapipe pose for worker process"""
    global worker_pose_instance
    worker_pose_instance = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    )

def calculate_angle_3d(a, b, c):
    """calculate angle at point b given 3 points"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    dot_product = np.dot(ba, bc)
    magnitude = np.linalg.norm(ba) * np.linalg.norm(bc)
    if magnitude == 0:
        return 0
    return np.degrees(np.arccos(np.clip(dot_product / magnitude, -1.0, 1.0)))

def extract_angles_from_image(image_path):
    """extract joint angles and key positions (54 features - expanded)"""
    global worker_pose_instance
    if worker_pose_instance is None:
        init_worker()
        
    image = cv2.imread(image_path)
    if image is None: 
        return None
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = worker_pose_instance.process(image_rgb)
    
    if not results.pose_world_landmarks:
        return None
        
    try:
        lm = results.pose_world_landmarks.landmark
        
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
            # left hand relative to right shoulder plane
            calculate_angle_3d([lm[12].x, lm[12].y, lm[12].z], [lm[11].x, lm[11].y, lm[11].z], [lm[15].x, lm[15].y, lm[15].z]),
            # right hand relative to left shoulder plane
            calculate_angle_3d([lm[11].x, lm[11].y, lm[11].z], [lm[12].x, lm[12].y, lm[12].z], [lm[16].x, lm[16].y, lm[16].z]),
            # diagonal: left shoulder to right hip to right knee
            calculate_angle_3d([lm[11].x, lm[11].y, lm[11].z], [lm[24].x, lm[24].y, lm[24].z], [lm[26].x, lm[26].y, lm[26].z]),
            # diagonal: right shoulder to left hip to left knee
            calculate_angle_3d([lm[12].x, lm[12].y, lm[12].z], [lm[23].x, lm[23].y, lm[23].z], [lm[25].x, lm[25].y, lm[25].z]),
        ]
        
        # ===== RELATIVE POSITIONS (18 features - original) =====
        left_wrist_rel = [lm[15].x - lm[11].x, lm[15].y - lm[11].y, lm[15].z - lm[11].z]
        right_wrist_rel = [lm[16].x - lm[12].x, lm[16].y - lm[12].y, lm[16].z - lm[12].z]
        
        hip_center = [(lm[23].x + lm[24].x)/2, (lm[23].y + lm[24].y)/2, (lm[23].z + lm[24].z)/2]
        left_hand_rel = [lm[19].x - hip_center[0], lm[19].y - hip_center[1]]
        right_hand_rel = [lm[20].x - hip_center[0], lm[20].y - hip_center[1]]
        
        left_foot_rel = [lm[31].x - hip_center[0], lm[31].y - hip_center[1]]
        right_foot_rel = [lm[32].x - hip_center[0], lm[32].y - hip_center[1]]
        
        shoulder_tilt = lm[11].y - lm[12].y
        hip_tilt = lm[23].y - lm[24].y
        stance_width = abs(lm[27].x - lm[28].x)
        facing_direction = lm[11].z - lm[12].z
        
        # ===== DISTANCE FEATURES (8 new) =====
        # hand-to-hand distance (important for blocking/striking)
        hand_distance = np.sqrt((lm[19].x - lm[20].x)**2 + (lm[19].y - lm[20].y)**2 + (lm[19].z - lm[20].z)**2)
        # wrist-to-wrist distance
        wrist_distance = np.sqrt((lm[15].x - lm[16].x)**2 + (lm[15].y - lm[16].y)**2 + (lm[15].z - lm[16].z)**2)
        # arm extension (wrist to hip)
        left_arm_extension = np.sqrt((lm[15].x - lm[23].x)**2 + (lm[15].y - lm[23].y)**2)
        right_arm_extension = np.sqrt((lm[16].x - lm[24].x)**2 + (lm[16].y - lm[24].y)**2)
        # elbow to hip center
        left_elbow_dist = np.sqrt((lm[13].x - hip_center[0])**2 + (lm[13].y - hip_center[1])**2)
        right_elbow_dist = np.sqrt((lm[14].x - hip_center[0])**2 + (lm[14].y - hip_center[1])**2)
        # knee spread
        knee_distance = np.sqrt((lm[25].x - lm[26].x)**2 + (lm[25].y - lm[26].y)**2)
        # foot spread (3D)
        foot_distance = np.sqrt((lm[31].x - lm[32].x)**2 + (lm[31].y - lm[32].y)**2 + (lm[31].z - lm[32].z)**2)
        
        distances = [hand_distance, wrist_distance, left_arm_extension, right_arm_extension,
                    left_elbow_dist, right_elbow_dist, knee_distance, foot_distance]
        
        # ===== SYMMETRY FEATURES (5 new) =====
        elbow_symmetry = angles[0] - angles[1]  # left - right elbow angle diff
        shoulder_symmetry = angles[2] - angles[3]  # shoulder angle diff
        knee_symmetry = angles[8] - angles[9]  # knee angle diff
        arm_raise_symmetry = angles[12] - angles[13]  # arm raise diff
        wrist_height_diff = lm[15].y - lm[16].y  # wrist height difference
        
        symmetry = [elbow_symmetry, shoulder_symmetry, knee_symmetry, arm_raise_symmetry, wrist_height_diff]
        
        # ===== BODY PROPORTIONS (4 new) =====
        arm_span = abs(lm[15].x - lm[16].x)
        body_height = abs(lm[0].y - (lm[27].y + lm[28].y)/2)
        arm_to_height_ratio = arm_span / (body_height + 0.001)  # avoid division by zero
        stance_depth = abs(lm[27].z - lm[28].z)  # front-back foot difference
        
        proportions = [arm_span, body_height, arm_to_height_ratio, stance_depth]
        
        # ===== COMBINE ALL FEATURES (54 total) =====
        features = (angles +                    # 15 features
                   cross_body_angles +          # 4 features
                   left_wrist_rel + right_wrist_rel +  # 6 features
                   left_hand_rel + right_hand_rel +    # 4 features
                   left_foot_rel + right_foot_rel +    # 4 features
                   [shoulder_tilt, hip_tilt, stance_width, facing_direction] +  # 4 features
                   distances +                  # 8 features
                   symmetry +                   # 5 features
                   proportions)                 # 4 features
        return features
        
    except Exception:
        return None

def extract_coordinates_from_image(image_path):
    """extract hip-centered coordinates (99 features)"""
    global worker_pose_instance
    if worker_pose_instance is None:
        init_worker()
        
    image = cv2.imread(image_path)
    if image is None: 
        return None
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = worker_pose_instance.process(image_rgb)
    
    if not results.pose_world_landmarks:
        return None
        
    try:
        landmarks = results.pose_world_landmarks.landmark
        hip_center = np.array([
            (landmarks[23].x + landmarks[24].x) / 2,
            (landmarks[23].y + landmarks[24].y) / 2,
            (landmarks[23].z + landmarks[24].z) / 2
        ])
        
        features = []
        for lm in landmarks:
            features.extend([lm.x - hip_center[0], lm.y - hip_center[1], lm.z - hip_center[2]])
        
        return features
        
    except Exception:
        return None

def _extract_wrapper(args):
    """wrapper for multiprocessing"""
    image_path, class_name, extraction_func = args
    features = extraction_func(image_path)
    if features is not None:
        return [class_name] + list(features)
    return None

def extract_features_from_dataset(dataset_path, csv_output_path, feature_mode='angles'):
    """extract features from all images in dataset and save to CSV"""
    print(f"\n[STAGE 1] Extracting {feature_mode.upper()} features...")
    
    if feature_mode == 'angles':
        extraction_func = extract_angles_from_image
        # 15 joint angles
        angle_names = ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
                      'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                      'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                      'left_arm_raise', 'right_arm_raise', 'torso_lean']
        # 4 cross-body angles
        cross_body_names = ['cross_left_hand', 'cross_right_hand', 'diagonal_left', 'diagonal_right']
        # 18 relative positions
        position_names = ['lwrist_rel_x', 'lwrist_rel_y', 'lwrist_rel_z',
                         'rwrist_rel_x', 'rwrist_rel_y', 'rwrist_rel_z',
                         'lhand_rel_x', 'lhand_rel_y', 'rhand_rel_x', 'rhand_rel_y',
                         'lfoot_rel_x', 'lfoot_rel_y', 'rfoot_rel_x', 'rfoot_rel_y',
                         'shoulder_tilt', 'hip_tilt', 'stance_width', 'facing_direction']
        # 8 distance features
        distance_names = ['hand_distance', 'wrist_distance', 'left_arm_ext', 'right_arm_ext',
                         'left_elbow_dist', 'right_elbow_dist', 'knee_distance', 'foot_distance']
        # 5 symmetry features
        symmetry_names = ['elbow_symmetry', 'shoulder_symmetry', 'knee_symmetry', 
                         'arm_raise_symmetry', 'wrist_height_diff']
        # 4 body proportions
        proportion_names = ['arm_span', 'body_height', 'arm_to_height_ratio', 'stance_depth']
        
        header = ['class'] + angle_names + cross_body_names + position_names + distance_names + symmetry_names + proportion_names
    else:
        extraction_func = extract_coordinates_from_image
        header = ['class'] + [f'{ax}_{i}' for i in range(33) for ax in ['x', 'y', 'z']]
    
    # collect all image paths
    image_tasks = []
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    print(f"  Found {len(classes)} classes")
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_tasks.append((os.path.join(class_path, img_name), class_name, extraction_func))
    
    print(f"  Processing {len(image_tasks)} images...")
    
    # extract features using multiprocessing
    n_workers = max(1, cpu_count() - 1)
    with Pool(n_workers, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(_extract_wrapper, image_tasks), total=len(image_tasks)))
    
    # filter out failed extractions
    valid_results = [r for r in results if r is not None]
    
    print(f"  Extracted: {len(valid_results)}/{len(image_tasks)} images")
    
    # save to CSV
    df = pd.DataFrame(valid_results, columns=header)
    df.to_csv(csv_output_path, index=False)
    
    print(f"  Saved to: {csv_output_path}")
    
    return df
