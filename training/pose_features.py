import numpy as np

def compute_angle(a, b, c):
    """
    Compute angle between three points (a, b, c) where b is the vertex.
    Returns angle in degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def compute_distance(a, b):
    """Compute Euclidean distance between two points."""
    return np.linalg.norm(np.array(a) - np.array(b))

def extract_geometric_features(keypoints):
    """
    Extract geometric features from 33 MediaPipe keypoints.
    
    Args:
        keypoints: [33, 3] array of (x, y, visibility)
        
    Returns:
        concatenated feature vector [num_features]
    """
    kpts = keypoints[:, :2]  # Use only x, y
    
    # 1. Joint Angles (8 important angles)
    # Indices: 11=left_shoulder, 13=left_elbow, 15=left_wrist, 23=left_hip, 25=left_knee, 27=left_ankle
    #          12=right_shoulder, 14=right_elbow, 16=right_wrist, 24=right_hip, 26=right_knee, 28=right_ankle
    
    angles = [
        # Left Arm
        compute_angle(kpts[11], kpts[13], kpts[15]),  # Left Elbow
        compute_angle(kpts[23], kpts[11], kpts[13]),  # Left Shoulder
        
        # Right Arm
        compute_angle(kpts[12], kpts[14], kpts[16]),  # Right Elbow
        compute_angle(kpts[24], kpts[12], kpts[14]),  # Right Shoulder
        
        # Left Leg
        compute_angle(kpts[23], kpts[25], kpts[27]),  # Left Knee
        compute_angle(kpts[11], kpts[23], kpts[25]),  # Left Hip
        
        # Right Leg
        compute_angle(kpts[24], kpts[26], kpts[28]),  # Right Knee
        compute_angle(kpts[12], kpts[24], kpts[26]),  # Right Hip
    ]
    
    # Normalize angles to [0, 1] (divide by 180)
    angles = np.array(angles) / 180.0
    
    # 2. Limb Distances (Normalized by torso height)
    # Torso height = distance(mid_shoulder, mid_hip)
    mid_shoulder = (kpts[11] + kpts[12]) / 2
    mid_hip = (kpts[23] + kpts[24]) / 2
    torso_height = compute_distance(mid_shoulder, mid_hip) + 1e-6
    
    distances = [
        compute_distance(kpts[15], kpts[16]), # Wrist separation
        compute_distance(kpts[27], kpts[28]), # Ankle separation
        compute_distance(kpts[15], kpts[27]), # Left wrist-ankle
        compute_distance(kpts[16], kpts[28]), # Right wrist-ankle
        compute_distance(kpts[15], mid_shoulder), # Left wrist to center
        compute_distance(kpts[16], mid_shoulder), # Right wrist to center
    ]
    
    # Normalize distances
    distances = np.array(distances) / torso_height
    
    # 3. Symmetry (Left vs Right y-coordinates)
    symmetry = [
        kpts[11][1] - kpts[12][1], # Shoulder height diff
        kpts[13][1] - kpts[14][1], # Elbow height diff
        kpts[15][1] - kpts[16][1], # Wrist height diff
        kpts[25][1] - kpts[26][1], # Knee height diff
    ]
    
    # Concatenate all features
    features = np.concatenate([angles, distances, symmetry])
    
    return features.astype(np.float32)

def augment_node_features(keypoints):
    """
    Augment each node's features with global geometric context.
    
    Args:
        keypoints: [33, 3] array
    
    Returns:
        [33, num_new_features] array
    """
    # Simply repeating the global features for each node is a simple strategy
    # to give every node awareness of the global pose
    global_features = extract_geometric_features(keypoints)
    
    # Repeat for all nodes
    node_features = np.tile(global_features, (len(keypoints), 1))
    
    return node_features
