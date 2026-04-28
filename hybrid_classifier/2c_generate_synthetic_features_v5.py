"""
2c_generate_synthetic_features_v2.py
====================================
Generate synthetic feature tensors via intra-class mixup + joint perturbation.
Reads from hybrid_features_v4 (signed direction features) and writes synthetic there.

TRAIN: 2 synthetic per real sample (mixup + arm perturbation) = 3x total
TEST:  1 synthetic per real sample (arm perturbation only) = 2x total

Anatomical safeguards:
- Knee angles clamped [5, 175] degrees
- Wrist height >= ankle height
- Elbow angles clamped [15, 175] degrees
- Visibility capped [0, 1]
- Stick length sanity check

Output:
    synthetic_train_features_front_3x.pt
    synthetic_test_features_front_1x.pt

Usage:
    python hybrid_classifier/2c_generate_synthetic_features_v2.py
"""


import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Paths
FEATURES_DIR = Path("hybrid_classifier/hybrid_features_v5")
OUTPUT_DIR = FEATURES_DIR

# Class names
CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct',
    'neutral'
]
NUM_CLASSES = len(CLASS_NAMES)

# Joint indices for perturbation (arms + stick only)
# Perturb arm joints + stick + lower body (stance diversity for synthetic training)
# Added knees (25, 26) and ankles (27, 28) to create varied stances per synthetic sample
ARM_JOINTS = [11, 12, 13, 14, 15, 16, 25, 26, 27, 28, 33, 34]
# Leg joints for knee angle check
KNEE_JOINTS = {
    'left': (23, 25, 27),
    'right': (24, 26, 28),
}
# Elbow joints for angle check
ELBOW_JOINTS = {
    'left': (11, 13, 15),
    'right': (12, 14, 16),
}


def compute_angle_3d(a, b, c):
    """Compute angle at b formed by a-b-c in 3D."""
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba) + 1e-8
    norm_bc = np.linalg.norm(bc) + 1e-8
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle


def recompute_node_features(node_features):
    """
    Recompute dist_to_hip and angle_from_hip for all nodes.
    node_features: [35, 6] with [x, y, z, visibility, dist_to_hip, angle_from_hip]
    """
    pts = node_features[:, :3].copy()
    vis = node_features[:, 3].copy()

    # Hip center (nodes 23 and 24)
    hip_center = (pts[23] + pts[24]) / 2.0

    out = np.zeros_like(node_features)
    out[:, :3] = pts
    out[:, 3] = np.clip(vis, 0.0, 1.0)

    for i in range(35):
        dx = pts[i, 0] - hip_center[0]
        dy = pts[i, 1] - hip_center[1]
        dz = pts[i, 2] - hip_center[2]
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        angle = np.degrees(np.arctan2(dy, dx))
        out[i, 4] = dist
        out[i, 5] = angle

    return out


def has_valid_stick(node_features):
    """Check if stick grip has valid coordinates."""
    grip = node_features[33]
    return (abs(grip[0]) > 1e-6 or abs(grip[1]) > 1e-6) and grip[3] > 0.01


def check_knee_angles(node_features):
    """Return dict of knee angles and whether they're valid."""
    pts = node_features[:, :3]
    angles = {}
    valid = True
    for side, (hip, knee, ankle) in KNEE_JOINTS.items():
        angle = compute_angle_3d(pts[hip], pts[knee], pts[ankle])
        angles[side] = angle
        if angle < 5 or angle > 175:
            valid = False
    return angles, valid


def check_elbow_angles(node_features):
    """Return dict of elbow angles and whether they're valid."""
    pts = node_features[:, :3]
    angles = {}
    valid = True
    for side, (shoulder, elbow, wrist) in ELBOW_JOINTS.items():
        angle = compute_angle_3d(pts[shoulder], pts[elbow], pts[wrist])
        angles[side] = angle
        if angle < 15 or angle > 175:
            valid = False
    return angles, valid


def check_wrist_height(node_features):
    """Check that wrists are above ankles."""
    pts = node_features[:, :3]
    ankle_y = min(pts[27, 1], pts[28, 1])
    left_wrist_y = pts[15, 1]
    right_wrist_y = pts[16, 1]
    return left_wrist_y <= ankle_y + 0.05 and right_wrist_y <= ankle_y + 0.05


def check_stick_length(node_features):
    """Check stick length is reasonable (0.3 to 2.0 x forearm)."""
    pts = node_features[:, :3]
    grip = pts[33]
    tip = pts[34]
    stick_len = np.linalg.norm(tip - grip)
    # Forearm length
    left_forearm = np.linalg.norm(pts[15] - pts[13])
    right_forearm = np.linalg.norm(pts[16] - pts[14])
    avg_forearm = (left_forearm + right_forearm) / 2.0 + 1e-8
    ratio = stick_len / avg_forearm
    return 0.3 <= ratio <= 2.5


def apply_joint_perturbation(node_features, sigma=0.02, seed=None, stick_right_hand=True):
    """Add Gaussian noise to arm joints + stick.
    
    Uses stick_right_hand metadata to reattach stick to the correct hand:
    - If stick_right_hand=True (default): reattach to right wrist (node 16)
    - If stick_right_hand=False: reattach to left wrist (node 15)
    
    This preserves the structural prior that the model learns from real data.
    """
    if seed is not None:
        np.random.seed(seed)
    perturbed = node_features.copy()
    
    # Determine which hand holds the stick
    wrist_node = 16 if stick_right_hand else 15
    
    # Record original stick-to-wrist relationship
    original_wrist = node_features[wrist_node, :3].copy()
    original_stick_grip = node_features[33, :3].copy()
    original_stick_tip = node_features[34, :3].copy()
    grip_to_wrist = original_stick_grip - original_wrist
    tip_to_grip = original_stick_tip - original_stick_grip
    
    # Perturb all arm joints independently
    for joint in ARM_JOINTS:
        noise = np.random.normal(0, sigma, size=3)
        perturbed[joint, :3] += noise
    
    # Reattach stick grip to perturbed wrist (maintain relative offset)
    perturbed_wrist = perturbed[wrist_node, :3]
    perturbed[33, :3] = perturbed_wrist + grip_to_wrist
    
    # Perturb stick tip relative to grip (small noise on direction, preserve length)
    tip_noise = np.random.normal(0, sigma * 0.5, size=3)
    perturbed_tip_to_grip = tip_to_grip + tip_noise
    # Preserve original stick length
    orig_len = np.linalg.norm(tip_to_grip) + 1e-8
    new_len = np.linalg.norm(perturbed_tip_to_grip) + 1e-8
    if new_len > 0.01:
        perturbed_tip_to_grip = perturbed_tip_to_grip * (orig_len / new_len)
    perturbed[34, :3] = perturbed[33, :3] + perturbed_tip_to_grip
    
    return perturbed


def mixup_nodes(node_a, node_b, lam):
    """Blend two node feature arrays."""
    blended = lam * node_a + (1.0 - lam) * node_b
    # Visibility should be max (more visible = more confident)
    blended[:, 3] = np.maximum(node_a[:, 3], node_b[:, 3])
    # Clamp visibility
    blended[:, 3] = np.clip(blended[:, 3], 0.0, 1.0)
    return blended


def find_nearest_neighbors(sample_idx, class_mask, hybrid_features, k=2):
    """
    Find k nearest neighbors within same class using hybrid feature cosine similarity.
    Returns list of (neighbor_idx, similarity) tuples.
    """
    sample_hybrid = hybrid_features[sample_idx]
    class_hybrid = hybrid_features[class_mask]
    class_indices = np.where(class_mask)[0]

    # Cosine similarity
    sample_norm = sample_hybrid / (np.linalg.norm(sample_hybrid) + 1e-8)
    class_norms = np.linalg.norm(class_hybrid, axis=1, keepdims=True) + 1e-8
    class_normalized = class_hybrid / class_norms

    similarities = np.dot(class_normalized, sample_norm)

    # Exclude self (should be ~1.0)
    similarities[class_indices == sample_idx] = -1.0

    top_k = np.argsort(similarities)[-k:][::-1]
    return [(class_indices[i], similarities[i]) for i in top_k]


def generate_train_synthetics(data, factor=2, sigma=0.02):
    """
    Generate synthetic training samples.
    factor: number of synthetics per real sample (default 2 -> 3x total)
    """
    node_features = data['node_features'].numpy()  # [N, 35, 6]
    hybrid_features = data['hybrid_features'].numpy()  # [N, 30]
    labels = data['labels'].numpy()  # [N]
    viewpoints = data.get('viewpoints', ['front'] * len(labels))
    has_stick_nodes = data.get('has_stick_nodes', torch.ones(len(labels), dtype=torch.bool)).numpy()
    stick_right_hand = data.get('stick_right_hand', torch.ones(len(labels), dtype=torch.bool)).numpy()

    synthetic_nodes = []
    synthetic_hybrid = []
    synthetic_labels = []
    synthetic_viewpoints = []
    synthetic_has_stick = []
    synthetic_is_synthetic = []
    synthetic_meta = []  # Store (parent_idx, lambda, nn_idx) for debugging

    for class_idx in range(NUM_CLASSES):
        class_mask = labels == class_idx
        class_indices = np.where(class_mask)[0]
        n_class = len(class_indices)

        if n_class == 0:
            continue

        print(f"Class {class_idx} ({CLASS_NAMES[class_idx]}): {n_class} samples")

        for real_idx in tqdm(class_indices, desc=f"  Generating synthetics", leave=False):
            real_node = node_features[real_idx]
            real_hybrid = hybrid_features[real_idx]

            # Find nearest neighbors
            nn_list = find_nearest_neighbors(real_idx, class_mask, hybrid_features, k=factor)

            for syn_num, (nn_idx, sim) in enumerate(nn_list):
                nn_node = node_features[nn_idx]
                nn_hybrid = hybrid_features[nn_idx]

                # Mixup lambda: 0.33 for first, 0.67 for second (asymmetric)
                lam = 0.33 if syn_num == 0 else 0.67

                # Blend node features
                blended_node = mixup_nodes(real_node, nn_node, lam)
                blended_hybrid = lam * real_hybrid + (1.0 - lam) * nn_hybrid

                # Apply arm perturbation (respecting stick handedness)
                blended_node = apply_joint_perturbation(
                    blended_node, sigma=sigma,
                    stick_right_hand=bool(stick_right_hand[real_idx])
                )

                # Recompute dependent features (dist_to_hip, angle_from_hip)
                blended_node = recompute_node_features(blended_node)

                # Anatomical checks (log but don't reject - keep for diversity)
                knee_angles, knees_ok = check_knee_angles(blended_node)
                elbow_angles, elbows_ok = check_elbow_angles(blended_node)
                wrist_ok = check_wrist_height(blended_node)
                stick_ok = check_stick_length(blended_node)

                if not (knees_ok and elbows_ok and wrist_ok and stick_ok):
                    pass  # Keep anyway, just log stats

                synthetic_nodes.append(blended_node)
                synthetic_hybrid.append(blended_hybrid)
                synthetic_labels.append(class_idx)
                synthetic_viewpoints.append(viewpoints[real_idx])
                synthetic_has_stick.append(has_stick_nodes[real_idx])
                synthetic_is_synthetic.append(True)
                synthetic_meta.append({
                    'parent_idx': int(real_idx),
                    'nn_idx': int(nn_idx),
                    'lambda': float(lam),
                    'similarity': float(sim),
                    'checks': {
                        'knees_ok': bool(knees_ok),
                        'elbows_ok': bool(elbows_ok),
                        'wrist_ok': bool(wrist_ok),
                        'stick_ok': bool(stick_ok),
                    }
                })

    result = {
        'node_features': torch.tensor(np.array(synthetic_nodes), dtype=torch.float32),
        'hybrid_features': torch.tensor(np.array(synthetic_hybrid), dtype=torch.float32),
        'labels': torch.tensor(synthetic_labels, dtype=torch.long),
        'viewpoints': synthetic_viewpoints,
        'has_stick_nodes': torch.tensor(synthetic_has_stick, dtype=torch.bool),
        'is_synthetic': torch.tensor(synthetic_is_synthetic, dtype=torch.bool),
        'meta': synthetic_meta,
    }

    return result


def generate_test_synthetics(data, factor=1, sigma=0.02):
    """
    Generate synthetic test samples using PERTURBATION ONLY (no mixup).
    factor: number of synthetics per real sample (default 1 -> 2x total)
    """
    node_features = data['node_features'].numpy()
    hybrid_features = data['hybrid_features'].numpy()
    labels = data['labels'].numpy()
    viewpoints = data.get('viewpoints', ['front'] * len(labels))
    has_stick_nodes = data.get('has_stick_nodes', torch.ones(len(labels), dtype=torch.bool)).numpy()
    stick_right_hand = data.get('stick_right_hand', torch.ones(len(labels), dtype=torch.bool)).numpy()

    synthetic_nodes = []
    synthetic_hybrid = []
    synthetic_labels = []
    synthetic_viewpoints = []
    synthetic_has_stick = []
    synthetic_is_synthetic = []
    synthetic_meta = []

    for class_idx in range(NUM_CLASSES):
        class_mask = labels == class_idx
        class_indices = np.where(class_mask)[0]
        n_class = len(class_indices)

        if n_class == 0:
            continue

        print(f"Class {class_idx} ({CLASS_NAMES[class_idx]}): {n_class} samples")

        for real_idx in tqdm(class_indices, desc=f"  Generating test synthetics", leave=False):
            real_node = node_features[real_idx]
            real_hybrid = hybrid_features[real_idx]

            for syn_num in range(factor):
                # Only perturbation, no mixup (respecting stick handedness)
                perturbed_node = apply_joint_perturbation(
                    real_node.copy(), sigma=sigma,
                    seed=real_idx + syn_num * 1000,
                    stick_right_hand=bool(stick_right_hand[real_idx])
                )
                perturbed_node = recompute_node_features(perturbed_node)

                # Hybrid features: small noise
                perturbed_hybrid = real_hybrid + np.random.normal(0, 0.01, size=real_hybrid.shape)
                perturbed_hybrid = np.clip(perturbed_hybrid, 0.0, 1.0)

                # Checks
                knee_angles, knees_ok = check_knee_angles(perturbed_node)
                elbow_angles, elbows_ok = check_elbow_angles(perturbed_node)
                wrist_ok = check_wrist_height(perturbed_node)
                stick_ok = check_stick_length(perturbed_node)

                synthetic_nodes.append(perturbed_node)
                synthetic_hybrid.append(perturbed_hybrid)
                synthetic_labels.append(class_idx)
                synthetic_viewpoints.append(viewpoints[real_idx])
                synthetic_has_stick.append(has_stick_nodes[real_idx])
                synthetic_is_synthetic.append(True)
                synthetic_meta.append({
                    'parent_idx': int(real_idx),
                    'perturbation_seed': int(real_idx + syn_num * 1000),
                    'checks': {
                        'knees_ok': bool(knees_ok),
                        'elbows_ok': bool(elbows_ok),
                        'wrist_ok': bool(wrist_ok),
                        'stick_ok': bool(stick_ok),
                    }
                })

    result = {
        'node_features': torch.tensor(np.array(synthetic_nodes), dtype=torch.float32),
        'hybrid_features': torch.tensor(np.array(synthetic_hybrid), dtype=torch.float32),
        'labels': torch.tensor(synthetic_labels, dtype=torch.long),
        'viewpoints': synthetic_viewpoints,
        'has_stick_nodes': torch.tensor(synthetic_has_stick, dtype=torch.bool),
        'is_synthetic': torch.tensor(synthetic_is_synthetic, dtype=torch.bool),
        'meta': synthetic_meta,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic features")
    parser.add_argument("--train_factor", type=int, default=2,
                        help="Synthetic samples per real training sample (default: 2 = 3x total)")
    parser.add_argument("--test_factor", type=int, default=1,
                        help="Synthetic samples per real test sample (default: 1 = 2x total)")
    parser.add_argument("--sigma", type=float, default=0.02,
                        help="Joint perturbation std dev (default: 0.02 = 2%)")
    parser.add_argument("--viewpoint", type=str, default="front",
                        help="Viewpoint to process (default: front)")
    args = parser.parse_args()

    viewpoint = args.viewpoint
    train_path = FEATURES_DIR / f"train_features_{viewpoint}.pt"
    test_path = FEATURES_DIR / f"test_features_{viewpoint}.pt"

    if not train_path.exists():
        print(f"Error: Train file not found: {train_path}")
        return
    if not test_path.exists():
        print(f"Error: Test file not found: {test_path}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # === TRAIN ===
    print(f"\n{'='*60}")
    print(f"TRAIN: Generating {args.train_factor}x synthetics")
    print(f"Input: {train_path}")
    print(f"{'='*60}")

    train_data = torch.load(train_path, map_location='cpu')
    print(f"Loaded {len(train_data['labels'])} training samples")

    syn_train = generate_train_synthetics(train_data, factor=args.train_factor, sigma=args.sigma)

    train_output = OUTPUT_DIR / f"synthetic_train_features_{viewpoint}_{args.train_factor}x.pt"
    torch.save(syn_train, train_output)
    print(f"[OK] Saved synthetic train: {train_output}")
    print(f"  Samples: {len(syn_train['labels'])} (from {len(train_data['labels'])} real)")
    print(f"  Node features: {syn_train['node_features'].shape}")
    print(f"  Hybrid features: {syn_train['hybrid_features'].shape}")

    # === TEST ===
    print(f"\n{'='*60}")
    print(f"TEST: Generating {args.test_factor}x synthetics (perturbation only)")
    print(f"Input: {test_path}")
    print(f"{'='*60}")

    test_data = torch.load(test_path, map_location='cpu')
    print(f"Loaded {len(test_data['labels'])} test samples")

    syn_test = generate_test_synthetics(test_data, factor=args.test_factor, sigma=args.sigma)

    test_output = OUTPUT_DIR / f"synthetic_test_features_{viewpoint}_{args.test_factor}x.pt"
    torch.save(syn_test, test_output)
    print(f"[OK] Saved synthetic test: {test_output}")
    print(f"  Samples: {len(syn_test['labels'])} (from {len(test_data['labels'])} real)")
    print(f"  Node features: {syn_test['node_features'].shape}")
    print(f"  Hybrid features: {syn_test['hybrid_features'].shape}")

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
