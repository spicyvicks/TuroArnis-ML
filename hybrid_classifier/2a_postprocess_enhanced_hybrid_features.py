"""
2a_postprocess_enhanced_hybrid_features.py
===========================================
Post-process existing .pt features to ADD enhanced hybrid features.
Computes 6 new geometric features from existing node coordinates,
then computes Gaussian similarities against enhanced templates.

Input: existing train_features_front.pt, test_features_front.pt
Output: train_features_front_enhanced.pt, test_features_front_enhanced.pt
Fast — no image re-processing needed!
"""

import numpy as np
import torch
from pathlib import Path
import json
from tqdm import tqdm

FEATURE_TEMPLATES = "hybrid_classifier/feature_templates_enhanced.json"
INPUT_DIR = Path("hybrid_classifier/hybrid_features_v3")
OUTPUT_DIR = Path("hybrid_classifier/hybrid_features_v3")

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct',
    'neutral'
]


def gaussian_similarity(value, mean, std):
    if std < 1e-6:
        return 1.0 if abs(value - mean) < 1e-6 else 0.0
    return np.exp(-0.5 * ((value - mean) / std) ** 2)


def compute_enhanced_features_from_nodes(node_features):
    """
    Compute 6 enhanced geometric features from node coordinates.
    node_features: [35, 6] tensor — [x, y, z, vis, dist_to_hip, angle]
    
    Node indices:
      0: nose
      11: left_shoulder, 12: right_shoulder
      13: left_elbow, 14: right_elbow
      15: left_wrist, 16: right_wrist
      23: left_hip, 24: right_hip
      25: left_knee, 26: right_knee
      27: left_ankle, 28: right_ankle
      33: stick_grip, 34: stick_tip
    """
    nf = node_features.numpy()
    
    # Extract key points
    nose = nf[0, :3]
    l_shoulder = nf[11, :3]
    r_shoulder = nf[12, :3]
    l_elbow = nf[13, :3]
    r_elbow = nf[14, :3]
    l_wrist = nf[15, :3]
    r_wrist = nf[16, :3]
    l_hip = nf[23, :3]
    r_hip = nf[24, :3]
    l_ankle = nf[27, :3]
    r_ankle = nf[28, :3]
    grip = nf[33, :3]
    tip = nf[34, :3]
    
    features = {}
    
    # 1. Stick angle vs forearm
    stick_vec = tip - grip
    stick_len = np.linalg.norm(stick_vec) + 1e-6
    stick_vec = stick_vec / stick_len
    
    # Determine which hand holds stick by proximity
    dist_l = np.linalg.norm(grip - l_wrist)
    dist_r = np.linalg.norm(grip - r_wrist)
    if dist_r < dist_l:
        forearm_vec = r_wrist - r_elbow
    else:
        forearm_vec = l_wrist - l_elbow
    
    forearm_len = np.linalg.norm(forearm_vec) + 1e-6
    if forearm_len > 0.01:
        forearm_vec = forearm_vec / forearm_len
        dot = np.dot(forearm_vec, stick_vec)
        features['stick_forearm_angle'] = np.degrees(np.arccos(np.clip(abs(dot), 0, 1)))
    else:
        features['stick_forearm_angle'] = 90.0
    
    # 2. Tip-to-nose distance
    features['tip_nose_dist_3d'] = np.linalg.norm(tip - nose)
    
    # 3. Tip-to-chest distance
    chest_center = (l_shoulder + r_shoulder) / 2
    features['tip_chest_dist_3d'] = np.linalg.norm(tip - chest_center)
    
    # 4. Body rotation
    hip_vec = np.array([r_hip[0] - l_hip[0], r_hip[1] - l_hip[1]])
    shoulder_vec = np.array([r_shoulder[0] - l_shoulder[0], r_shoulder[1] - l_shoulder[1]])
    hip_len = np.linalg.norm(hip_vec) + 1e-6
    shoulder_len = np.linalg.norm(shoulder_vec) + 1e-6
    if hip_len > 0.01 and shoulder_len > 0.01:
        twist_dot = np.dot(hip_vec / hip_len, shoulder_vec / shoulder_len)
        features['body_rotation'] = np.degrees(np.arccos(np.clip(abs(twist_dot), 0, 1)))
    else:
        features['body_rotation'] = 0.0
    
    # 5. Stance width
    features['stance_width'] = np.linalg.norm(r_ankle - l_ankle)
    
    # 6. Stick-to-forearm ratio
    if dist_r < dist_l:
        forearm_len = np.linalg.norm(r_wrist - r_elbow)
    else:
        forearm_len = np.linalg.norm(l_wrist - l_elbow)
    features['stick_forearm_ratio'] = stick_len / (forearm_len + 1e-6)
    
    return features


def compute_enhanced_hybrid_features(node_features, templates, class_name, viewpoint='front'):
    """Compute enhanced hybrid features for a single sample."""
    key = f"{viewpoint}_{class_name}"
    
    if key not in templates:
        return np.zeros(6)
    
    template = templates[key]
    raw_features = compute_enhanced_features_from_nodes(node_features)
    
    hybrid = []
    for feat_name in ['stick_forearm_angle', 'tip_nose_dist_3d', 'tip_chest_dist_3d',
                      'body_rotation', 'stance_width', 'stick_forearm_ratio']:
        if feat_name in template and feat_name in raw_features:
            mean = template[feat_name]['mean']
            std = template[feat_name]['std']
            hybrid.append(gaussian_similarity(raw_features[feat_name], mean, std))
        else:
            hybrid.append(0.0)
    
    return np.array(hybrid, dtype=np.float32)


def postprocess_dataset(input_path, output_path, templates, viewpoint='front'):
    """Add enhanced hybrid features to existing .pt file."""
    print(f"\nLoading: {input_path}")
    data = torch.load(input_path, map_location='cpu')
    
    node_features = data['node_features']
    existing_hybrid = data['hybrid_features']
    labels = data['labels']
    
    print(f"  Samples: {len(labels)}")
    print(f"  Existing hybrid dims: {existing_hybrid.shape[1]}")
    
    # Compute enhanced hybrid features for each sample
    enhanced_list = []
    for i in tqdm(range(len(labels)), desc="Computing enhanced features"):
        class_name = CLASS_NAMES[int(labels[i])]
        enh = compute_enhanced_hybrid_features(node_features[i], templates, class_name, viewpoint)
        enhanced_list.append(torch.tensor(enh, dtype=torch.float32))
    
    enhanced_hybrid = torch.stack(enhanced_list)
    print(f"  Enhanced hybrid dims: {enhanced_hybrid.shape[1]}")
    
    # Concatenate: original hybrid + enhanced hybrid
    combined_hybrid = torch.cat([existing_hybrid, enhanced_hybrid], dim=1)
    print(f"  Combined hybrid dims: {combined_hybrid.shape[1]}")
    
    # Save with all original data
    output_data = {
        'node_features': node_features,
        'hybrid_features': combined_hybrid,
        'labels': labels,
        'viewpoints': data.get('viewpoints', ['front'] * len(labels)),
        'has_stick_nodes': data.get('has_stick_nodes', torch.ones(len(labels), dtype=torch.bool)),
        'stick_right_hand': data.get('stick_right_hand', torch.ones(len(labels), dtype=torch.bool)),
        'is_enhanced': True,
        'enhanced_dims': enhanced_hybrid.shape[1]
    }
    
    torch.save(output_data, output_path)
    print(f"  Saved: {output_path}")
    print(f"  Node shape: {node_features.shape}")
    print(f"  Hybrid shape: {combined_hybrid.shape}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewpoint', type=str, default='front', choices=['front', 'left', 'right'])
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"Post-processing enhanced hybrid features")
    print(f"Viewpoint: {args.viewpoint}")
    print(f"{'='*60}")
    
    # Load enhanced templates
    with open(FEATURE_TEMPLATES, 'r') as f:
        templates = json.load(f)
    print(f"Loaded {len(templates)} enhanced templates")
    
    # Process train
    train_input = INPUT_DIR / f"train_features_{args.viewpoint}.pt"
    train_output = OUTPUT_DIR / f"train_features_{args.viewpoint}_enhanced.pt"
    if train_input.exists():
        postprocess_dataset(train_input, train_output, templates, args.viewpoint)
    else:
        print(f"WARNING: {train_input} not found")
    
    # Process test
    test_input = INPUT_DIR / f"test_features_{args.viewpoint}.pt"
    test_output = OUTPUT_DIR / f"test_features_{args.viewpoint}_enhanced.pt"
    if test_input.exists():
        postprocess_dataset(test_input, test_output, templates, args.viewpoint)
    else:
        print(f"WARNING: {test_input} not found")
    
    print(f"\n{'='*60}")
    print("Done! Enhanced features created without re-processing images.")
    print(f"{'='*60}")
