"""
2a_generate_enhanced_hybrid_features.py
==========================================
Generate node features + ENHANCED hybrid features using enhanced templates.
Uses same node features as 2b, but computes hybrid features against
feature_templates_enhanced.json (with 6 additional geometric features).

Output: hybrid_features_v3/*_enhanced.pt
Does NOT overwrite existing *_front.pt files.
"""

import sys
import importlib.util
from pathlib import Path

# Load 2b module dynamically (filename starts with number, can't import normally)
spec = importlib.util.spec_from_file_location(
    "_mod_2b",
    str(Path(__file__).parent / "2b_generate_node_hybrid_features.py")
)
_mod_2b = importlib.util.module_from_spec(spec)
sys.modules["_mod_2b"] = _mod_2b
spec.loader.exec_module(_mod_2b)

# Re-use functions from 2b
extract_raw_features = _mod_2b.extract_raw_features
extract_node_features = _mod_2b.extract_node_features
gaussian_similarity = _mod_2b.gaussian_similarity

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import json
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Config - DISTINCT filenames
FEATURE_TEMPLATES = "hybrid_classifier/feature_templates_enhanced.json"
DATASET_ROOT = Path("dataset_split")
OUTPUT_DIR = Path("hybrid_classifier/hybrid_features_v3")
STICK_MODEL = "runs/pose/stick_detector_20260425_212025/weights/best.pt"

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct',
    'neutral'
]


def compute_enhanced_hybrid_features(raw_features, templates, viewpoint, class_name):
    """Convert raw geometric features to similarity scores using ENHANCED templates."""
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


def process_single_image_enhanced(args):
    """Process single image with enhanced hybrid features."""
    img_path, class_idx, viewpoint, templates, stick_detector = args
    
    try:
        raw_data = extract_raw_features(img_path, stick_detector, viewpoint=viewpoint, class_idx=class_idx)
        if raw_data is None:
            return None
        
        node_features = extract_node_features(
            raw_data['pose_keypoints'],
            raw_data['stick_keypoints'],
            include_stick=True
        )
        
        class_name = CLASS_NAMES[class_idx]
        hybrid_features = compute_enhanced_hybrid_features(
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
            'has_stick_nodes': True,
            'stick_right_hand': raw_data.get('stick_right_hand', True)
        }
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def process_dataset_enhanced(viewpoint_filter=None, num_workers=None):
    """Process all images and generate enhanced features"""
    with open(FEATURE_TEMPLATES, 'r') as f:
        templates = json.load(f)
    
    print(f"Loaded {len(templates)} ENHANCED feature templates")
    print(f"Features per template: ~{len(next(iter(templates.values())))}")
    
    stick_detector = YOLO(STICK_MODEL)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    splits = ['train', 'test']
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} set with ENHANCED features")
        print(f"{'='*60}")
        
        viewpoint_dir = DATASET_ROOT / split
        if viewpoint_filter:
            viewpoints = [viewpoint_filter]
        else:
            viewpoints = ['front', 'left', 'right']
        
        all_features = []
        
        for viewpoint in viewpoints:
            vp_dir = viewpoint_dir / viewpoint
            if not vp_dir.exists():
                continue
            
            for class_idx, class_name in enumerate(CLASS_NAMES):
                class_dir = vp_dir / class_name
                if not class_dir.exists():
                    continue
                
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                
                args_list = [(img, class_idx, viewpoint, templates, stick_detector) for img in images]
                
                if num_workers is None:
                    num_workers = max(1, cpu_count() - 1)
                
                with Pool(num_workers) as pool:
                    results = list(tqdm(
                        pool.imap(process_single_image_enhanced, args_list),
                        total=len(args_list),
                        desc=f"{viewpoint}/{class_name}"
                    ))
                
                valid_results = [r for r in results if r is not None]
                all_features.extend(valid_results)
                print(f"  {viewpoint}/{class_name}: {len(valid_results)}/{len(images)} accepted")
        
        if not all_features:
            print(f"No features extracted for {split}")
            continue
        
        node_features = torch.stack([torch.tensor(f['node_features'], dtype=torch.float32) for f in all_features])
        hybrid_features = torch.stack([torch.tensor(f['hybrid_features'], dtype=torch.float32) for f in all_features])
        labels = torch.tensor([f['label'] for f in all_features], dtype=torch.long)
        viewpoints_list = [f['viewpoint'] for f in all_features]
        has_stick = torch.tensor([f['has_stick_nodes'] for f in all_features], dtype=torch.bool)
        stick_right = torch.tensor([f['stick_right_hand'] for f in all_features], dtype=torch.bool)
        
        if viewpoint_filter:
            output_path = OUTPUT_DIR / f"{split}_features_{viewpoint_filter}_enhanced.pt"
        else:
            output_path = OUTPUT_DIR / f"{split}_features_enhanced.pt"
        
        torch.save({
            'node_features': node_features,
            'hybrid_features': hybrid_features,
            'labels': labels,
            'viewpoints': viewpoints_list,
            'has_stick_nodes': has_stick,
            'stick_right_hand': stick_right,
            'is_enhanced': True
        }, output_path)
        
        print(f"\nSaved {split} features: {output_path}")
        print(f"  Samples: {len(labels)}")
        print(f"  Node shape: {node_features.shape}")
        print(f"  Hybrid shape: {hybrid_features.shape} (enhanced)")
        print(f"  Classes: {torch.bincount(labels)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewpoint', type=str, default=None, choices=['front', 'left', 'right'])
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()
    
    process_dataset_enhanced(viewpoint_filter=args.viewpoint, num_workers=args.workers)
