"""
Left Viewpoint Inference Example
Demonstrates loading and running left-viewpoint models.
"""

import sys
from pathlib import Path

# Add paths for imports
script_dir = Path(__file__).parent
left_dir = script_dir.parent  # deployment_package/left/
repo_root = left_dir.parent.parent  # repo root
# Add lower-priority paths first
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'deployment_package' / 'src'))
# Add local script_dir LAST so it has highest priority (index 0)
sys.path.insert(0, str(script_dir))

import torch
import json

# For v6 inference (recommended for production)
from model_v6 import load_deployment_model, SKELETON_EDGES

# Feature extraction imports
from feature_extraction_v6 import (
    extract_raw_features, compute_hybrid_features, extract_node_features, create_node_mask
)

# For v5 inference (if needed for higher accuracy)
# from model_v5 import load_deployment_model, SKELETON_EDGES
# from feature_extraction_v5 import ...

import numpy as np


class LeftViewpointInference:
    """Inference wrapper for left-viewpoint models."""
    
    CLASS_NAMES = [
        'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
        'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
        'right_chest_thrust_correct', 'right_elbow_block_correct',
        'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
        'solar_plexus_thrust_correct', 'neutral'
    ]
    
    def __init__(self, model_path, templates_path, stick_model_path=None, device='cpu'):
        self.device = torch.device(device)
        
        # Load model
        self.model, self.class_names, self.config = load_deployment_model(
            model_path, device=self.device
        )
        
        # Load templates
        with open(templates_path) as f:
            self.templates = json.load(f)
        
        # Load stick detector (optional for inference, can be None if stick data provided)
        self.stick_model = None
        if stick_model_path and Path(stick_model_path).exists():
            from ultralytics import YOLO
            self.stick_model = YOLO(stick_model_path)
    
    def predict(self, image, pose_results=None, stick_results=None):
        """
        Run inference on a single image.
        
        Args:
            image: numpy array (BGR image)
            pose_results: Optional pre-computed MediaPipe pose results
            stick_results: Optional pre-computed YOLO stick detection results
            
        Returns:
            dict with 'class', 'confidence', 'all_probabilities'
        """
        # TODO: Implement full preprocessing pipeline
        # This is a placeholder showing the expected interface
        raise NotImplementedError(
            "Full inference pipeline not yet implemented. "
            "Use hybrid_classifier/6_evaluate_left_models.py as reference."
        )


def demo_load_models():
    """Demonstrate loading both left viewpoint models."""
    base_dir = Path(__file__).parent.parent  # deployment_package/left/
    
    print("="*60)
    print("LEFT VIEWPOINT MODEL LOADING DEMO")
    print("="*60)
    
    # Load v6 standard (production model)
    print("\n1. Loading v6 standard model (production)...")
    v6_model_path = base_dir / "models" / "model_left_v6_standard.pth"
    v6_templates_path = base_dir / "src" / "feature_templates.json"
    
    if v6_model_path.exists():
        model_v6, _, config_v6 = load_deployment_model(v6_model_path, device='cpu')
        print(f"   OK: v6 model loaded")
        print(f"   Hidden dim: {config_v6.get('hidden_dim', 'N/A')}")
        print(f"   Num layers: {config_v6.get('num_layers', 'N/A')}")
        print(f"   Dropout: {config_v6.get('dropout', 'N/A')}")
    else:
        print(f"   SKIP: {v6_model_path} not found")
    
    # Load v5 mirrored (best accuracy)
    print("\n2. Loading v5 mirrored model (best accuracy)...")
    v5_model_path = base_dir / "models" / "model_left_v5_mirrored.pth"
    v5_templates_path = base_dir / "src" / "feature_templates_mirrored.json"
    
    if v5_model_path.exists():
        from model_v5 import load_deployment_model as load_v5
        model_v5, _, config_v5 = load_v5(v5_model_path, device='cpu')
        print(f"   OK: v5 model loaded")
        print(f"   Hidden dim: {config_v5.get('hidden_dim', 'N/A')}")
        print(f"   Num layers: {config_v5.get('num_layers', 'N/A')}")
        print(f"   Dropout: {config_v5.get('dropout', 'N/A')}")
    else:
        print(f"   SKIP: {v5_model_path} not found")
    
    # Check templates
    print("\n3. Checking templates...")
    if v6_templates_path.exists():
        with open(v6_templates_path) as f:
            templates = json.load(f)
        print(f"   Standard templates: {len(templates)} keys")
        for k in sorted(templates.keys())[:3]:
            print(f"     {k}")
    
    if v5_templates_path.exists():
        with open(v5_templates_path) as f:
            templates = json.load(f)
        print(f"   Mirrored templates: {len(templates)} keys")
        for k in sorted(templates.keys())[:3]:
            print(f"     {k}")
    
    print("\n" + "="*60)
    print("All left viewpoint models loaded successfully!")
    print("="*60)


if __name__ == '__main__':
    demo_load_models()
