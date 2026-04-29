"""
V6 Deployment Inference Wrapper
High-level interface for running v6 model inference in the TuroArnis app.

V6 Key Differences from V5:
- node_mask is REQUIRED in forward pass (for masked global_mean_pool)
- True zero-stick fallback (not origin-based)
- 49 hybrid features (includes has_stick binary)

Usage:
    from deployment_package.src.inference_v6 import V6Inference
    
    inf = V6Inference(
        model_path='deployment_package/models/model_front_v6_deploy.pth',
        templates_path='hybrid_classifier/feature_templates.json',
        stick_model_path='runs/pose/stick_detector_20260425_212025/weights/best.pt',
        device='cpu'
    )
    
    image = cv2.imread('test.jpg')
    result = inf.predict(image, viewpoint='front')
    # result = {'class': 'crown_thrust_correct', 'confidence': 0.88, ...}
"""

import numpy as np
import torch
from pathlib import Path
import json

from .model_v6 import HybridGCN, load_deployment_model, SKELETON_EDGES
from .feature_extraction_v6 import (
    extract_raw_features, compute_hybrid_features, extract_node_features, create_node_mask
)


class V6Inference:
    """V6 model inference wrapper for single-image classification."""
    
    CLASS_NAMES = [
        'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
        'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
        'right_chest_thrust_correct', 'right_elbow_block_correct',
        'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
        'solar_plexus_thrust_correct', 'neutral'
    ]
    
    def __init__(self, model_path, templates_path, stick_model_path, device='cpu'):
        self.device = torch.device(device)
        
        # Load model
        self.model, self.class_names, self.config = load_deployment_model(
            model_path, device=self.device
        )
        
        # Load templates
        with open(templates_path) as f:
            self.templates = json.load(f)
        
        # Load stick detector
        from ultralytics import YOLO
        self.stick_detector = YOLO(stick_model_path)
        
        # Precompute edge_index
        self.edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t().contiguous()
    
    def predict(self, image, viewpoint='front', top_k=3):
        """
        Run inference on a single image.
        
        Args:
            image: numpy array (BGR) or path string
            viewpoint: 'front', 'left', or 'right'
            top_k: number of top predictions to return
        
        Returns:
            dict with 'class', 'confidence', 'all_probs', 'top_k'
        """
        # Extract features
        raw_data = extract_raw_features(image, self.stick_detector)
        if raw_data is None:
            return {'class': None, 'confidence': 0.0, 'error': 'No pose detected'}
        
        # Build hybrid features using all class templates
        hybrid_features_per_class = []
        for class_name in self.class_names:
            hf = compute_hybrid_features(
                raw_data['global_features'],
                self.templates,
                viewpoint,
                class_name
            )
            hybrid_features_per_class.append(hf)
        
        hybrid_tensor = torch.from_numpy(
            np.stack(hybrid_features_per_class, axis=0)
        ).float().to(self.device)
        
        # Build node features (7-dim, true zeros, has_stick as 7th dimension)
        node_features = extract_node_features(
            raw_data['pose_keypoints'],
            raw_data['stick_keypoints'],
            has_stick_detected=raw_data['has_stick_detected']
        )
        node_tensor = torch.from_numpy(node_features).float().to(self.device)
        
        # Build node_mask (V6: mask out stick nodes if not detected)
        node_mask_np = create_node_mask(raw_data['has_stick_detected'])
        node_mask_tensor = torch.from_numpy(node_mask_np).float().to(self.device)
        
        # Build PyG Data object for each class
        from torch_geometric.data import Data
        
        graphs = []
        for i in range(len(self.class_names)):
            graphs.append(Data(
                x=node_tensor,
                edge_index=self.edge_index.to(self.device),
                hybrid_features=hybrid_tensor[i],
                y=torch.tensor([i], device=self.device),
                node_mask=node_mask_tensor  # V6: required for masked pooling
            ))
        
        # Run inference
        with torch.no_grad():
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(graphs)
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=-1)
        
        # Get top prediction
        top_probs, top_indices = torch.topk(probs[0], top_k)
        
        return {
            'class': self.class_names[top_indices[0].item()],
            'confidence': top_probs[0].item(),
            'all_probs': {name: prob.item() for name, prob in zip(self.class_names, probs[0])},
            'top_k': [
                {'class': self.class_names[idx.item()], 'confidence': prob.item()}
                for prob, idx in zip(top_probs, top_indices)
            ]
        }


def create_single_class_graph(node_features, hybrid_features, node_mask, edge_index, device='cpu'):
    """
    Create a PyG Data object for a single graph (V6 with node_mask).
    
    Args:
        node_features: [35, 7] numpy array
        hybrid_features: [49] numpy array
        node_mask: [35] numpy array (1.0 for valid, 0.0 for masked)
        edge_index: [2, num_edges] torch tensor
        device: torch device
    
    Returns:
        PyG Data object
    """
    from torch_geometric.data import Data
    
    return Data(
        x=torch.from_numpy(node_features).float().to(device),
        edge_index=edge_index.to(device),
        hybrid_features=torch.from_numpy(hybrid_features).float().to(device),
        y=torch.tensor([0], device=device),
        node_mask=torch.from_numpy(node_mask).float().to(device)
    )
