"""
V5 Deployment Inference Wrapper
High-level interface for running v5 model inference in the TuroArnis app.

Usage:
    from deployment_package.src.inference_v5 import V5Inference
    
    inf = V5Inference(
        model_path='deployment_package/models/model_front_v5_deploy.pth',
        templates_path='hybrid_classifier/feature_templates.json',
        stick_model_path='runs/pose/stick_detector_20260425_212025/weights/best.pt',
        device='cpu'
    )
    
    image = cv2.imread('test.jpg')
    result = inf.predict(image, viewpoint='front')
    # result = {'class': 'crown_thrust_correct', 'confidence': 0.92, 'all_probs': [...]}
"""

import numpy as np
import torch
from pathlib import Path
import json

from .model_v5 import HybridGCN, load_deployment_model, SKELETON_EDGES
from .feature_extraction_v5 import extract_raw_features, compute_hybrid_features, extract_node_features


class V5Inference:
    """V5 model inference wrapper for single-image classification."""
    
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
        
        # Build hybrid features using all class templates (per-class comparison)
        # For inference, we compute hybrid features against each class template
        # and the model chooses the best match
        hybrid_features_per_class = []
        for class_name in self.class_names:
            hf = compute_hybrid_features(
                raw_data['global_features'],
                self.templates,
                viewpoint,
                class_name
            )
            hybrid_features_per_class.append(hf)
        
        # Stack to [num_classes, num_hybrid_features]
        hybrid_tensor = torch.from_numpy(
            np.stack(hybrid_features_per_class, axis=0)
        ).float().to(self.device)
        
        # Build node features
        node_features = extract_node_features(
            raw_data['pose_keypoints'],
            raw_data['stick_keypoints']
        )
        node_tensor = torch.from_numpy(node_features).float().to(self.device)
        
        # Build PyG Data object for each class
        from torch_geometric.data import Data
        
        graphs = []
        for i in range(len(self.class_names)):
            # Use same node features and edges for all classes,
            # only hybrid features differ (per-class template comparison)
            graphs.append(Data(
                x=node_tensor,
                edge_index=self.edge_index.to(self.device),
                hybrid_features=hybrid_tensor[i],
                y=torch.tensor([i], device=self.device)
            ))
        
        # Run inference
        with torch.no_grad():
            # Batch all class comparisons
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


def create_single_class_graph(node_features, hybrid_features, edge_index, device='cpu'):
    """
    Create a PyG Data object for a single graph.
    
    Args:
        node_features: [35, 6] numpy array
        hybrid_features: [46] numpy array
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
        y=torch.tensor([0], device=device)
    )
