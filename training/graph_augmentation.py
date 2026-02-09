"""
Graph Augmentation Module for Spatial GCN
Conservative augmentation to avoid overfitting (learned from XGBoost experiment)
"""

import torch
import copy


def augment_spatial_jitter(graph, noise_std=0.02):
    """
    Add Gaussian noise to keypoint coordinates.
    Simulates MediaPipe detection variance.
    
    Args:
        graph: PyG Data object
        noise_std: Standard deviation of noise (default: 0.02)
    
    Returns:
        Augmented graph
    """
    aug_graph = copy.deepcopy(graph)
    
    # Add noise to x, y coordinates only (not confidence/visibility)
    noise = torch.randn_like(aug_graph.x[:, :2]) * noise_std
    aug_graph.x[:, :2] = aug_graph.x[:, :2] + noise
    
    # Clip to valid range [0, 1]
    aug_graph.x[:, :2] = torch.clamp(aug_graph.x[:, :2], 0.0, 1.0)
    
    return aug_graph


def augment_node_dropout(graph, drop_prob=0.1):
    """
    Randomly zero out keypoints to simulate occlusion.
    
    Args:
        graph: PyG Data object
        drop_prob: Probability of dropping each node (default: 0.1)
    
    Returns:
        Augmented graph
    """
    aug_graph = copy.deepcopy(graph)
    
    # Create dropout mask (don't drop stick nodes 33, 34)
    num_pose_nodes = 33
    mask = torch.rand(num_pose_nodes) > drop_prob
    
    # Zero out dropped nodes
    aug_graph.x[:num_pose_nodes][~mask] = 0.0
    
    return aug_graph


def augment_stick_dropout(graph, drop_prob=0.3):
    """
    Randomly remove stick nodes to simulate detection failures.
    Forces model to handle missing stick information.
    
    Args:
        graph: PyG Data object
        drop_prob: Probability of dropping stick (default: 0.3)
    
    Returns:
        Augmented graph
    """
    aug_graph = copy.deepcopy(graph)
    
    # Randomly drop stick nodes (indices 33, 34)
    if torch.rand(1).item() < drop_prob:
        aug_graph.x[33:35] = 0.0
    
    return aug_graph


def augment_graph_conservative(graph):
    """
    Apply conservative augmentation pipeline.
    Creates 3 augmented variants (4x total with original).
    
    Args:
        graph: PyG Data object
    
    Returns:
        List of augmented graphs [original, jitter, dropout, stick_dropout]
    """
    augmented = [graph]  # Original
    
    # Apply ONE augmentation per variant (not stacked)
    augmented.append(augment_spatial_jitter(graph))
    augmented.append(augment_node_dropout(graph))
    augmented.append(augment_stick_dropout(graph))
    
    return augmented


def augment_graph_moderate(graph):
    """
    Apply moderate augmentation pipeline (if conservative underperforms).
    Creates 4 augmented variants (5x total with original).
    
    Args:
        graph: PyG Data object
    
    Returns:
        List of augmented graphs
    """
    augmented = [graph]  # Original
    
    # Single augmentations
    augmented.append(augment_spatial_jitter(graph))
    augmented.append(augment_node_dropout(graph))
    
    # Combined augmentations
    aug = augment_spatial_jitter(graph)
    augmented.append(augment_stick_dropout(aug))
    
    aug = augment_node_dropout(graph)
    augmented.append(augment_stick_dropout(aug))
    
    return augmented


if __name__ == "__main__":
    # Test augmentation functions
    from torch_geometric.data import Data
    
    # Create dummy graph
    x = torch.rand(35, 3)  # 35 nodes, 3 features
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
    y = torch.tensor([0])
    
    graph = Data(x=x, edge_index=edge_index, y=y)
    
    # Test conservative augmentation
    augmented = augment_graph_conservative(graph)
    print(f"Conservative augmentation: {len(augmented)} graphs")
    
    # Test moderate augmentation
    augmented = augment_graph_moderate(graph)
    print(f"Moderate augmentation: {len(augmented)} graphs")
    
    print("✓ Augmentation tests passed")
