import torch
from torch_geometric.data import Data
import numpy as np

class ArnisGraphBuilder:
    def __init__(self):
        # 1. Define Nodes (17 Body + 2 Weapon = 19 Nodes)
        # Body indices match COCO format (0-16)
        # 17: Grip, 18: Tip
        self.num_nodes = 19
        self.body_indices = list(range(17))
        self.grip_idx = 17
        self.tip_idx = 18
        
        # 2. Define Edges (Undirected)
        # (source, target) pairs
        self.skeleton_edges = [
            (0, 1), (0, 2), (1, 3), (2, 4), # Face
            (5, 6), (5, 7), (7, 9), # Left Arm
            (6, 8), (8, 10), # Right Arm
            (11, 12), (11, 13), (13, 15), # Left Leg
            (12, 14), (14, 16), # Right Leg
            (5, 11), (6, 12) # Torso
        ]
        
        self.weapon_edges = [
            (17, 18) # Grip to Tip (The Stick)
        ]
        
        # We need to dynamically determine which hand holds the stick.
        # But the graph topology should ideally be fixed or have a "virtual" edge.
        # For this implementation, we will connect Grip to BOTH wrists for semantic context,
        # or use the 'holding_hand' feature if available to weight them.
        # For simplicity in MVP GAT, we add edges to both wrists.
        self.attachment_edges = [
            (9, 17), # Left Wrist -> Grip
            (10, 17) # Right Wrist -> Grip
        ]
        
        # Semantic Edges (Optional, for Arnis specific moves)
        # e.g., Tip to Target areas (Head, Chest, etc.)
        # For MVP, we can simulate spatial attention by fully connecting Tip to Body? 
        # No, let's keep it sparse for GAT.
        
        self.edges = self.skeleton_edges + self.weapon_edges + self.attachment_edges
        
        # Make edges bidirectional
        self.edge_index = self._build_edge_index(self.edges)
        
    def _build_edge_index(self, edges):
        src = [s for s, t in edges] + [t for s, t in edges]
        dst = [t for s, t in edges] + [s for s, t in edges]
        return torch.tensor([src, dst], dtype=torch.long)

    def compute_edge_attributes(self, x, edge_index):
        """
        Computes edge attributes: [distance, sin(angle), cos(angle), rigid_flag]
        """
        num_edges = edge_index.shape[1]
        row, col = edge_index
        
        # Calculate displacement vectors
        diff = x[col] - x[row]
        
        # Distance (Euclidean)
        dist = torch.norm(diff, p=2, dim=-1).view(-1, 1)
        
        # Angles (relative to x-axis)
        # eps to avoid div by zero
        eps = 1e-8
        sin_angle = (diff[:, 1] / (dist[:, 0] + eps)).view(-1, 1)
        cos_angle = (diff[:, 0] / (dist[:, 0] + eps)).view(-1, 1)
        
        # Rigid flag (1 for bone/stick connections, 0 for virtual/attachment)
        # This is a bit complex to map back to edge list efficiently without a map.
        # For MVP, we can just use 1.0 for all for now, or approximate.
        # Let's use 1.0 for all as 'connected'.
        rigid = torch.ones((num_edges, 1), dtype=torch.float32)
        
        edge_attr = torch.cat([dist, sin_angle, cos_angle, rigid], dim=-1)
        return edge_attr

    def normalize_nodes(self, nodes):
        """
        Normalizes node coordinates by torso scale.
        """
        # Torso height: Avg(Shoulders) to Avg(Hips)
        # Shoulders: 5, 6. Hips: 11, 12
        shoulders = (nodes[5] + nodes[6]) / 2.0
        hips = (nodes[11] + nodes[12]) / 2.0
        
        torso_center = (shoulders + hips) / 2.0
        torso_height = np.linalg.norm(shoulders - hips)
        
        if torso_height == 0:
            scale = 1.0
        else:
            scale = torso_height
            
        nodes_centered = (nodes - torso_center) / scale
        
        return torch.tensor(nodes_centered, dtype=torch.float32)

    def build_graph(self, body_keypoints, weapon_keypoints, label=None):
        """
        Constructs a PyG Data object.
        
        Args:
            body_keypoints: (17, 2) or (17, 3) numpy array
            weapon_keypoints: (2, 2) or (2, 3) numpy array [Grip, Tip]
            label: (int or None) Class label
            
        Returns:
            data: PyG Data object
        """
        # Concatenate nodes
        nodes = np.vstack([body_keypoints, weapon_keypoints])
        
        # Normalize
        x = self.normalize_nodes(nodes)
        
        # Compute Edge Attributes
        edge_attr = self.compute_edge_attributes(x, self.edge_index)
        
        # Label
        y = torch.tensor([label], dtype=torch.long) if label is not None else None
        
        data = Data(x=x, edge_index=self.edge_index, edge_attr=edge_attr, y=y)
        return data
