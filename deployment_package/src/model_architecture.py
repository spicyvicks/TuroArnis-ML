"""
Hybrid GCN V2 Model Architecture
Optimized for CPU inference with node-specific features + global context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]

SKELETON_EDGES = [
    (11, 12), (12, 11), (11, 23), (23, 11), (12, 24), (24, 12),
    (23, 24), (24, 23), (11, 13), (13, 11), (13, 15), (15, 13),
    (12, 14), (14, 12), (14, 16), (16, 14), (23, 25), (25, 23),
    (25, 27), (27, 25), (24, 26), (26, 24), (26, 28), (28, 26),
    (15, 33), (33, 15), (16, 33), (33, 16), (33, 34), (34, 33)
]

class HybridGCN(nn.Module):
    """
    GCN with Node-Specific Features + Global Hybrid Context
    - GCN processes spatial node features
    - Global hybrid features provide expert knowledge
    - Both are combined for final classification
    """
    def __init__(self, node_in_channels, hybrid_in_channels, hidden_channels, num_classes, num_layers=3, dropout=0.5, embedding_dim=8):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        
        # Node identity embeddings (35 nodes: 0-34)
        self.node_embedding = nn.Embedding(35, embedding_dim)
        
        # GCN layers for node features (geometric + embedding)
        gcn_input_dim = node_in_channels + embedding_dim
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer (geometric features + node embeddings)
        self.convs.append(GCNConv(gcn_input_dim, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # MLP for global hybrid features
        self.hybrid_fc1 = nn.Linear(hybrid_in_channels, hidden_channels)
        self.hybrid_bn1 = nn.BatchNorm1d(hidden_channels)
        self.hybrid_fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.hybrid_bn2 = nn.BatchNorm1d(hidden_channels)
        
        # Fusion layer (combines GCN output + hybrid features)
        self.fusion_fc = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fusion_bn = nn.BatchNorm1d(hidden_channels)
        
        # Classification head
        self.fc = nn.Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, batch, hybrid_features):
        # Generate node indices (0-34 for each graph in batch)
        num_nodes_per_graph = 35
        num_graphs = batch.max().item() + 1
        node_indices = torch.arange(num_nodes_per_graph, device=x.device).repeat(num_graphs)
        
        # Get node embeddings and concatenate with geometric features
        node_emb = self.node_embedding(node_indices)
        x = torch.cat([x, node_emb], dim=1)
        
        # Process node features with GCN
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling (graph-level representation)
        x_graph = global_mean_pool(x, batch)
        
        # Process global hybrid features
        h = self.hybrid_fc1(hybrid_features)
        h = self.hybrid_bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = self.hybrid_fc2(h)
        h = self.hybrid_bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Fusion: concatenate GCN output + hybrid features
        combined = torch.cat([x_graph, h], dim=1)
        combined = self.fusion_fc(combined)
        combined = self.fusion_bn(combined)
        combined = F.relu(combined)
        combined = F.dropout(combined, p=self.dropout, training=self.training)
        
        # Classification
        out = self.fc(combined)
        return out
