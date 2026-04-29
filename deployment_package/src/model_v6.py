"""
HybridGCN V6 Deployment Model Architecture
Exact match to training model from 4e_train_hybrid_gcn_v2_with_synthetic_v6.py
- 6-dim node features
- Masked global_mean_pool using node_mask + global_add_pool
- 49 hybrid features (48 base/signed + 1 has_stick)
- 13 classes (including neutral)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool


class HybridGCN(nn.Module):
    """
    GCN with Node-Specific Features + Global Hybrid Context + Node Masking
    - GCN processes spatial node features with per-node masking
    - Missing stick nodes (zero-stick fallback) are masked out of pooling
    - Global hybrid features provide expert knowledge
    - Both are combined for final classification
    """
    def __init__(self, num_node_features, num_hybrid_features, num_classes=13,
                 hidden_dim=128, num_layers=3, dropout=0.5, node_embed_dim=8):
        super(HybridGCN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.node_embed_dim = node_embed_dim
        
        # Node identity embeddings (35 nodes: 0-34)
        self.node_embedding = nn.Embedding(35, node_embed_dim)
        
        # GCN layers for node features (geometric + embedding)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features + node_embed_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim, track_running_stats=False))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim, track_running_stats=False))
        
        # MLP for global hybrid features
        self.hybrid_mlp = nn.Sequential(
            nn.Linear(num_hybrid_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Fusion and classification
        self.fc1 = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, data):
        """
        Args:
            data: PyG Data object with:
                - x: [num_nodes, num_node_features] node features
                - edge_index: [2, num_edges] graph connectivity
                - batch: [num_nodes] batch assignment
                - hybrid_features: [batch_size, num_hybrid_features] global features
                - node_mask: [num_nodes] 1.0 for valid nodes, 0.0 for masked nodes
        
        Returns:
            logits: [batch_size, num_classes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        hybrid_features = data.hybrid_features
        node_mask = data.node_mask
        
        batch_size = batch.max().item() + 1
        
        # Generate node indices (0-34 for each graph in batch)
        node_indices = torch.arange(35, device=x.device).unsqueeze(0).expand(batch_size, -1)
        node_emb = self.node_embedding(node_indices).view(-1, self.node_embed_dim)
        x = torch.cat([x, node_emb], dim=-1)
        
        # Process node features with GCN
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout_layer(x_new)
            if x_new.size(-1) == x.size(-1):
                x = x_new + x
            else:
                x = x_new
        
        # Masked global mean pool: zero out missing nodes, divide by valid count
        x = x * node_mask.unsqueeze(-1)
        x_sum = global_add_pool(x, batch)
        mask_sum = global_add_pool(node_mask.unsqueeze(-1), batch)
        x_pool = x_sum / (mask_sum + 1e-8)
        
        # Process global hybrid features
        hybrid_out = self.hybrid_mlp(hybrid_features)
        
        # Fusion
        combined = torch.cat([x_pool, hybrid_out], dim=-1)
        x_out = F.relu(self.fc1(combined))
        x_out = self.dropout_layer(x_out)
        logits = self.fc2(x_out)
        return logits


def load_deployment_model(checkpoint_path, device='cpu'):
    """Load deployment model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt['config']
    
    model = HybridGCN(
        num_node_features=config['num_node_features'],
        num_hybrid_features=config['num_hybrid_features'],
        num_classes=config['num_classes'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        node_embed_dim=config['node_embed_dim']
    ).to(device)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    return model, ckpt.get('class_names', []), config


# Skeleton edges for Arnis pose graph
SKELETON_EDGES = [
    (11, 12), (12, 11), (11, 23), (23, 11), (12, 24), (24, 12),
    (23, 24), (24, 23), (11, 13), (13, 11), (13, 15), (15, 13),
    (12, 14), (14, 12), (14, 16), (16, 14), (23, 25), (25, 23),
    (25, 27), (27, 25), (24, 26), (26, 24), (26, 28), (28, 26),
    (15, 33), (33, 15), (16, 33), (33, 16), (33, 34), (34, 33)
]
