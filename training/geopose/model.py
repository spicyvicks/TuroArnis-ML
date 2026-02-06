import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GeoPoseNet(torch.nn.Module):
    def __init__(self, num_classes=12, hidden_channels=64, heads=4, edge_dim=4):
        super(GeoPoseNet, self).__init__()
        
        # 1. GATv2 Layer 1
        # Input: Node features (2D coords normalized) -> Hidden * Heads
        self.conv1 = GATv2Conv(2, hidden_channels, heads=heads, edge_dim=edge_dim, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        
        # 2. GATv2 Layer 2
        # Input: Hidden * Heads -> Hidden * 2
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels * 2, heads=2, edge_dim=edge_dim, concat=False)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)
        
        # Classifier
        self.lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)
        
        self.dropout = nn.Dropout(0.4)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # x: [num_nodes, 2]
        # edge_attr: [num_edges, 4]
        
        # Layer 1
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Global Pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
            
        # Classifier
        x = self.lin1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        
        return x
