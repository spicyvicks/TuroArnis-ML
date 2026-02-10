import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

# Config
OUTPUT_DIR = Path("hybrid_classifier/analysis")
HYBRID_FEATURES_DIR = Path("hybrid_classifier/hybrid_features_v2")
MODELS_DIR = Path("hybrid_classifier/models")

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]

# Skeleton edges (same as original GCN)
SKELETON_EDGES = [
    (11, 12), (12, 11), (11, 23), (23, 11), (12, 24), (24, 12),
    (23, 24), (24, 23), (11, 13), (13, 11), (13, 15), (15, 13),
    (12, 14), (14, 12), (14, 16), (16, 14), (23, 25), (25, 23),
    (25, 27), (27, 25), (24, 26), (26, 24), (26, 28), (28, 26),
    # Stick connections
    (15, 33), (33, 15), (16, 33), (33, 16), (33, 34), (34, 33)
]

class HybridGCN(nn.Module):
    """
    GCN with Node-Specific Features + Global Hybrid Context
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
        
        # Input layer
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
        
        # Fusion layer
        self.fusion_fc = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fusion_bn = nn.BatchNorm1d(hidden_channels)
        
        # Classification head
        self.fc = nn.Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, batch, hybrid_features):
        num_nodes_per_graph = 35
        num_graphs = batch.max().item() + 1
        node_indices = torch.arange(num_nodes_per_graph, device=x.device).repeat(num_graphs)
        
        node_emb = self.node_embedding(node_indices)
        x = torch.cat([x, node_emb], dim=1)
        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x_graph = global_mean_pool(x, batch)
        
        h = self.hybrid_fc1(hybrid_features)
        h = self.hybrid_bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = self.hybrid_fc2(h)
        h = self.hybrid_bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        combined = torch.cat([x_graph, h], dim=1)
        combined = self.fusion_fc(combined)
        combined = self.fusion_bn(combined)
        combined = F.relu(combined)
        combined = F.dropout(combined, p=self.dropout, training=self.training)
        
        out = self.fc(combined)
        return out

def load_hybrid_graph_data(viewpoint_filter=None):
    """Load hybrid features and convert to graph format"""
    suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
    test_file = HYBRID_FEATURES_DIR / f"test_features{suffix}.pt"
    
    print(f"Loading features from: {test_file}")
    
    if not test_file.exists():
        raise FileNotFoundError(f"{test_file} not found. Run 2b_generate_node_hybrid_features.py first")
    
    test_data = torch.load(test_file)
    
    test_graphs = []
    for i in range(len(test_data['node_features'])):
        node_feat = test_data['node_features'][i]
        hybrid_feat = test_data['hybrid_features'][i]
        label = test_data['labels'][i]
        
        edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t()
        graph = Data(x=node_feat, edge_index=edge_index, y=label, hybrid=hybrid_feat)
        test_graphs.append(graph)
    
    return test_graphs

def analyze_model(viewpoint=None):
    """Load model and generate performance report"""
    
    suffix = f"_{viewpoint}" if viewpoint else ""
    model_path = MODELS_DIR / f"hybrid_gcn_v2{suffix}.pth"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading test data...")
    test_graphs = load_hybrid_graph_data(viewpoint)
    
    # Load model checkpoint
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model with saved config
    model = HybridGCN(
        node_in_channels=checkpoint['node_feat_dim'],
        hybrid_in_channels=checkpoint['hybrid_feat_dim'],
        hidden_channels=checkpoint['hidden_dim'],
        num_classes=len(CLASS_NAMES),
        num_layers=checkpoint.get('num_layers', 3),
        dropout=checkpoint['dropout'],
        embedding_dim=checkpoint.get('embedding_dim', 8)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get predictions
    y_true = []
    y_pred = []
    
    from torch_geometric.loader import DataLoader
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            hybrid_batch = batch.hybrid.view(batch.num_graphs, -1)
            
            out = model(batch.x, batch.edge_index, batch.batch, hybrid_batch)
            pred = out.argmax(dim=1)
            
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            
    # generate report
    title = f"Hybrid GCN V2 ({viewpoint.upper() if viewpoint else 'ALL'})"
    print("\n" + "="*60)
    print(f"Classification Report: {title}")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix: {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f'confusion_matrix{suffix}.png'
    plt.savefig(out_file)
    print(f"\nConfusion matrix saved to {out_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewpoint', type=str, default=None,
                        choices=['front', 'left', 'right'],
                        help='Analyze specific viewpoint model')
    args = parser.parse_args()
    
    analyze_model(args.viewpoint)
