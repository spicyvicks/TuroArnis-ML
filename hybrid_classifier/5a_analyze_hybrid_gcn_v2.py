import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from training.train_gcn import CLASS_NAMES
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Config
HYBRID_FEATURES_DIR = Path("hybrid_classifier/hybrid_features_v2")
OUTPUT_DIR = Path("hybrid_classifier/analysis")

SKELETON_EDGES = [
    (11, 12), (12, 11), (11, 23), (23, 11), (12, 24), (24, 12),
    (23, 24), (24, 23), (11, 13), (13, 11), (13, 15), (15, 13),
    (12, 14), (14, 12), (14, 16), (16, 14), (23, 25), (25, 23),
    (25, 27), (27, 25), (24, 26), (26, 24), (26, 28), (28, 26),
    (15, 33), (33, 15), (16, 33), (33, 16), (33, 34), (34, 33)
]


# Import model architectures
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class HybridGCN(nn.Module):
    """GCN with Node-Specific Features + Global Hybrid Context"""
    def __init__(self, node_in_channels, hybrid_in_channels, hidden_channels, num_classes, dropout=0.5):
        super().__init__()
        
        # GCN layers for node features
        self.conv1 = GCNConv(node_in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
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
        self.dropout = dropout
        
    def forward(self, x, edge_index, batch, hybrid_features):
        # Process node features with GCN
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
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
        
        # Fusion
        combined = torch.cat([x_graph, h], dim=1)
        combined = self.fusion_fc(combined)
        combined = self.fusion_bn(combined)
        combined = F.relu(combined)
        combined = F.dropout(combined, p=self.dropout, training=self.training)
        
        # Classification
        out = self.fc(combined)
        return out


class HybridGAT(nn.Module):
    """GAT with Node-Specific Features + Global Hybrid Context"""
    def __init__(self, node_in_channels, hybrid_in_channels, hidden_channels, num_classes, heads=4, dropout=0.5):
        super().__init__()
        
        # GAT layers for node features
        self.conv1 = GATConv(node_in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
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
        self.dropout = dropout
        
    def forward(self, x, edge_index, batch, hybrid_features):
        # Process node features with GAT
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
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
        
        # Fusion
        combined = torch.cat([x_graph, h], dim=1)
        combined = self.fusion_fc(combined)
        combined = self.fusion_bn(combined)
        combined = F.relu(combined)
        combined = F.dropout(combined, p=self.dropout, training=self.training)
        
        # Classification
        out = self.fc(combined)
        return out


def load_hybrid_graph_data(viewpoint_filter=None):
    """Load node features + hybrid features and convert to graph format"""
    suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
    
    test_file = HYBRID_FEATURES_DIR / f"test_features{suffix}.pt"
    
    if not test_file.exists():
        raise FileNotFoundError(f"{test_file} not found. Run 2b_generate_node_hybrid_features.py first")
    
    print(f"Loading features from: {test_file}")
    test_data = torch.load(test_file)
    
    # Convert to graph format
    test_graphs = []
    for i in range(len(test_data['node_features'])):
        node_feat = test_data['node_features'][i]
        hybrid_feat = test_data['hybrid_features'][i]
        label = test_data['labels'][i]
        
        edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t()
        graph = Data(x=node_feat, edge_index=edge_index, y=label, hybrid=hybrid_feat)
        test_graphs.append(graph)
    
    return test_graphs


def analyze_model(model_path, viewpoint='front', model_type='gcn'):
    """Load model and generate performance report"""
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading test data...")
    test_graphs = load_hybrid_graph_data(viewpoint)
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device)
    
    node_feat_dim = checkpoint['node_feat_dim']
    hybrid_feat_dim = checkpoint['hybrid_feat_dim']
    hidden_dim = checkpoint['hidden_dim']
    dropout = checkpoint['dropout']
    
    # Initialize model based on type
    if model_type.lower() == 'gat':
        heads = checkpoint.get('heads', 4)
        model = HybridGAT(
            node_in_channels=node_feat_dim,
            hybrid_in_channels=hybrid_feat_dim,
            hidden_channels=hidden_dim,
            num_classes=len(CLASS_NAMES),
            heads=heads,
            dropout=dropout
        ).to(device)
        model_name = f"Hybrid GAT (heads={heads})"
    else:
        model = HybridGCN(
            node_in_channels=node_feat_dim,
            hybrid_in_channels=hybrid_feat_dim,
            hidden_channels=hidden_dim,
            num_classes=len(CLASS_NAMES),
            dropout=dropout
        ).to(device)
        model_name = "Hybrid GCN"
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model: {model_name}")
    print(f"Test accuracy (from training): {checkpoint['test_accuracy']:.4f}")
    
    # Get predictions
    y_true = []
    y_pred = []
    
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Reshape hybrid features (PyG flattens custom attributes)
            hybrid_batch = batch.hybrid.view(batch.num_graphs, -1)
            
            out = model(batch.x, batch.edge_index, batch.batch, hybrid_batch)
            pred = out.argmax(dim=1)
            
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    
    # Generate report
    print("\n" + "="*60)
    print(f"Classification Report - {model_name} ({viewpoint.upper()})")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'{model_name} Confusion Matrix ({viewpoint.upper()})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f'confusion_matrix_{model_type}_v2_{viewpoint}.png'
    plt.savefig(output_file)
    print(f"\nConfusion matrix saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model file (e.g., hybrid_classifier/models/hybrid_gcn_v2_front.pth)')
    parser.add_argument('--viewpoint', type=str, default='front',
                        choices=['front', 'left', 'right'])
    parser.add_argument('--type', type=str, default='gcn',
                        choices=['gcn', 'gat'],
                        help='Model type: gcn or gat')
    
    args = parser.parse_args()
    
    analyze_model(args.model, args.viewpoint, args.type)
