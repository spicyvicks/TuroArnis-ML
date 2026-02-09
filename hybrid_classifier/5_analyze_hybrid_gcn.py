"""
Analyze Hybrid GCN Performance
Generates classification report and confusion matrix for the trained model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import sys

# Add parent directory to path to import model
sys.path.append(str(Path(__file__).parent.parent))
from models.spatial_gcn import SpatialGCN
from training.train_gcn import CLASS_NAMES
from torch_geometric.data import Data

# Config
# Check which model exists or passed as arg
MODEL_PATH = Path("hybrid_classifier/models/hybrid_gcn.pth") 
if not MODEL_PATH.exists():
    MODEL_PATH = Path("hybrid_classifier/models/hybrid_gcn_front.pth")

OUTPUT_DIR = Path("hybrid_classifier/analysis")
HYBRID_FEATURES_DIR = Path("hybrid_classifier/hybrid_features")

# Skeleton edges (same as original GCN)
SKELETON_EDGES = [
    (11, 12), (12, 11), (11, 23), (23, 11), (12, 24), (24, 12),
    (23, 24), (24, 23), (11, 13), (13, 11), (13, 15), (15, 13),
    (12, 14), (14, 12), (14, 16), (16, 14), (23, 25), (25, 23),
    (25, 27), (27, 25), (24, 26), (26, 24), (26, 28), (28, 26),
    # Stick connections
    (15, 33), (33, 15), (16, 33), (33, 16), (33, 34), (34, 33)
]

def load_hybrid_graph_data(viewpoint_filter=None):
    """Load hybrid features and convert to graph format"""
    
    # If viewpoint_filter is None, look for 'test_features.pt' (combined)
    # If viewpoint_filter is 'front', look for 'test_features_front.pt'
    if viewpoint_filter:
        suffix = f"_{viewpoint_filter}"
    else:
        suffix = ""
    
    train_file = HYBRID_FEATURES_DIR / f"train_features{suffix}.pt"
    test_file = HYBRID_FEATURES_DIR / f"test_features{suffix}.pt"
    
    print(f"Loading features from: {test_file}")
    
    if not train_file.exists():
        raise FileNotFoundError(f"{train_file} not found. Run 2_generate_hybrid_features.py first")
    
    train_data = torch.load(train_file)
    test_data = torch.load(test_file)
    
    # Convert to graph format
    train_graphs = []
    for i in range(len(train_data['features'])):
        hybrid_feat = train_data['features'][i]
        label = train_data['labels'][i]
        node_features = hybrid_feat.unsqueeze(0).repeat(35, 1)
        edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t()
        graph = Data(x=node_features, edge_index=edge_index, y=label)
        train_graphs.append(graph)
    
    test_graphs = []
    for i in range(len(test_data['features'])):
        hybrid_feat = test_data['features'][i]
        label = test_data['labels'][i]
        node_features = hybrid_feat.unsqueeze(0).repeat(35, 1)
        edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t()
        graph = Data(x=node_features, edge_index=edge_index, y=label)
        test_graphs.append(graph)
    
    return train_graphs, test_graphs

def analyze_model():
    """Load model and generate performance report"""
    
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading test data...")
    _, test_graphs = load_hybrid_graph_data(None)
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    num_features = checkpoint['num_features']
    
    model = SpatialGCN(
        in_channels=num_features,
        hidden_channels=64,
        num_classes=len(CLASS_NAMES),
        dropout=0.5
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
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            
    # generate report
    print("\n" + "="*60)
    print("Classification Report (Front View)")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Hybrid GCN Confusion Matrix (Front View)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / 'confusion_matrix_front.png')
    print(f"\nConfusion matrix saved to {OUTPUT_DIR / 'confusion_matrix_front.png'}")

if __name__ == "__main__":
    analyze_model()
