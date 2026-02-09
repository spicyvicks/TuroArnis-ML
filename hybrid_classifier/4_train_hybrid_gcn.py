"""
Hybrid GCN: Train GCN with rule-based node features
Combines geometric feature engineering with graph structure
"""

import torch
import numpy as np
from pathlib import Path
import json
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.spatial_gcn import SpatialGCN
from training.train_gcn import CLASS_NAMES

# Config
FEATURE_TEMPLATES = Path("hybrid_classifier/feature_templates.json")
HYBRID_FEATURES_DIR = Path("hybrid_classifier/hybrid_features")
OUTPUT_DIR = Path("hybrid_classifier/models")

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
    
    suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
    
    train_file = HYBRID_FEATURES_DIR / f"train_features{suffix}.pt"
    test_file = HYBRID_FEATURES_DIR / f"test_features{suffix}.pt"
    
    if not train_file.exists():
        raise FileNotFoundError(f"{train_file} not found. Run 2_generate_hybrid_features.py first")
    
    train_data = torch.load(train_file)
    test_data = torch.load(test_file)
    
    # Convert to graph format
    # Each sample has hybrid features (similarity scores)
    # We'll broadcast these to all 35 nodes as node attributes
    
    train_graphs = []
    for i in range(len(train_data['features'])):
        hybrid_feat = train_data['features'][i]  # [num_features]
        label = train_data['labels'][i]
        
        # Broadcast hybrid features to all 35 nodes
        # Each node gets: [hybrid_feature_vector]
        node_features = hybrid_feat.unsqueeze(0).repeat(35, 1)  # [35, num_features]
        
        # Build graph
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


def train_hybrid_gcn(viewpoint_filter=None, epochs=100, lr=0.001):
    """Train GCN on hybrid features"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    viewpoint_str = f" ({viewpoint_filter} only)" if viewpoint_filter else ""
    print(f"Training Hybrid GCN{viewpoint_str}")
    
    # Load data
    train_graphs, test_graphs = load_hybrid_graph_data(viewpoint_filter)
    
    print(f"Train samples: {len(train_graphs)}")
    print(f"Test samples: {len(test_graphs)}")
    print(f"Node features: {train_graphs[0].x.shape[1]}")
    
    # Create dataloaders
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    # Initialize model
    num_features = train_graphs[0].x.shape[1]
    model = SpatialGCN(
        in_channels=num_features,
        hidden_channels=64,
        num_classes=len(CLASS_NAMES),
        dropout=0.5
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_acc = 0
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = out.argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)
        
        train_acc = train_correct / train_total
        
        # Test
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                test_correct += (pred == batch.y).sum().item()
                test_total += batch.y.size(0)
        
        test_acc = test_correct / test_total
        
        print(f"Epoch {epoch+1:3d} | Train: {train_acc:.4f} | Test: {test_acc:.4f}")
        
        # Early stopping
        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            
            # Save best model
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
            torch.save({
                'model_state_dict': model.state_dict(),
                'test_accuracy': test_acc,
                'num_features': num_features
            }, OUTPUT_DIR / f"hybrid_gcn{suffix}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print(f"\n{'='*60}")
    print(f"Best Test Accuracy: {best_acc:.4f}")
    print(f"{'='*60}")
    
    return model, best_acc


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewpoint', type=str, default=None,
                        choices=['front', 'left', 'right'],
                        help='Train on specific viewpoint only')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    
    # If viewpoint is specified, train only that one. Otherwise, train all 3.
    viewpoints_to_train = [args.viewpoint] if args.viewpoint else ['front', 'left', 'right']
    
    for vp in viewpoints_to_train:
        print(f"\n{'='*40}")
        print(f"TRAINING MODEL FOR VIEWPOINT: {vp.upper()}")
        print(f"{'='*40}")
        
        try:
            train_hybrid_gcn(viewpoint_filter=vp, epochs=args.epochs, lr=args.lr)
        except Exception as e:
            print(f"Skipping {vp} due to error: {e}")
