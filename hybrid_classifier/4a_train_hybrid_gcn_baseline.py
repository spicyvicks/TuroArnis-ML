"""
Baseline Hybrid GCN with Research-Backed Improvements
- Learning Rate Scheduler (ReduceLROnPlateau)
- Batch Normalization
- Class Weighting (handles imbalance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import sys

sys.path.append(str(Path(__file__).parent.parent))
from training.train_gcn import CLASS_NAMES

# Config
FEATURE_TEMPLATES = Path("hybrid_classifier/feature_templates.json")
HYBRID_FEATURES_DIR = Path("hybrid_classifier/hybrid_features")
OUTPUT_DIR = Path("hybrid_classifier/models")

SKELETON_EDGES = [
    (11, 12), (12, 11), (11, 23), (23, 11), (12, 24), (24, 12),
    (23, 24), (24, 23), (11, 13), (13, 11), (13, 15), (15, 13),
    (12, 14), (14, 12), (14, 16), (16, 14), (23, 25), (25, 23),
    (25, 27), (27, 25), (24, 26), (26, 24), (26, 28), (28, 26),
    (15, 33), (33, 15), (16, 33), (33, 16), (33, 34), (34, 33)
]


class ImprovedGCN(nn.Module):
    """GCN with Batch Normalization"""
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout
        
    def forward(self, x, edge_index, batch):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.fc(x)
        return x


def load_hybrid_graph_data(viewpoint_filter=None):
    """Load hybrid features and convert to graph format"""
    suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
    
    train_file = HYBRID_FEATURES_DIR / f"train_features{suffix}.pt"
    test_file = HYBRID_FEATURES_DIR / f"test_features{suffix}.pt"
    
    if not train_file.exists():
        raise FileNotFoundError(f"{train_file} not found")
    
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


def compute_class_weights(train_graphs):
    """Compute class weights for imbalanced dataset"""
    labels = [g.y.item() for g in train_graphs]
    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    
    # Inverse frequency weighting
    total = len(labels)
    weights = total / (len(CLASS_NAMES) * class_counts + 1e-6)
    
    return torch.FloatTensor(weights)


def train_baseline_gcn(viewpoint_filter=None, epochs=150, lr=0.001, hidden_dim=128, dropout=0.5):
    """Train GCN with research-backed improvements"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    viewpoint_str = f" ({viewpoint_filter})" if viewpoint_filter else ""
    print(f"\n{'='*60}")
    print(f"Training Baseline GCN{viewpoint_str}")
    print(f"Improvements: LR Scheduler + BatchNorm + Class Weighting")
    print(f"{'='*60}\n")
    
    # Load data
    train_graphs, test_graphs = load_hybrid_graph_data(viewpoint_filter)
    
    print(f"Train samples: {len(train_graphs)}")
    print(f"Test samples: {len(test_graphs)}")
    print(f"Node features: {train_graphs[0].x.shape[1]}")
    print(f"Hidden dim: {hidden_dim}, Dropout: {dropout}\n")
    
    # Compute class weights
    class_weights = compute_class_weights(train_graphs).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}\n")
    
    # Create dataloaders
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    # Initialize model
    num_features = train_graphs[0].x.shape[1]
    model = ImprovedGCN(
        in_channels=num_features,
        hidden_channels=hidden_dim,
        num_classes=len(CLASS_NAMES),
        dropout=dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, verbose=True
    )
    
    best_acc = 0
    patience = 30
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
        
        # Update scheduler
        scheduler.step(test_acc)
        
        print(f"Epoch {epoch+1:3d} | Train: {train_acc:.4f} | Test: {test_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
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
                'num_features': num_features,
                'hidden_dim': hidden_dim,
                'dropout': dropout
            }, OUTPUT_DIR / f"baseline_gcn{suffix}.pth")
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
                        choices=['front', 'left', 'right'])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    args = parser.parse_args()
    
    # Train all viewpoints or specific one
    viewpoints_to_train = [args.viewpoint] if args.viewpoint else ['front', 'left', 'right']
    
    for vp in viewpoints_to_train:
        print(f"\n{'='*60}")
        print(f"TRAINING BASELINE MODEL: {vp.upper()}")
        print(f"{'='*60}")
        
        try:
            train_baseline_gcn(
                viewpoint_filter=vp,
                epochs=args.epochs,
                lr=args.lr,
                hidden_dim=args.hidden,
                dropout=args.dropout
            )
        except Exception as e:
            print(f"Error training {vp}: {e}")
