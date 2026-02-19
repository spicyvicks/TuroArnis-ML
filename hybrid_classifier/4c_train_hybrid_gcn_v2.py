"""
Optimized Hybrid GCN V2 with Node-Specific Features + Global Context
- Node features: Per-node geometric data (x, y, visibility, angles, distances)
- Global features: Hybrid similarity scores
- Architecture: 3-layer GCN + MLP fusion
- Optimizations: Data augmentation, larger hidden dim (256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct',
    'neutral'
]

# Config
HYBRID_FEATURES_DIR = Path("hybrid_classifier/hybrid_features_v3")
OUTPUT_DIR = Path("hybrid_classifier/models")

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


def load_hybrid_graph_data(viewpoint_filter=None):
    """Load node features + hybrid features and convert to graph format"""
    suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
    
    train_file = HYBRID_FEATURES_DIR / f"train_features{suffix}.pt"
    test_file = HYBRID_FEATURES_DIR / f"test_features{suffix}.pt"
    
    if not train_file.exists():
        raise FileNotFoundError(f"{train_file} not found. Run 2b_generate_node_hybrid_features.py first")
    
    train_data = torch.load(train_file)
    test_data = torch.load(test_file)
    
    # Convert to graph format
    train_graphs = []
    for i in range(len(train_data['node_features'])):
        node_feat = train_data['node_features'][i]  # [35, node_feat_dim]
        hybrid_feat = train_data['hybrid_features'][i]  # [hybrid_feat_dim]
        label = train_data['labels'][i]
        
        edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t()
        graph = Data(x=node_feat, edge_index=edge_index, y=label, hybrid=hybrid_feat)
        train_graphs.append(graph)
    
    test_graphs = []
    for i in range(len(test_data['node_features'])):
        node_feat = test_data['node_features'][i]
        hybrid_feat = test_data['hybrid_features'][i]
        label = test_data['labels'][i]
        
        edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t()
        graph = Data(x=node_feat, edge_index=edge_index, y=label, hybrid=hybrid_feat)
        test_graphs.append(graph)
    
    return train_graphs, test_graphs


def compute_class_weights(train_graphs):
    """Compute class weights for imbalanced dataset"""
    labels = [g.y.item() for g in train_graphs]
    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    total = len(labels)
    weights = total / (len(CLASS_NAMES) * class_counts + 1e-6)
    return torch.FloatTensor(weights)


def train_hybrid_gcn(viewpoint_filter=None, epochs=150, lr=0.001, hidden_dim=128, dropout=0.5, num_layers=3, augment=True):
    """Train Hybrid GCN with node + global features"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    viewpoint_str = f" ({viewpoint_filter})" if viewpoint_filter else ""
    print(f"\n{'='*60}")
    print(f"Training Hybrid GCN V2 (Optimized){viewpoint_str}")
    print(f"Architecture: Node Features (GCN) + Global Features (MLP)")
    print(f"Layers: {num_layers}, Hidden: {hidden_dim}, Augment: {augment}, Node Emb: 8")
    print(f"{'='*60}\n")
    
    # Load data
    train_graphs, test_graphs = load_hybrid_graph_data(viewpoint_filter)
    
    print(f"Train samples: {len(train_graphs)}")
    print(f"Test samples: {len(test_graphs)}")
    print(f"Node features: {train_graphs[0].x.shape}")
    print(f"Hybrid features: {train_graphs[0].hybrid.shape}")
    print(f"Hidden dim: {hidden_dim}, Dropout: {dropout}\n")
    
    # Verify nodes have different features
    print("Verifying node diversity:")
    sample_graph = train_graphs[0]
    print(f"  Node 0 features: {sample_graph.x[0, :3].numpy()}")
    print(f"  Node 15 features: {sample_graph.x[15, :3].numpy()}")
    print(f"  Node 33 features: {sample_graph.x[33, :3].numpy()}")
    are_different = not torch.allclose(sample_graph.x[0], sample_graph.x[15])
    print(f"  Nodes are different: {are_different}\n")
    
    # Compute class weights
    class_weights = compute_class_weights(train_graphs).to(device)
    
    # Create dataloaders
    # drop_last=True prevents BatchNorm error when last batch has only 1 sample
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    # Initialize model
    node_feat_dim = train_graphs[0].x.shape[1]
    hybrid_feat_dim = train_graphs[0].hybrid.shape[0]
    
    model = HybridGCN(
        node_in_channels=node_feat_dim,
        hybrid_in_channels=hybrid_feat_dim,
        hidden_channels=hidden_dim,
        num_classes=len(CLASS_NAMES),
        num_layers=num_layers,
        dropout=dropout,
        embedding_dim=8  # 8-dim learnable node identity embeddings
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15
    )
    
    best_acc = 0
    patience = 20  # Reduced from 30 for faster stopping
    patience_counter = 0
    overfitting_threshold = 0.20  # Stop if gap > 20%
    
    # Initialize history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_correct = 0
        train_total = 0
        running_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Extract hybrid features from batch
            # PyG collates [30] tensors into [batch_size * 30], so we need to reshape
            hybrid_batch = batch.hybrid.view(batch.num_graphs, -1)
            
            # Data Augmentation: Scale + Jitter
            x = batch.x
            if augment:
                # 1. Random Scaling (0.85 to 1.15) - Handle height/distance variations
                scale = 0.85 + (0.3 * torch.rand(1, device=device).item())
                x = x.clone()
                
                # Scale coordinates (x, y, z) - indices 0, 1, 2
                x[:, :3] *= scale
                
                # Scale distance feature (index 4) if present
                if x.shape[1] >= 5:
                    x[:, 4] *= scale
                
                # 2. Add Jitter (Noise robustness)
                noise = torch.randn_like(x[:, :3]) * 0.02  # 2% jitter
                x[:, :3] += noise
            
            out = model(x, batch.edge_index, batch.batch, hybrid_batch)
            loss = criterion(out, batch.y)
            
            loss.backward()
            optimizer.step()
            
            pred = out.argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)
            running_loss += loss.item() * batch.y.size(0)
        
        train_acc = train_correct / train_total
        epoch_loss = running_loss / train_total
        
        # Test
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                hybrid_batch = batch.hybrid.view(batch.num_graphs, -1)
                
                out = model(batch.x, batch.edge_index, batch.batch, hybrid_batch)
                pred = out.argmax(dim=1)
                test_correct += (pred == batch.y).sum().item()
                test_total += batch.y.size(0)
        
        test_acc = test_correct / test_total
        
        # Update history
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        # Update scheduler
        scheduler.step(test_acc)
        
        print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss:.4f} | Train: {train_acc:.4f} | Test: {test_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Check for severe overfitting
        gap = train_acc - test_acc
        if gap > overfitting_threshold:
            print(f"\n⚠️  OVERFITTING DETECTED: Gap = {gap:.2%} (Train: {train_acc:.2%}, Test: {test_acc:.2%})")
            print(f"Stopping training to prevent wasted time.")
            break
        
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
                'node_feat_dim': node_feat_dim,
                'hybrid_feat_dim': hybrid_feat_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout': dropout
            }, OUTPUT_DIR / f"hybrid_gcn_v2{suffix}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Save history to JSON
    import json
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
    with open(OUTPUT_DIR / f"history{suffix}.json", 'w') as f:
        json.dump(history, f)
    print(f"History saved to {OUTPUT_DIR / f'history{suffix}.json'}")
    
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
    parser.add_argument('--merged', action='store_true',
                        help='Train a single model on all viewpoints combined')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--no-augment', action='store_true', help="Disable data augmentation")
    
    args = parser.parse_args()
    
    # Determine training mode
    if args.merged:
        # Train single merged model on all viewpoints
        print(f"\n{'='*60}")
        print(f"TRAINING MERGED MODEL (All Viewpoints)")
        print(f"{'='*60}")
        
        try:
            train_hybrid_gcn(
                viewpoint_filter=None,  # No filter = use all data
                epochs=args.epochs,
                lr=args.lr,
                hidden_dim=args.hidden,
                dropout=args.dropout,
                num_layers=args.layers,
                augment=not args.no_augment
            )
        except Exception as e:
            print(f"Error training merged model: {e}")
    
    elif args.viewpoint:
        # Train single specialist model
        print(f"\n{'='*60}")
        print(f"TRAINING HYBRID GCN V2: {args.viewpoint.upper()}")
        print(f"{'='*60}")
        
        try:
            train_hybrid_gcn(
                viewpoint_filter=args.viewpoint,
                epochs=args.epochs,
                lr=args.lr,
                hidden_dim=args.hidden,
                dropout=args.dropout,
                num_layers=args.layers,
                augment=not args.no_augment
            )
        except Exception as e:
            print(f"Error training {args.viewpoint}: {e}")
    
    else:
        # Default: Train all 3 specialist models
        viewpoints_to_train = ['front', 'left', 'right']
        
        for vp in viewpoints_to_train:
            print(f"\n{'='*60}")
            print(f"TRAINING HYBRID GCN V2: {vp.upper()}")
            print(f"{'='*60}")
            
            try:
                train_hybrid_gcn(
                    viewpoint_filter=vp,
                    epochs=args.epochs,
                    lr=args.lr,
                    hidden_dim=args.hidden,
                    dropout=args.dropout,
                    num_layers=args.layers,
                    augment=not args.no_augment
                )
            except Exception as e:
                print(f"Error training {vp}: {e}")
