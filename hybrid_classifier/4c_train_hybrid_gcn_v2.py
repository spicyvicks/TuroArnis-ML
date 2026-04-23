"""
HybridGCN V2 Training Script - Optimized Version
==============================================

Optimizations applied (Phase 5.7):
- HIDDEN_DIM: 256 → 128 (reduce overfitting)
- DROPOUT: 0.5 → 0.7 (aggressive regularization)
- NODE_EMBED_DIM: 8 → 16 (better node representation)
- PATIENCE: 20 → 15 (faster overfit detection)
- BatchNorm: track_running_stats=False (fix inference variance)
- WEIGHT_DECAY: 1e-4 (L2 regularization)
- ReduceLROnPlateau scheduler (better convergence)
- WeightedRandomSampler (fix 6.7:1 class imbalance)
- Xavier initialization (better convergence)
- Overfitting gap monitoring (train-test gap detection)

Usage:
    python 4c_train_hybrid_gcn_v2.py --merged
    python 4c_train_hybrid_gcn_v2.py --viewpoint front
    python 4c_train_hybrid_gcn_v2.py --viewpoint left
    python 4c_train_hybrid_gcn_v2.py --viewpoint right
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader as GeoDataLoader

from tqdm import tqdm

# =============================================================================
# CONFIGURATION - OPTIMIZED FOR ~1,871 TRAINING SAMPLES
# =============================================================================

# Model architecture - March 2 proven configuration (achieved 80%+ accuracy)
# NOTE: These hyperparameters MUST match the March 2 successful models
HIDDEN_DIM = 128              # Front: 128, Left/Right: 256 (set dynamically in training)
NUM_LAYERS = 3                # Keep: 3 layers sufficient
DROPOUT = 0.5                 # March 2 used 0.5
NODE_EMBED_DIM = 8            # CRITICAL: March 2 models used 8 (not 16!)

# Training - OPTION A Configuration
LEARNING_RATE = 0.005         # OPTION A: 5x faster than 0.001, more stable than 0.01
WEIGHT_DECAY = 5e-5           # OPTION A: Light regularization
EPOCHS = 150                  # Keep: 150 epochs max
PATIENCE = 20                 # OPTION A: 20 epochs patience for convergence
BATCH_SIZE = 64               # Keep: 64 for stable gradients

# Early stopping overfit detection
MAX_OVERFIT_GAP = 35.0        # Keep: Allow 35% gap during learning phase

# Data paths
DATA_DIR = Path("hybrid_classifier/hybrid_features_v3")
MODELS_DIR = Path("hybrid_classifier/models")
HISTORY_DIR = Path("hybrid_classifier/models")

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names (13 classes: 12 techniques + neutral)
CLASS_NAMES = [
    'front_left_chest_thrust', 'front_right_chest_thrust',
    'front_crown_thrust', 'left_crown_thrust', 'right_crown_thrust',
    'left_jab', 'right_jab',
    'front_left_downward_block', 'front_right_downward_block',
    'left_outward_block', 'right_outward_block', 'left_waist_block',
    'neutral'
]
NUM_CLASSES = len(CLASS_NAMES)

# Skeleton edges (28 bidirectional edges)
SKELETON_EDGES = [
    # Shoulders (bidirectional)
    (11, 12), (12, 11),
    # Torso
    (11, 23), (23, 11),  # L shoulder ↔ L hip
    (12, 24), (24, 12),  # R shoulder ↔ R hip
    (23, 24), (24, 23),  # Hips ↔
    # Left arm
    (11, 13), (13, 11),  # Shoulder ↔ elbow
    (13, 15), (15, 13),  # Elbow ↔ wrist
    # Right arm
    (12, 14), (14, 12),
    (14, 16), (16, 14),
    # Left leg
    (23, 25), (25, 23),  # Hip ↔ knee
    (25, 27), (27, 25),  # Knee ↔ ankle
    # Right leg
    (24, 26), (26, 24),
    (26, 28), (28, 26),
    # Stick connections
    (15, 33), (33, 15),  # L wrist ↔ stick grip
    (16, 33), (33, 16),  # R wrist ↔ stick grip
    (33, 34), (34, 33),  # Grip ↔ tip
]

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class HybridGCN(nn.Module):
    """
    HybridGCN with node features + global hybrid features.
    
    Architecture:
        1. Node embedding: 35 nodes → learnable embeddings
        2. GCN layers: 3 layers with BatchNorm + ReLU + Dropout
        3. Global pooling: Mean pool node features
        4. Hybrid MLP: 2-layer MLP for 30 hybrid features
        5. Fusion: Concatenate GCN output + hybrid MLP output
        6. Classification: Linear layer → class logits
    """
    
    def __init__(self, num_node_features, num_hybrid_features, num_classes, hidden_dim=HIDDEN_DIM):
        super(HybridGCN, self).__init__()
        
        # Node embedding layer
        self.node_embedding = nn.Embedding(35, NODE_EMBED_DIM)
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer: node_features + embedding -> hidden
        self.convs.append(GCNConv(num_node_features + NODE_EMBED_DIM, hidden_dim))
        # OPTIMIZED: track_running_stats=False for small batch stability
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim, track_running_stats=False))
        
        # Hidden layers
        for _ in range(NUM_LAYERS - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            # OPTIMIZED: track_running_stats=False
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim, track_running_stats=False))
        
        # Hybrid feature MLP
        self.hybrid_mlp = nn.Sequential(
            nn.Linear(num_hybrid_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Classification layers
        self.fc1 = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(DROPOUT)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        hybrid_features = data.hybrid_features
        
        # Get node embeddings - FIX: Proper batch handling
        batch_size = batch.max().item() + 1
        node_indices = torch.arange(35, device=x.device).unsqueeze(0).expand(batch_size, -1)
        node_emb = self.node_embedding(node_indices).view(-1, NODE_EMBED_DIM)  # [B*35, NODE_EMBED_DIM]
        
        # Concatenate node features with embeddings
        x = torch.cat([x, node_emb], dim=-1)
        
        # GCN layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            
            # Residual connection (if dimensions match)
            if x_new.size(-1) == x.size(-1):
                x = x_new + x
            else:
                x = x_new
        
        # Global pooling
        x_pool = global_mean_pool(x, batch)
        
        # Hybrid feature processing - FIX: Reshape from [B*30] to [B, 30]
        batch_size = batch.max().item() + 1
        hybrid_features = hybrid_features.view(batch_size, -1)  # [B, 30]
        hybrid_out = self.hybrid_mlp(hybrid_features)
        
        # Fusion
        combined = torch.cat([x_pool, hybrid_out], dim=-1)
        
        # Classification
        x_out = F.relu(self.fc1(combined))
        x_out = self.dropout(x_out)
        logits = self.fc2(x_out)
        
        return logits


# =============================================================================
# DATA LOADING
# =============================================================================

class GraphDataset(Dataset):
    """Dataset for loading pre-computed graph features with NaN filtering and stick node masking."""
    
    def __init__(self, features_path, viewpoint=None, filter_nan=True):
        self.data = torch.load(features_path, map_location='cpu')
        self.viewpoint = viewpoint
        
        # Check if viewpoints are available in the data (merged file) or using per-viewpoint file
        has_viewpoints = 'viewpoints' in self.data
        is_per_viewpoint_file = viewpoint and f"_{viewpoint}" in str(features_path)
        
        # Filter by viewpoint if specified and available
        if viewpoint and has_viewpoints:
            # Merged file with viewpoint filtering
            mask = [v == viewpoint for v in self.data['viewpoints']]
            self.node_features = self.data['node_features'][mask]
            self.hybrid_features = self.data['hybrid_features'][mask]
            self.labels = self.data['labels'][mask]
            self.viewpoints = [v for v, m in zip(self.data['viewpoints'], mask) if m]
        elif viewpoint and not has_viewpoints and not is_per_viewpoint_file:
            # Error only if not using per-viewpoint file
            raise ValueError(f"Viewpoint filtering requested ({viewpoint}) but 'viewpoints' key not found in {features_path}. "
                           f"Regenerate features with viewpoint information or run without --viewpoint.")
        else:
            # Per-viewpoint file or no viewpoint specified - use all data
            self.node_features = self.data['node_features']
            self.hybrid_features = self.data['hybrid_features']
            self.labels = self.data['labels']
            self.viewpoints = self.data.get('viewpoints', [viewpoint] * len(self.labels))
        
        # Load has_stick_nodes mask if present (for samples where stick nodes were excluded)
        if 'has_stick_nodes' in self.data:
            self.has_stick_nodes = self.data['has_stick_nodes']
            if viewpoint and has_viewpoints:
                # Apply same viewpoint filter
                self.has_stick_nodes = self.has_stick_nodes[mask]
        else:
            # Default: all samples have stick nodes
            self.has_stick_nodes = torch.ones(len(self.labels), dtype=torch.bool)
        
        # CRITICAL FIX: Filter out NaN samples (from failed stick detection)
        if filter_nan:
            nan_mask = self._get_nan_mask()
            if nan_mask.sum() > 0:
                print(f"[WARN] Found {nan_mask.sum().item()} NaN samples, filtering them out")
                valid_mask = ~nan_mask
                self.node_features = self.node_features[valid_mask]
                self.hybrid_features = self.hybrid_features[valid_mask]
                self.labels = self.labels[valid_mask]
                self.viewpoints = [v for v, m in zip(self.viewpoints, valid_mask.tolist()) if m]
                self.has_stick_nodes = self.has_stick_nodes[valid_mask]
        
        # Print stats about stick node presence
        no_stick_count = (~self.has_stick_nodes).sum().item()
        if no_stick_count > 0:
            print(f"[INFO] {no_stick_count}/{len(self)} samples ({100*no_stick_count/len(self):.1f}%) without stick nodes")
        
        print(f"Loaded {len(self)} samples" + (f" for {viewpoint} view" if viewpoint else " for all views"))
    
    def _get_nan_mask(self):
        """Identify samples with NaN values."""
        node_nan = torch.isnan(self.node_features).any(dim=(1, 2))
        hybrid_nan = torch.isnan(self.hybrid_features).any(dim=1)
        return node_nan | hybrid_nan
        
        # Print class distribution
        class_counts = np.bincount(self.labels.numpy(), minlength=NUM_CLASSES)
        print("Class distribution:")
        for i, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
            if count > 0:
                print(f"  {name}: {count}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Create edge index
        edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t().contiguous()
        
        # If sample doesn't have stick nodes, mask edges connecting to stick nodes (indices 33, 34)
        # to prevent message passing to/from zero-padded nodes
        if not self.has_stick_nodes[idx]:
            # Filter out edges involving stick nodes (33, 34)
            mask = (edge_index[0] < 33) & (edge_index[1] < 33)
            edge_index = edge_index[:, mask]
        
        data = Data(
            x=self.node_features[idx],
            edge_index=edge_index,
            hybrid_features=self.hybrid_features[idx],
            y=self.labels[idx],
            has_stick_nodes=self.has_stick_nodes[idx]
        )
        return data


def create_weighted_sampler(dataset):
    """OPTIMIZED: Create WeightedRandomSampler for class imbalance."""
    labels = dataset.labels.numpy()
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    
    # Compute weights (inverse frequency)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[label] for label in labels]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print(f"Class weights: {class_weights.round(4)}")
    return sampler


def collate_fn(batch):
    """Custom collate for PyG Data objects."""
    return Batch.from_data_list(batch)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def compute_class_weights(train_dataset):
    """Compute inverse frequency class weights for loss function."""
    labels = train_dataset.labels.numpy()
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    total = len(labels)
    weights = total / (NUM_CLASSES * class_counts + 1e-6)
    return torch.FloatTensor(weights).to(DEVICE)


def train_epoch(model, loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        
        out = model(batch)
        loss = criterion(out, batch.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
    
    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(DEVICE)
            out = model(batch)
            loss = criterion(out, batch.y)
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    accuracy = 100.0 * correct / total
    return total_loss / len(loader), accuracy, all_preds, all_labels


def train_model(train_dataset, val_dataset, viewpoint=None, merged=False, config=None):
    """Main training loop with optimizations."""
    
    # Use provided config or fall back to module constants
    if config is None:
        config = {
            'epochs': EPOCHS,
            'patience': PATIENCE,
            'dropout': DROPOUT,
            'hidden_dim': HIDDEN_DIM,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'batch_size': BATCH_SIZE,
            'max_overfit_gap': MAX_OVERFIT_GAP
        }
    
    # Extract config values
    epochs = config['epochs']
    patience = config['patience']
    dropout = config['dropout']
    hidden_dim = config['hidden_dim']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    batch_size = config['batch_size']
    max_overfit_gap = config['max_overfit_gap']
    
    # Create output directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    
    # Model name
    view_suffix = "merged" if merged else viewpoint
    model_name = f"model_{view_suffix}"
    best_model_path = MODELS_DIR / f"{model_name}.pth"
    history_path = HISTORY_DIR / f"history_{view_suffix}.json"
    
    print(f"\n{'='*60}")
    print(f"Training HybridGCN V2 - {view_suffix.upper()}")
    print(f"{'='*60}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Dropout: {dropout}")
    print(f"Node embed: {NODE_EMBED_DIM}")
    print(f"Weight decay: {weight_decay}")
    print(f"Patience: {patience}")
    print(f"Device: {DEVICE}")
    
    # OPTIMIZED: Create weighted sampler for class balance
    sampler = create_weighted_sampler(train_dataset)
    
    # Create data loaders
    train_loader = GeoDataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # OPTIMIZED: Use weighted sampler instead of shuffle
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = GeoDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Get feature dimensions from first sample
    sample = train_dataset[0]
    num_node_features = sample.x.size(1)
    num_hybrid_features = sample.hybrid_features.size(0)
    
    print(f"Node features: {num_node_features}, Hybrid features: {num_hybrid_features}")
    
    # Initialize model
    model = HybridGCN(
        num_node_features=num_node_features,
        num_hybrid_features=num_hybrid_features,
        num_classes=NUM_CLASSES,
        hidden_dim=hidden_dim
    ).to(DEVICE)
    
    # OPTIMIZED: Xavier initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    print("[OK] Applied Xavier initialization")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # OPTIMIZED: Adam optimizer with weight decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # OPTIMIZED: Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )
    
    # Class weights for loss function
    class_weights = compute_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training history
    history = {
        'config': {
            'hidden_dim': hidden_dim,
            'num_layers': NUM_LAYERS,
            'dropout': dropout,
            'node_embed_dim': NODE_EMBED_DIM,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'patience': patience,
            'viewpoint': viewpoint,
            'merged': merged
        },
        'epochs': []
    }
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\nStarting training for up to {epochs} epochs...")
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        
        # Validate
        val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion)
        
        # OPTIMIZED: Compute overfitting gap
        overfit_gap = train_acc - val_acc
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Record history
        history['epochs'].append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'gap': overfit_gap,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.1f}%, "
              f"val_acc={val_acc:.1f}%, gap={overfit_gap:.1f}%, "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")
        
        # Check if best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'config': history['config']
            }, best_model_path)
            print(f"  [OK] Saved best model (val_acc={val_acc:.1f}%)")
        else:
            patience_counter += 1
        
        # OPTIMIZED: Early stop on severe overfitting
        if overfit_gap > max_overfit_gap:
            print(f"  [WARN] Severe overfitting detected (gap={overfit_gap:.1f}%), stopping...")
            break
        
        # Standard early stopping
        if patience_counter >= patience:
            print(f"  Early stopping after {epoch+1} epochs (no improvement for {patience} epochs)")
            break
    
    # Save final history
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {best_model_path}")
    print(f"History saved to: {history_path}")
    
    return best_val_acc, history


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Store default config values locally (avoid global modification)
    default_config = {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'dropout': DROPOUT,
        'hidden_dim': HIDDEN_DIM,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'batch_size': BATCH_SIZE,
        'max_overfit_gap': MAX_OVERFIT_GAP
    }
    
    parser = argparse.ArgumentParser(description='Train HybridGCN V2 (Optimized)')
    parser.add_argument('--viewpoint', choices=['front', 'left', 'right'],
                        help='Train specialist model for single viewpoint')
    parser.add_argument('--merged', action='store_true',
                        help='Train merged model using all viewpoints')
    parser.add_argument('--epochs', type=int, default=default_config['epochs'],
                        help=f"Number of epochs (default: {default_config['epochs']})")
    parser.add_argument('--patience', type=int, default=default_config['patience'],
                        help=f"Early stopping patience (default: {default_config['patience']})")
    parser.add_argument('--dropout', type=float, default=default_config['dropout'],
                        help=f"Dropout rate (default: {default_config['dropout']})")
    parser.add_argument('--hidden-dim', type=int, default=default_config['hidden_dim'],
                        help=f"Hidden dimension (default: {default_config['hidden_dim']})")
    parser.add_argument('--learning-rate', type=float, default=default_config['learning_rate'],
                        help=f"Learning rate (default: {default_config['learning_rate']})")
    
    args = parser.parse_args()
    
    # CRITICAL: March 2 models used different hidden_dims per viewpoint
    # Front: 128, Left/Right: 256
    if args.viewpoint and args.hidden_dim == default_config['hidden_dim']:
        if args.viewpoint in ['left', 'right']:
            args.hidden_dim = 256
            print(f"[INFO] Using HIDDEN_DIM=256 for {args.viewpoint} viewpoint (March 2 config)")
        else:
            args.hidden_dim = 128
            print(f"[INFO] Using HIDDEN_DIM=128 for {args.viewpoint} viewpoint (March 2 config)")
    
    # Build config dict from args (no globals modified)
    config = {
        'epochs': args.epochs,
        'patience': args.patience,
        'dropout': args.dropout,
        'hidden_dim': args.hidden_dim,
        'learning_rate': args.learning_rate,
        'weight_decay': default_config['weight_decay'],
        'batch_size': default_config['batch_size'],
        'max_overfit_gap': default_config['max_overfit_gap']
    }
    
    # Check data exists - support both per-viewpoint and merged files
    if args.viewpoint:
        # Try viewpoint-specific file first (e.g., train_features_front.pt)
        train_path = DATA_DIR / f"train_features_{args.viewpoint}.pt"
        test_path = DATA_DIR / f"test_features_{args.viewpoint}.pt"
        
        if not train_path.exists():
            # Fall back to merged file with filtering
            train_path = DATA_DIR / "train_features.pt"
            test_path = DATA_DIR / "test_features.pt"
            print(f"Viewpoint-specific file not found, using merged file with filtering")
    else:
        # Use merged file (default behavior)
        train_path = DATA_DIR / "train_features.pt"
        test_path = DATA_DIR / "test_features.pt"
    
    if not train_path.exists():
        print(f"Error: Training data not found at {train_path}")
        print("Run 2b_generate_node_hybrid_features.py first")
        sys.exit(1)
    
    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        sys.exit(1)
    
    # Load datasets with NaN filtering (CRITICAL FIX)
    print(f"\nLoading training data from {train_path}...")
    train_dataset = GraphDataset(train_path, viewpoint=args.viewpoint, filter_nan=True)
    
    print(f"\nLoading validation data from {test_path}...")
    val_dataset = GraphDataset(test_path, viewpoint=args.viewpoint, filter_nan=True)
    
    # Train with config dict (clean parameter passing, no globals)
    best_acc, history = train_model(
        train_dataset,
        val_dataset,
        viewpoint=args.viewpoint,
        merged=args.merged,
        config=config
    )
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULT: {best_acc:.1f}% validation accuracy")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
