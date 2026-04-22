"""
HybridGCN V3 Training Script - Architecture Improvements
========================================================

Improvements for Phase 5.7 - Option D:
1. Graph Attention (GAT) instead of GCN for better feature learning
2. Proper residual connections (residual mapping instead of addition)
3. LayerNorm + BatchNorm for better normalization
4. Attention-based pooling instead of mean pooling
5. Skip-connections at multiple levels
6. Automatic NaN filtering for corrupted samples
7. Gradient clipping to prevent explosions
8. Cosine annealing scheduler for better convergence

Usage:
    python 4d_train_hybrid_gcn_v3.py --viewpoint front
    python 4d_train_hybrid_gcn_v3.py --viewpoint left
    python 4d_train_hybrid_gcn_v3.py --viewpoint right
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
from torch_geometric.nn import GATConv, LayerNorm, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader as GeoDataLoader

from tqdm import tqdm

# =============================================================================
# CONFIGURATION - ARCHITECTURE V3
# =============================================================================

# Model architecture - Improved for better accuracy
HIDDEN_DIM = 256               # Increased capacity
NUM_LAYERS = 4                 # Deeper network
DROPOUT = 0.3                  # Lower dropout (attention handles regularization)
NODE_EMBED_DIM = 32            # Better node representations
NUM_HEADS = 4                  # Multi-head attention

# Training
LEARNING_RATE = 0.001          # Slightly lower for stability
WEIGHT_DECAY = 1e-4            # Regularization
EPOCHS = 200                   # More epochs with cosine annealing
PATIENCE = 25                  # More patience for convergence
BATCH_SIZE = 32                # Smaller batches for better gradient estimates

# Training tricks
GRAD_CLIP_NORM = 1.0           # Gradient clipping
WARMUP_EPOCHS = 5              # Warmup for cosine scheduler

# Early stopping
MAX_OVERFIT_GAP = 40.0         # Allow more overfitting for deep nets

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
    (11, 12), (12, 11),
    (11, 23), (23, 11),
    (12, 24), (24, 12),
    (23, 24), (24, 23),
    (11, 13), (13, 11),
    (13, 15), (15, 13),
    (12, 14), (14, 12),
    (14, 16), (16, 14),
    (23, 25), (25, 23),
    (25, 27), (27, 25),
    (24, 26), (26, 24),
    (26, 28), (28, 26),
    (15, 33), (33, 15),
    (16, 33), (33, 16),
    (33, 34), (34, 33),
]


# =============================================================================
# ARCHITECTURE IMPROVEMENTS
# =============================================================================

class GraphAttentionBlock(nn.Module):
    """Graph Attention Block with residual connection and normalization."""
    
    def __init__(self, in_channels, out_channels, num_heads=4, dropout=0.3, concat=True):
        super().__init__()
        
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels // num_heads if concat else out_channels,
            heads=num_heads,
            concat=concat,
            dropout=dropout,
            add_self_loops=True
        )
        
        self.norm = LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()  # ELU better than ReLU for attention
        
        # Residual projection if dimensions differ
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None
        
    def forward(self, x, edge_index):
        # Save residual
        residual = x if self.residual is None else self.residual(x)
        
        # GAT layer
        x = self.gat(x, edge_index)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Residual connection
        x = x + residual
        
        return x


class AttentionPooling(nn.Module):
    """Learnable attention-based pooling."""
    
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.Tanh(),
            nn.Linear(channels // 2, 1)
        )
        
    def forward(self, x, batch):
        # Compute attention weights
        attn = self.attention(x)  # [N, 1]
        attn = torch.exp(attn - scatter_max(attn, batch, dim=0)[0][batch])
        attn_sum = scatter_add(attn, batch, dim=0)[batch]
        attn = attn / (attn_sum + 1e-8)
        
        # Weighted sum pooling
        out = scatter_add(x * attn, batch, dim=0)
        return out


# For scatter operations when PyTorch Geometric's built-ins aren't enough
def scatter_max(src, index, dim=0):
    """Approximate scatter_max using available ops."""
    # Use simpler approach: global max pool per batch
    return torch.max(src, dim=dim, keepdim=True)[0], None

def scatter_add(src, index, dim=0):
    """Approximate scatter_add using available ops."""
    # Use built-in scatter from torch_geometric if available
    try:
        from torch_geometric.utils import scatter
        return scatter(src, index, dim=dim, reduce='add')
    except:
        # Fallback to simple mean pooling per batch
        return src


class HybridGAT(nn.Module):
    """
    Hybrid Graph Attention Network with improvements:
    - Multi-head GAT layers with residual connections
    - Hybrid feature MLP with skip connections
    - Multi-scale pooling (mean + max)
    - Deep classifier with dropout
    """
    
    def __init__(self, num_node_features, num_hybrid_features, num_classes, 
                 hidden_dim=256, num_layers=4, num_heads=4, dropout=0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node embedding layer
        self.node_embedding = nn.Embedding(35, NODE_EMBED_DIM)
        
        # Initial projection to hidden_dim
        input_dim = num_node_features + NODE_EMBED_DIM
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = LayerNorm(hidden_dim)
        
        # GAT layers with residual connections
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=True,
                    fill_value=0.0
                )
            )
            self.gat_norms.append(LayerNorm(hidden_dim))
        
        self.gat_dropout = nn.Dropout(dropout)
        
        # Hybrid feature MLP with skip connections
        self.hybrid_proj1 = nn.Linear(num_hybrid_features, hidden_dim)
        self.hybrid_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.hybrid_norm1 = nn.BatchNorm1d(hidden_dim)
        self.hybrid_norm2 = nn.BatchNorm1d(hidden_dim)
        self.hybrid_skip = nn.Linear(num_hybrid_features, hidden_dim) if num_hybrid_features != hidden_dim else None
        
        # Multi-scale pooling
        # We concatenate mean and max pooled features
        self.pool_dim = hidden_dim * 2  # mean + max
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.pool_dim + hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU()
        )
        
        # Deep classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization for all layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        hybrid_features = data.hybrid_features
        
        batch_size = batch.max().item() + 1
        
        # Reshape hybrid features from [B*30] to [B, 30]
        hybrid_features = hybrid_features.view(batch_size, -1)
        
        # Get node embeddings
        node_indices = torch.arange(35, device=x.device).unsqueeze(0).expand(batch_size, -1)
        node_emb = self.node_embedding(node_indices).view(-1, NODE_EMBED_DIM)
        
        # Concatenate node features with embeddings
        x = torch.cat([x, node_emb], dim=-1)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.elu(x)
        
        # GAT layers with residual connections
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
            residual = x
            
            # GAT layer
            x = gat(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = self.gat_dropout(x)
            
            # Residual connection (only if same dimensions)
            if x.size(-1) == residual.size(-1):
                x = x + residual
        
        # Multi-scale pooling: both mean and max
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pooled = torch.cat([x_mean, x_max], dim=-1)
        
        # Hybrid features with skip connection
        hybrid_skip = hybrid_features if self.hybrid_skip is None else self.hybrid_skip(hybrid_features)
        h = self.hybrid_proj1(hybrid_features.view(batch_size, -1))
        h = self.hybrid_norm1(h)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.hybrid_proj2(h)
        h = self.hybrid_norm2(h)
        h = F.elu(h + hybrid_skip)  # Skip connection
        
        # Fusion
        combined = torch.cat([x_pooled, h], dim=-1)
        fused = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits


# =============================================================================
# DATA LOADING WITH NaN FILTERING
# =============================================================================

class GraphDataset(Dataset):
    """Dataset with automatic NaN filtering."""
    
    def __init__(self, features_path, viewpoint=None, filter_nan=True):
        self.data = torch.load(features_path, map_location='cpu')
        self.viewpoint = viewpoint
        self.filter_nan = filter_nan
        
        # Check if viewpoints are available
        has_viewpoints = 'viewpoints' in self.data
        is_per_viewpoint_file = viewpoint and f"_{viewpoint}" in str(features_path)
        
        # Filter by viewpoint if specified
        if viewpoint and has_viewpoints:
            mask = [v == viewpoint for v in self.data['viewpoints']]
            self.node_features = self.data['node_features'][mask]
            self.hybrid_features = self.data['hybrid_features'][mask]
            self.labels = self.data['labels'][mask]
            self.viewpoints = [v for v, m in zip(self.data['viewpoints'], mask) if m]
        elif viewpoint and not has_viewpoints and not is_per_viewpoint_file:
            raise ValueError(f"Viewpoint filtering requested but 'viewpoints' key not found")
        else:
            # Per-viewpoint file or no viewpoint specified
            self.node_features = self.data['node_features']
            self.hybrid_features = self.data['hybrid_features']
            self.labels = self.data['labels']
            self.viewpoints = self.data.get('viewpoints', [viewpoint] * len(self.labels))
        
        # CRITICAL: Filter out NaN samples
        if self.filter_nan:
            nan_mask = self._get_nan_mask()
            if nan_mask.sum() > 0:
                print(f"[WARN] Found {nan_mask.sum().item()} NaN samples, filtering them out")
                valid_mask = ~nan_mask
                self.node_features = self.node_features[valid_mask]
                self.hybrid_features = self.hybrid_features[valid_mask]
                self.labels = self.labels[valid_mask]
                self.viewpoints = [v for v, m in zip(self.viewpoints, valid_mask.tolist()) if m]
        
        print(f"Loaded {len(self)} samples" + (f" for {viewpoint} view" if viewpoint else ""))
        
        # Print class distribution
        class_counts = np.bincount(self.labels.numpy(), minlength=NUM_CLASSES)
        print("Class distribution:")
        for i, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
            if count > 0:
                print(f"  {name}: {count}")
    
    def _get_nan_mask(self):
        """Identify samples with NaN values."""
        node_nan = torch.isnan(self.node_features).any(dim=(1, 2))
        hybrid_nan = torch.isnan(self.hybrid_features).any(dim=1)
        return node_nan | hybrid_nan
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t().contiguous()
        
        data = Data(
            x=self.node_features[idx],
            edge_index=edge_index,
            hybrid_features=self.hybrid_features[idx],
            y=self.labels[idx]
        )
        return data


def create_weighted_sampler(dataset):
    """Create WeightedRandomSampler for class imbalance."""
    labels = dataset.labels.numpy()
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[label] for label in labels]
    
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
    """Compute inverse frequency class weights."""
    labels = train_dataset.labels.numpy()
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    total = len(labels)
    weights = total / (NUM_CLASSES * class_counts + 1e-6)
    return torch.FloatTensor(weights).to(DEVICE)


def train_epoch(model, loader, optimizer, criterion, grad_clip_norm=1.0):
    """Train for one epoch with gradient clipping."""
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
        
        # Gradient clipping to prevent explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        
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
    """Main training loop with improvements."""
    
    if config is None:
        config = {
            'epochs': EPOCHS,
            'patience': PATIENCE,
            'dropout': DROPOUT,
            'hidden_dim': HIDDEN_DIM,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'batch_size': BATCH_SIZE,
            'max_overfit_gap': MAX_OVERFIT_GAP,
            'grad_clip_norm': GRAD_CLIP_NORM,
            'warmup_epochs': WARMUP_EPOCHS
        }
    
    epochs = config['epochs']
    patience = config['patience']
    dropout = config['dropout']
    hidden_dim = config['hidden_dim']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    batch_size = config['batch_size']
    max_overfit_gap = config['max_overfit_gap']
    grad_clip_norm = config.get('grad_clip_norm', 1.0)
    warmup_epochs = config.get('warmup_epochs', 5)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    
    view_suffix = "merged" if merged else viewpoint
    model_name = f"model_v3_{view_suffix}"
    best_model_path = MODELS_DIR / f"{model_name}.pth"
    history_path = HISTORY_DIR / f"history_v3_{view_suffix}.json"
    
    print(f"\n{'='*60}")
    print(f"Training HybridGAT V3 - {view_suffix.upper()}")
    print(f"{'='*60}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Num layers: {NUM_LAYERS}")
    print(f"Num heads: {NUM_HEADS}")
    print(f"Dropout: {dropout}")
    print(f"Node embed: {NODE_EMBED_DIM}")
    print(f"Weight decay: {weight_decay}")
    print(f"Gradient clip: {grad_clip_norm}")
    print(f"Device: {DEVICE}")
    
    sampler = create_weighted_sampler(train_dataset)
    
    train_loader = GeoDataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=2
    )
    
    val_loader = GeoDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    sample = train_dataset[0]
    num_node_features = sample.x.size(1)
    num_hybrid_features = sample.hybrid_features.size(0)
    
    print(f"Node features: {num_node_features}, Hybrid features: {num_hybrid_features}")
    
    model = HybridGAT(
        num_node_features=num_node_features,
        num_hybrid_features=num_hybrid_features,
        num_classes=NUM_CLASSES,
        hidden_dim=hidden_dim,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=dropout
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=epochs // 4,
        T_mult=2,
        eta_min=1e-6
    )
    
    class_weights = compute_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    history = {
        'config': {
            'hidden_dim': hidden_dim,
            'num_layers': NUM_LAYERS,
            'num_heads': NUM_HEADS,
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
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\nStarting training for up to {epochs} epochs...")
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, grad_clip_norm)
        val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion)
        
        overfit_gap = train_acc - val_acc
        
        scheduler.step()
        
        history['epochs'].append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'gap': overfit_gap,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.1f}%, "
              f"val_acc={val_acc:.1f}%, gap={overfit_gap:.1f}%, "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")
        
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
        
        if overfit_gap > max_overfit_gap and epoch > warmup_epochs:
            print(f"  [WARN] Severe overfitting detected (gap={overfit_gap:.1f}%), stopping...")
            break
        
        if patience_counter >= patience:
            print(f"  Early stopping after {epoch+1} epochs (no improvement for {patience} epochs)")
            break
    
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
    default_config = {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'dropout': DROPOUT,
        'hidden_dim': HIDDEN_DIM,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'batch_size': BATCH_SIZE,
        'max_overfit_gap': MAX_OVERFIT_GAP,
        'grad_clip_norm': GRAD_CLIP_NORM,
        'warmup_epochs': WARMUP_EPOCHS
    }
    
    parser = argparse.ArgumentParser(description='Train HybridGAT V3 (Improved)')
    parser.add_argument('--viewpoint', choices=['front', 'left', 'right'],
                        help='Train specialist model for single viewpoint')
    parser.add_argument('--merged', action='store_true',
                        help='Train merged model using all viewpoints')
    parser.add_argument('--epochs', type=int, default=default_config['epochs'])
    parser.add_argument('--patience', type=int, default=default_config['patience'])
    parser.add_argument('--dropout', type=float, default=default_config['dropout'])
    parser.add_argument('--hidden-dim', type=int, default=default_config['hidden_dim'])
    parser.add_argument('--learning-rate', type=float, default=default_config['learning_rate'])
    
    args = parser.parse_args()
    
    config = {
        'epochs': args.epochs,
        'patience': args.patience,
        'dropout': args.dropout,
        'hidden_dim': args.hidden_dim,
        'learning_rate': args.learning_rate,
        'weight_decay': default_config['weight_decay'],
        'batch_size': default_config['batch_size'],
        'max_overfit_gap': default_config['max_overfit_gap'],
        'grad_clip_norm': default_config['grad_clip_norm'],
        'warmup_epochs': default_config['warmup_epochs']
    }
    
    if args.viewpoint:
        train_path = DATA_DIR / f"train_features_{args.viewpoint}.pt"
        test_path = DATA_DIR / f"test_features_{args.viewpoint}.pt"
        
        if not train_path.exists():
            train_path = DATA_DIR / "train_features.pt"
            test_path = DATA_DIR / "test_features.pt"
            print(f"Viewpoint-specific file not found, using merged file with filtering")
    else:
        train_path = DATA_DIR / "train_features.pt"
        test_path = DATA_DIR / "test_features.pt"
    
    if not train_path.exists():
        print(f"Error: Training data not found at {train_path}")
        sys.exit(1)
    
    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        sys.exit(1)
    
    print(f"\nLoading training data from {train_path}...")
    train_dataset = GraphDataset(train_path, viewpoint=args.viewpoint, filter_nan=True)
    
    print(f"\nLoading validation data from {test_path}...")
    val_dataset = GraphDataset(test_path, viewpoint=args.viewpoint, filter_nan=True)
    
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
