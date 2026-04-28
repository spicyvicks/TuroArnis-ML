"""
HybridGCN V2 Training with Synthetic Data
===========================================
Train HybridGCN using combined real + synthetic feature tensors.
Based on 4c_train_hybrid_gcn_v2.py with identical architecture/hyperparameters.

Usage:
    # Train with 3x synthetic data (front view)
    python hybrid_classifier/4e_train_hybrid_gcn_v2_with_synthetic.py \
        --viewpoint front \
        --synthetic_factor 3 \
        --epochs 150

    # Train with 1x synthetic test data
    python hybrid_classifier/4e_train_hybrid_gcn_v2_with_synthetic.py \
        --viewpoint front \
        --synthetic_factor 3 \
        --test_synthetic_factor 1 \
        --epochs 150

Output: model_{viewpoint}_with_synthetic_{factor}x.pth
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
# CONFIGURATION (IDENTICAL TO 4c_train_hybrid_gcn_v2.py)
# =============================================================================

HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.5
NODE_EMBED_DIM = 8

LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-5
EPOCHS = 150
PATIENCE = 20
BATCH_SIZE = 64

MAX_OVERFIT_GAP = 35.0

DATA_DIR = Path("hybrid_classifier/hybrid_features_v4")
MODELS_DIR = Path("hybrid_classifier/models")
HISTORY_DIR = Path("hybrid_classifier/models")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct',
    'neutral'
]
NUM_CLASSES = len(CLASS_NAMES)

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
# MODEL ARCHITECTURE (IDENTICAL TO 4c)
# =============================================================================

class HybridGCN(nn.Module):
    def __init__(self, num_node_features, num_hybrid_features, num_classes, hidden_dim=HIDDEN_DIM):
        super(HybridGCN, self).__init__()
        self.node_embedding = nn.Embedding(35, NODE_EMBED_DIM)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features + NODE_EMBED_DIM, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim, track_running_stats=False))
        for _ in range(NUM_LAYERS - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim, track_running_stats=False))
        self.hybrid_mlp = nn.Sequential(
            nn.Linear(num_hybrid_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        hybrid_features = data.hybrid_features
        batch_size = batch.max().item() + 1
        node_indices = torch.arange(35, device=x.device).unsqueeze(0).expand(batch_size, -1)
        node_emb = self.node_embedding(node_indices).view(-1, NODE_EMBED_DIM)
        x = torch.cat([x, node_emb], dim=-1)
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            if x_new.size(-1) == x.size(-1):
                x = x_new + x
            else:
                x = x_new
        x_pool = global_mean_pool(x, batch)
        batch_size = batch.max().item() + 1
        hybrid_features = hybrid_features.view(batch_size, -1)
        hybrid_out = self.hybrid_mlp(hybrid_features)
        combined = torch.cat([x_pool, hybrid_out], dim=-1)
        x_out = F.relu(self.fc1(combined))
        x_out = self.dropout(x_out)
        logits = self.fc2(x_out)
        return logits


# =============================================================================
# DATASET (IDENTICAL TO 4c)
# =============================================================================

class GraphDataset(Dataset):
    def __init__(self, features_path, viewpoint=None, filter_nan=True):
        self.data = torch.load(features_path, map_location='cpu')
        self.viewpoint = viewpoint
        has_viewpoints = 'viewpoints' in self.data
        is_per_viewpoint_file = viewpoint and f"_{viewpoint}" in str(features_path)

        if viewpoint and has_viewpoints:
            mask = [v == viewpoint for v in self.data['viewpoints']]
            self.node_features = self.data['node_features'][mask]
            self.hybrid_features = self.data['hybrid_features'][mask]
            self.labels = self.data['labels'][mask]
            self.viewpoints = [v for v, m in zip(self.data['viewpoints'], mask) if m]
        elif viewpoint and not has_viewpoints and not is_per_viewpoint_file:
            raise ValueError(f"Viewpoint filtering requested ({viewpoint}) but 'viewpoints' key not found in {features_path}.")
        else:
            self.node_features = self.data['node_features']
            self.hybrid_features = self.data['hybrid_features']
            self.labels = self.data['labels']
            self.viewpoints = self.data.get('viewpoints', [viewpoint] * len(self.labels))

        if 'has_stick_nodes' in self.data:
            self.has_stick_nodes = self.data['has_stick_nodes']
            if viewpoint and has_viewpoints:
                self.has_stick_nodes = self.has_stick_nodes[mask]
        else:
            self.has_stick_nodes = torch.ones(len(self.labels), dtype=torch.bool)

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

        no_stick_count = (~self.has_stick_nodes).sum().item()
        if no_stick_count > 0:
            print(f"[INFO] {no_stick_count}/{len(self)} samples ({100*no_stick_count/len(self):.1f}%) without stick nodes")

        print(f"Loaded {len(self)} samples" + (f" for {viewpoint} view" if viewpoint else " for all views"))

        class_counts = np.bincount(self.labels.numpy(), minlength=NUM_CLASSES)
        print("Class distribution:")
        for i, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
            if count > 0:
                print(f"  {name}: {count}")

    def _get_nan_mask(self):
        node_nan = torch.isnan(self.node_features).any(dim=(1, 2))
        hybrid_nan = torch.isnan(self.hybrid_features).any(dim=1)
        return node_nan | hybrid_nan

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t().contiguous()
        if not self.has_stick_nodes[idx]:
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
    labels = dataset.labels.numpy()
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    print(f"Class weights: {class_weights.round(4)}")
    return sampler


def collate_fn(batch):
    return Batch.from_data_list(batch)


# =============================================================================
# TRAINING FUNCTIONS (IDENTICAL TO 4c)
# =============================================================================

def compute_class_weights(train_dataset):
    labels = train_dataset.labels.numpy()
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    total = len(labels)
    weights = total / (NUM_CLASSES * class_counts + 1e-6)
    return torch.FloatTensor(weights).to(DEVICE)


def train_epoch(model, loader, optimizer, criterion):
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


# =============================================================================
# SYNTHETIC DATA LOADING (NEW)
# =============================================================================

def load_combined_dataset(real_path, synthetic_path, viewpoint=None):
    """Load real data and synthetic data, combine into single dataset object."""
    real_data = torch.load(real_path, map_location='cpu')
    syn_data = torch.load(synthetic_path, map_location='cpu')

    # Verify keys match
    for key in ['node_features', 'hybrid_features', 'labels', 'has_stick_nodes']:
        if key not in real_data:
            raise ValueError(f"Real data missing key: {key}")
        if key not in syn_data:
            raise ValueError(f"Synthetic data missing key: {key}")

    # Combine tensors
    real_stick_hand = real_data.get('stick_right_hand', torch.ones(len(real_data['labels']), dtype=torch.bool))
    syn_stick_hand = syn_data.get('stick_right_hand', torch.ones(len(syn_data['labels']), dtype=torch.bool))
    combined = {
        'node_features': torch.cat([real_data['node_features'], syn_data['node_features']]),
        'hybrid_features': torch.cat([real_data['hybrid_features'], syn_data['hybrid_features']]),
        'labels': torch.cat([real_data['labels'], syn_data['labels']]),
        'viewpoints': real_data.get('viewpoints', ['front'] * len(real_data['labels'])) + \
                      syn_data.get('viewpoints', ['front'] * len(syn_data['labels'])),
        'has_stick_nodes': torch.cat([
            real_data.get('has_stick_nodes', torch.ones(len(real_data['labels']), dtype=torch.bool)),
            syn_data.get('has_stick_nodes', torch.ones(len(syn_data['labels']), dtype=torch.bool))
        ]),
        'is_synthetic': torch.cat([
            torch.zeros(len(real_data['labels']), dtype=torch.bool),
            syn_data.get('is_synthetic', torch.ones(len(syn_data['labels']), dtype=torch.bool))
        ]),
        'stick_right_hand': torch.cat([real_stick_hand, syn_stick_hand]),
    }

    # Save combined for potential reuse
    combined_path = synthetic_path.parent / f"combined_{synthetic_path.stem}.pt"
    torch.save(combined, combined_path)
    print(f"[OK] Saved combined dataset: {combined_path}")

    # Build a dataset-like object from combined dict
    class CombinedDataset(Dataset):
        def __init__(self, combined_dict, viewpoint=None):
            self.data = combined_dict
            self.viewpoint = viewpoint
            self.node_features = self.data['node_features']
            self.hybrid_features = self.data['hybrid_features']
            self.labels = self.data['labels']
            self.viewpoints = self.data['viewpoints']
            self.has_stick_nodes = self.data['has_stick_nodes']
            self.is_synthetic = self.data['is_synthetic']

            # Filter by viewpoint if needed
            if viewpoint:
                mask = [v == viewpoint for v in self.viewpoints]
                self.node_features = self.node_features[mask]
                self.hybrid_features = self.hybrid_features[mask]
                self.labels = self.labels[mask]
                self.viewpoints = [v for v, m in zip(self.viewpoints, mask) if m]
                self.has_stick_nodes = self.has_stick_nodes[mask]
                self.is_synthetic = self.is_synthetic[mask]

            # NaN filter
            node_nan = torch.isnan(self.node_features).any(dim=(1, 2))
            hybrid_nan = torch.isnan(self.hybrid_features).any(dim=1)
            nan_mask = node_nan | hybrid_nan
            if nan_mask.sum() > 0:
                print(f"[WARN] Found {nan_mask.sum().item()} NaN samples in combined data, filtering")
                valid = ~nan_mask
                self.node_features = self.node_features[valid]
                self.hybrid_features = self.hybrid_features[valid]
                self.labels = self.labels[valid]
                self.viewpoints = [v for v, m in zip(self.viewpoints, valid.tolist()) if m]
                self.has_stick_nodes = self.has_stick_nodes[valid]
                self.is_synthetic = self.is_synthetic[valid]

            print(f"Combined dataset: {len(self)} total samples")
            real_count = (~self.is_synthetic).sum().item()
            syn_count = self.is_synthetic.sum().item()
            print(f"  Real: {real_count}, Synthetic: {syn_count}")

            class_counts = np.bincount(self.labels.numpy(), minlength=NUM_CLASSES)
            print("Class distribution:")
            for i, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
                if count > 0:
                    print(f"  {name}: {count}")

        def _get_nan_mask(self):
            node_nan = torch.isnan(self.node_features).any(dim=(1, 2))
            hybrid_nan = torch.isnan(self.hybrid_features).any(dim=1)
            return node_nan | hybrid_nan

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t().contiguous()
            if not self.has_stick_nodes[idx]:
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

    return CombinedDataset(combined, viewpoint=viewpoint)


# =============================================================================
# MAIN TRAINING LOOP (BASED ON 4c)
# =============================================================================

def train_model(train_dataset, val_dataset, viewpoint=None, config=None, synthetic_factor=3):
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

    epochs = config['epochs']
    patience = config['patience']
    dropout = config['dropout']
    hidden_dim = config['hidden_dim']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    batch_size = config['batch_size']
    max_overfit_gap = config['max_overfit_gap']

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    view_suffix = viewpoint
    model_name = f"model_{view_suffix}_with_synthetic_{synthetic_factor}x"
    best_model_path = MODELS_DIR / f"{model_name}.pth"
    history_path = HISTORY_DIR / f"history_{view_suffix}_with_synthetic_{synthetic_factor}x.json"

    print(f"\n{'='*60}")
    print(f"Training HybridGCN V2 WITH SYNTHETIC - {view_suffix.upper()}")
    print(f"Synthetic factor: {synthetic_factor}x")
    print(f"{'='*60}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Dropout: {dropout}")
    print(f"Node embed: {NODE_EMBED_DIM}")
    print(f"Weight decay: {weight_decay}")
    print(f"Patience: {patience}")
    print(f"Device: {DEVICE}")

    sampler = create_weighted_sampler(train_dataset)

    train_loader = GeoDataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = GeoDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    sample = train_dataset[0]
    num_node_features = sample.x.size(1)
    num_hybrid_features = sample.hybrid_features.size(0)

    print(f"Node features: {num_node_features}, Hybrid features: {num_hybrid_features}")

    model = HybridGCN(
        num_node_features=num_node_features,
        num_hybrid_features=num_hybrid_features,
        num_classes=NUM_CLASSES,
        hidden_dim=hidden_dim
    ).to(DEVICE)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)
    print("[OK] Applied Xavier initialization")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    class_weights = compute_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

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
            'synthetic_factor': synthetic_factor,
        },
        'epochs': []
    }

    best_val_acc = 0.0
    patience_counter = 0

    print(f"\nStarting training for up to {epochs} epochs...")

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion)
        overfit_gap = train_acc - val_acc
        scheduler.step(val_acc)

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

        if overfit_gap > max_overfit_gap:
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


def main():
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

    parser = argparse.ArgumentParser(description='Train HybridGCN V2 with Synthetic Data')
    parser.add_argument('--viewpoint', choices=['front', 'left', 'right'],
                        help='Train specialist model for single viewpoint')
    parser.add_argument('--synthetic_factor', type=int, default=3,
                        help='Training synthetic multiplication factor (default: 3)')
    parser.add_argument('--test_synthetic_factor', type=int, default=1,
                        help='Test synthetic multiplication factor (default: 1)')
    parser.add_argument('--epochs', type=int, default=default_config['epochs'])
    parser.add_argument('--patience', type=int, default=default_config['patience'])
    parser.add_argument('--dropout', type=float, default=default_config['dropout'])
    parser.add_argument('--hidden-dim', type=int, default=default_config['hidden_dim'])
    parser.add_argument('--learning-rate', type=float, default=default_config['learning_rate'])
    parser.add_argument('--no_synthetic', action='store_true',
                        help='Train WITHOUT synthetic data (baseline comparison)')

    args = parser.parse_args()

    if args.viewpoint and args.hidden_dim == default_config['hidden_dim']:
        if args.viewpoint in ['left', 'right']:
            args.hidden_dim = 256
            print(f"[INFO] Using HIDDEN_DIM=256 for {args.viewpoint} viewpoint")
        else:
            args.hidden_dim = 128
            print(f"[INFO] Using HIDDEN_DIM=128 for {args.viewpoint} viewpoint")

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

    viewpoint = args.viewpoint
    if not viewpoint:
        print("Error: --viewpoint is required (front, left, or right)")
        sys.exit(1)

    # Paths
    real_train_path = DATA_DIR / f"train_features_{viewpoint}.pt"
    real_test_path = DATA_DIR / f"test_features_{viewpoint}.pt"

    if not real_train_path.exists():
        print(f"Error: Real train data not found: {real_train_path}")
        sys.exit(1)
    if not real_test_path.exists():
        print(f"Error: Real test data not found: {real_test_path}")
        sys.exit(1)

    if args.no_synthetic:
        # Baseline: train without synthetic
        print("\n=== BASELINE MODE (no synthetic data) ===")
        train_dataset = GraphDataset(real_train_path, viewpoint=viewpoint, filter_nan=True)
        val_dataset = GraphDataset(real_test_path, viewpoint=viewpoint, filter_nan=True)
        best_acc, history = train_model(
            train_dataset, val_dataset,
            viewpoint=viewpoint, config=config, synthetic_factor=0
        )
    else:
        # With synthetic
        syn_train_path = DATA_DIR / f"synthetic_train_features_{viewpoint}_{args.synthetic_factor}x.pt"
        syn_test_path = DATA_DIR / f"synthetic_test_features_{viewpoint}_{args.test_synthetic_factor}x.pt"

        if not syn_train_path.exists():
            print(f"Error: Synthetic train data not found: {syn_train_path}")
            print(f"Run: python hybrid_classifier/2c_generate_synthetic_features.py --viewpoint {viewpoint}")
            sys.exit(1)
        if not syn_test_path.exists():
            print(f"Error: Synthetic test data not found: {syn_test_path}")
            print(f"Run: python hybrid_classifier/2c_generate_synthetic_features.py --viewpoint {viewpoint}")
            sys.exit(1)

        print(f"\n=== SYNTHETIC MODE ===")
        print(f"Train synthetic: {syn_train_path}")
        print(f"Test synthetic: {syn_test_path}")

        train_dataset = load_combined_dataset(real_train_path, syn_train_path, viewpoint=viewpoint)
        val_dataset = load_combined_dataset(real_test_path, syn_test_path, viewpoint=viewpoint)

        best_acc, history = train_model(
            train_dataset, val_dataset,
            viewpoint=viewpoint, config=config, synthetic_factor=args.synthetic_factor
        )

    print(f"\n{'='*60}")
    print(f"FINAL RESULT: {best_acc:.1f}% validation accuracy")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
