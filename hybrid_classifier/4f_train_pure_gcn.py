"""
4f_train_pure_gcn.py — Plan A: Pure GCN Training
==================================================
Train PureGCN with person-invariant node features and attention pooling.
No hybrid MLP branch — classifies from body geometry alone.

Usage:
    python hybrid_classifier/4f_train_pure_gcn.py --viewpoint front --epochs 150
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeoDataLoader

from tqdm import tqdm

# Import PureGCN
sys.path.insert(0, str(Path(__file__).parent))
from models.pure_gcn import PureGCN

# =============================================================================
# PLAN A CONFIGURATION
# =============================================================================

HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.7
NODE_EMBED_DIM = 4

LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-3
EPOCHS = 150
PATIENCE = 20
BATCH_SIZE = 32

MAX_OVERFIT_GAP = 15.0

DATA_DIR = Path("hybrid_classifier/hybrid_features_v3")
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

# Pose-only edges (without stick connections)
POSE_EDGES = [
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
]

# Dynamic stick edges based on handedness
STICK_EDGES_RIGHT = [(16, 33), (33, 16), (33, 34), (34, 33)]
STICK_EDGES_LEFT = [(15, 33), (33, 15), (33, 34), (34, 33)]

# Binary task definitions
LEFT_CLASSES = {1, 2, 3, 4, 5}
RIGHT_CLASSES = {6, 7, 8, 9, 10}
BLOCK_CLASSES = {2, 4, 5, 7, 9, 10}
THRUST_CLASSES = {0, 1, 3, 6, 8, 11}


# =============================================================================
# FOCAL LOSS (Plan A)
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=1.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


# =============================================================================
# DATASET (Plan A: dynamic edges, no hybrid features)
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
            self.labels = self.data['labels'][mask]
            self.viewpoints = [v for v, m in zip(self.data['viewpoints'], mask) if m]
            self.has_stick_nodes = self.data['has_stick_nodes'][mask]
            self.stick_right_hand = self.data.get('stick_right_hand', torch.ones(len(self.data['labels']), dtype=torch.bool))
            if isinstance(self.stick_right_hand, torch.Tensor):
                self.stick_right_hand = self.stick_right_hand[mask]
            else:
                self.stick_right_hand = torch.tensor([self.stick_right_hand[i] for i, m in enumerate(mask) if m], dtype=torch.bool)
        elif viewpoint and not has_viewpoints and not is_per_viewpoint_file:
            raise ValueError(f"Viewpoint filtering requested ({viewpoint}) but 'viewpoints' key not found in {features_path}.")
        else:
            self.node_features = self.data['node_features']
            self.labels = self.data['labels']
            self.viewpoints = self.data.get('viewpoints', [viewpoint] * len(self.labels))
            self.has_stick_nodes = self.data.get('has_stick_nodes', torch.ones(len(self.labels), dtype=torch.bool))
            self.stick_right_hand = self.data.get('stick_right_hand', torch.ones(len(self.labels), dtype=torch.bool))
            if not isinstance(self.stick_right_hand, torch.Tensor):
                self.stick_right_hand = torch.tensor(self.stick_right_hand, dtype=torch.bool)

        if filter_nan:
            nan_mask = torch.isnan(self.node_features).any(dim=(1, 2))
            if nan_mask.sum() > 0:
                print(f"[WARN] Found {nan_mask.sum().item()} NaN samples, filtering them out")
                valid_mask = ~nan_mask
                self.node_features = self.node_features[valid_mask]
                self.labels = self.labels[valid_mask]
                self.viewpoints = [v for v, m in zip(self.viewpoints, valid_mask.tolist()) if m]
                self.has_stick_nodes = self.has_stick_nodes[valid_mask]
                self.stick_right_hand = self.stick_right_hand[valid_mask]

        print(f"Loaded {len(self)} samples" + (f" for {viewpoint} view" if viewpoint else " for all views"))

        class_counts = np.bincount(self.labels.numpy(), minlength=NUM_CLASSES)
        print("Class distribution:")
        for i, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
            if count > 0:
                print(f"  {name}: {count}")

    def _get_nan_mask(self):
        return torch.isnan(self.node_features).any(dim=(1, 2))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Build dynamic edges based on stick handedness
        edges = list(POSE_EDGES)
        if self.has_stick_nodes[idx]:
            if self.stick_right_hand[idx]:
                edges.extend(STICK_EDGES_RIGHT)
            else:
                edges.extend(STICK_EDGES_LEFT)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        data = Data(
            x=self.node_features[idx],
            edge_index=edge_index,
            y=self.labels[idx]
        )
        return data


def collate_fn(batch):
    return Batch.from_data_list(batch)


def create_weighted_sampler(dataset):
    labels = dataset.labels.numpy()
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    print(f"Class weights: {class_weights.round(4)}")
    return sampler


# =============================================================================
# CLASS WEIGHTS — Effective Number (Plan A)
# =============================================================================

def compute_effective_number_weights(train_dataset):
    labels = train_dataset.labels.numpy()
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * NUM_CLASSES
    print(f"Effective number weights: {weights.round(4)}")
    return torch.FloatTensor(weights).to(DEVICE)


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

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
    return total_loss / len(loader), accuracy, np.array(all_preds), np.array(all_labels)


def compute_detailed_metrics(preds, labels):
    """Compute Plan A requested metrics."""
    metrics = {}
    
    # Overall accuracy
    metrics['accuracy'] = 100.0 * (preds == labels).mean()
    
    # Per-class recall
    per_class_recall = {}
    for c in range(NUM_CLASSES):
        mask = (labels == c)
        if mask.sum() > 0:
            per_class_recall[c] = 100.0 * (preds[mask] == c).mean()
        else:
            per_class_recall[c] = 0.0
    metrics['per_class_recall'] = per_class_recall
    
    # Binary left vs right (excluding 0, 11, 12)
    left_right_mask = np.isin(labels, list(LEFT_CLASSES | RIGHT_CLASSES))
    if left_right_mask.sum() > 0:
        left_right_preds = np.isin(preds[left_right_mask], list(RIGHT_CLASSES)).astype(int)
        left_right_labels = np.isin(labels[left_right_mask], list(RIGHT_CLASSES)).astype(int)
        metrics['left_right_accuracy'] = 100.0 * (left_right_preds == left_right_labels).mean()
    else:
        metrics['left_right_accuracy'] = 0.0
    
    # Binary block vs thrust (excluding 12 neutral)
    block_thrust_mask = np.isin(labels, list(BLOCK_CLASSES | THRUST_CLASSES))
    if block_thrust_mask.sum() > 0:
        block_thrust_preds = np.isin(preds[block_thrust_mask], list(BLOCK_CLASSES)).astype(int)
        block_thrust_labels = np.isin(labels[block_thrust_mask], list(BLOCK_CLASSES)).astype(int)
        metrics['block_thrust_accuracy'] = 100.0 * (block_thrust_preds == block_thrust_labels).mean()
    else:
        metrics['block_thrust_accuracy'] = 0.0
    
    return metrics


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train_model(train_dataset, val_dataset, viewpoint=None, config=None):
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

    view_suffix = viewpoint or "all"
    model_name = f"pure_gcn_{view_suffix}"
    best_model_path = MODELS_DIR / f"{model_name}.pth"
    history_path = HISTORY_DIR / f"history_pure_gcn_{view_suffix}.json"

    print(f"\n{'='*60}")
    print(f"Training PureGCN — Plan A — {view_suffix.upper()}")
    print(f"{'='*60}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Num layers: {NUM_LAYERS}")
    print(f"Node embed: {NODE_EMBED_DIM}")
    print(f"Dropout: {dropout}")
    print(f"Weight decay: {weight_decay}")
    print(f"Batch size: {batch_size}")
    print(f"Patience: {patience}")
    print(f"Max overfit gap: {max_overfit_gap}%")
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

    print(f"Node features: {num_node_features}")

    model = PureGCN(
        num_node_features=num_node_features,
        num_classes=NUM_CLASSES,
        hidden_dim=hidden_dim,
        num_layers=NUM_LAYERS,
        node_embed_dim=NODE_EMBED_DIM,
        dropout=dropout
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
    
    class_weights = compute_effective_number_weights(train_dataset)
    criterion = FocalLoss(gamma=1.5, weight=class_weights)

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

        # Detailed metrics
        detailed = compute_detailed_metrics(preds, labels)

        history['epochs'].append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'gap': overfit_gap,
            'lr': optimizer.param_groups[0]['lr'],
            'left_right_acc': detailed['left_right_accuracy'],
            'block_thrust_acc': detailed['block_thrust_accuracy'],
            'per_class_recall': {int(k): float(v) for k, v in detailed['per_class_recall'].items()}
        })

        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.1f}%, "
              f"val_acc={val_acc:.1f}%, gap={overfit_gap:.1f}%, "
              f"L/R={detailed['left_right_accuracy']:.1f}%, "
              f"B/T={detailed['block_thrust_accuracy']:.1f}%, "
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

    parser = argparse.ArgumentParser(description='Train PureGCN — Plan A')
    parser.add_argument('--viewpoint', choices=['front', 'left', 'right'],
                        help='Train specialist model for single viewpoint')
    parser.add_argument('--epochs', type=int, default=default_config['epochs'])
    parser.add_argument('--patience', type=int, default=default_config['patience'])
    parser.add_argument('--dropout', type=float, default=default_config['dropout'])
    parser.add_argument('--hidden-dim', type=int, default=default_config['hidden_dim'])
    parser.add_argument('--learning-rate', type=float, default=default_config['learning_rate'])
    parser.add_argument('--weight-decay', type=float, default=default_config['weight_decay'])
    parser.add_argument('--batch-size', type=int, default=default_config['batch_size'])
    parser.add_argument('--max-overfit-gap', type=float, default=default_config['max_overfit_gap'])

    args = parser.parse_args()

    config = {
        'epochs': args.epochs,
        'patience': args.patience,
        'dropout': args.dropout,
        'hidden_dim': args.hidden_dim,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'max_overfit_gap': args.max_overfit_gap
    }

    viewpoint = args.viewpoint
    if not viewpoint:
        print("Error: --viewpoint is required (front, left, or right)")
        sys.exit(1)

    # Paths (look for PureGCN person-normalized features first)
    train_path = DATA_DIR / f"train_features_{viewpoint}_pure_gcn.pt"
    test_path = DATA_DIR / f"test_features_{viewpoint}_pure_gcn.pt"
    
    if not train_path.exists():
        print(f"Error: PureGCN train data not found: {train_path}")
        print(f"Run: python hybrid_classifier/2b_generate_node_hybrid_features.py --viewpoint {viewpoint} --pure_gcn")
        sys.exit(1)
    if not test_path.exists():
        print(f"Error: PureGCN test data not found: {test_path}")
        print(f"Run: python hybrid_classifier/2b_generate_node_hybrid_features.py --viewpoint {viewpoint} --pure_gcn")
        sys.exit(1)

    print(f"\n=== PureGCN Training ===")
    print(f"Train: {train_path}")
    print(f"Test:  {test_path}")

    train_dataset = GraphDataset(train_path, viewpoint=viewpoint, filter_nan=True)
    val_dataset = GraphDataset(test_path, viewpoint=viewpoint, filter_nan=True)

    best_acc, history = train_model(
        train_dataset, val_dataset,
        viewpoint=viewpoint, config=config
    )

    print(f"\n{'='*60}")
    print(f"FINAL RESULT: {best_acc:.1f}% validation accuracy")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
