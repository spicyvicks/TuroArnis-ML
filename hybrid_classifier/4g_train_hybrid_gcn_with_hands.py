"""
4g_train_hybrid_gcn_with_hands.py
==================================
Train HybridGCN with hand landmarks (41 nodes total).
Based on 4e but with expanded graph for hand nodes.

Node layout:
  0-32: MediaPipe Pose
  33-34: Stick (grip, tip)
  35: Hand wrist
  36-40: Hand fingertips (thumb, index, middle, ring, pinky)

Output: model_{viewpoint}_with_hands_{synthetic_factor}x.pth
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
# CONFIGURATION (same as 4e)
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

# 41 nodes: 33 pose + 2 stick + 6 hand
NUM_NODES = 41

# Skeleton edges including hand connections
SKELETON_EDGES = [
    # Body edges (same as 4e)
    (11, 12), (12, 11), (11, 23), (23, 11), (12, 24), (24, 12),
    (23, 24), (24, 23), (11, 13), (13, 11), (13, 15), (15, 13),
    (12, 14), (14, 12), (14, 16), (16, 14), (23, 25), (25, 23),
    (25, 27), (27, 25), (24, 26), (26, 24), (26, 28), (28, 26),
    (15, 33), (33, 15), (16, 33), (33, 16), (33, 34), (34, 33),
    # Hand internal edges (wrist to each fingertip)
    (35, 36), (36, 35), (35, 37), (37, 35), (35, 38), (38, 35),
    (35, 39), (39, 35), (35, 40), (40, 35),
    # Hand to pose wrists (allow model to learn association)
    (35, 15), (15, 35), (35, 16), (16, 35),
    # Hand to stick grip
    (35, 33), (33, 35),
]


# =============================================================================
# MODEL (same architecture, 41 nodes)
# =============================================================================

class HybridGCNWithHands(nn.Module):
    def __init__(self, num_node_features, num_hybrid_features, num_classes, hidden_dim=HIDDEN_DIM):
        super(HybridGCNWithHands, self).__init__()
        self.node_embedding = nn.Embedding(NUM_NODES, NODE_EMBED_DIM)
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
        node_indices = torch.arange(NUM_NODES, device=x.device).unsqueeze(0).expand(batch_size, -1)
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
# DATASET
# =============================================================================

class GraphDatasetWithHands(Dataset):
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
            raise ValueError(f"Viewpoint filtering requested but 'viewpoints' key not found")
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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        nf = self.node_features[idx]
        hf = self.hybrid_features[idx]
        label = self.labels[idx]

        edge_index = torch.tensor(
            [[s, t] for s, t in SKELETON_EDGES] + [[t, s] for s, t in SKELETON_EDGES],
            dtype=torch.long
        ).t()

        return Data(x=nf, edge_index=edge_index, y=label, hybrid_features=hf.unsqueeze(0))


def collate_fn(batch):
    return Batch.from_data_list(batch)


# =============================================================================
# TRAINING (same as 4e)
# =============================================================================

def create_weighted_sampler(dataset):
    labels = dataset.labels.numpy()
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs
    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.num_graphs
    return total_loss / total, 100.0 * correct / total


def train_model(train_dataset, val_dataset, viewpoint, config, synthetic_factor=0):
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

    model_name = f"model_{viewpoint}_with_hands_{synthetic_factor}x"
    best_model_path = MODELS_DIR / f"{model_name}.pth"
    history_path = HISTORY_DIR / f"history_{viewpoint}_with_hands_{synthetic_factor}x.json"

    print(f"\n{'='*60}")
    print(f"Training HybridGCN WITH HANDS - {viewpoint.upper()}")
    print(f"Nodes: {NUM_NODES} (33 body + 2 stick + 6 hand)")
    print(f"Synthetic factor: {synthetic_factor}x")
    print(f"{'='*60}")

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
    num_hybrid_features = sample.hybrid_features.size(1)

    model = HybridGCNWithHands(num_node_features, num_hybrid_features, NUM_CLASSES, hidden_dim).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [], 'lr': []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        gap = train_acc - val_acc
        scheduler.step(val_acc)

        print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.1f}%, "
              f"val_acc={val_acc:.1f}%, gap={gap:.1f}%, lr={current_lr:.6f}")

        if gap > max_overfit_gap:
            print(f"  [WARN] Severe overfitting detected (gap={gap:.1f}%), stopping...")
            break

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'config': config
            }, best_model_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping after {epoch} epochs (no improvement for {patience} epochs)")
                break

    with open(history_path, 'w') as f:
        json.dump(history, f)

    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {best_model_path}")
    print(f"History saved to: {history_path}")

    return best_val_acc, history


class CombinedGraphDatasetWithHands(Dataset):
    """Dataset that works like GraphDatasetWithHands but from in-memory tensors."""
    def __init__(self, node_features, hybrid_features, labels, viewpoints=None, has_stick_nodes=None):
        self.node_features = node_features
        self.hybrid_features = hybrid_features
        self.labels = labels
        self.viewpoints = viewpoints if viewpoints else ['front'] * len(labels)
        self.has_stick_nodes = has_stick_nodes if has_stick_nodes else torch.ones(len(labels), dtype=torch.bool)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        nf = self.node_features[idx]
        hf = self.hybrid_features[idx]
        label = self.labels[idx]

        edge_index = torch.tensor(
            [[s, t] for s, t in SKELETON_EDGES] + [[t, s] for s, t in SKELETON_EDGES],
            dtype=torch.long
        ).t()

        return Data(x=nf, edge_index=edge_index, y=label, hybrid_features=hf.unsqueeze(0))


def load_combined_dataset(real_path, synthetic_path, viewpoint=None):
    """Combine real + synthetic datasets."""
    real_data = torch.load(real_path, map_location='cpu')
    syn_data = torch.load(synthetic_path, map_location='cpu')

    if viewpoint and 'viewpoints' in real_data:
        mask = torch.tensor([v == viewpoint for v in real_data['viewpoints']])
        real_node = real_data['node_features'][mask]
        real_hybrid = real_data['hybrid_features'][mask]
        real_labels = real_data['labels'][mask]
    else:
        real_node = real_data['node_features']
        real_hybrid = real_data['hybrid_features']
        real_labels = real_data['labels']

    if viewpoint and 'viewpoints' in syn_data:
        mask = torch.tensor([v == viewpoint for v in syn_data['viewpoints']])
        syn_node = syn_data['node_features'][mask]
        syn_hybrid = syn_data['hybrid_features'][mask]
        syn_labels = syn_data['labels'][mask]
    else:
        syn_node = syn_data['node_features']
        syn_hybrid = syn_data['hybrid_features']
        syn_labels = syn_data['labels']

    combined_node = torch.cat([real_node, syn_node])
    combined_hybrid = torch.cat([real_hybrid, syn_hybrid])
    combined_labels = torch.cat([real_labels, syn_labels])

    return CombinedGraphDatasetWithHands(combined_node, combined_hybrid, combined_labels)


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

    parser = argparse.ArgumentParser(description='Train HybridGCN with Hand Landmarks')
    parser.add_argument('--viewpoint', choices=['front', 'left', 'right'], required=True)
    parser.add_argument('--synthetic_factor', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=default_config['epochs'])
    parser.add_argument('--patience', type=int, default=default_config['patience'])
    args = parser.parse_args()

    config = default_config.copy()
    config['epochs'] = args.epochs
    config['patience'] = args.patience

    viewpoint = args.viewpoint
    syn_factor = args.synthetic_factor

    real_train_path = DATA_DIR / f"train_features_{viewpoint}_hands.pt"
    real_test_path = DATA_DIR / f"test_features_{viewpoint}_hands.pt"
    syn_train_path = DATA_DIR / f"synthetic_train_features_{viewpoint}_{syn_factor}x.pt"
    syn_test_path = DATA_DIR / f"synthetic_test_features_{viewpoint}_1x.pt"

    if not real_train_path.exists():
        print(f"ERROR: {real_train_path} not found. Run 2d_generate_features_with_hands.py first.")
        sys.exit(1)

    # For synthetic, we'd need to also generate synthetic with hands
    # For now, support no-synthetic mode since synthetic generation needs updating too
    if not syn_train_path.exists():
        print(f"WARNING: {syn_train_path} not found. Training with real data only.")
        train_dataset = GraphDatasetWithHands(real_train_path, viewpoint=viewpoint)
        val_dataset = GraphDatasetWithHands(real_test_path, viewpoint=viewpoint)
        best_acc, history = train_model(train_dataset, val_dataset, viewpoint, config, synthetic_factor=0)
    else:
        print(f"Loading combined dataset: real + {syn_factor}x synthetic")
        train_dataset = load_combined_dataset(real_train_path, syn_train_path, viewpoint=viewpoint)
        val_dataset = load_combined_dataset(real_test_path, syn_test_path, viewpoint=viewpoint)
        best_acc, history = train_model(train_dataset, val_dataset, viewpoint, config, synthetic_factor=syn_factor)

    print(f"\n{'='*60}")
    print(f"FINAL RESULT: {best_acc:.1f}% validation accuracy")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
