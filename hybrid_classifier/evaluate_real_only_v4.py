"""
Evaluate trained HybridGCN model on REAL-ONLY test data.
Reports true generalization (no synthetic inflation).
"""
import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader as GeoDataLoader

# --- CONFIG (must match training) ---
NUM_CLASSES = 13
NUM_NODES = 35  # 34 pose + 1 stick
NODE_FEATURES = 6
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.5
NODE_EMBED_DIM = 8
CLASS_NAMES = [
    'crown_thrust_correct',
    'left_chest_thrust_correct',
    'left_elbow_block_correct',
    'left_eye_thrust_correct',
    'left_knee_block_correct',
    'left_temple_block_correct',
    'right_chest_thrust_correct',
    'right_elbow_block_correct',
    'right_eye_thrust_correct',
    'right_knee_block_correct',
    'right_temple_block_correct',
    'solar_plexus_thrust_correct',
    'neutral'
]

SKELETON_EDGES = [
    (0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),(10,11),
    (8,12),(12,13),(13,14),(0,15),(15,17),(0,16),(16,18),(2,9),
    (12,11),(12,14),(13,14),(10,11),(6,8),(7,12),(3,22),(22,23),
    (20,18),(21,19),(23,24),(25,26),(26,27),(28,29),(30,31),(32,33),
    (16,33),(33,16),(33,34),(34,33),
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = Path('hybrid_classifier/hybrid_features_v4')

# --- MODEL (exact match to training script) ---
class HybridGCN(nn.Module):
    def __init__(self, num_node_features, num_hybrid_features, num_classes, hidden_dim=HIDDEN_DIM):
        super(HybridGCN, self).__init__()
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


def load_real_test_data(real_path, viewpoint='front'):
    data = torch.load(real_path, map_location='cpu')
    print(f"Loaded real test data: {len(data['labels'])} samples")
    class_counts = torch.bincount(data['labels'], minlength=NUM_CLASSES)
    for i, name in enumerate(CLASS_NAMES):
        if class_counts[i] > 0:
            print(f"  {name}: {class_counts[i]}")
    # Filter viewpoint
    viewpoints = data.get('viewpoints', ['front'] * len(data['labels']))
    mask = [v == viewpoint for v in viewpoints]
    node_features = data['node_features'][mask]
    hybrid_features = data['hybrid_features'][mask]
    labels = data['labels'][mask]
    has_stick_nodes = data.get('has_stick_nodes', torch.ones(len(labels), dtype=torch.bool))[mask]
    # NaN filter
    node_nan = torch.isnan(node_features).any(dim=(1, 2))
    hybrid_nan = torch.isnan(hybrid_features).any(dim=1)
    nan_mask = node_nan | hybrid_nan
    if nan_mask.sum() > 0:
        print(f"[WARN] Filtering {nan_mask.sum()} NaN samples")
        valid = ~nan_mask
        node_features = node_features[valid]
        hybrid_features = hybrid_features[valid]
        labels = labels[valid]
        has_stick_nodes = has_stick_nodes[valid]
    print(f"Final real test samples: {len(labels)}")
    return node_features, hybrid_features, labels, has_stick_nodes


def create_graph_data(node_features, hybrid_features, labels, has_stick_nodes):
    edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t().contiguous()
    graphs = []
    for i in range(len(labels)):
        ei = edge_index
        if not has_stick_nodes[i]:
            mask = (ei[0] < 33) & (ei[1] < 33)
            ei = ei[:, mask]
        graphs.append(Data(
            x=node_features[i],
            edge_index=ei,
            hybrid_features=hybrid_features[i],
            y=labels[i],
            has_stick_nodes=has_stick_nodes[i]
        ))
    return graphs


def evaluate_model(model, graphs):
    model.eval()
    loader = GeoDataLoader(graphs, batch_size=32, shuffle=False)
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    acc = 100.0 * correct / total
    return acc, all_preds, all_labels


def print_confusion_matrix(labels, preds):
    print("\n=== PER-CLASS ACCURACY (Real-Only Test) ===")
    per_class_correct = {i: 0 for i in range(NUM_CLASSES)}
    per_class_total = {i: 0 for i in range(NUM_CLASSES)}
    for l, p in zip(labels, preds):
        per_class_total[l] += 1
        if l == p:
            per_class_correct[l] += 1
    for i, name in enumerate(CLASS_NAMES):
        if per_class_total[i] > 0:
            pct = 100.0 * per_class_correct[i] / per_class_total[i]
            print(f"  {name:22s}: {per_class_correct[i]}/{per_class_total[i]} = {pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Evaluate HybridGCN on real-only test data')
    parser.add_argument('--viewpoint', type=str, default='front',
                        choices=['front', 'left', 'right'],
                        help='Viewpoint to evaluate')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained model checkpoint (default: auto-detect)')
    args = parser.parse_args()

    viewpoint = args.viewpoint
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = f'hybrid_classifier/models/model_{viewpoint}_with_synthetic_3x.pth'
    real_test_path = DATA_DIR / f'test_features_{viewpoint}.pt'

    if not Path(model_path).exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    if not real_test_path.exists():
        print(f"Error: Test data not found: {real_test_path}")
        sys.exit(1)

    # Load real-only test data first to get feature dims
    node_features, hybrid_features, labels, has_stick_nodes = load_real_test_data(real_test_path, viewpoint)
    num_node_features = node_features.size(2)
    num_hybrid_features = hybrid_features.size(1)
    print(f"Node features: {num_node_features}, Hybrid features: {num_hybrid_features}")

    # Load model
    model = HybridGCN(
        num_node_features=num_node_features,
        num_hybrid_features=num_hybrid_features,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[OK] Loaded checkpoint: {model_path} (epoch {checkpoint.get('epoch', '?')}, val_acc={checkpoint.get('val_acc', '?')})")
    else:
        model.load_state_dict(checkpoint)
        print(f"[OK] Loaded model: {model_path}")

    graphs = create_graph_data(node_features, hybrid_features, labels, has_stick_nodes)

    # Evaluate
    acc, preds, labels_list = evaluate_model(model, graphs)
    print(f"\n{'='*60}")
    print(f"REAL-ONLY TEST ACCURACY: {acc:.1f}%")
    print(f"{'='*60}")

    print_confusion_matrix(labels_list, preds)


if __name__ == '__main__':
    main()
