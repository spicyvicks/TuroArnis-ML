"""
Show example predictions from v6_new model on real test data.
Displays correct and misclassified samples per class.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from torch_geometric.loader import DataLoader as GeoDataLoader

# Same config as evaluate_real_only_v6.py
NUM_CLASSES = 13
NUM_NODES = 35
NODE_FEATURES = 7
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.5
NODE_EMBED_DIM = 8

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct',
    'neutral'
]

SKELETON_EDGES = [
    (11, 12), (12, 11), (11, 23), (23, 11), (12, 24), (24, 12),
    (23, 24), (24, 23), (11, 13), (13, 11), (13, 15), (15, 13),
    (12, 14), (14, 12), (14, 16), (16, 14), (23, 25), (25, 23),
    (25, 27), (27, 25), (24, 26), (26, 24), (26, 28), (28, 26),
    (15, 33), (33, 15), (16, 33), (33, 16), (33, 34), (34, 33),
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HybridGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_hybrid_features, num_classes, hidden_dim=HIDDEN_DIM):
        super(HybridGCN, self).__init__()
        self.node_embedding = torch.nn.Embedding(NUM_NODES, NODE_EMBED_DIM)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_node_features + NODE_EMBED_DIM, hidden_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim, track_running_stats=False))
        for _ in range(NUM_LAYERS - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim, track_running_stats=False))
        self.hybrid_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_hybrid_features, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 2),
            torch.nn.ReLU()
        )
        self.fc1 = torch.nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(DROPOUT)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        hybrid_features = data.hybrid_features
        node_mask = getattr(data, 'node_mask', None)
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
        if node_mask is not None:
            x = x * node_mask.unsqueeze(-1)
            x_sum = global_add_pool(x, batch)
            mask_sum = global_add_pool(node_mask.unsqueeze(-1), batch)
            x_pool = x_sum / (mask_sum + 1e-8)
        else:
            x_pool = global_mean_pool(x, batch)
        batch_size = batch.max().item() + 1
        hybrid_features = hybrid_features.view(batch_size, -1)
        hybrid_out = self.hybrid_mlp(hybrid_features)
        combined = torch.cat([x_pool, hybrid_out], dim=-1)
        x_out = F.relu(self.fc1(combined))
        x_out = self.dropout(x_out)
        logits = self.fc2(x_out)
        return logits


def load_data():
    data = torch.load('hybrid_classifier/hybrid_features_v6/test_features_front.pt', map_location='cpu')
    print(f"Loaded real test data: {len(data['labels'])} samples")
    viewpoints = data.get('viewpoints', ['front'] * len(data['labels']))
    mask = [v == 'front' for v in viewpoints]
    node_features = data['node_features'][mask]
    hybrid_features = data['hybrid_features'][mask]
    labels = data['labels'][mask]
    has_stick_nodes = data.get('has_stick_nodes', torch.ones(len(labels), dtype=torch.bool))[mask]

    node_nan = torch.isnan(node_features).any(dim=(1, 2))
    hybrid_nan = torch.isnan(hybrid_features).any(dim=1)
    nan_mask = node_nan | hybrid_nan
    if nan_mask.sum() > 0:
        valid = ~nan_mask
        node_features = node_features[valid]
        hybrid_features = hybrid_features[valid]
        labels = labels[valid]
        has_stick_nodes = has_stick_nodes[valid]

    num_nodes = node_features.size(1)
    node_mask = torch.ones(len(labels), num_nodes, dtype=torch.float32)
    for i in range(len(labels)):
        if not has_stick_nodes[i]:
            node_mask[i, 33:] = 0.0

    edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t().contiguous()
    graphs = []
    for i in range(len(labels)):
        ei = edge_index
        if not has_stick_nodes[i]:
            mask_ei = (ei[0] < 33) & (ei[1] < 33)
            ei = ei[:, mask_ei]
        graphs.append(Data(
            x=node_features[i],
            edge_index=ei,
            hybrid_features=hybrid_features[i],
            y=labels[i],
            has_stick_nodes=has_stick_nodes[i],
            node_mask=node_mask[i]
        ))
    return graphs, labels


def main():
    model_path = 'hybrid_classifier/models/model_front_with_synthetic_2x_v6_new.pth'
    graphs, labels = load_data()

    sample = graphs[0]
    model = HybridGCN(
        num_node_features=sample.x.size(1),
        num_hybrid_features=sample.hybrid_features.size(0),
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    loader = GeoDataLoader(graphs, batch_size=32, shuffle=False)

    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            probs = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Group by class
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS (v6_new model)")
    print("="*80)

    for class_idx in range(NUM_CLASSES):
        class_name = CLASS_NAMES[class_idx]
        indices = [i for i, y in enumerate(labels) if y == class_idx]
        correct = [i for i in indices if all_preds[i] == class_idx]
        incorrect = [i for i in indices if all_preds[i] != class_idx]

        total = len(indices)
        acc = 100.0 * len(correct) / total if total > 0 else 0

        print(f"\n{class_name} ({len(correct)}/{total} = {acc:.1f}%)")
        print("-" * 60)

        # Show up to 3 correct examples
        if correct:
            print("  CORRECT examples:")
            for idx in correct[:3]:
                prob = all_probs[idx][class_idx]
                print(f"    Sample {idx}: predicted={CLASS_NAMES[all_preds[idx]]}, confidence={prob:.3f}")

        # Show all incorrect examples (with what they were misclassified as)
        if incorrect:
            print("  MISCLASSIFIED examples:")
            for idx in incorrect:
                pred_class = all_preds[idx]
                prob = all_probs[idx][pred_class]
                true_prob = all_probs[idx][class_idx]
                print(f"    Sample {idx}: predicted={CLASS_NAMES[pred_class]} (conf={prob:.3f}), "
                      f"true={CLASS_NAMES[class_idx]} (conf={true_prob:.3f})")

    # Confusion matrix summary
    print("\n" + "="*80)
    print("CONFUSION PATTERNS (most common misclassifications)")
    print("="*80)
    from collections import Counter
    misclass = Counter()
    for i in range(len(labels)):
        if all_preds[i] != labels[i]:
            pair = (CLASS_NAMES[labels[i]], CLASS_NAMES[all_preds[i]])
            misclass[pair] += 1

    for (true, pred), count in misclass.most_common(15):
        print(f"  {true} -> {pred}: {count} times")


if __name__ == '__main__':
    main()
