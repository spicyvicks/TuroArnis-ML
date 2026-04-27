"""
Compare 1x, 2x, 3x synthetic models on real-only test.
All models use standard (non-regularized) 4e training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

NUM_CLASSES = 13
NUM_NODES = 35
NODE_EMBED_DIM = 8
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.5

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct', 'neutral'
]

SKELETON_EDGES = [
    (0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),(10,11),
    (8,12),(12,13),(13,14),(0,15),(15,17),(0,16),(16,18),(2,9),
    (12,11),(12,14),(13,14),(10,11),(6,8),(7,12),(3,22),(22,23),
    (20,18),(21,19),(23,24),(25,26),(26,27),(28,29),(30,31),(32,33),
    (16,33),(33,16),(33,34),(34,33),
]

DEVICE = torch.device('cpu')
DATA_DIR = Path('hybrid_classifier/hybrid_features_v3')


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


def create_graph_data(node_features, hybrid_features, labels):
    graphs = []
    for i in range(len(labels)):
        nf = node_features[i]
        hf = hybrid_features[i]
        edge_index = torch.tensor(
            [[s, t] for s, t in SKELETON_EDGES] + [[t, s] for s, t in SKELETON_EDGES],
            dtype=torch.long
        ).t()
        g = Data(x=nf, edge_index=edge_index, y=labels[i], hybrid_features=hf.unsqueeze(0))
        g.batch = torch.zeros(nf.size(0), dtype=torch.long)
        graphs.append(g)
    return graphs


def evaluate_model(model, graphs):
    model.eval()
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for g in graphs:
            g = g.to(DEVICE)
            out = model(g)
            pred = out.argmax().item()
            all_preds.append(pred)
            all_labels.append(g.y.item())
            correct += (pred == g.y.item())
    acc = 100.0 * correct / len(graphs)
    return acc, all_preds, all_labels


def evaluate_track(factor, test_path, results):
    model_path = f'hybrid_classifier/models/model_front_with_synthetic_{factor}x.pth'
    print(f"\n{'='*60}")
    print(f"EVALUATING: {factor}x Synthetic Model")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    if not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        return None

    data = torch.load(test_path, map_location=DEVICE)
    node_features = data['node_features']
    hybrid_features = data['hybrid_features']
    labels = data['labels']

    num_nf = node_features.size(2)
    num_hf = hybrid_features.size(1)
    model = HybridGCN(num_nf, num_hf, NUM_CLASSES, HIDDEN_DIM).to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        val_acc = checkpoint.get('val_acc', 0)
        epoch = checkpoint.get('epoch', '?')
        print(f"[OK] Loaded checkpoint (epoch {epoch}, val_acc={val_acc:.1f}%)")
    else:
        model.load_state_dict(checkpoint)
        val_acc = 0
        print(f"[OK] Loaded model state dict")

    graphs = create_graph_data(node_features, hybrid_features, labels)
    acc, preds, labels_list = evaluate_model(model, graphs)

    # Per-class accuracy
    per_class = {}
    for c in range(NUM_CLASSES):
        mask = [l == c for l in labels_list]
        total = sum(mask)
        if total > 0:
            c_correct = sum(1 for i, m in enumerate(mask) if m and preds[i] == c)
            per_class[c] = 100.0 * c_correct / total
        else:
            per_class[c] = 0.0

    print(f"\nREAL-ONLY TEST ACCURACY: {acc:.1f}%")

    results[factor] = {
        'real_acc': acc,
        'val_acc': val_acc,
        'per_class': per_class
    }
    return acc


if __name__ == '__main__':
    viewpoint = 'front'
    test_path = DATA_DIR / f'test_features_{viewpoint}.pt'

    if not test_path.exists():
        print(f"ERROR: Test features not found: {test_path}")
        exit(1)

    results = {}

    # Evaluate all three standard models
    for factor in ['1', '2', '3']:
        evaluate_track(factor, test_path, results)

    # Comparison table
    print(f"\n{'='*60}")
    print("COMPARISON: Standard Training (Non-Regularized)")
    print(f"{'='*60}")
    print(f"{'Factor':>8} | {'Val Acc':>8} | {'Real-Only':>10} | {'Gap':>8}")
    print("-" * 42)
    for factor in ['1', '2', '3']:
        if factor in results:
            r = results[factor]
            gap = r['val_acc'] - r['real_acc']
            print(f"{factor+'x':>8} | {r['val_acc']:>7.1f}% | {r['real_acc']:>9.1f}% | {gap:>7.1f}%")

    print(f"\n{'='*60}")
    print("PER-CLASS BREAKDOWN (Best Factor per Class)")
    print(f"{'='*60}")
    print(f"{'Class':<35} | {'1x':>6} | {'2x':>6} | {'3x':>6} | {'Best':>6}")
    print("-" * 72)
    for i, name in enumerate(CLASS_NAMES):
        accs = []
        for factor in ['1', '2', '3']:
            if factor in results and i in results[factor]['per_class']:
                accs.append(results[factor]['per_class'][i])
            else:
                accs.append(0.0)
        best = max(accs) if accs else 0.0
        best_factor = ['1x', '2x', '3x'][accs.index(best)] if accs else '?'
        print(f"{name:<35} | {accs[0]:>5.1f}% | {accs[1]:>5.1f}% | {accs[2]:>5.1f}% | {best_factor:>6}")
    print(f"{'='*60}")
