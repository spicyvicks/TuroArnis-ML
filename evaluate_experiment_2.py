"""
Evaluate Experiment 2 model (regularized 2x) on real-only test.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

NUM_CLASSES = 13
NUM_NODES = 35
NODE_EMBED_DIM = 8
HIDDEN_DIM = 96  # Matches experiment 2
NUM_LAYERS = 3
DROPOUT = 0.7

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
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

def create_graph_data(node_features, hybrid_features, labels, has_stick_nodes):
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
            print(f"  {name:35s}: {per_class_correct[i]}/{per_class_total[i]} = {pct:.1f}%")

if __name__ == '__main__':
    viewpoint = 'front'
    test_path = DATA_DIR / f'test_features_{viewpoint}.pt'
    model_path = 'hybrid_classifier/models/model_front_with_synthetic_2x_regularized.pth'

    if not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        print("Run experiment_2_train_regularized.py first")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"EVALUATING: Experiment 2 Model (2x Regularized)")
    print(f"Model: {model_path}")
    print(f"Test:  {test_path}")
    print(f"{'='*60}")

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
        print(f"[OK] Loaded checkpoint (epoch {checkpoint.get('epoch', '?')}, val_acc={checkpoint.get('val_acc', 0):.1f}%)")
    else:
        model.load_state_dict(checkpoint)
        print(f"[OK] Loaded model state dict")

    graphs = create_graph_data(node_features, hybrid_features, labels, None)
    acc, preds, labels_list = evaluate_model(model, graphs)

    print(f"\n{'='*60}")
    print(f"REAL-ONLY TEST ACCURACY: {acc:.1f}%")
    print(f"{'='*60}")

    print_confusion_matrix(labels_list, preds)

    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"Baseline 1x:     32.5%")
    print(f"Baseline 3x:     40.6%")
    print(f"Regularized 2x:  {acc:.1f}%")
    print(f"\nIf {acc:.1f}% > 45%: Regularization helped, consider more tuning")
    print(f"If {acc:.1f}% < 40%: Bottleneck is features, not regularization")
    print(f"{'='*60}")
