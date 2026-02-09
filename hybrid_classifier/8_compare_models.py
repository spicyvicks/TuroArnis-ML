"""
Model Comparison Testing Script
Compares two approaches:
1. Viewpoint-Specific Ensemble (Manual Selection) - uses specialist models
2. Merged Viewpoint Model - single model trained on all viewpoints

Usage:
    python hybrid_classifier/8_compare_models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import sys

sys.path.append(str(Path(__file__).parent.parent))
from training.train_gcn import CLASS_NAMES

# Config
HYBRID_FEATURES_DIR = Path("hybrid_classifier/hybrid_features_v2")
MODELS_DIR = Path("hybrid_classifier/models")

SKELETON_EDGES = [
    (11, 12), (12, 11), (11, 23), (23, 11), (12, 24), (24, 12),
    (23, 24), (24, 23), (11, 13), (13, 11), (13, 15), (15, 13),
    (12, 14), (14, 12), (14, 16), (16, 14), (23, 25), (25, 23),
    (25, 27), (27, 25), (24, 26), (26, 24), (26, 28), (28, 26),
    (15, 33), (33, 15), (16, 33), (33, 16), (33, 34), (34, 33)
]


class HybridGCN(nn.Module):
    def __init__(self, node_in_channels, hybrid_in_channels, hidden_channels, num_classes, num_layers=3, dropout=0.5, embedding_dim=8):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        
        self.node_embedding = nn.Embedding(35, embedding_dim)
        
        gcn_input_dim = node_in_channels + embedding_dim
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GCNConv(gcn_input_dim, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.hybrid_fc1 = nn.Linear(hybrid_in_channels, hidden_channels)
        self.hybrid_bn1 = nn.BatchNorm1d(hidden_channels)
        self.hybrid_fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.hybrid_bn2 = nn.BatchNorm1d(hidden_channels)
        
        self.fusion_fc = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fusion_bn = nn.BatchNorm1d(hidden_channels)
        
        self.fc = nn.Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, batch, hybrid_features):
        num_nodes_per_graph = 35
        num_graphs = batch.max().item() + 1
        node_indices = torch.arange(num_nodes_per_graph, device=x.device).repeat(num_graphs)
        
        node_emb = self.node_embedding(node_indices)
        x = torch.cat([x, node_emb], dim=1)
        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x_graph = global_mean_pool(x, batch)
        
        h = self.hybrid_fc1(hybrid_features)
        h = self.hybrid_bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = self.hybrid_fc2(h)
        h = self.hybrid_bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        combined = torch.cat([x_graph, h], dim=1)
        combined = self.fusion_fc(combined)
        combined = self.fusion_bn(combined)
        combined = F.relu(combined)
        combined = F.dropout(combined, p=self.dropout, training=self.training)
        
        out = self.fc(combined)
        return out


def load_model(model_path):
    """Load a trained model"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = HybridGCN(
        node_in_channels=checkpoint['node_feat_dim'],
        hybrid_in_channels=checkpoint['hybrid_feat_dim'],
        hidden_channels=checkpoint['hidden_dim'],
        num_classes=len(CLASS_NAMES),
        num_layers=checkpoint.get('num_layers', 3),
        dropout=checkpoint['dropout'],
        embedding_dim=checkpoint.get('embedding_dim', 8)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def load_test_data(viewpoint=None):
    """Load test data for a specific viewpoint or all viewpoints"""
    suffix = f"_{viewpoint}" if viewpoint else ""
    test_file = HYBRID_FEATURES_DIR / f"test_features{suffix}.pt"
    
    if not test_file.exists():
        raise FileNotFoundError(f"{test_file} not found")
    
    test_data = torch.load(test_file)
    
    graphs = []
    for i in range(len(test_data['node_features'])):
        node_feat = test_data['node_features'][i]
        hybrid_feat = test_data['hybrid_features'][i]
        label = test_data['labels'][i]
        
        edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t()
        graph = Data(x=node_feat, edge_index=edge_index, y=label, hybrid=hybrid_feat)
        
        # Store viewpoint info if available
        if 'viewpoints' in test_data:
            graph.viewpoint = test_data['viewpoints'][i]
        elif viewpoint:
            graph.viewpoint = viewpoint
        
        graphs.append(graph)
    
    return graphs


def evaluate_specialist_ensemble():
    """
    Approach 1: Viewpoint-Specific Ensemble (Manual Selection)
    Uses the correct specialist model for each viewpoint
    """
    print("\n" + "="*60)
    print("APPROACH 1: Viewpoint-Specific Ensemble (Manual Selection)")
    print("="*60)
    
    # Load specialist models
    models = {}
    for vp in ['front', 'left', 'right']:
        model_path = MODELS_DIR / f"hybrid_gcn_v2_{vp}.pth"
        if model_path.exists():
            models[vp] = load_model(model_path)
            print(f"Loaded {vp} specialist model")
        else:
            print(f"Warning: {vp} model not found at {model_path}")
    
    if not models:
        print("No specialist models found!")
        return None
    
    # Load test data for each viewpoint
    all_correct = 0
    all_total = 0
    viewpoint_results = {}
    
    for vp in models.keys():
        test_graphs = load_test_data(vp)
        model = models[vp]
        
        correct = 0
        total = 0
        
        for graph in test_graphs:
            with torch.no_grad():
                x = graph.x
                batch = torch.zeros(x.shape[0], dtype=torch.long)
                edge_index = graph.edge_index
                hybrid = graph.hybrid.unsqueeze(0)
                
                out = model(x, edge_index, batch, hybrid)
                pred = out.argmax(dim=1).item()
                
                if pred == graph.y.item():
                    correct += 1
                total += 1
        
        acc = correct / total if total > 0 else 0
        viewpoint_results[vp] = {
            'correct': correct,
            'total': total,
            'accuracy': acc
        }
        
        all_correct += correct
        all_total += total
        
        print(f"  {vp.upper()}: {correct}/{total} = {acc:.4f}")
    
    overall_acc = all_correct / all_total if all_total > 0 else 0
    print(f"\n  OVERALL: {all_correct}/{all_total} = {overall_acc:.4f}")
    print("="*60)
    
    return {
        'approach': 'specialist_ensemble',
        'overall_accuracy': overall_acc,
        'viewpoint_results': viewpoint_results,
        'total_correct': all_correct,
        'total_samples': all_total
    }


def evaluate_merged_model():
    """
    Approach 2: Merged Viewpoint Model
    Single model trained on all viewpoints
    """
    print("\n" + "="*60)
    print("APPROACH 2: Merged Viewpoint Model")
    print("="*60)
    
    # Check if merged model exists
    merged_model_path = MODELS_DIR / "hybrid_gcn_v2.pth"
    
    if not merged_model_path.exists():
        print(f"Merged model not found at {merged_model_path}")
        print("To train a merged model, run:")
        print("  python hybrid_classifier/4c_train_hybrid_gcn_v2.py --merged --epochs 150")
        return None
    
    # Load merged model
    model = load_model(merged_model_path)
    print(f"Loaded merged model from {merged_model_path}")
    
    # Load test data (all viewpoints)
    all_test_graphs = []
    for vp in ['front', 'left', 'right']:
        try:
            graphs = load_test_data(vp)
            all_test_graphs.extend(graphs)
        except FileNotFoundError:
            print(f"Warning: Test data for {vp} not found")
    
    if not all_test_graphs:
        print("No test data found!")
        return None
    
    # Evaluate
    correct = 0
    total = 0
    viewpoint_results = {'front': {'correct': 0, 'total': 0}, 
                        'left': {'correct': 0, 'total': 0}, 
                        'right': {'correct': 0, 'total': 0}}
    
    for graph in all_test_graphs:
        with torch.no_grad():
            x = graph.x
            batch = torch.zeros(x.shape[0], dtype=torch.long)
            edge_index = graph.edge_index
            hybrid = graph.hybrid.unsqueeze(0)
            
            out = model(x, edge_index, batch, hybrid)
            pred = out.argmax(dim=1).item()
            
            is_correct = (pred == graph.y.item())
            if is_correct:
                correct += 1
            total += 1
            
            # Track by viewpoint
            vp = graph.viewpoint
            viewpoint_results[vp]['total'] += 1
            if is_correct:
                viewpoint_results[vp]['correct'] += 1
    
    # Calculate accuracies
    for vp in viewpoint_results:
        vp_correct = viewpoint_results[vp]['correct']
        vp_total = viewpoint_results[vp]['total']
        viewpoint_results[vp]['accuracy'] = vp_correct / vp_total if vp_total > 0 else 0
        print(f"  {vp.upper()}: {vp_correct}/{vp_total} = {viewpoint_results[vp]['accuracy']:.4f}")
    
    overall_acc = correct / total if total > 0 else 0
    print(f"\n  OVERALL: {correct}/{total} = {overall_acc:.4f}")
    print("="*60)
    
    return {
        'approach': 'merged_model',
        'overall_accuracy': overall_acc,
        'viewpoint_results': viewpoint_results,
        'total_correct': correct,
        'total_samples': total
    }


def compare_results(specialist_results, merged_results):
    """Compare the two approaches"""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    if specialist_results:
        print(f"\nSpecialist Ensemble (Manual Selection):")
        print(f"  Overall Accuracy: {specialist_results['overall_accuracy']:.4f}")
        print(f"  Total Samples: {specialist_results['total_samples']}")
    
    if merged_results:
        print(f"\nMerged Model:")
        print(f"  Overall Accuracy: {merged_results['overall_accuracy']:.4f}")
        print(f"  Total Samples: {merged_results['total_samples']}")
    
    if specialist_results and merged_results:
        diff = merged_results['overall_accuracy'] - specialist_results['overall_accuracy']
        print(f"\nDifference: {diff:+.4f}")
        if diff > 0:
            print("✅ Merged model is better")
        elif diff < -0.01:
            print("✅ Specialist ensemble is better")
        else:
            print("≈ Both approaches are similar")
    
    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MODEL COMPARISON TEST")
    print("="*60)
    
    # Evaluate both approaches
    specialist_results = evaluate_specialist_ensemble()
    merged_results = evaluate_merged_model()
    
    # Compare
    compare_results(specialist_results, merged_results)
