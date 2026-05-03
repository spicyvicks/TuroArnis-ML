"""
Phase 6: Real-Only Evaluation of Right Viewpoint Models
Evaluate trained models on real (non-synthetic) test data.
Produces per-class accuracy, confusion matrix, precision, recall, F1.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeoDataLoader

import sys
sys.path.insert(0, 'hybrid_classifier')
sys.path.insert(0, 'deployment_package/src')

from model_v5 import HybridGCN as HybridGCNv5
from model_v6 import HybridGCN as HybridGCNv6

SKELETON_EDGES = [
    (11, 12), (12, 11), (11, 23), (23, 11), (12, 24), (24, 12),
    (23, 24), (24, 23), (11, 13), (13, 11), (13, 15), (15, 13),
    (12, 14), (14, 12), (14, 16), (16, 14), (23, 25), (25, 23),
    (25, 27), (27, 25), (24, 26), (26, 24), (26, 28), (28, 26),
    (15, 33), (33, 15), (16, 33), (33, 16), (33, 34), (34, 33),
]

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct',
    'neutral'
]
NUM_CLASSES = len(CLASS_NAMES)


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
            raise ValueError(f"Viewpoint filtering requested but 'viewpoints' key not found.")
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

        num_nodes = self.node_features.size(1)
        if 'node_mask' in self.data:
            self.node_mask = self.data['node_mask']
            if viewpoint and has_viewpoints:
                self.node_mask = self.node_mask[mask]
        else:
            self.node_mask = torch.ones(len(self.labels), num_nodes, dtype=torch.float32)
            for i in range(len(self.labels)):
                if not self.has_stick_nodes[i]:
                    self.node_mask[i, 33:] = 0.0

        if filter_nan:
            node_nan = torch.isnan(self.node_features).any(dim=(1, 2))
            hybrid_nan = torch.isnan(self.hybrid_features).any(dim=1)
            nan_mask = node_nan | hybrid_nan
            if nan_mask.sum() > 0:
                valid_mask = ~nan_mask
                self.node_features = self.node_features[valid_mask]
                self.hybrid_features = self.hybrid_features[valid_mask]
                self.labels = self.labels[valid_mask]
                self.viewpoints = [v for v, m in zip(self.viewpoints, valid_mask.tolist()) if m]
                self.has_stick_nodes = self.has_stick_nodes[valid_mask]

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
            has_stick_nodes=self.has_stick_nodes[idx],
            node_mask=self.node_mask[idx]
        )
        return data


def collate_fn(batch):
    return Batch.from_data_list(batch)


def load_model(model_path, model_version, device):
    ckpt = torch.load(model_path, map_location=device)
    config = ckpt.get('config', {})
    state_dict = ckpt['model_state_dict']

    node_feat_dim = None
    hybrid_feat_dim = None
    for key, val in state_dict.items():
        if 'convs.0.lin.weight' in key or 'convs.0.lin_rel.weight' in key:
            node_feat_dim = val.shape[1] - 8
            break
        elif 'convs.0.weight' in key:
            node_feat_dim = val.shape[1] - 8
            break

    for key, val in state_dict.items():
        if 'hybrid_mlp.0.weight' in key:
            hybrid_feat_dim = val.shape[1]
            break

    hidden_dim = config.get('hidden_dim', 128)

    if model_version == 'v5':
        model = HybridGCNv5(
            num_node_features=node_feat_dim or 6,
            num_hybrid_features=hybrid_feat_dim or 46,
            num_classes=NUM_CLASSES,
            hidden_dim=hidden_dim
        )
    else:
        model = HybridGCNv6(
            num_node_features=node_feat_dim or 7,
            num_hybrid_features=hybrid_feat_dim or 49,
            num_classes=NUM_CLASSES,
            hidden_dim=hidden_dim
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, dataset, device, batch_size=64):
    loader = GeoDataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    overall_acc = 100.0 * (all_preds == all_labels).sum() / len(all_labels)

    per_class = {}
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    for pred, label in zip(all_preds, all_labels):
        confusion[label, pred] += 1

    for i in range(NUM_CLASSES):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        tn = confusion.sum() - tp - fp - fn

        total_actual = confusion[i, :].sum()
        total_predicted = confusion[:, i].sum()

        per_class[CLASS_NAMES[i]] = {
            'total_samples': int(total_actual),
            'correct': int(tp),
            'accuracy_pct': round(100.0 * tp / total_actual, 2) if total_actual > 0 else 0.0,
            'precision_pct': round(100.0 * tp / total_predicted, 2) if total_predicted > 0 else 0.0,
            'recall_pct': round(100.0 * tp / (tp + fn), 2) if (tp + fn) > 0 else 0.0,
            'f1_pct': round(100.0 * 2 * tp / (2 * tp + fp + fn), 2) if (2 * tp + fp + fn) > 0 else 0.0
        }

    top2_correct = 0
    for i, label in enumerate(all_labels):
        top2_preds = np.argsort(all_probs[i])[-2:]
        if label in top2_preds:
            top2_correct += 1
    top2_acc = 100.0 * top2_correct / len(all_labels)

    confusion_pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and confusion[i, j] > 0:
                confusion_pairs.append({
                    'actual': CLASS_NAMES[i],
                    'predicted': CLASS_NAMES[j],
                    'count': int(confusion[i, j])
                })
    confusion_pairs.sort(key=lambda x: x['count'], reverse=True)

    return {
        'overall_accuracy_pct': round(overall_acc, 2),
        'top2_accuracy_pct': round(top2_acc, 2),
        'total_samples': len(all_labels),
        'per_class': per_class,
        'confusion_matrix': confusion.tolist(),
        'top_confusion_pairs': confusion_pairs[:15],
        'mean_per_class_accuracy_pct': round(
            np.mean([v['accuracy_pct'] for v in per_class.values() if v['total_samples'] > 0]), 2
        )
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    models = [
        {
            'name': 'v5_standard',
            'model_path': 'hybrid_classifier/models/model_right_with_synthetic_3x_v5.pth',
            'test_path': 'hybrid_classifier/hybrid_features_v5/test_features_right.pt',
            'version': 'v5',
            'template': 'standard'
        },
        {
            'name': 'v6_standard',
            'model_path': 'hybrid_classifier/models/model_right_with_synthetic_2x_v6.pth',
            'test_path': 'hybrid_classifier/hybrid_features_v6/test_features_right.pt',
            'version': 'v6',
            'template': 'standard'
        },
        {
            'name': 'v5_cleaned_3x',
            'model_path': 'hybrid_classifier/models/model_right_with_synthetic_3x_v5_cleaned.pth',
            'test_path': 'hybrid_classifier/hybrid_features_v5_cleaned_right/test_features_right.pt',
            'version': 'v5',
            'template': 'standard'
        },
        {
            'name': 'v5_cleaned_5x',
            'model_path': 'hybrid_classifier/models/model_right_with_synthetic_5x_v5_cleaned.pth',
            'test_path': 'hybrid_classifier/hybrid_features_v5_cleaned_right/test_features_right.pt',
            'version': 'v5',
            'template': 'standard'
        },
        {
            'name': 'v6_cleaned_2x',
            'model_path': 'hybrid_classifier/models/model_right_with_synthetic_2x_v6_cleaned.pth',
            'test_path': 'hybrid_classifier/hybrid_features_v6_cleaned_right/test_features_right.pt',
            'version': 'v6',
            'template': 'standard'
        },
        {
            'name': 'v6_cleaned_5x',
            'model_path': 'hybrid_classifier/models/model_right_with_synthetic_5x_v6_cleaned.pth',
            'test_path': 'hybrid_classifier/hybrid_features_v6_cleaned_right/test_features_right.pt',
            'version': 'v6',
            'template': 'standard'
        },
        {
            'name': 'v5_reaugmented',
            'model_path': 'hybrid_classifier/models/model_right_with_synthetic_3x_v5_reaugmented_standard.pth',
            'test_path': 'hybrid_classifier/hybrid_features_v5_right_reaugmented/test_features_right.pt',
            'version': 'v5',
            'template': 'standard'
        },
        {
            'name': 'v5_reaugmented_mirrored',
            'model_path': 'hybrid_classifier/models/model_right_with_synthetic_3x_v5_reaugmented_mirrored.pth',
            'test_path': 'hybrid_classifier/hybrid_features_v5_right_reaugmented_mirrored/test_features_right.pt',
            'version': 'v5',
            'template': 'mirrored'
        },
        {
            'name': 'v6_reaugmented',
            'model_path': 'hybrid_classifier/models/model_right_with_synthetic_2x_v6_reaugmented.pth',
            'test_path': 'hybrid_classifier/hybrid_features_v6_right_reaugmented/test_features_right.pt',
            'version': 'v6',
            'template': 'standard'
        },
        {
            'name': 'v6_reaugmented_mirrored',
            'model_path': 'hybrid_classifier/models/model_right_with_synthetic_2x_v6_reaugmented_mirrored.pth',
            'test_path': 'hybrid_classifier/hybrid_features_v6_right_reaugmented_mirrored/test_features_right.pt',
            'version': 'v6',
            'template': 'mirrored'
        }
    ]

    all_results = {}

    for model_info in models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_info['name']}")
        print(f"Model: {model_info['model_path']}")
        print(f"Test data: {model_info['test_path']}")
        print(f"{'='*60}")

        if not Path(model_info['model_path']).exists():
            print(f"  [SKIP] Model file not found")
            continue
        if not Path(model_info['test_path']).exists():
            print(f"  [SKIP] Test data not found")
            continue

        dataset = GraphDataset(model_info['test_path'], viewpoint='right', filter_nan=True)
        print(f"  Test samples: {len(dataset)}")

        labels = dataset.labels.numpy()
        class_counts = np.bincount(labels, minlength=NUM_CLASSES)
        for i, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
            if count > 0:
                print(f"    {name}: {count}")

        model = load_model(model_info['model_path'], model_info['version'], device)
        print(f"  Model loaded successfully")

        results = evaluate_model(model, dataset, device, batch_size=1)
        all_results[model_info['name']] = {
            'template': model_info['template'],
            'version': model_info['version'],
            **results
        }

        print(f"\n  Overall Accuracy: {results['overall_accuracy_pct']}%")
        print(f"  Top-2 Accuracy: {results['top2_accuracy_pct']}%")
        print(f"  Mean Per-Class Accuracy: {results['mean_per_class_accuracy_pct']}%")
        print(f"\n  Per-Class Results:")
        for class_name, metrics in results['per_class'].items():
            if metrics['total_samples'] > 0:
                print(f"    {class_name:35s} | n={metrics['total_samples']:3d} | "
                      f"Acc={metrics['accuracy_pct']:5.1f}% | "
                      f"P={metrics['precision_pct']:5.1f}% | "
                      f"R={metrics['recall_pct']:5.1f}% | "
                      f"F1={metrics['f1_pct']:5.1f}%")

        print(f"\n  Top Confusion Pairs:")
        for pair in results['top_confusion_pairs'][:10]:
            print(f"    {pair['actual']:35s} -> {pair['predicted']:35s} : {pair['count']}")

    output_path = Path('hybrid_classifier/models/right_viewpoint_real_only_evaluation.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    for name, results in all_results.items():
        print(f"{name:20s}: Overall={results['overall_accuracy_pct']:5.1f}% | "
              f"Top-2={results['top2_accuracy_pct']:5.1f}% | "
              f"Mean/Class={results['mean_per_class_accuracy_pct']:5.1f}%")

    print(f"\nFull results saved to: {output_path}")


if __name__ == '__main__':
    main()
