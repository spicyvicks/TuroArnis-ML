"""
Debug script for investigating NaN issues in right viewpoint features.
Checks for data corruption, invalid values, and feature statistics.
"""

import torch
import numpy as np
from pathlib import Path

DATA_DIR = Path("hybrid_classifier/hybrid_features_v3")

def analyze_features(features_path, name):
    """Analyze feature file for issues."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"Path: {features_path}")
    print(f"{'='*60}")
    
    if not features_path.exists():
        print("ERROR: File does not exist!")
        return None
    
    # Load data
    data = torch.load(features_path, map_location='cpu')
    
    node_features = data['node_features']
    hybrid_features = data['hybrid_features']
    labels = data['labels']
    
    print(f"\nDataset Info:")
    print(f"  Samples: {len(labels)}")
    print(f"  Node features shape: {node_features.shape}")
    print(f"  Hybrid features shape: {hybrid_features.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique labels: {torch.unique(labels).tolist()}")
    
    # Check for NaN
    node_nan = torch.isnan(node_features).sum().item()
    hybrid_nan = torch.isnan(hybrid_features).sum().item()
    labels_nan = torch.isnan(labels.float()).sum().item()
    
    node_inf = torch.isinf(node_features).sum().item()
    hybrid_inf = torch.isinf(hybrid_features).sum().item()
    
    print(f"\nNaN Counts:")
    print(f"  Node features NaN: {node_nan}")
    print(f"  Hybrid features NaN: {hybrid_nan}")
    print(f"  Labels NaN: {labels_nan}")
    
    print(f"\nInf Counts:")
    print(f"  Node features Inf: {node_inf}")
    print(f"  Hybrid features Inf: {hybrid_inf}")
    
    # Statistics
    print(f"\nNode Features Statistics:")
    print(f"  Min: {node_features.min():.4f}")
    print(f"  Max: {node_features.max():.4f}")
    print(f"  Mean: {node_features.mean():.4f}")
    print(f"  Std: {node_features.std():.4f}")
    
    print(f"\nHybrid Features Statistics:")
    print(f"  Min: {hybrid_features.min():.4f}")
    print(f"  Max: {hybrid_features.max():.4f}")
    print(f"  Mean: {hybrid_features.mean():.4f}")
    print(f"  Std: {hybrid_features.std():.4f}")
    
    # Check per-sample for NaN (which samples have issues)
    if node_nan > 0 or hybrid_nan > 0:
        print(f"\nNaN by Sample:")
        for i in range(min(20, len(labels))):  # Check first 20
            n_nan = torch.isnan(node_features[i]).sum().item()
            h_nan = torch.isnan(hybrid_features[i]).sum().item()
            if n_nan > 0 or h_nan > 0:
                print(f"  Sample {i}: node_nan={n_nan}, hybrid_nan={h_nan}, label={labels[i]}")
    
    # Check feature distribution (are all samples same?)
    print(f"\nFeature Variance Check:")
    node_var_per_sample = node_features.view(len(labels), -1).var(dim=1)
    hybrid_var_per_sample = hybrid_features.var(dim=1)
    
    zero_var_node = (node_var_per_sample == 0).sum().item()
    zero_var_hybrid = (hybrid_var_per_sample == 0).sum().item()
    
    print(f"  Samples with zero node variance: {zero_var_node}")
    print(f"  Samples with zero hybrid variance: {zero_var_hybrid}")
    
    # Check if features are all identical
    if len(labels) > 1:
        first_node = node_features[0]
        identical_nodes = sum(torch.allclose(node_features[i], first_node) for i in range(1, len(labels)))
        print(f"  Samples with identical node features to sample 0: {identical_nodes}")
    
    # Check extreme values
    print(f"\nExtreme Value Check:")
    node_large = (torch.abs(node_features) > 100).sum().item()
    hybrid_large = (torch.abs(hybrid_features) > 100).sum().item()
    print(f"  Node values |x| > 100: {node_large}")
    print(f"  Hybrid values |x| > 100: {hybrid_large}")
    
    # Per-class statistics
    print(f"\nPer-Class Sample Counts:")
    for label in torch.unique(labels):
        count = (labels == label).sum().item()
        print(f"  Class {label}: {count} samples")
    
    return {
        'node_nan': node_nan,
        'hybrid_nan': hybrid_nan,
        'node_inf': node_inf,
        'hybrid_inf': hybrid_inf,
        'samples': len(labels)
    }

if __name__ == "__main__":
    print("="*60)
    print("FEATURE DEBUG ANALYSIS")
    print("="*60)
    
    viewpoints = ['front', 'left', 'right']
    results = {}
    
    for vp in viewpoints:
        train_path = DATA_DIR / f"train_features_{vp}.pt"
        test_path = DATA_DIR / f"test_features_{vp}.pt"
        
        results[f'{vp}_train'] = analyze_features(train_path, f"{vp.upper()} - TRAIN")
        results[f'{vp}_test'] = analyze_features(test_path, f"{vp.upper()} - TEST")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for name, result in results.items():
        if result:
            print(f"{name:20s}: {result['samples']:4d} samples | "
                  f"NaN: node={result['node_nan']:3d} hybrid={result['hybrid_nan']:3d} | "
                  f"Inf: node={result['node_inf']:3d} hybrid={result['hybrid_inf']:3d}")
        else:
            print(f"{name:20s}: FILE NOT FOUND")
