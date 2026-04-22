import torch
import json
from pathlib import Path
import numpy as np

# Load model to check
model_path = Path('hybrid_classifier/models/model_merged.pth')
if model_path.exists():
    checkpoint = torch.load(model_path, map_location='cpu')
    print('=== Model Checkpoint ===')
    print(f'Keys: {list(checkpoint.keys())}')
    if 'model_state_dict' in checkpoint:
        print(f'Model state dict: {len(checkpoint["model_state_dict"])} tensors')
    if 'val_acc' in checkpoint:
        print(f'Final val_acc: {checkpoint["val_acc"]}')
    if 'train_acc' in checkpoint:
        print(f'Final train_acc: {checkpoint["train_acc"]}')
    if 'epoch' in checkpoint:
        print(f'Trained epochs: {checkpoint["epoch"]}')
    print()

# Load training features
train_path = Path('hybrid_classifier/hybrid_features_v3/train_features.pt')
if train_path.exists():
    data = torch.load(train_path)
    print('=== Training Features ===')
    print(f'Node features: {data["node_features"].shape}')
    print(f'Hybrid features: {data["hybrid_features"].shape}')
    print(f'Labels: {data["labels"].shape}')
    print(f'Unique labels: {torch.unique(data["labels"]).tolist()}')
    
    # Check class distribution
    counts = torch.bincount(data['labels'])
    print(f'Label counts: {counts.tolist()}')
    print(f'Min class size: {counts.min().item()}')
    print(f'Max class size: {counts.max().item()}')
