"""
Training Script for Spatial GCN
Trains on graph dataset with strong regularization to prevent overfitting
Target: 80%+ accuracy with <15% train-val gap
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

import sys
sys.path.append('models')
from spatial_gcn import SpatialGCN


# Class names
CLASS_NAMES = [
    'crown_thrust_correct',
    'left_chest_thrust_correct',
    'left_elbow_block_correct',
    'left_eye_thrust_correct',
    'left_knee_block_correct',
    'left_temple_block_correct',
    'neutral_stance',
    'right_chest_thrust_correct',
    'right_elbow_block_correct',
    'right_eye_thrust_correct',
    'right_knee_block_correct',
    'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]


def load_graph_dataset(dataset_root, split='train'):
    """Load all .pt graph files from dataset."""
    dataset_root = Path(dataset_root)
    graphs = []
    
    split_path = dataset_root / split
    
    for viewpoint in ['front', 'left', 'right']:
        viewpoint_path = split_path / viewpoint
        
        if not viewpoint_path.exists():
            continue
        
        for class_dir in viewpoint_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            # Load all .pt files
            for graph_file in class_dir.glob('*.pt'):
                graph = torch.load(graph_file)
                graphs.append(graph)
    
    return graphs


def compute_class_weights(dataset):
    """Compute class weights for imbalanced dataset."""
    class_counts = torch.zeros(len(CLASS_NAMES))
    
    for graph in dataset:
        class_counts[graph.y.item()] += 1
    
    # Inverse frequency weighting
    total = class_counts.sum()
    class_weights = total / (len(CLASS_NAMES) * class_counts)
    
    return class_weights


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            batch = batch.to(device)
            
            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            
            # Metrics
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def evaluate_per_viewpoint(model, dataset_root, device):
    """Evaluate model separately for each viewpoint."""
    model.eval()
    
    results = {}
    
    for viewpoint in ['front', 'left', 'right']:
        graphs = []
        dataset_path = Path(dataset_root) / 'test' / viewpoint
        
        if not dataset_path.exists():
            continue
        
        # Load graphs for this viewpoint
        for class_dir in dataset_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            for graph_file in class_dir.glob('*.pt'):
                graph = torch.load(graph_file)
                graphs.append(graph)
        
        if len(graphs) == 0:
            continue
        
        # Evaluate
        loader = DataLoader(graphs, batch_size=32, shuffle=False)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        
        accuracy = correct / total
        results[viewpoint] = accuracy
    
    return results


def train_model(
    dataset_root='dataset_graphs',
    batch_size=32,
    hidden_channels=64,
    dropout=0.5,
    lr=0.001,
    weight_decay=1e-3,
    max_epochs=200,
    early_stop_patience=20,
    output_dir='models/gcn_checkpoints'
):
    """Main training function."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = load_graph_dataset(dataset_root, 'train')
    test_dataset = load_graph_dataset(dataset_root, 'test')
    
    print(f"Train graphs: {len(train_dataset)}")
    print(f"Test graphs: {len(test_dataset)}")
    
    # Compute class weights
    class_weights = compute_class_weights(train_dataset)
    class_weights = class_weights.to(device)
    print(f"Class weights computed: {class_weights}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    num_node_features = train_dataset[0].x.shape[1]
    print(f"Input node features: {num_node_features}")
    
    model = SpatialGCN(
        in_channels=num_node_features,
        hidden_channels=hidden_channels,
        num_classes=len(CLASS_NAMES),
        dropout=dropout
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, test_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Train-Val Gap: {abs(train_acc - val_acc):.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_acc': train_acc,
                'history': history
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"✓ Saved best model (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"Early stopping: {patience_counter}/{early_stop_patience}")
            
            if patience_counter >= early_stop_patience:
                print("\n⚠ Early stopping triggered")
                break
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    print(f"Best Val Accuracy: {checkpoint['val_acc']:.4f}")
    print(f"Best Train Accuracy: {checkpoint['train_acc']:.4f}")
    print(f"Train-Val Gap: {abs(checkpoint['train_acc'] - checkpoint['val_acc']):.4f}")
    
    # Per-viewpoint evaluation
    print("\nPer-Viewpoint Accuracy:")
    viewpoint_results = evaluate_per_viewpoint(model, dataset_root, device)
    for viewpoint, acc in viewpoint_results.items():
        print(f"  {viewpoint}: {acc:.4f}")
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n✓ Training complete!")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Spatial GCN")
    parser.add_argument('--dataset_root', type=str, default='dataset_graphs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='models/gcn_checkpoints')
    
    args = parser.parse_args()
    
    train_model(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        early_stop_patience=args.early_stop_patience,
        output_dir=args.output_dir
    )
