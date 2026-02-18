"""
Hybrid ST-GCN for Arnis Pose Classification
Combines ST-GCN's deep residual architecture with Hybrid GCN's expert features
Tests if deeper networks can better leverage expert geometric features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
from tqdm import tqdm
import json

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]

FEATURES_DIR = Path("hybrid_classifier/hybrid_features_v4")
OUTPUT_DIR = Path("baseline_comparison/hybrid_stgcn/results")

class GraphConv(nn.Module):
    """Spatial graph convolution with fixed skeleton adjacency"""
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.A = nn.Parameter(A.clone(), requires_grad=True)  # Learnable adjacency
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        # x: (batch, C, T, V) = (N, channels, time, joints)
        N, C, T, V = x.size()
        
        # Graph convolution: A ∈ (V, V), x ∈ (N, C, T, V)
        x_reshaped = x.permute(0, 1, 2, 3).contiguous().view(N*C*T, V)
        x_graph = x_reshaped @ self.A  # Apply adjacency
        x_graph = x_graph.view(N, C, T, V)
        
        # Standard 1x1 conv + BN
        x_out = self.conv(x_graph)
        return self.bn(x_out)

class STGCNBlock(nn.Module):
    """ST-GCN block with residual connection and dropout"""
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, dropout=0.3):
        super().__init__()
        
        self.gcn = GraphConv(in_channels, out_channels, A)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.dropout(x)  # Apply dropout before activation
        x = self.relu(x + res)
        return x

class HybridSTGCN(nn.Module):
    """ST-GCN using EXPERT FEATURES from Hybrid GCN with regularization"""
    def __init__(self, num_classes=12, num_joints=35, 
                 in_channels=6, num_frames=1, dropout=0.5):
        super().__init__()
        
        # Create fixed skeleton adjacency matrix (35, 35)
        A = self._create_adjacency(num_joints)
        self.register_buffer('A_fixed', A)
        
        # ST-GCN layers: 5 layers (reduced from 9 to prevent overfitting)
        # USING EXPERT FEATURES (6D: angles, distances, velocities)
        self.st_gcn_networks = nn.ModuleList([
            STGCNBlock(6, 64, A, stride=1, residual=False, dropout=dropout),
            STGCNBlock(64, 64, A, stride=1, dropout=dropout),
            STGCNBlock(64, 128, A, stride=1, dropout=dropout),
            STGCNBlock(128, 128, A, stride=1, dropout=dropout),
            STGCNBlock(128, 256, A, stride=1, dropout=dropout),
        ])
        
        self.fc = nn.Linear(256, num_classes)
        
    def _create_adjacency(self, num_joints):
        """Create normalized adjacency from skeleton edges"""
        A = torch.zeros(num_joints, num_joints)
        
        # Define skeleton edges (same as Hybrid GCN)
        edges = [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),  # Right eye
            (0, 4), (4, 5), (5, 6), (6, 8),  # Left eye
            (9, 10),  # Mouth
            
            # Torso
            (11, 12),  # Shoulders
            (11, 23), (12, 24),  # Shoulder to hip
            (23, 24),  # Hips
            
            # Right arm
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            
            # Left arm
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            
            # Right leg
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
            
            # Left leg
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
            
            # Stick connections (nodes 33 and 34)
            (15, 33), (33, 34)  # Right hand to stick tip
        ]
        
        for i, j in edges:
            if i < num_joints and j < num_joints:
                A[i, j] = 1
                A[j, i] = 1  # Undirected
        
        # Add self-loops and normalize
        A = A + torch.eye(num_joints)
        D = A.sum(dim=1, keepdim=True)
        A = A / D  # Row-normalized adjacency
        return A
        
    def forward(self, x):
        # x: (batch, T, V, C) = (N, 1, 35, 6) for static
        # ST-GCN expects: (N, C, T, V)
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, 6, 1, 35)
        
        for gcn in self.st_gcn_networks:
            x = gcn(x)
            
        # Global pooling: N, C, T, V → N, C
        x = x.mean(dim=(2, 3))  # Average over time and joints
        
        return self.fc(x)

def load_data(viewpoint_filter=None):
    """Load node features from hybrid_features_v4 (WITH EXPERT FEATURES)"""
    suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
    
    train_data = torch.load(FEATURES_DIR / f"train_features{suffix}.pt")
    test_data = torch.load(FEATURES_DIR / f"test_features{suffix}.pt")
    
    X_train = train_data['node_features']  # (N, 35, 6) - EXPERT FEATURES
    y_train = train_data['labels']
    
    X_test = test_data['node_features']
    y_test = test_data['labels']
    
    # Add temporal dimension: (N, 35, 6) → (N, 1, 35, 6)
    X_train = X_train.unsqueeze(1)
    X_test = X_test.unsqueeze(1)
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print("Using EXPERT FEATURES (angles, distances, velocities)")
    return X_train, y_train, X_test, y_test

def compute_class_weights(y_train):
    """Compute class weights for imbalanced dataset"""
    class_counts = torch.bincount(y_train)
    total = len(y_train)
    weights = total / (len(CLASS_NAMES) * class_counts.float())
    return weights

def train_hybrid_stgcn(viewpoint_filter=None, epochs=150, lr=0.001, batch_size=32, dropout=0.5):
    """Train Hybrid ST-GCN classifier with regularization"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print(f"Hybrid ST-GCN Training (Device: {device})")
    print("5-Layer ST-GCN + Expert Features + Regularization")
    print(f"Dropout: {dropout}, Weight Decay: 5e-3, Label Smoothing: 0.1")
    print("="*60)
    
    # Load data
    X_train, y_train, X_test, y_test = load_data(viewpoint_filter)
    
    # Compute class weights
    class_weights = compute_class_weights(y_train).to(device)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model with dropout
    model = HybridSTGCN(
        num_classes=len(CLASS_NAMES),
        num_joints=35,
        in_channels=6,
        num_frames=1,
        dropout=dropout
    ).to(device)
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Optimizer with increased weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_test_acc = 0.0
    patience = 15  # Reduced from 20 for more aggressive early stopping
    patience_counter = 0
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += batch_y.size(0)
                test_correct += predicted.eq(batch_y).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = test_correct / test_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
        
        # Early stopping
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            # Save best model
            suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
            torch.save(model.state_dict(), OUTPUT_DIR / f"hybrid_stgcn_best{suffix}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model for evaluation
    suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
    model.load_state_dict(torch.load(OUTPUT_DIR / f"hybrid_stgcn_best{suffix}.pth"))
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    final_acc = accuracy_score(all_labels, all_preds)
    
    print(f"\n{'='*60}")
    print(f"Best Test Accuracy: {best_test_acc:.4f}")
    print(f"Final Test Accuracy: {final_acc:.4f}")
    print(f"{'='*60}\n")
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))
    
    # Visualizations
    plot_training_history(history, suffix)
    plot_confusion_matrix(all_labels, all_preds, suffix)
    
    # Save results
    results = {
        'model': 'Hybrid ST-GCN (Expert Features, 5 Layers)',
        'best_test_accuracy': float(best_test_acc),
        'final_test_accuracy': float(final_acc),
        'architecture': {
            'num_layers': 5,
            'num_frames': 1,
            'residual_connections': True,
            'learnable_adjacency': True,
            'expert_features': True,
            'dropout': dropout
        },
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'classification_report': classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    }
    
    results_path = OUTPUT_DIR / f"results{suffix}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")

def plot_training_history(history, suffix=""):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['test_loss'], label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Hybrid ST-GCN - Training History (Loss)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['test_acc'], label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Hybrid ST-GCN - Training History (Accuracy)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"training_history{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {output_path}")

def plot_confusion_matrix(y_true, y_pred, suffix=""):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[c.replace('_correct', '').replace('_', ' ').title() for c in CLASS_NAMES],
                yticklabels=[c.replace('_correct', '').replace('_', ' ').title() for c in CLASS_NAMES])
    plt.title('Hybrid ST-GCN - Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"confusion_matrix{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Hybrid ST-GCN with regularization")
    parser.add_argument('--viewpoint', type=str, required=True,
                        choices=['front', 'left', 'right'],
                        help='Viewpoint to train on (required)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')
    
    args = parser.parse_args()
    train_hybrid_stgcn(args.viewpoint, args.epochs, args.lr, args.batch_size, args.dropout)
