"""
Hybrid ST-GCN for Arnis Pose Classification - AGGRESSIVE REGULARIZATION
9-layer deep network with extreme regularization measures
Tests if deep networks can work with very aggressive overfitting prevention
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
OUTPUT_DIR = Path("baseline_comparison/hybrid_stgcn/results_aggressive")

class GraphConv(nn.Module):
    """Spatial graph convolution with fixed skeleton adjacency"""
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.A = nn.Parameter(A.clone(), requires_grad=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        N, C, T, V = x.size()
        x_reshaped = x.permute(0, 1, 2, 3).contiguous().view(N*C*T, V)
        x_graph = x_reshaped @ self.A
        x_graph = x_graph.view(N, C, T, V)
        x_out = self.conv(x_graph)
        return self.bn(x_out)

class STGCNBlock(nn.Module):
    """ST-GCN block with residual connection and aggressive dropout"""
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, dropout=0.5):
        super().__init__()
        
        self.gcn = GraphConv(in_channels, out_channels, A)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
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
        x = self.dropout(x)
        x = self.relu(x + res)
        return x

class HybridSTGCNAggressive(nn.Module):
    """9-layer ST-GCN with AGGRESSIVE regularization"""
    def __init__(self, num_classes=12, num_joints=35, 
                 in_channels=6, num_frames=1, dropout=0.5):
        super().__init__()
        
        A = self._create_adjacency(num_joints)
        self.register_buffer('A_fixed', A)
        
        # 9 layers with AGGRESSIVE dropout
        self.st_gcn_networks = nn.ModuleList([
            STGCNBlock(6, 64, A, stride=1, residual=False, dropout=dropout),
            STGCNBlock(64, 64, A, stride=1, dropout=dropout),
            STGCNBlock(64, 64, A, stride=1, dropout=dropout),
            STGCNBlock(64, 64, A, stride=1, dropout=dropout),
            STGCNBlock(64, 128, A, stride=1, dropout=dropout),
            STGCNBlock(128, 128, A, stride=1, dropout=dropout),
            STGCNBlock(128, 128, A, stride=1, dropout=dropout),
            STGCNBlock(128, 256, A, stride=1, dropout=dropout),
            STGCNBlock(256, 256, A, stride=1, dropout=dropout),
        ])
        
        self.fc = nn.Linear(256, num_classes)
        
    def _create_adjacency(self, num_joints):
        A = torch.zeros(num_joints, num_joints)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10),
            (11, 12), (11, 23), (12, 24), (23, 24),
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
            (15, 33), (33, 34)
        ]
        
        for i, j in edges:
            if i < num_joints and j < num_joints:
                A[i, j] = 1
                A[j, i] = 1
        
        A = A + torch.eye(num_joints)
        D = A.sum(dim=1, keepdim=True)
        A = A / D
        return A
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        for gcn in self.st_gcn_networks:
            x = gcn(x)
        x = x.mean(dim=(2, 3))
        return self.fc(x)

def load_data(viewpoint_filter=None):
    suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
    train_data = torch.load(FEATURES_DIR / f"train_features{suffix}.pt")
    test_data = torch.load(FEATURES_DIR / f"test_features{suffix}.pt")
    
    X_train = train_data['node_features'].unsqueeze(1)
    y_train = train_data['labels']
    X_test = test_data['node_features'].unsqueeze(1)
    y_test = test_data['labels']
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print("Using EXPERT FEATURES (angles, distances, velocities)")
    return X_train, y_train, X_test, y_test

def compute_class_weights(y_train):
    class_counts = torch.bincount(y_train)
    total = len(y_train)
    weights = total / (len(CLASS_NAMES) * class_counts.float())
    return weights

def train_aggressive(viewpoint_filter=None, epochs=150, lr=0.0005, batch_size=32, dropout=0.5):
    """Train with ALL aggressive measures"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print(f"AGGRESSIVE 9-Layer ST-GCN (Device: {device})")
    print("Extreme Regularization:")
    print(f"  - Dropout: {dropout}")
    print(f"  - Weight Decay: 1e-2")
    print(f"  - Label Smoothing: 0.1")
    print(f"  - Learning Rate: {lr}")
    print(f"  - Gradient Clipping: 1.0")
    print("="*60)
    
    X_train, y_train, X_test, y_test = load_data(viewpoint_filter)
    class_weights = compute_class_weights(y_train).to(device)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = HybridSTGCNAggressive(
        num_classes=len(CLASS_NAMES),
        num_joints=35,
        in_channels=6,
        num_frames=1,
        dropout=dropout
    ).to(device)
    
    # AGGRESSIVE: Higher weight decay
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_test_acc = 0.0
    patience = 15
    patience_counter = 0
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
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
            
            # AGGRESSIVE: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
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
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
            torch.save(model.state_dict(), OUTPUT_DIR / f"aggressive_best{suffix}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Evaluation
    suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
    model.load_state_dict(torch.load(OUTPUT_DIR / f"aggressive_best{suffix}.pth"))
    
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
    
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))
    
    # Save results
    results = {
        'model': 'Hybrid ST-GCN (9 Layers, Aggressive Regularization)',
        'best_test_accuracy': float(best_test_acc),
        'final_test_accuracy': float(final_acc),
        'architecture': {
            'num_layers': 9,
            'dropout': dropout,
            'weight_decay': 1e-2,
            'learning_rate': lr,
            'gradient_clipping': 1.0,
            'label_smoothing': 0.1
        },
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'classification_report': classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    }
    
    results_path = OUTPUT_DIR / f"results{suffix}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Aggressive 9-Layer Hybrid ST-GCN")
    parser.add_argument('--viewpoint', type=str, required=True,
                        choices=['front', 'left', 'right'])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate (default: 0.0005, lower than standard)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')
    
    args = parser.parse_args()
    train_aggressive(args.viewpoint, args.epochs, args.lr, args.batch_size, args.dropout)
