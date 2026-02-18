"""
Hybrid EdgeConv for Arnis Pose Classification
Uses expert features with dynamic graph learning
Optimized based on Hybrid ST-GCN lessons: moderate depth, strong regularization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import EdgeConv, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import json

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]

FEATURES_DIR = Path("hybrid_classifier/hybrid_features_v4")
OUTPUT_DIR = Path("baseline_comparison/hybrid_edgeconv/results")

class HybridEdgeConv(nn.Module):
    """EdgeConv with expert features - learns optimal graph structure dynamically"""
    def __init__(self, num_classes=12, k=15, dropout=0.4):
        """
        Args:
            num_classes: Number of output classes
            k: Number of nearest neighbors (reduced from 20 to prevent overfitting)
            dropout: Dropout rate (0.4 - moderate regularization)
        """
        super().__init__()
        self.k = k
        
        # 3 EdgeConv layers (moderate depth based on Hybrid ST-GCN lessons)
        # Input: 6D expert features
        self.conv1 = EdgeConv(nn.Sequential(
            nn.Linear(2 * 6, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        ), aggr='max')
        
        self.conv2 = EdgeConv(nn.Sequential(
            nn.Linear(2 * 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        ), aggr='max')
        
        self.conv3 = EdgeConv(nn.Sequential(
            nn.Linear(2 * 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        ), aggr='max')
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, data):
        x, batch = data.x, data.batch
        
        # EdgeConv dynamically computes k-NN graph at each layer
        from torch_geometric.nn import knn_graph
        
        # Layer 1: Compute k-NN edges and apply EdgeConv
        edge_index = knn_graph(x, k=self.k, batch=batch)
        x = self.conv1(x, edge_index)
        
        # Layer 2: Recompute k-NN in new feature space
        edge_index = knn_graph(x, k=self.k, batch=batch)
        x = self.conv2(x, edge_index)
        
        # Layer 3: Recompute k-NN again
        edge_index = knn_graph(x, k=self.k, batch=batch)
        x = self.conv3(x, edge_index)
        
        # Global max pooling
        x = global_max_pool(x, batch)
        
        return self.fc(x)

def load_data(viewpoint_filter=None):
    """Load expert features from hybrid_features_v4"""
    suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
    
    train_data = torch.load(FEATURES_DIR / f"train_features{suffix}.pt")
    test_data = torch.load(FEATURES_DIR / f"test_features{suffix}.pt")
    
    X_train = train_data['node_features']  # (N, 35, 6) - EXPERT FEATURES
    y_train = train_data['labels']
    
    X_test = test_data['node_features']
    y_test = test_data['labels']
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print("Using EXPERT FEATURES (angles, distances, velocities)")
    return X_train, y_train, X_test, y_test

def create_graph_data(node_features, labels):
    """Convert node features to PyG Data objects"""
    graphs = []
    for i in range(len(node_features)):
        # Each sample: 35 nodes × 6 features
        x = node_features[i]  # (35, 6)
        y = labels[i]
        
        # EdgeConv will dynamically compute edges based on k-NN
        # We don't need to provide edge_index
        data = Data(x=x, y=y)
        graphs.append(data)
    
    return graphs

def compute_class_weights(y_train):
    class_counts = torch.bincount(y_train)
    total = len(y_train)
    weights = total / (len(CLASS_NAMES) * class_counts.float())
    return weights

def train_hybrid_edgeconv(viewpoint_filter=None, epochs=150, lr=0.001, batch_size=32, 
                          k=15, dropout=0.4):
    """Train Hybrid EdgeConv with optimized parameters"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print(f"Hybrid EdgeConv Training (Device: {device})")
    print("Dynamic Graph Learning + Expert Features")
    print(f"k={k} neighbors, Dropout={dropout}, Weight Decay=5e-3")
    print("="*60)
    
    # Load data
    X_train, y_train, X_test, y_test = load_data(viewpoint_filter)
    
    # Convert to graph format
    train_graphs = create_graph_data(X_train, y_train)
    test_graphs = create_graph_data(X_test, y_test)
    
    # Create dataloaders
    from torch_geometric.loader import DataLoader as PyGDataLoader
    train_loader = PyGDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = PyGDataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    # Compute class weights
    class_weights = compute_class_weights(y_train).to(device)
    
    # Initialize model
    model = HybridEdgeConv(
        num_classes=len(CLASS_NAMES),
        k=k,
        dropout=dropout
    ).to(device)
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Optimizer with moderate weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_test_acc = 0.0
    patience = 15
    patience_counter = 0
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch.y.size(0)
            train_correct += predicted.eq(batch.y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                outputs = model(batch)
                loss = criterion(outputs, batch.y)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += batch.y.size(0)
                test_correct += predicted.eq(batch.y).sum().item()
        
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
            suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
            torch.save(model.state_dict(), OUTPUT_DIR / f"hybrid_edgeconv_best{suffix}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model for evaluation
    suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
    model.load_state_dict(torch.load(OUTPUT_DIR / f"hybrid_edgeconv_best{suffix}.pth"))
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
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
        'model': 'Hybrid EdgeConv (Expert Features)',
        'best_test_accuracy': float(best_test_acc),
        'final_test_accuracy': float(final_acc),
        'architecture': {
            'num_layers': 3,
            'k_neighbors': k,
            'dropout': dropout,
            'weight_decay': 5e-3,
            'label_smoothing': 0.1,
            'expert_features': True
        },
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'classification_report': classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    }
    
    results_path = OUTPUT_DIR / f"results{suffix}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")

def plot_training_history(history, suffix=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['test_loss'], label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Hybrid EdgeConv - Training History (Loss)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['test_acc'], label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Hybrid EdgeConv - Training History (Accuracy)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"training_history{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {output_path}")

def plot_confusion_matrix(y_true, y_pred, suffix=""):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[c.replace('_correct', '').replace('_', ' ').title() for c in CLASS_NAMES],
                yticklabels=[c.replace('_correct', '').replace('_', ' ').title() for c in CLASS_NAMES])
    plt.title('Hybrid EdgeConv - Normalized Confusion Matrix')
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
    
    parser = argparse.ArgumentParser(description="Train Hybrid EdgeConv")
    parser.add_argument('--viewpoint', type=str, required=True,
                        choices=['front', 'left', 'right'])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--k', type=int, default=15,
                        help='Number of nearest neighbors (default: 15)')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate (default: 0.4)')
    
    args = parser.parse_args()
    train_hybrid_edgeconv(args.viewpoint, args.epochs, args.lr, args.batch_size, args.k, args.dropout)
