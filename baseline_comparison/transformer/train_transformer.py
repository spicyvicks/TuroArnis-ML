"""
Pose Transformer Baseline for Arnis Pose Classification
Tests self-attention vs graph convolution for joint relationships
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
from tqdm import tqdm
import json
import math

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]

FEATURES_DIR = Path("hybrid_classifier/hybrid_features_v4")
OUTPUT_DIR = Path("baseline_comparison/transformer/results")

class PoseTransformer(nn.Module):
    def __init__(self, num_joints=35, feat_dim=6, 
                 d_model=64, nhead=4, num_layers=4, 
                 num_classes=12, dropout=0.1):
        super().__init__()
        
        # Project 6D features to model dimension
        self.input_proj = nn.Linear(feat_dim, d_model)
        
        # Learnable positional encoding per joint
        self.pos_embed = nn.Parameter(torch.randn(1, num_joints, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, 35, 6)
        batch_size = x.size(0)
        
        # Project and add position encoding
        x = self.input_proj(x) + self.pos_embed  # (batch, 35, 64)
        
        # Transformer: self-attention across all joint pairs
        x = self.transformer(x)  # (batch, 35, 64)
        
        # Global pooling: average over joints
        x = x.mean(dim=1)  # (batch, 64)
        
        return self.classifier(x)

def load_data(viewpoint_filter=None):
    """Load node features from hybrid_features_v4"""
    suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
    
    train_data = torch.load(FEATURES_DIR / f"train_features{suffix}.pt")
    test_data = torch.load(FEATURES_DIR / f"test_features{suffix}.pt")
    
    X_train = train_data['node_features']  # (N, 35, 6)
    y_train = train_data['labels']
    
    X_test = test_data['node_features']
    y_test = test_data['labels']
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def compute_class_weights(y_train):
    """Compute class weights for imbalanced dataset"""
    class_counts = torch.bincount(y_train)
    total = len(y_train)
    weights = total / (len(CLASS_NAMES) * class_counts.float())
    return weights

def train_transformer(viewpoint_filter=None, epochs=150, lr=0.0001, batch_size=32):
    """Train Pose Transformer classifier"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print(f"Pose Transformer Baseline Training (Device: {device})")
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
    
    # Initialize model
    model = PoseTransformer(
        num_joints=35,
        feat_dim=6,
        d_model=64,
        nhead=4,
        num_layers=4,
        num_classes=len(CLASS_NAMES),
        dropout=0.1
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Cosine annealing scheduler with warmup
    warmup_epochs = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs - warmup_epochs,
        eta_min=1e-6
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_test_acc = 0.0
    patience = 20
    patience_counter = 0
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        # Warmup learning rate
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
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
        
        # Step scheduler after warmup
        if epoch >= warmup_epochs:
            scheduler.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}] LR: {current_lr:.6f} "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
        
        # Early stopping
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            # Save best model
            suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
            torch.save(model.state_dict(), OUTPUT_DIR / f"transformer_best{suffix}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model for evaluation
    suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
    model.load_state_dict(torch.load(OUTPUT_DIR / f"transformer_best{suffix}.pth"))
    
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
        'model': 'Pose Transformer',
        'best_test_accuracy': float(best_test_acc),
        'final_test_accuracy': float(final_acc),
        'architecture': {
            'd_model': 64,
            'nhead': 4,
            'num_layers': 4,
            'dropout': 0.1
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
    ax1.set_title('Pose Transformer - Training History (Loss)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['test_acc'], label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Pose Transformer - Training History (Accuracy)')
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
    plt.title('Pose Transformer - Normalized Confusion Matrix')
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
    
    parser = argparse.ArgumentParser(description="Train Pose Transformer baseline")
    parser.add_argument('--viewpoint', type=str, required=True,
                        choices=['front', 'left', 'right'],
                        help='Viewpoint to train on (required)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001 for Transformer)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    
    args = parser.parse_args()
    train_transformer(args.viewpoint, args.epochs, args.lr, args.batch_size)
