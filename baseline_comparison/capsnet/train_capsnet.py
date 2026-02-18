"""
CapsNet Baseline for Arnis Pose Classification
Tests if capsule routing handles viewpoint variation better than GCN
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
OUTPUT_DIR = Path("baseline_comparison/capsnet/results")

def squash(s, dim=-1):
    """Capsule non-linearity: preserves direction, scales magnitude"""
    norm = torch.norm(s, dim=dim, keepdim=True)
    scale = norm / (1 + norm**2)
    return scale * s / (norm + 1e-8)

class CapsNet(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        # 7 anatomical groups: Left Arm, Right Arm, Torso, Left Leg, Right Leg, Stick, Head
        # Each group has 5 joints × 6 features = 30 features
        self.primary_caps = nn.ModuleList([
            nn.Linear(30, 8) for _ in range(7)
        ])
        
        # Routing parameters: 7 primary caps → 12 digit caps
        self.W = nn.Parameter(torch.randn(7, 12, 8, 16))  # 16D digit caps
        
        self.num_classes = num_classes
        
    def forward(self, x, num_iterations=3):
        batch = x.size(0)
        
        # Group 35 joints into 7 anatomical regions
        # Joints 0-4: Head region (nose, eyes, ears)
        # Joints 5-9: Left arm (shoulder, elbow, wrist, hand)
        # Joints 10-14: Right arm
        # Joints 15-19: Torso (shoulders, hips)
        # Joints 20-24: Left leg
        # Joints 25-29: Right leg
        # Joints 30-34: Stick (grip, tip, extras)
        
        # Reshape: (batch, 35, 6) -> (batch, 7, 5, 6) -> (batch, 7, 30)
        # Note: This assumes we can evenly divide 35 nodes into 7 groups
        # In practice, we pad to 35 nodes (33 pose + 2 stick = 35)
        groups = x.view(batch, 7, 5, 6).reshape(batch, 7, 30)
        
        # Primary capsules: (batch, 7, 8)
        u = torch.stack([squash(self.primary_caps[i](groups[:, i])) 
                        for i in range(7)], dim=1)
        
        # Expand for routing: (batch, 7, 12, 8) @ (7, 12, 8, 16) -> (batch, 7, 12, 16)
        u_expanded = u.unsqueeze(2).expand(-1, -1, 12, -1)  # (batch, 7, 12, 8)
        u_hat = torch.einsum('bnmj,nmjk->bnmk', u_expanded, self.W)  # (batch, 7, 12, 16)
        
        # Dynamic routing
        b = torch.zeros(batch, 7, 12, device=x.device)
        for iteration in range(num_iterations):
            c = F.softmax(b, dim=2)  # (batch, 7, 12)
            s = torch.einsum('bnm,bnmk->bmk', c, u_hat)  # (batch, 12, 16)
            v = squash(s)  # (batch, 12, 16)
            
            if iteration < num_iterations - 1:
                # Update routing coefficients
                b = b + torch.einsum('bnmk,bmk->bnm', u_hat, v)
        
        # Class probabilities from capsule lengths
        return torch.norm(v, dim=-1)  # (batch, 12)

def margin_loss(v_lengths, labels, m_plus=0.9, m_minus=0.1, lambda_=0.5):
    """Margin loss for capsule network"""
    batch_size = v_lengths.size(0)
    num_classes = v_lengths.size(1)
    
    # One-hot encode labels
    labels_one_hot = F.one_hot(labels, num_classes).float()
    
    # Compute loss
    loss_positive = labels_one_hot * F.relu(m_plus - v_lengths) ** 2
    loss_negative = lambda_ * (1 - labels_one_hot) * F.relu(v_lengths - m_minus) ** 2
    
    loss = (loss_positive + loss_negative).sum(dim=1).mean()
    return loss

def load_data(viewpoint_filter=None):
    """Load node features from hybrid_features_v3"""
    suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
    
    train_data = torch.load(FEATURES_DIR / f"train_features{suffix}.pt")
    test_data = torch.load(FEATURES_DIR / f"test_features{suffix}.pt")
    
    X_train = train_data['node_features']  # (N, 35, 6)
    y_train = train_data['labels']
    
    X_test = test_data['node_features']
    y_test = test_data['labels']
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def train_capsnet(viewpoint_filter=None, epochs=200, lr=5e-4, batch_size=32):
    """Train CapsNet classifier"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print(f"CapsNet Baseline Training (Device: {device})")
    print("="*60)
    
    # Load data
    X_train, y_train, X_test, y_test = load_data(viewpoint_filter)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = CapsNet(num_classes=len(CLASS_NAMES)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_test_acc = 0.0
    patience = 30  # CapsNet needs more patience
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
            v_lengths = model(batch_x)
            loss = margin_loss(v_lengths, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = v_lengths.max(1)
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
                
                v_lengths = model(batch_x)
                loss = margin_loss(v_lengths, batch_y)
                
                test_loss += loss.item()
                _, predicted = v_lengths.max(1)
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
            torch.save(model.state_dict(), OUTPUT_DIR / f"capsnet_best{suffix}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model for evaluation
    suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
    model.load_state_dict(torch.load(OUTPUT_DIR / f"capsnet_best{suffix}.pth"))
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            v_lengths = model(batch_x)
            _, predicted = v_lengths.max(1)
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
        'best_test_accuracy': float(best_test_acc),
        'final_test_accuracy': float(final_acc),
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
    ax1.set_title('CapsNet - Training History (Loss)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['test_acc'], label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('CapsNet - Training History (Accuracy)')
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
    plt.title('CapsNet - Normalized Confusion Matrix')
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
    
    parser = argparse.ArgumentParser(description="Train CapsNet baseline")
    parser.add_argument('--viewpoint', type=str, required=True,
                        choices=['front', 'left', 'right'],
                        help='Viewpoint to train on (required)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    
    args = parser.parse_args()
    train_capsnet(args.viewpoint, args.epochs, args.lr, args.batch_size)
