import os
import sys
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from training.geopose.model import GeoPoseNet

class PoseAugmenter:
    """ geometric data augmentation for graphs """
    def __init__(self, rotation_range=15, scale_range=0.1, jitter=0.01):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.jitter = jitter
        
    def __call__(self, data):
        data = data.clone()
        
        # 1. Random Rotation (2D)
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            theta = np.radians(angle)
            c, s = np.cos(theta), np.sin(theta)
            R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
            data.x = torch.matmul(data.x, R)
            
        # 2. Random Scaling
        if self.scale_range > 0:
            scale = 1.0 + np.random.uniform(-self.scale_range, self.scale_range)
            data.x = data.x * scale
            
        # 3. Jitter
        if self.jitter > 0:
            noise = torch.randn_like(data.x) * self.jitter
            data.x = data.x + noise
            
        # Re-compute edge attributes if position changed significantly?
        # Ideally yes, but for MVP we assume edge attrs (dist, angle) follow x.
        # Rotation preserves distance, scaling scales distance.
        # But sine/cosine might change if we didn't rotate edge attributes.
        # However, GraphBuilder computes edge attributes from x.
        # If we augment x, we should technically re-compute edge attributes.
        # But that requires importing GraphBuilder logic here which is slow.
        # For MVP, let's assume the network learns robustness or we limit augs that invalidate edge features.
        # Rotation invalides cos/sin OF edges if edge features rely on absolute orientation.
        # Our GraphBuilder uses relative diffs.
        # If we rotate nodes, the relative vector rotates.
        # If we don't update edge_attr, there is a mismatch.
        # Let's skip rotation for now or accept the noise.
        # Better: Jitter and Scale are safe-ish. Rotation is hard without recomputing.
        # Let's drop rotation for now to be safe.
        
        return data

def train_model(data_dir='data/processed', batch_size=32, epochs=50, folds=5, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    train_path = os.path.join(project_root, data_dir, 'train_graphs.pt')
    if not os.path.exists(train_path):
        print("Dataset not found. Run create_dataset.py first.")
        return
        
    dataset = torch.load(train_path)
    labels = [d.y.item() for d in dataset]
    
    # Load dataset info for num_classes
    info_path = os.path.join(project_root, data_dir, 'dataset_info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
            num_classes = info['num_classes']
            class_names = info['classes']
    else:
        num_classes = len(set(labels))
        class_names = [str(i) for i in range(num_classes)]
        
    print(f"Loaded {len(dataset)} samples, {num_classes} classes.")
    
    # Cross Validation
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    fold_results = []
    
    augmenter = PoseAugmenter(rotation_range=0, scale_range=0.1, jitter=0.02)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, labels)):
        print(f"\nFold {fold+1}/{folds}")
        
        train_subset = [dataset[i] for i in train_idx]
        val_subset = [dataset[i] for i in val_idx]
        
        # Apply augmentation to training data (online augmentation via loader? No, simple offline for now or pre-generate)
        # PyG DataLoader doesn't support transform easily on list.
        # We can implement a custom collate or just augment in loop.
        # Let's keep it simple: No aug in DataLoader for MVP speed, or basic augmentation.
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        model = GeoPoseNet(num_classes=num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0.0
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for data in train_loader:
                data = data.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Augmentation (simple implementation)
                # if np.random.random() > 0.5:
                #    data = augmenter(data) # Requires moving back/forth or implementation on Tensor
                
                # Forward
                # Check dimensions
                # print(data.x.shape, data.edge_index.shape, data.edge_attr.shape)
                
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = criterion(out, data.y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * data.num_graphs
                pred = out.argmax(dim=1)
                correct += int((pred == data.y).sum())
                total += data.num_graphs
                
            train_acc = correct / total
            train_loss = total_loss / total
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0 # Track validation loss as well
            
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                    loss = criterion(out, data.y)
                    val_loss += loss.item() * data.num_graphs
                    
                    pred = out.argmax(dim=1)
                    val_correct += int((pred == data.y).sum())
                    val_total += data.num_graphs
            
            val_acc = val_correct / val_total
            val_loss_avg = val_loss / val_total
            
            scheduler.step()
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss_avg:.4f} Acc={val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                # Save best model for this fold
                torch.save(model.state_dict(), os.path.join(project_root, data_dir, f'model_fold{fold}.pt'))
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print("Early stopping")
                break
                
        fold_results.append(best_acc)
        print(f"Fold {fold+1} Best Val Acc: {best_acc:.4f}")
        
    print(f"\nAverage Accuracy: {np.mean(fold_results):.4f} +/- {np.std(fold_results):.4f}")
    
    # Retrain on full dataset and save final model
    print("\nRetraining on full dataset...")
    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    final_model = GeoPoseNet(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-4) # Reset optimizer
    
    # Train for mean epochs of folds or fixed number? Fixed is safer for MVP.
    for epoch in range(int(epochs * 0.8)): # Slightly less to avoid overfit? Or same.
        final_model.train()
        for data in full_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = final_model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            
    torch.save(final_model.state_dict(), os.path.join(project_root, 'models', 'geopose_model.pt'))
    print("Final model saved to models/geopose_model.pt")

if __name__ == '__main__':
    # usage: python train.py
    train_model()
