
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from training.train_gcn import load_graph_dataset, CLASS_NAMES
from models.spatial_gcn import SpatialGCN
from torch_geometric.loader import DataLoader
from pathlib import Path

# Config
MODEL_PATH = "models/gcn_checkpoints/best_model.pth"
DATASET_ROOT = "dataset_graphs"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def analyze_errors():
    print(f"Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Load attributes
    hidden_channels = 64 # Default from train_gcn
    dropout = 0.5
    
    # Load dataset to get input features logic
    print("Loading test dataset...")
    test_graphs = load_graph_dataset(DATASET_ROOT, 'test')
    if len(test_graphs) == 0:
        print("Error: No test graphs found.")
        return

    num_node_features = test_graphs[0].x.shape[1]
    print(f"Input features: {num_node_features}")
    
    # Initialize model
    model = SpatialGCN(
        in_channels=num_node_features,
        hidden_channels=hidden_channels,
        num_classes=len(CLASS_NAMES),
        dropout=dropout
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Run inference
    loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    all_preds = []
    all_labels = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds = out.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalize CM
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")
    
    # Text Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

if __name__ == "__main__":
    analyze_errors()
