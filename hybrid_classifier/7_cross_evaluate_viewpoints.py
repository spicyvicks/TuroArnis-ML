"""
Step 7: Cross-Evaluate Specialist Models
Generates a heatmap showing how well each specialist model performs on different viewpoints.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path to import model
sys.path.append(str(Path(__file__).parent.parent))
from hybrid_classifier import HybridGCN, load_hybrid_graph_data, MODELS_DIR, OUTPUT_DIR

VIEWPOINTS = ['front', 'left', 'right']

def evaluate_cross_matrix():
    """Evaluate every model on every viewpoint dataset"""
    
    results = np.zeros((3, 3))
    
    print("\n" + "="*60)
    print("CROSS-VIEWPOINT EVALUATION")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Iterate through models (Rows)
    for i, model_vp in enumerate(VIEWPOINTS):
        model_path = MODELS_DIR / f"hybrid_gcn_v2_{model_vp}.pth"
        
        if not model_path.exists():
            print(f"Skipping {model_vp} model (not found)")
            continue
            
        print(f"\nLoading Model: {model_vp.upper()}")
        checkpoint = torch.load(model_path, map_location=device)
        
        model = HybridGCN(
            node_in_channels=checkpoint['node_feat_dim'],
            hybrid_in_channels=checkpoint['hybrid_feat_dim'],
            hidden_channels=checkpoint['hidden_dim'],
            num_classes=12, # Hardcoded based on CLASS_NAMES length
            num_layers=checkpoint.get('num_layers', 3),
            dropout=checkpoint['dropout'],
            embedding_dim=checkpoint.get('embedding_dim', 8)
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Iterate through datasets (Columns)
        for j, data_vp in enumerate(VIEWPOINTS):
            print(f"  Testing on {data_vp} data...", end="")
            
            try:
                test_graphs = load_hybrid_graph_data(data_vp)
            except FileNotFoundError:
                print(" [Data Not Found]")
                continue
                
            from torch_geometric.loader import DataLoader
            test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    hybrid_batch = batch.hybrid.view(batch.num_graphs, -1)
                    
                    out = model(batch.x, batch.edge_index, batch.batch, hybrid_batch)
                    pred = out.argmax(dim=1)
                    
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
            
            acc = correct / total if total > 0 else 0
            results[i, j] = acc
            print(f" Accuracy: {acc:.4f}")

    # Plot Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(results, annot=True, fmt='.2%', cmap='RdYlGn', vmin=0, vmax=1,
                xticklabels=[vp.capitalize() for vp in VIEWPOINTS],
                yticklabels=[vp.capitalize() for vp in VIEWPOINTS])
    
    plt.title('Cross-Viewpoint Model Performance')
    plt.xlabel('Test Data Viewpoint')
    plt.ylabel('Trained Model Viewpoint')
    plt.tight_layout()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / 'cross_viewpoint_matrix.png'
    plt.savefig(out_file)
    print(f"\n✓ Matrix saved to {out_file}")

if __name__ == "__main__":
    evaluate_cross_matrix()
