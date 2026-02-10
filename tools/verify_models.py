import torch
import sys
from pathlib import Path

# Add src to path to load architecture
sys.path.append(str(Path("deployment_package/src").resolve()))
from model_architecture import HybridGCN

def verify_models():
    models_dir = Path("deployment_package/models")
    models = ["hybrid_gcn_v2_front.pth", "hybrid_gcn_v2_left.pth", "hybrid_gcn_v2_right.pth"]
    
    for m in models:
        path = models_dir / m
        print(f"Checking {m} ({path.stat().st_size / 1024:.1f} KB)...")
        try:
            checkpoint = torch.load(path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                print("  - Contains state_dict")
                # Try loading into architecture
                # We need the params from the checkpoint usually, or defaults
                # The checkpoint in 4c saves: node_feat_dim, hybrid_feat_dim, hidden_dim, etc.
                
                args = {
                    'node_in_channels': checkpoint.get('node_feat_dim', 6), # default 6
                    'hybrid_in_channels': checkpoint.get('hybrid_feat_dim', 30), # default 30
                    'hidden_channels': checkpoint.get('hidden_dim', 256),
                    'num_classes': 12, # We know it's 12 now
                    'num_layers': checkpoint.get('num_layers', 3),
                    'dropout': checkpoint.get('dropout', 0.5)
                }
                
                model = HybridGCN(**args)
                model.load_state_dict(checkpoint['model_state_dict'])
                print("  - Successfully loaded into HybridGCN")
            else:
                print("  - Warning: No model_state_dict found (maybe direct model save?)")
                
        except Exception as e:
            print(f"  - FAILED to load: {e}")

if __name__ == "__main__":
    verify_models()
