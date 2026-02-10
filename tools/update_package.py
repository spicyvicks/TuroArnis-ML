import shutil
import os
from pathlib import Path
import json
import re

# Config
ROOT_DIR = Path(__file__).resolve().parent.parent
DEPLOY_DIR = ROOT_DIR / "deployment_package"
HYBRID_DIR = ROOT_DIR / "hybrid_classifier"
SRC_DIR = DEPLOY_DIR / "src"
MODELS_DIR = DEPLOY_DIR / "models"

# 1. Update src/model_architecture.py
def update_model_architecture():
    print("Updating model_architecture.py...")
    source_file = HYBRID_DIR / "4c_train_hybrid_gcn_v2.py"
    dest_file = SRC_DIR / "model_architecture.py"
    
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract HybridGCN class
    class_match = re.search(r'class HybridGCN\(nn\.Module\):.*?(?=\n\n\n)', content, re.DOTALL)
    if not class_match:
        print("Error: HybridGCN class not found in source.")
        return
    
    class_code = class_match.group(0)
    
    # Extract CLASS_NAMES
    class_names_match = re.search(r'CLASS_NAMES = \[.*?\]', content, re.DOTALL)
    if not class_names_match:
        print("Error: CLASS_NAMES not found in source.")
        return
        
    class_names_code = class_names_match.group(0)
    
    # Extract SKELETON_EDGES
    edges_match = re.search(r'SKELETON_EDGES = \[.*?\]', content, re.DOTALL)
    if not edges_match:
        print("Error: SKELETON_EDGES not found in source.")
        return

    edges_code = edges_match.group(0)

    # Create new content
    new_content = f'''"""
Hybrid GCN V2 Model Architecture
Optimized for CPU inference with node-specific features + global context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

{class_names_code}

{edges_code}

{class_code}
'''
    with open(dest_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("model_architecture.py updated.")

# 2. Update src/feature_extraction.py
def update_feature_extraction():
    print("Updating feature_extraction.py...")
    # For now, we'll manually check this or copy if we had a clean source. 
    # Since 2b_generate_node_hybrid_features.py has the logic but is a generation script,
    # we should check if we need to port changes.
    # The current feature_extraction.py seems to have the correct logic structure, 
    # but let's make sure it matches the latest "extract_node_features" from 2b.
    
    source_file = HYBRID_DIR / "2b_generate_node_hybrid_features.py"
    dest_file = SRC_DIR / "feature_extraction.py"
    
    with open(source_file, 'r', encoding='utf-8') as f:
        source_content = f.read()
        
    with open(dest_file, 'r', encoding='utf-8') as f:
        dest_content = f.read()

    # Extract extract_node_features from source
    func_match = re.search(r'def extract_node_features\(pose_keypoints, stick_keypoints\):.*?(?=\n\n)', source_content, re.DOTALL)
    if func_match:
        new_func = func_match.group(0)
        # Replace in dest
        dest_content = re.sub(r'def extract_node_features\(pose_keypoints, stick_keypoints\):.*?(?=\n\n)', new_func, dest_content, flags=re.DOTALL)
        
        with open(dest_file, 'w', encoding='utf-8') as f:
            f.write(dest_content)
        print("feature_extraction.py updated with latest node extraction logic.")
    else:
        print("Warning: Could not extract node feature logic from source.")

# 3. Update feature_templates.json
def update_templates():
    print("Updating feature_templates.json...")
    source = HYBRID_DIR / "feature_templates.json"
    dest = SRC_DIR / "feature_templates.json"
    shutil.copy(source, dest)
    print("feature_templates.json updated.")

# 4. Update Models
def update_models():
    print("Updating models...")
    models_to_copy = [
        "hybrid_gcn_v2_front.pth",
        "hybrid_gcn_v2_left.pth",
        "hybrid_gcn_v2_right.pth"
    ]
    
    for model_name in models_to_copy:
        source = HYBRID_DIR / "models" / model_name
        dest = MODELS_DIR / model_name
        if source.exists():
            shutil.copy(source, dest)
            print(f"Copied {model_name}")
        else:
            print(f"Warning: {model_name} not found in source!")

# 5. Update MANIFEST.md
def update_manifest():
    print("Updating MANIFEST.md...")
    manifest_path = DEPLOY_DIR / "MANIFEST.md"
    
    # Get model sizes
    sizes = []
    for model_name in ["hybrid_gcn_v2_front.pth", "hybrid_gcn_v2_left.pth", "hybrid_gcn_v2_right.pth"]:
        p = MODELS_DIR / model_name
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            sizes.append(f"- models/{model_name} ({size_mb:.2f} MB)")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update Class Count in description if present (it was 13, now 12)
    # Actually, let's just update the Version History to reflect the update
    
    new_version_entry = """
### v1.1 (2026-02-11)
- Updated Class List (Removed 'neutral_stance', 13 -> 12 classes)
- Syncing latest Hybrid GCN V2 models
- Updated feature extraction logic
"""
    if "### v1.1" not in content:
         content += new_version_entry
         
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("MANIFEST.md updated.")

if __name__ == "__main__":
    update_model_architecture()
    update_feature_extraction()
    update_templates()
    update_models()
    update_manifest()
    print("\nPackage update complete!")
