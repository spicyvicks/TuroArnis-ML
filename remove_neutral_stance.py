"""
Remove Neutral Stance Class

Deletes all neutral_stance folders and data from the dataset.
This includes:
- dataset_split/train/*/neutral_stance/
- dataset_split/test/*/neutral_stance/
- reference_poses/*/neutral_stance/

After running this, you'll need to:
1. Update CLASS_NAMES in all Python files (use update_class_names.py)
2. Regenerate feature templates
3. Regenerate features
4. Retrain models
"""

import shutil
from pathlib import Path

# Directories to clean
DATASET_SPLIT = Path("dataset_split")
REFERENCE_POSES = Path("reference_poses")

VIEWPOINTS = ['front', 'left', 'right']

def remove_neutral_stance():
    print(f"\n{'='*60}")
    print(f"Removing Neutral Stance Class")
    print(f"{'='*60}\n")
    
    removed_folders = []
    removed_images = 0
    
    # Remove from dataset_split (train and test)
    for split in ['train', 'test']:
        split_dir = DATASET_SPLIT / split
        if not split_dir.exists():
            continue
            
        print(f"Processing {split} data...")
        
        for viewpoint in VIEWPOINTS:
            viewpoint_dir = split_dir / viewpoint / "neutral_stance"
            
            if viewpoint_dir.exists():
                # Count images before deletion
                images = list(viewpoint_dir.glob("*.jpg")) + list(viewpoint_dir.glob("*.png"))
                removed_images += len(images)
                
                # Remove folder
                shutil.rmtree(viewpoint_dir)
                removed_folders.append(str(viewpoint_dir))
                print(f"  ✓ Removed {viewpoint_dir} ({len(images)} images)")
    
    # Remove from reference_poses
    ref_dir = REFERENCE_POSES
    if ref_dir.exists():
        print(f"\nProcessing reference poses...")
        
        for viewpoint in VIEWPOINTS:
            viewpoint_dir = ref_dir / viewpoint / "neutral_stance"
            
            if viewpoint_dir.exists():
                images = list(viewpoint_dir.glob("*.jpg")) + list(viewpoint_dir.glob("*.png"))
                removed_images += len(images)
                
                shutil.rmtree(viewpoint_dir)
                removed_folders.append(str(viewpoint_dir))
                print(f"  ✓ Removed {viewpoint_dir} ({len(images)} images)")
    
    print(f"\n{'='*60}")
    print(f"✅ Removed {len(removed_folders)} folders, {removed_images} images")
    print(f"{'='*60}\n")
    
    print("Next steps:")
    print("1. Run: python update_class_names.py")
    print("2. Run: python hybrid_classifier/1_extract_reference_features.py")
    print("3. Run: python hybrid_classifier/2b_generate_node_hybrid_features.py")
    print("4. Run: python hybrid_classifier/4c_train_hybrid_gcn_v2.py --epochs 150")

if __name__ == "__main__":
    import sys
    
    print("\n⚠️  WARNING: This will permanently delete all neutral_stance data")
    print(f"📁 Directories to be removed:")
    print("   - dataset_split/train/*/neutral_stance/")
    print("   - dataset_split/test/*/neutral_stance/")
    print("   - reference_poses/*/neutral_stance/")
    
    response = input("\nProceed? (yes/no): ").strip().lower()
    
    if response == 'yes':
        remove_neutral_stance()
    else:
        print("Operation cancelled.")
