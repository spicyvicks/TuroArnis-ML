"""
Update CLASS_NAMES in All Files

Automatically updates CLASS_NAMES list in all Python files to remove 'neutral_stance'.

Files updated:
- 0c_augment_training_data.py
- 0c_flip_test_images.py
- 0c_remove_augmented_data.py
- 0_verify_split.py
- 0_organize_and_verify_split.py
- 0_move_test_to_train.py
- hybrid_classifier/*.py
- deployment_package/src/model_architecture.py
- inference_realtime_gcn.py
"""

import re
from pathlib import Path

# Old CLASS_NAMES (13 classes)
OLD_CLASS_NAMES = """CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'neutral_stance',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]"""

# New CLASS_NAMES (12 classes, no neutral_stance)
NEW_CLASS_NAMES = """CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]"""

# Files to update
FILES_TO_UPDATE = [
    "0c_augment_training_data.py",
    "0c_flip_test_images.py",
    "0c_remove_augmented_data.py",
    "0_verify_split.py",
    "0_organize_and_verify_split.py",
    "0_move_test_to_train.py",
    "inference_realtime_gcn.py",
    "hybrid_classifier/1_extract_reference_features.py",
    "hybrid_classifier/2_generate_hybrid_features.py",
    "hybrid_classifier/2b_generate_node_hybrid_features.py",
    "hybrid_classifier/2c_extract_test_features.py",
    "hybrid_classifier/3_train_classifier.py",
    "hybrid_classifier/4c_train_hybrid_gcn_v2.py",
    "hybrid_classifier/5_analyze_hybrid_gcn.py",
    "hybrid_classifier/auto_select_references.py",
    "deployment_package/src/model_architecture.py",
]

def update_class_names():
    print(f"\n{'='*60}")
    print(f"Updating CLASS_NAMES in All Files")
    print(f"{'='*60}\n")
    
    updated_files = []
    skipped_files = []
    
    for file_path in FILES_TO_UPDATE:
        path = Path(file_path)
        
        if not path.exists():
            skipped_files.append(file_path)
            print(f"⚠️  Skipped (not found): {file_path}")
            continue
        
        # Read file content
        content = path.read_text(encoding='utf-8')
        
        # Check if old CLASS_NAMES exists
        if "'neutral_stance'," in content:
            # Replace old CLASS_NAMES with new one (removing neutral_stance)
            # Use regex to find the CLASS_NAMES block
            pattern = r"CLASS_NAMES = \[\s*'crown_thrust_correct'[^\]]+\]"
            
            if re.search(pattern, content, re.DOTALL):
                updated_content = re.sub(pattern, NEW_CLASS_NAMES, content, flags=re.DOTALL)
                
                # Write back
                path.write_text(updated_content, encoding='utf-8')
                updated_files.append(file_path)
                print(f"✓ Updated: {file_path}")
            else:
                skipped_files.append(file_path)
                print(f"⚠️  Skipped (couldn't match pattern): {file_path}")
        else:
            skipped_files.append(file_path)
            print(f"⚠️  Skipped (already updated or no CLASS_NAMES): {file_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ Updated {len(updated_files)} files")
    if skipped_files:
        print(f"⚠️  Skipped {len(skipped_files)} files")
    print(f"{'='*60}\n")
    
    if updated_files:
        print("CLASS_NAMES updated from 13 → 12 classes")
        print("Removed: 'neutral_stance'")
        print("\nNext steps:")
        print("1. Run: python hybrid_classifier/1_extract_reference_features.py")
        print("2. Run: python hybrid_classifier/2b_generate_node_hybrid_features.py")
        print("3. Run: python hybrid_classifier/4c_train_hybrid_gcn_v2.py --epochs 150")

if __name__ == "__main__":
    import sys
    
    print("\nℹ️  This will update CLASS_NAMES in all Python files")
    print("   Removes: 'neutral_stance'")
    print("   Result: 13 classes → 12 classes")
    
    response = input("\nProceed? (yes/no): ").strip().lower()
    
    if response == 'yes':
        update_class_names()
    else:
        print("Operation cancelled.")
