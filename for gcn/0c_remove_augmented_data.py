"""
Remove Augmented Training Data

Removes all augmented images (files with _aug1, _aug2, etc. suffixes)
from dataset_split/train directory, keeping only original images.
"""

import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# Configuration
INPUT_DIR = Path("dataset_split/train")

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]

VIEWPOINTS = ['front', 'left', 'right']

def remove_augmented_images():
    print(f"\n{'='*60}")
    print(f"Removing Augmented Training Data")
    print(f"{'='*60}")
    
    if not INPUT_DIR.exists():
        print(f"❌ Input directory not found: {INPUT_DIR}")
        return

    total_removed = 0

    # Process each viewpoint
    for viewpoint in VIEWPOINTS:
        viewpoint_dir = INPUT_DIR / viewpoint
        if not viewpoint_dir.exists():
            continue
            
        print(f"\nProcessing {viewpoint} view...")
        
        for class_name in CLASS_NAMES:
            class_dir = viewpoint_dir / class_name
            if not class_dir.exists():
                continue
                
            # Find all augmented images (contain _aug in filename)
            augmented_images = [
                img for img in class_dir.iterdir() 
                if img.is_file() and '_aug' in img.stem and img.suffix in ['.jpg', '.png', '.jpeg']
            ]
            
            if augmented_images:
                print(f"  - {class_name}: Removing {len(augmented_images)} augmented images")
                
                for img_path in tqdm(augmented_images, desc=f"    Deleting"):
                    try:
                        img_path.unlink()
                        total_removed += 1
                    except Exception as e:
                        print(f"    Error deleting {img_path.name}: {e}")

    print(f"\n{'='*60}")
    print(f"✅ Removed {total_removed} augmented images!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import sys
    
    # Safety confirmation
    print("\n⚠️  WARNING: This will permanently delete all augmented images (_aug1, _aug2, etc.)")
    print(f"📁 Directory: {INPUT_DIR.absolute()}")
    
    response = input("\nProceed? (yes/no): ").strip().lower()
    
    if response == 'yes':
        remove_augmented_images()
    else:
        print("Operation cancelled.")
