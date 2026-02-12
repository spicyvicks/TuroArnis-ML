"""
Copy Images from Test to Train

Copies a specified number of images from test set to train set,
maintaining the directory structure (viewpoint/class).

Usage:
1. Update NUM_IMAGES_TO_COPY
2. Run script
3. Images will be copied (test set remains intact)
"""

import shutil
from pathlib import Path
from collections import defaultdict

# Configuration
DATASET_DIR = Path("dataset_split")
NUM_IMAGES_TO_COPY = 3  # Number of images to copy per class

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'neutral_stance',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]

VIEWPOINTS = ['front', 'left', 'right']


def copy_images():
    print(f"\n{'='*60}")
    print(f"Copying {NUM_IMAGES_TO_COPY} images from Test to Train")
    print(f"{'='*60}\n")
    
    total_copied = 0
    
    for viewpoint in VIEWPOINTS:
        print(f"\n{viewpoint.upper()} View:")
        print("-" * 40)
        
        for class_name in CLASS_NAMES:
            test_dir = DATASET_DIR / "test" / viewpoint / class_name
            train_dir = DATASET_DIR / "train" / viewpoint / class_name
            
            if not test_dir.exists():
                continue
            
            # Get images from test directory
            images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
            
            if not images:
                print(f"  {class_name}: No images in test set")
                continue
            
            # Determine how many to copy (don't copy more than available)
            num_to_copy = min(NUM_IMAGES_TO_COPY, len(images))
            
            if num_to_copy == 0:
                continue
            
            # Ensure train directory exists
            train_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy images
            copied_count = 0
            for img in images[:num_to_copy]:
                dest = train_dir / img.name
                shutil.copy2(str(img), str(dest))
                copied_count += 1
            
            total_copied += copied_count
            print(f"  {class_name}: Copied {copied_count} images")
    
    print(f"\n{'='*60}")
    print(f"✅ Total images copied: {total_copied}")
    print(f"{'='*60}\n")


def main():
    if not DATASET_DIR.exists():
        print(f"❌ Error: Dataset directory not found: {DATASET_DIR}")
        return
    
    test_dir = DATASET_DIR / "test"
    if not test_dir.exists():
        print(f"❌ Error: Test directory not found: {test_dir}")
        return
    
    copy_images()
    
    print("Run 0_verify_split.py to check the new ratios.")


if __name__ == "__main__":
    main()
