"""
Flip All Test Images

Horizontally flips all test images to match the app's mirrored camera view.
Since the application ALWAYS sends mirrored frames to the model,
test images should also be mirrored to get accurate real-world performance metrics.

Input: dataset_split/test (original images)
Output: dataset_split/test (flipped images, replaces originals)
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
TEST_DIR = Path("dataset_split/test")

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]

VIEWPOINTS = ['front', 'left', 'right']

def flip_test_images():
    print(f"\n{'='*60}")
    print(f"Flipping Test Images (Horizontal Flip)")
    print(f"{'='*60}")
    
    if not TEST_DIR.exists():
        print(f"❌ Test directory not found: {TEST_DIR}")
        return

    total_flipped = 0

    # Process each viewpoint
    for viewpoint in VIEWPOINTS:
        viewpoint_dir = TEST_DIR / viewpoint
        if not viewpoint_dir.exists():
            continue
            
        print(f"\nProcessing {viewpoint} view...")
        
        for class_name in CLASS_NAMES:
            class_dir = viewpoint_dir / class_name
            if not class_dir.exists():
                continue
                
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            if images:
                print(f"  - {class_name}: Flipping {len(images)} images")
                
                for img_path in tqdm(images, desc=f"    Flipping"):
                    try:
                        # Read image
                        image = cv2.imread(str(img_path))
                        
                        # Flip horizontally
                        flipped = cv2.flip(image, 1)
                        
                        # Overwrite original with flipped version
                        cv2.imwrite(str(img_path), flipped)
                        total_flipped += 1
                        
                    except Exception as e:
                        print(f"    Error flipping {img_path.name}: {e}")

    print(f"\n{'='*60}")
    print(f"✅ Flipped {total_flipped} test images!")
    print(f"Test images now match app's mirrored camera view.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import sys
    
    # Safety confirmation
    print("\n⚠️  WARNING: This will OVERWRITE all test images with horizontally flipped versions")
    print(f"📁 Directory: {TEST_DIR.absolute()}")
    print("\nThis is recommended because the app ALWAYS sends mirrored frames to the model.")
    print("Test accuracy will then reflect real-world deployment performance.")
    
    response = input("\nProceed? (yes/no): ").strip().lower()
    
    if response == 'yes':
        flip_test_images()
    else:
        print("Operation cancelled.")
