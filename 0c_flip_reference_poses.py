"""
Flip All Reference Pose Images

Horizontally flips all reference pose images to match the app's mirrored camera view.
Since the application ALWAYS sends mirrored frames to the model, reference poses
should also be mirrored so that feature templates accurately represent deployment reality.

This ensures hybrid similarity features are computed correctly.

Input: reference_poses (original images)
Output: reference_poses (flipped images, replaces originals)
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
REFERENCE_DIR = Path("reference_poses")

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]

VIEWPOINTS = ['front', 'left', 'right']

def flip_reference_images():
    print(f"\n{'='*60}")
    print(f"Flipping Reference Pose Images (Horizontal Flip)")
    print(f"{'='*60}")
    
    if not REFERENCE_DIR.exists():
        print(f"❌ Reference directory not found: {REFERENCE_DIR}")
        return

    total_flipped = 0

    # Process each viewpoint
    for viewpoint in VIEWPOINTS:
        viewpoint_dir = REFERENCE_DIR / viewpoint
        if not viewpoint_dir.exists():
            continue
            
        print(f"\nProcessing {viewpoint} view...")
        
        for class_name in CLASS_NAMES:
            class_dir = viewpoint_dir / class_name
            if not class_dir.exists():
                continue
                
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            if images:
                print(f"  - {class_name}: Flipping {len(images)} reference images")
                
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
    print(f"✅ Flipped {total_flipped} reference images!")
    print(f"Reference poses now match app's mirrored camera view.")
    print(f"{'='*60}\n")
    
    print("Next steps:")
    print("1. Run: python hybrid_classifier/1_extract_reference_features.py")
    print("2. Run: python hybrid_classifier/2b_generate_node_hybrid_features.py")
    print("3. Run: python hybrid_classifier/4c_train_hybrid_gcn_v2.py --epochs 150")

if __name__ == "__main__":
    import sys
    
    # Safety confirmation
    print("\n⚠️  WARNING: This will OVERWRITE all reference pose images with horizontally flipped versions")
    print(f"📁 Directory: {REFERENCE_DIR.absolute()}")
    print("\nThis is recommended because:")
    print("  - App ALWAYS sends mirrored frames to the model")
    print("  - Feature templates should represent actual deployment orientation")
    print("  - Hybrid similarity features will be more accurate")
    
    response = input("\nProceed? (yes/no): ").strip().lower()
    
    if response == 'yes':
        flip_reference_images()
    else:
        print("Operation cancelled.")
