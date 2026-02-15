"""
Step 0c: Augment Training Data (Research-Backed Methods)

Generates augmented copies of training images using research-backed augmentation
strategies proven effective for pose estimation tasks:

1. Rotation (±10°) - Standard in pose estimation literature
2. Scale/Zoom (0.8-1.2x) - Simulates distance variation
3. Perspective Transform - Simulates camera angle variation
4. Horizontal Flip - Applied to half of augmented images

Data distribution per original image:
- 1 original (non-flipped)
- 1 augmented (non-flipped, aug1)
- 2 augmented (ALWAYS flipped, aug2-3)
= 50% flipped data (2/4)

Input: dataset_split/train
Output: Augmented images saved in-place with _aug1, _aug2, _aug3 suffixes
"""

import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import albumentations as A

# Configuration
INPUT_DIR = Path("dataset_split/train")
OUTPUT_DIR = Path("dataset_split/train")  # Augment IN-PLACE (or change to dataset_augmented)
AUGMENT_FACTOR = 3  # Number of augmented copies per image (1 non-flipped + 2 flipped)

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]

VIEWPOINTS = ['front', 'left', 'right']

def get_augmentation_pipeline(viewpoint, always_flip=False):
    """Define augmentation pipeline based on research-backed methods for pose estimation"""
    transforms = [
        # 1. Rotation (±10°) - Standard in pose estimation
        A.Rotate(limit=10, p=0.7),
        
        # 2. Scale/Zoom (0.8-1.2x) - Simulates distance variation
        A.RandomScale(scale_limit=0.2, p=0.6),
        
        # 3. Perspective Transform - Simulates camera angle variation
        A.Perspective(scale=(0.02, 0.05), p=0.3),
        
        # 4. Horizontal Flip - Makes models mirror-invariant (solves camera mirroring issue)
        # ALWAYS flip augmented images to match app's mirrored camera view
        # The stick is always held in the right hand, labels stay the same
        A.HorizontalFlip(p=1.0 if always_flip else 0.0),
    ]
    
    return A.Compose(transforms)

def augment_dataset():
    print(f"\n{'='*60}")
    print(f"Augmenting Training Data (x{AUGMENT_FACTOR})")
    print(f"{'='*60}")
    
    if not INPUT_DIR.exists():
        print(f"❌ Input directory not found: {INPUT_DIR}")
        return

    # Process each viewpoint
    for viewpoint in VIEWPOINTS:
        viewpoint_dir = INPUT_DIR / viewpoint
        if not viewpoint_dir.exists():
            continue
            
        print(f"\nProcessing {viewpoint} view...")
        # Create both augmentation pipelines
        augmentor_no_flip = get_augmentation_pipeline(viewpoint, always_flip=False)
        augmentor_with_flip = get_augmentation_pipeline(viewpoint, always_flip=True)
        
        for class_name in CLASS_NAMES:
            class_dir = viewpoint_dir / class_name
            if not class_dir.exists():
                continue
                
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            # Filter out previously augmented images (only process originals)
            original_images = [img for img in images if '_aug' not in img.stem]
            print(f"  - {class_name}: {len(original_images)} original → {len(original_images) * (AUGMENT_FACTOR + 1)} total (1 original + 1 aug no-flip + 2 aug flipped)")
            
            for img_path in tqdm(original_images, desc=f"    Augmenting"):
                try:
                    image = cv2.imread(str(img_path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    for i in range(AUGMENT_FACTOR):
                        # aug1 (i=0): No flip, just other augmentations
                        # aug2-4 (i=1,2,3): Always flip + other augmentations
                        if i == 0:
                            augmented = augmentor_no_flip(image=image)['image']
                        else:
                            augmented = augmentor_with_flip(image=image)['image']
                        
                        # Save augmented image
                        aug_img = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                        output_filename = f"{img_path.stem}_aug{i+1}{img_path.suffix}"
                        output_path = class_dir / output_filename
                        
                        cv2.imwrite(str(output_path), aug_img)
                        
                except Exception as e:
                    print(f"    Error processing {img_path.name}: {e}")

    print(f"\n{'='*60}")
    print("✅ Augmentation Complete!")
    print(f"Data Distribution per original image:")
    print(f"  - 1 original (non-flipped)")
    print(f"  - 1 augmented (non-flipped, aug1)")
    print(f"  - 2 augmented (flipped, aug2-3)")
    print(f"Flip Ratio: 50% flipped data (2/4)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Check if albumentations is installed
    try:
        import albumentations
        augment_dataset()
    except ImportError:
        print("❌ Error: albumentations not installed.")
        print("Run: pip install albumentations")
