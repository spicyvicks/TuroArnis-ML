"""
Step 0c: Augment Training Data (Research-Backed Methods)

Generates augmented copies of training images using research-backed augmentation
strategies proven effective for pose estimation tasks:

1. Rotation (±10°) - Standard in pose estimation literature
2. Scale/Zoom (0.8-1.2x) - Simulates distance variation
3. Random Occlusion (CoarseDropout) - Improves robustness to partial visibility
4. Perspective Transform - Simulates camera angle variation
5. Horizontal Flip (front view only) - Preserves left/right semantics

Input: dataset_split/train
Output: Augmented images saved in-place with _aug1, _aug2 suffixes
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
AUGMENT_FACTOR = 2  # Number of augmented copies per image

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'neutral_stance',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]

VIEWPOINTS = ['front', 'left', 'right']

def get_augmentation_pipeline(viewpoint):
    """Define augmentation pipeline based on research-backed methods for pose estimation"""
    transforms = [
        # 1. Rotation (±10°) - Standard in pose estimation
        A.Rotate(limit=10, p=0.7),
        
        # 2. Scale/Zoom (0.8-1.2x) - Simulates distance variation
        A.RandomScale(scale_limit=0.2, p=0.6),
        
        # 3. Random Occlusion - REMOVED (Can hide critical keypoints)
        # A.CoarseDropout(
        #     max_holes=3, 
        #     max_height=50, 
        #     max_width=50, 
        #     min_holes=1,
        #     min_height=20,
        #     min_width=20,
        #     p=0.4
        # ),
        
        # 4. Perspective Transform - Simulates camera angle variation
        A.Perspective(scale=(0.02, 0.05), p=0.3),
    ]
    
    # Horizontal Flip is REMOVED for front view because it swaps the arm usage
    # (e.g. Right Knee Block uses Right Arm -> Flipped becomes Right Knee Block using Left Arm -> INVALID)
    # Side views already excluded it, but front view must also exclude it.
    
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
        augmentor = get_augmentation_pipeline(viewpoint)
        
        for class_name in CLASS_NAMES:
            class_dir = viewpoint_dir / class_name
            if not class_dir.exists():
                continue
                
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            print(f"  - {class_name}: {len(images)} original images")
            
            for img_path in tqdm(images, desc=f"    Augmenting"):
                try:
                    image = cv2.imread(str(img_path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    for i in range(AUGMENT_FACTOR):
                        # Apply augmentation
                        augmented = augmentor(image=image)['image']
                        
                        # Save augmented image
                        aug_img = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                        output_filename = f"{img_path.stem}_aug{i+1}{img_path.suffix}"
                        output_path = class_dir / output_filename
                        
                        cv2.imwrite(str(output_path), aug_img)
                        
                except Exception as e:
                    print(f"    Error processing {img_path.name}: {e}")

    print(f"\n{'='*60}")
    print("✅ Augmentation Complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Check if albumentations is installed
    try:
        import albumentations
        augment_dataset()
    except ImportError:
        print("❌ Error: albumentations not installed.")
        print("Run: pip install albumentations")
