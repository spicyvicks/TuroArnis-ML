"""
Step 0c: Augment Training Data

Generates augmented copies of training images to increase dataset size.
Applies:
- Random horizontal flip (for front view only)
- Random rotation (±10 degrees)
- Color jitter (brightness/contrast)
- Gaussian blur (robustness)

Input: dataset_split/train
Output: dataset_augmented/train
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
    """Define augmentation pipeline based on viewpoint"""
    transforms = [
        A.Rotate(limit=10, p=0.7),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    ]
    
    # Only flip horizontally for front view (side views have distinct L/R meaning)
    if viewpoint == 'front':
        transforms.append(A.HorizontalFlip(p=0.5))
        
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
