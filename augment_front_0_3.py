"""
Step 5.7.2: Augment Front View Classes 0-3 (Conservative 2x)

Target: Moderately increase classes 0-3 from ~40 to ~120 samples (3x total with originals)
Method: 2x augmentation (1 original + 2 augmented = 3 total per original image)

Conservative approach: Less aggressive augmentation to avoid introducing noise
Expected results:
- crown_thrust_correct (0): 42 → ~126 samples
- left_chest_thrust_correct (1): 37 → ~111 samples  
- left_elbow_block_correct (2): 39 → ~117 samples
- left_eye_thrust_correct (3): 41 → ~123 samples
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A

# Configuration
INPUT_DIR = Path("dataset_split/train/front")
TARGET_CLASSES = [
    'crown_thrust_correct',        # Class 0
    'left_chest_thrust_correct',   # Class 1
    'left_elbow_block_correct',    # Class 2
    'left_eye_thrust_correct'      # Class 3
]

AUGMENT_FACTOR = 2  # 2 augmented copies per original image (conservative)

def get_augmentation_pipeline(seed=42, always_flip=False):
    """Define CONSERVATIVE augmentation pipeline - less aggressive to avoid noise"""
    np.random.seed(seed)
    
    transforms = [
        # 1. Rotation (±8°) - Conservative, small angle changes
        A.Rotate(limit=8, p=0.6),
        
        # 2. Scale/Zoom (0.9-1.1x) - Minimal scaling to preserve pose
        A.RandomScale(scale_limit=0.1, p=0.5),
        
        # 3. Slight Translation (±5% of image) - Minimal movement
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0, rotate_limit=0, p=0.4),
        
        # 4. Minimal Perspective - Very slight camera angle change
        A.Perspective(scale=(0.02, 0.04), p=0.3),
        
        # 5. Subtle Brightness/Contrast - Avoid dramatic lighting changes
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
        
        # 6. Horizontal Flip (50% of augmented images) - Essential for front view
        A.HorizontalFlip(p=0.5 if not always_flip else 1.0),
        
        # NOTE: Removed blur - can obscure stick detection which is already difficult
    ]
    
    return A.Compose(transforms)

def augment_class(class_name):
    """Augment a single class"""
    class_dir = INPUT_DIR / class_name
    
    if not class_dir.exists():
        print(f"[ERROR] Directory not found: {class_dir}")
        return 0, 0
    
    # Get only original images (no _aug in filename)
    images = [f for f in class_dir.glob("*.jpg") if '_aug' not in f.name]
    
    if len(images) == 0:
        print(f"[WARN] No original images found for {class_name}")
        return 0, 0
    
    original_count = len(images)
    target_count = original_count * (AUGMENT_FACTOR + 1)  # +1 for originals
    
    print(f"\n[CLASS] {class_name}")
    print(f"   Original: {original_count} images")
    print(f"   Augmenting: {AUGMENT_FACTOR}x -> Target: {target_count} total")
    
    augmented_count = 0
    
    for img_path in tqdm(images, desc=f"   Augmenting", leave=False):
        try:
            # Read image
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Generate AUGMENT_FACTOR augmented versions
            for aug_idx in range(AUGMENT_FACTOR):
                # Create augmentor with different seed for variety
                augmentor = get_augmentation_pipeline(seed=aug_idx * 1000 + hash(str(img_path)) % 1000)
                
                # Apply augmentation
                augmented = augmentor(image=image)['image']
                
                # Save augmented image
                aug_img = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                output_filename = f"{img_path.stem}_aug{aug_idx+1}{img_path.suffix}"
                output_path = class_dir / output_filename
                
                cv2.imwrite(str(output_path), aug_img)
                augmented_count += 1
                
        except Exception as e:
            print(f"   [WARN] Error processing {img_path.name}: {e}")
    
    print(f"   [OK] Created: {augmented_count} augmented images")
    print(f"   [INFO] New total: {original_count + augmented_count} images")
    
    return original_count, augmented_count

def main():
    print("="*60)
    print("FRONT VIEW CLASSES 0-3 AUGMENTATION")
    print("="*60)
    print(f"Target: {AUGMENT_FACTOR}x augmentation per original image")
    print(f"Classes: {len(TARGET_CLASSES)}")
    print(f"Input directory: {INPUT_DIR}")
    print("="*60)
    
    total_original = 0
    total_augmented = 0
    
    for class_name in TARGET_CLASSES:
        orig, aug = augment_class(class_name)
        total_original += orig
        total_augmented += aug
    
    print("\n" + "="*60)
    print("AUGMENTATION COMPLETE")
    print("="*60)
    print(f"Total original images: {total_original}")
    print(f"Total augmented images created: {total_augmented}")
    print(f"New total samples: {total_original + total_augmented}")
    print(f"Augmentation factor: {total_augmented / total_original:.1f}x")
    print("="*60)
    
    # Verify results
    print("\n[VERIFICATION]")
    for class_name in TARGET_CLASSES:
        class_dir = INPUT_DIR / class_name
        if class_dir.exists():
            all_images = list(class_dir.glob("*.jpg"))
            originals = len([f for f in all_images if '_aug' not in f.name])
            augmented = len([f for f in all_images if '_aug' in f.name])
            print(f"  {class_name:35s}: {originals:3d} orig + {augmented:3d} aug = {originals+augmented:3d} total")

if __name__ == "__main__":
    import sys
    try:
        import albumentations
    except ImportError:
        print("[ERROR] albumentations not installed.")
        print("Run: pip install albumentations")
        sys.exit(1)
    
    main()
