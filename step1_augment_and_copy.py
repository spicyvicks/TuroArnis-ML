"""
Step 1a + 1b: Augment images in front/ folder (no flipping), then copy to dataset_split/train/front/
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A
import shutil

# Paths
SOURCE_DIR = Path("front")
TARGET_DIR = Path("dataset_split/train/front")

# Class mapping
CLASS_MAP = {
    'crown': 'crown_thrust_correct',
    'left_chest': 'left_chest_thrust_correct',
    'left_elbow': 'left_elbow_block_correct',
    'left_eye': 'left_eye_thrust_correct',
    'left_knee': 'left_knee_block_correct',
    'left_temple': 'left_temple_block_correct',
    'right_chest': 'right_chest_thrust_correct',
    'right_elbow': 'right_elbow_block_correct',
}

AUGMENT_FACTOR = 2  # 2 augmented copies per original

def get_augmentation_pipeline(seed=42):
    """Conservative augmentation pipeline — NO horizontal flip"""
    np.random.seed(seed)
    transforms = [
        A.Rotate(limit=8, p=0.6),
        A.RandomScale(scale_limit=0.1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0, rotate_limit=0, p=0.4),
        A.Perspective(scale=(0.02, 0.04), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
        # NOTE: NO HorizontalFlip — removed intentionally per user request
    ]
    return A.Compose(transforms)

def process_class(src_class_name, tgt_class_name):
    src_dir = SOURCE_DIR / src_class_name
    tgt_dir = TARGET_DIR / tgt_class_name
    
    if not src_dir.exists():
        print(f"[WARN] Source not found: {src_dir}")
        return 0, 0, 0
    
    tgt_dir.mkdir(parents=True, exist_ok=True)
    
    # Get source images (jpg, png)
    images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
    if not images:
        print(f"[WARN] No images in {src_dir}")
        return 0, 0, 0
    
    orig_copied = 0
    aug_created = 0
    
    for img_path in tqdm(images, desc=f"  {src_class_name} -> {tgt_class_name}", leave=False):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[WARN] Could not read {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Copy original to target (prefix to avoid filename collision with existing)
        orig_filename = f"new_{img_path.name}"
        orig_target_path = tgt_dir / orig_filename
        shutil.copy2(str(img_path), str(orig_target_path))
        orig_copied += 1
        
        # Generate augmented versions
        for aug_idx in range(AUGMENT_FACTOR):
            augmentor = get_augmentation_pipeline(seed=aug_idx * 1000 + hash(str(img_path)) % 1000)
            augmented = augmentor(image=image)['image']
            aug_img = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            
            aug_filename = f"new_{img_path.stem}_aug{aug_idx+1}{img_path.suffix}"
            aug_target_path = tgt_dir / aug_filename
            cv2.imwrite(str(aug_target_path), aug_img)
            aug_created += 1
    
    return orig_copied, aug_created, len(images)

def main():
    print("="*60)
    print("STEP 1a + 1b: AUGMENT front/ IMAGES & COPY TO dataset_split/train/front/")
    print("="*60)
    print(f"Augmentation factor: {AUGMENT_FACTOR}x (NO horizontal flipping)")
    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {TARGET_DIR}")
    print("="*60)
    
    total_orig = 0
    total_aug = 0
    total_src = 0
    
    print("\n[PROCESSING]")
    for src_name, tgt_name in CLASS_MAP.items():
        orig, aug, src_count = process_class(src_name, tgt_name)
        total_orig += orig
        total_aug += aug
        total_src += src_count
        print(f"  {src_name:15s} -> {tgt_name:30s}: {orig:3d} orig + {aug:3d} aug = {orig+aug:3d} total (from {src_count} source)")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Source images processed: {total_src}")
    print(f"Originals copied:      {total_orig}")
    print(f"Augmented created:     {total_aug}")
    print(f"New samples added:     {total_orig + total_aug}")
    print("="*60)
    
    # Verify target counts
    print("\n[VERIFICATION] Target folder counts after copy:")
    for tgt_name in CLASS_MAP.values():
        tgt_dir = TARGET_DIR / tgt_name
        if tgt_dir.exists():
            all_imgs = list(tgt_dir.glob("*.jpg")) + list(tgt_dir.glob("*.png"))
            new_imgs = [f for f in all_imgs if f.name.startswith("new_")]
            old_imgs = [f for f in all_imgs if not f.name.startswith("new_")]
            print(f"  {tgt_name:35s}: {len(old_imgs):3d} old + {len(new_imgs):3d} new = {len(all_imgs):3d} total")

if __name__ == "__main__":
    main()
