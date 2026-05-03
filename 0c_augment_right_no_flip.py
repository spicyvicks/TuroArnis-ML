"""
Step 0c-right: Re-Augment Right Viewpoint Training Data (NO Horizontal Flip)

Generates 3 augmented copies per original image with NO horizontal flipping.
Conservative augmentation for right viewpoint to avoid left/right label confusion.

Data distribution per original image:
- 1 original (non-flipped)
- 3 augmented (non-flipped, aug1-3)
= 4 total files per original

Input: dataset_split/train/right
Output: Augmented images saved in-place with _aug1, _aug2, _aug3 suffixes
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A

# Configuration
INPUT_DIR = Path("dataset_split/train/right")
AUGMENT_FACTOR = 3  # 3 augmented copies per image (4 total files)

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct',
    'neutral'
]


def get_augmentation_pipeline(seed=42):
    """Conservative augmentation pipeline - NO horizontal flip for right viewpoint"""
    transforms = [
        # 1. Rotation (+-10deg) - Standard in pose estimation
        A.Rotate(limit=10, p=0.7),

        # 2. Scale/Zoom (0.8-1.2x) - Simulates distance variation
        A.RandomScale(scale_limit=0.2, p=0.6),

        # 3. Perspective Transform - Simulates camera angle variation
        A.Perspective(scale=(0.02, 0.05), p=0.3),

        # 4. Brightness/Contrast variation
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),

        # NO HorizontalFlip - right viewpoint already suffers from left/right ambiguity
    ]

    return A.Compose(transforms)


def augment_right_dataset():
    print(f"\n{'='*60}")
    print(f"Augmenting Right Viewpoint Training Data (x{AUGMENT_FACTOR}, NO FLIP)")
    print(f"{'='*60}")

    if not INPUT_DIR.exists():
        print(f"X Input directory not found: {INPUT_DIR}")
        return

    total_original = 0
    total_augmented = 0

    for class_name in CLASS_NAMES:
        class_dir = INPUT_DIR / class_name
        if not class_dir.exists():
            continue

        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        # Filter out previously augmented images (only process originals)
        original_images = [img for img in images if '_aug' not in img.stem]
        total_original += len(original_images)

        print(f"  - {class_name}: {len(original_images)} original -> {len(original_images) * (AUGMENT_FACTOR + 1)} total")

        for img_path in tqdm(original_images, desc=f"    Augmenting", leave=False):
            try:
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                for i in range(AUGMENT_FACTOR):
                    augmentor = get_augmentation_pipeline(seed=i * 1000 + hash(str(img_path)) % 1000)
                    augmented = augmentor(image=image)['image']

                    # Save augmented image
                    aug_img = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                    output_filename = f"{img_path.stem}_aug{i+1}{img_path.suffix}"
                    output_path = class_dir / output_filename

                    cv2.imwrite(str(output_path), aug_img)
                    total_augmented += 1

            except Exception as e:
                print(f"    Error processing {img_path.name}: {e}")

    print(f"\n{'='*60}")
    print("OK Augmentation Complete!")
    print(f"Data Distribution per original image:")
    print(f"  - 1 original (non-flipped)")
    print(f"  - {AUGMENT_FACTOR} augmented (non-flipped, aug1-{AUGMENT_FACTOR})")
    print(f"  Total original images: {total_original}")
    print(f"  Total augmented images created: {total_augmented}")
    print(f"  New total samples: {total_original + total_augmented}")
    print(f"  Horizontal flip ratio: 0% (intentionally disabled for right viewpoint)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys

    # Check if albumentations is installed
    try:
        import albumentations
    except ImportError:
        print("X Error: albumentations not installed.")
        print("Run: pip install albumentations")
        sys.exit(1)

    augment_right_dataset()
