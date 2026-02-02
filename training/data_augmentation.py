import os
import cv2
import albumentations as A
import time
import numpy as np
from tqdm import tqdm

# configuration
INPUT_DATASET_FOLDER = "dataset"
OUTPUT_DATASET_FOLDER = "dataset_aug"
IMAGES_PER_ORIGINAL = 15  

def get_pose_augmentation_pipeline():
    """
    pose-specific augmentation pipeline
    designed to maintain body structure while adding variety
    """
    return A.Compose([
        # spatial transforms (preserve pose structure)
        A.HorizontalFlip(p=0.5),
        
        A.OneOf([
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.08, 0.08),
                rotate=(-12, 12),
                shear=(-5, 5),
                keep_ratio=True,
                p=1.0
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=10,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0
            ),
        ], p=0.9),
        
        # color/lighting transforms (simulate different environments)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=25,
                p=1.0
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=1.0
            ),
        ], p=0.8),
        
        # simulate different camera conditions
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 40.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.ImageCompression(quality_lower=70, quality_upper=95, p=1.0),
        ], p=0.5),
        
        # simulate different lighting
        A.OneOf([
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=4,
                p=1.0
            ),
            A.RandomToneCurve(scale=0.1, p=1.0),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.3),
        
        # optional grayscale (some may train in different lighting)
        A.ToGray(p=0.1),
    ])

def get_conservative_augmentation():
    """
    lighter augmentation for more reliable landmark extraction
    use this if aggressive augmentation causes mediapipe to fail
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=8,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.6
        ),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
    ])

def augment_and_save_images(input_folder, output_folder, num_variations, aggressive=True):
    """
    augment images for pose training
    
    args:
        aggressive: True for full augmentation, False for conservative
    """
    transform = get_pose_augmentation_pipeline() if aggressive else get_conservative_augmentation()
    
    print("\n" + "="*50)
    print("  DATA AUGMENTATION FOR POSE TRAINING")
    print("="*50)
    print(f"  Input:  {input_folder}")
    print(f"  Output: {output_folder}")
    print(f"  Variations per image: {num_variations}")
    print(f"  Mode: {'Aggressive' if aggressive else 'Conservative'}")
    print("="*50)
    
    start_time = time.time()
    total_generated_count = 0
    failed_count = 0

    all_image_paths = [
        os.path.join(dp, f) 
        for dp, dn, fn in os.walk(input_folder) 
        for f in fn if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    print(f"\n[INFO] Found {len(all_image_paths)} source images")
    print(f"[INFO] Will generate {len(all_image_paths) * num_variations} augmented images\n")

    for image_path in tqdm(all_image_paths, desc="Augmenting"):
        try:
            relative_path = os.path.relpath(os.path.dirname(image_path), input_folder)
            filename = os.path.basename(image_path)
            output_dir_path = os.path.join(output_folder, relative_path)
            os.makedirs(output_dir_path, exist_ok=True)

            image = cv2.imread(image_path)
            if image is None:
                print(f"[WARN] Could not read: {filename}")
                failed_count += 1
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            base_filename, file_extension = os.path.splitext(filename)

            # save original too
            original_save_path = os.path.join(output_dir_path, filename)
            if not os.path.exists(original_save_path):
                cv2.imwrite(original_save_path, image)
                total_generated_count += 1

            # generate augmented versions
            for i in range(num_variations):
                augmented = transform(image=image_rgb)
                augmented_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)

                new_filename = f"{base_filename}_aug{i+1:02d}{file_extension}"
                save_path = os.path.join(output_dir_path, new_filename)
                cv2.imwrite(save_path, augmented_bgr)
                total_generated_count += 1

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            failed_count += 1

    end_time = time.time()
    
    print("\n" + "="*50)
    print("  AUGMENTATION COMPLETE")
    print("="*50)
    print(f"  Images generated: {total_generated_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Time: {end_time - start_time:.1f} seconds")
    print(f"  Output: {output_folder}")
    print("="*50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Augment pose dataset')
    parser.add_argument('--input', '-i', default=INPUT_DATASET_FOLDER, help='Input folder')
    parser.add_argument('--output', '-o', default=OUTPUT_DATASET_FOLDER, help='Output folder')
    parser.add_argument('--num', '-n', type=int, default=IMAGES_PER_ORIGINAL, help='Variations per image')
    parser.add_argument('--conservative', '-c', action='store_true', help='Use lighter augmentation')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        print(f"[ERROR] Input folder not found: {args.input}")
        print("Make sure the folder exists.")
    else:
        augment_and_save_images(
            input_folder=args.input,
            output_folder=args.output,
            num_variations=args.num,
            aggressive=not args.conservative
        )