"""
Horizontal Flip Augmentation with Label Swapping

This script creates flipped versions of images AND correctly swaps left↔right labels.
This is critical for handling subjects facing different directions.

Example:
  - left_chest_thrust_correct → flip image → right_chest_thrust_correct
  - right_elbow_block_correct → flip image → left_elbow_block_correct
  - neutral_stance → flip image → neutral_stance (no change)
"""
import os
import cv2
from tqdm import tqdm
import shutil

# Label swap mapping
LABEL_SWAP = {
    'left_chest_thrust_correct': 'right_chest_thrust_correct',
    'right_chest_thrust_correct': 'left_chest_thrust_correct',
    'left_elbow_block_correct': 'right_elbow_block_correct',
    'right_elbow_block_correct': 'left_elbow_block_correct',
    'left_eye_thrust_correct': 'right_eye_thrust_correct',
    'right_eye_thrust_correct': 'left_eye_thrust_correct',
    'left_knee_block_correct': 'right_knee_block_correct',
    'right_knee_block_correct': 'left_knee_block_correct',
    'left_temple_block_correct': 'right_temple_block_correct',
    'right_temple_block_correct': 'left_temple_block_correct',
    # These don't change
    'neutral_stance': 'neutral_stance',
    'crown_thrust_correct': 'crown_thrust_correct',
    'solar_plexus_thrust_correct': 'solar_plexus_thrust_correct',
}


def flip_augment_with_label_swap(input_folder, output_folder):
    """
    Create horizontally flipped versions of all images with correct label swapping.
    
    Args:
        input_folder: Source dataset (e.g., dataset_split/train)
        output_folder: Where to save flipped images (e.g., dataset_split/train_flipped)
    """
    print("\n" + "="*60)
    print("   HORIZONTAL FLIP AUGMENTATION WITH LABEL SWAPPING")
    print("="*60)
    print(f"   Input:  {input_folder}")
    print(f"   Output: {output_folder}")
    print("="*60)
    
    # Get all class folders
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input folder not found: {input_folder}")
        return
    
    class_folders = [d for d in os.listdir(input_folder) 
                     if os.path.isdir(os.path.join(input_folder, d))]
    
    print(f"\n[INFO] Found {len(class_folders)} classes")
    
    total_original = 0
    total_flipped = 0
    
    for class_name in tqdm(class_folders, desc="Processing classes"):
        src_class_dir = os.path.join(input_folder, class_name)
        
        # Determine target class (swapped label)
        target_class = LABEL_SWAP.get(class_name, class_name)
        dst_class_dir = os.path.join(output_folder, target_class)
        
        # Also copy originals to their original class folder
        orig_dst_dir = os.path.join(output_folder, class_name)
        
        os.makedirs(dst_class_dir, exist_ok=True)
        os.makedirs(orig_dst_dir, exist_ok=True)
        
        # Get all images in this class
        images = [f for f in os.listdir(src_class_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in images:
            src_path = os.path.join(src_class_dir, img_name)
            
            # 1. Copy original to original class
            orig_dst_path = os.path.join(orig_dst_dir, img_name)
            if not os.path.exists(orig_dst_path):
                shutil.copy2(src_path, orig_dst_path)
                total_original += 1
            
            # 2. Create flipped version in swapped class
            img = cv2.imread(src_path)
            if img is None:
                continue
            
            flipped = cv2.flip(img, 1)  # 1 = horizontal flip
            
            # Save with _flip suffix to avoid name collision
            base, ext = os.path.splitext(img_name)
            flip_name = f"{base}_flip{ext}"
            flip_dst_path = os.path.join(dst_class_dir, flip_name)
            
            cv2.imwrite(flip_dst_path, flipped)
            total_flipped += 1
    
    print("\n" + "="*60)
    print("   FLIP AUGMENTATION COMPLETE")
    print("="*60)
    print(f"   Original images copied: {total_original}")
    print(f"   Flipped images created: {total_flipped}")
    print(f"   Total images: {total_original + total_flipped}")
    print(f"   Output: {output_folder}")
    print("="*60)
    print("\n[NEXT] Run feature extraction on the new folder:")
    print(f"       python training/run_extraction.py --train-dir {output_folder}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Flip augmentation with label swapping')
    parser.add_argument('--input', '-i', default='dataset_split/train_aug', 
                        help='Input training folder (default: train_aug with existing augmentations)')
    parser.add_argument('--output', '-o', default='dataset_split/train_flip_aug',
                        help='Output folder for augmented data')
    
    args = parser.parse_args()
    
    flip_augment_with_label_swap(args.input, args.output)
