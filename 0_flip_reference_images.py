"""
Script to flip reference images horizontally.
This creates mirrored versions of reference poses, useful for data augmentation
and creating symmetrical pose variations.
"""

import os
import cv2
from pathlib import Path
import shutil
from tqdm import tqdm


def swap_left_right_in_name(name):
    """Swap 'left' and 'right' in directory or file names."""
    if 'left' in name.lower():
        return name.replace('left', 'right').replace('Left', 'Right').replace('LEFT', 'RIGHT')
    elif 'right' in name.lower():
        return name.replace('right', 'left').replace('Right', 'Left').replace('RIGHT', 'LEFT')
    return name


def flip_reference_images(source_dir, output_dir, swap_lr_names=True, add_suffix=True):
    """
    Flip all reference images horizontally.
    
    Args:
        source_dir: Source directory containing reference images
        output_dir: Output directory for flipped images
        swap_lr_names: If True, swap 'left' and 'right' in directory names
        add_suffix: If True, add '_flipped' suffix to filenames
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Collect all image files
    image_files = []
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(Path(root) / file)
    
    print(f"Found {len(image_files)} images to flip")
    
    # Process each image
    flipped_count = 0
    error_count = 0
    
    for img_path in tqdm(image_files, desc="Flipping images"):
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read {img_path}")
                error_count += 1
                continue
            
            # Flip horizontally
            flipped_img = cv2.flip(img, 1)
            
            # Determine output path
            relative_path = img_path.relative_to(source_path)
            
            # Process directory names
            parts = list(relative_path.parts)
            if swap_lr_names:
                parts = [swap_left_right_in_name(part) for part in parts]
            
            # Process filename
            if add_suffix:
                filename = parts[-1]
                stem = Path(filename).stem
                suffix = Path(filename).suffix
                parts[-1] = f"{stem}_flipped{suffix}"
            
            # Create output path
            out_file = output_path / Path(*parts)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save flipped image
            cv2.imwrite(str(out_file), flipped_img)
            flipped_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            error_count += 1
    
    print(f"\nFlipping complete!")
    print(f"Successfully flipped: {flipped_count}")
    print(f"Errors: {error_count}")
    print(f"Output saved to: {output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Flip reference images horizontally for data augmentation"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="reference_poses",
        help="Source directory containing reference images (default: reference_poses)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reference_poses_flipped",
        help="Output directory for flipped images (default: reference_poses_flipped)"
    )
    parser.add_argument(
        "--no-swap-lr",
        action="store_true",
        help="Don't swap 'left' and 'right' in directory names"
    )
    parser.add_argument(
        "--no-suffix",
        action="store_true",
        help="Don't add '_flipped' suffix to filenames"
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Save flipped images in the same directory structure (with _flipped suffix)"
    )
    
    args = parser.parse_args()
    
    # If in-place, use source as output
    output_dir = args.source if args.in_place else args.output
    
    flip_reference_images(
        source_dir=args.source,
        output_dir=output_dir,
        swap_lr_names=not args.no_swap_lr,
        add_suffix=not args.no_suffix or not args.in_place
    )


if __name__ == "__main__":
    main()
