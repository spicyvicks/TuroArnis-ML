"""
Helper: Auto-select reference images based on confidence scores
Randomly samples 5 high-confidence images per class/viewpoint
"""

import shutil
from pathlib import Path
import random

DATASET_ROOT = Path("dataset_split/train")
OUTPUT_ROOT = Path("reference_poses")
NUM_SAMPLES = 5

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]

def auto_select_references(viewpoint_filter=None):
    """Randomly sample reference images from training set"""
    
    viewpoints = [viewpoint_filter] if viewpoint_filter else ['front', 'left', 'right']
    
    for viewpoint in viewpoints:
        for class_name in CLASS_NAMES:
            source_dir = DATASET_ROOT / viewpoint / class_name
            target_dir = OUTPUT_ROOT / viewpoint / class_name
            
            if not source_dir.exists():
                print(f"Warning: {source_dir} does not exist")
                continue
            
            # Get all images
            images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
            
            if len(images) < NUM_SAMPLES:
                print(f"Warning: {source_dir} has only {len(images)} images (need {NUM_SAMPLES})")
                selected = images
            else:
                selected = random.sample(images, NUM_SAMPLES)
            
            # Copy to reference folder
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for i, img_path in enumerate(selected, 1):
                target_path = target_dir / f"{i}{img_path.suffix}"
                shutil.copy(img_path, target_path)
            
            print(f"✓ {viewpoint}/{class_name}: copied {len(selected)} images")
    
    print(f"\n✓ Auto-selection complete!")
    print(f"  Reference images saved to: {OUTPUT_ROOT}")
    print(f"\nYou can manually replace any images if needed.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewpoint', type=str, default=None,
                        choices=['front', 'left', 'right'],
                        help='Select references for specific viewpoint only (default: all)')
    args = parser.parse_args()
    
    auto_select_references(args.viewpoint)
