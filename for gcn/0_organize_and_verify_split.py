"""
Step 0b: Organize Pre-Split Data and Verify Split Ratio

Use this when you already have data separated into train/test folders.
This script will:
1. Copy your pre-split data into the correct structure
2. Verify the train/test split ratio for each class
3. Report statistics

Expected input structure (flexible):
source_train/
    ├── front/
    │   ├── crown_thrust_correct/
    │   └── ...
    └── ...

source_test/
    └── ... (same structure)

Output structure:
dataset_split/
    ├── train/
    │   ├── front/
    │   │   ├── crown_thrust_correct/
    │   │   └── ...
    │   ├── left/
    │   └── right/
    └── test/
        └── ... (same structure)
"""

import shutil
from pathlib import Path
from collections import defaultdict

# Configuration
SOURCE_TRAIN = Path("dataset_split/train")  # UPDATE THIS
SOURCE_TEST = Path("dataset_split/test")    # UPDATE THIS
OUTPUT_DIR = Path("dataset_split_final")

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]

VIEWPOINTS = ['front', 'left', 'right']


def count_images(directory):
    """Count images in a directory"""
    if not directory.exists():
        return 0
    return len(list(directory.glob("*.jpg")) + list(directory.glob("*.png")))


def organize_data(source_dir, output_split_name):
    """Copy data from source to output with correct structure"""
    print(f"\n{'='*60}")
    print(f"Organizing {output_split_name} data...")
    print(f"{'='*60}")
    
    stats = defaultdict(lambda: defaultdict(int))
    
    for viewpoint in VIEWPOINTS:
        viewpoint_src = source_dir / viewpoint
        if not viewpoint_src.exists():
            print(f"⚠️  Warning: {viewpoint_src} not found, skipping...")
            continue
        
        for class_name in CLASS_NAMES:
            class_src = viewpoint_src / class_name
            if not class_src.exists():
                continue
            
            # Create destination directory
            class_dst = OUTPUT_DIR / output_split_name / viewpoint / class_name
            class_dst.mkdir(parents=True, exist_ok=True)
            
            # Copy images
            images = list(class_src.glob("*.jpg")) + list(class_src.glob("*.png"))
            for img in images:
                shutil.copy2(img, class_dst / img.name)
            
            count = len(images)
            stats[viewpoint][class_name] = count
            print(f"  ✓ {viewpoint}/{class_name}: {count} images")
    
    return stats


def verify_split_ratio(train_stats, test_stats):
    """Verify and report train/test split ratios with actionable advice"""
    print(f"\n{'='*80}")
    print(f"{'SPLIT RATIO VERIFICATION':^80}")
    print(f"{'='*80}\n")
    
    total_train = 0
    total_test = 0
    target_ratio = 0.80
    
    # Header
    print(f"{'Viewpoint/Class':<35} {'Train':>6} {'Test':>6} {'Ratio':>8} {'Status':>4} {'Action Needed':<20}")
    print("-" * 88)
    
    for viewpoint in VIEWPOINTS:
        for class_name in CLASS_NAMES:
            train_count = train_stats[viewpoint][class_name]
            test_count = test_stats[viewpoint][class_name]
            total = train_count + test_count
            
            if total == 0:
                continue
            
            ratio = train_count / total
            total_train += train_count
            total_test += test_count
            
            # Check status
            status = "✓" if 0.75 <= ratio <= 0.85 else "⚠️"
            
            # Calculate action needed
            target_train = int(total * target_ratio)
            diff = train_count - target_train
            
            if abs(diff) <= 1:
                action = "None"
            elif diff > 0:
                action = f"Move {diff} to Test"
            else:
                action = f"Move {abs(diff)} to Train"
            
            print(f"{viewpoint}/{class_name:<30} {train_count:>6} {test_count:>6} {ratio:>8.1%} {status:>4} {action:<20}")
    
    print("-" * 88)
    overall_total = total_train + total_test
    overall_ratio = total_train / overall_total if overall_total > 0 else 0
    overall_status = "✓" if 0.75 <= overall_ratio <= 0.85 else "⚠️"
    
    print(f"{'OVERALL':<35} {total_train:>6} {total_test:>6} {overall_ratio:>8.1%} {overall_status:>4}")
    
    print(f"\n{'='*80}")
    if 0.75 <= overall_ratio <= 0.85:
        print("✅ Split ratio is within acceptable range (75-85% train)")
    else:
        print(f"⚠️  Split ratio ({overall_ratio:.1%}) is outside recommended range.")
        print("   Please follow the 'Action Needed' column to balance your dataset.")
    print(f"{'='*80}\n")


def main():
    print(f"\n{'='*60}")
    print("DATA ORGANIZATION AND SPLIT VERIFICATION")
    print(f"{'='*60}")
    
    # Check source directories
    if not SOURCE_TRAIN.exists():
        print(f"❌ Error: Training data directory not found: {SOURCE_TRAIN}")
        print("Please update SOURCE_TRAIN in the script.")
        return
    
    if not SOURCE_TEST.exists():
        print(f"❌ Error: Test data directory not found: {SOURCE_TEST}")
        print("Please update SOURCE_TEST in the script.")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Organize data
    train_stats = organize_data(SOURCE_TRAIN, "train")
    test_stats = organize_data(SOURCE_TEST, "test")
    
    # Verify split ratio
    verify_split_ratio(train_stats, test_stats)
    
    print(f"\n✅ Data organized successfully!")
    print(f"Output directory: {OUTPUT_DIR.absolute()}\n")


if __name__ == "__main__":
    main()
