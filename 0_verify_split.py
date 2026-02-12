"""
Step 0: Verify Train/Test Split Ratio

Analyzes your existing dataset_split/ directory and reports:
- Train/test split ratio for each class
- Actionable advice on how many images to move to achieve 80/20 split
- Overall dataset statistics

No copying - just verification and recommendations.
"""

from pathlib import Path
from collections import defaultdict

# Configuration
DATASET_DIR = Path("dataset_split")
TARGET_RATIO = 0.80  # 80% train, 20% test

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'neutral_stance',
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


def collect_stats(split_name):
    """Collect image counts for a split (train or test)"""
    stats = defaultdict(lambda: defaultdict(int))
    
    for viewpoint in VIEWPOINTS:
        for class_name in CLASS_NAMES:
            class_dir = DATASET_DIR / split_name / viewpoint / class_name
            count = count_images(class_dir)
            stats[viewpoint][class_name] = count
    
    return stats


def verify_split_ratio(train_stats, test_stats):
    """Verify and report train/test split ratios with actionable advice"""
    print(f"\n{'='*88}")
    print(f"{'SPLIT RATIO VERIFICATION':^88}")
    print(f"{'='*88}\n")
    
    total_train = 0
    total_test = 0
    
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
            target_train = int(total * TARGET_RATIO)
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
    
    print(f"\n{'='*88}")
    if 0.75 <= overall_ratio <= 0.85:
        print("✅ Split ratio is within acceptable range (75-85% train)")
    else:
        print(f"⚠️  Split ratio ({overall_ratio:.1%}) is outside recommended range.")
        print("   Please follow the 'Action Needed' column to balance your dataset.")
    print(f"{'='*88}\n")


def main():
    print(f"\n{'='*88}")
    print(f"{'DATASET SPLIT VERIFICATION':^88}")
    print(f"{'='*88}")
    print(f"Analyzing: {DATASET_DIR.absolute()}\n")
    
    # Check if dataset directory exists
    if not DATASET_DIR.exists():
        print(f"❌ Error: Dataset directory not found: {DATASET_DIR}")
        print("Please ensure your data is in dataset_split/train and dataset_split/test")
        return
    
    train_dir = DATASET_DIR / "train"
    test_dir = DATASET_DIR / "test"
    
    if not train_dir.exists():
        print(f"❌ Error: Training directory not found: {train_dir}")
        return
    
    if not test_dir.exists():
        print(f"❌ Error: Test directory not found: {test_dir}")
        return
    
    # Collect statistics
    train_stats = collect_stats("train")
    test_stats = collect_stats("test")
    
    # Verify split ratio
    verify_split_ratio(train_stats, test_stats)


if __name__ == "__main__":
    main()
