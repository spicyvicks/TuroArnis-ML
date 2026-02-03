import os
import shutil
import random
from tqdm import tqdm

# configuration
SOURCE_DATASET = "dataset"
OUTPUT_BASE = "dataset_split"
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
RANDOM_SEED = 42

def split_dataset(source_folder, output_folder, train_ratio=0.8, test_ratio=0.2, seed=42):
    """
    split dataset into train/test sets BEFORE augmentation
    this prevents data leakage by ensuring no image variations leak into test set
    note: validation is handled via cross-validation during training (RF/XGB)
    """
    random.seed(seed)
    
    print("\n" + "="*50)
    print("  DATASET SPLITTING (PREVENTS DATA LEAKAGE)")
    print("="*50)
    print(f"  Source: {source_folder}")
    print(f"  Output: {output_folder}")
    print(f"  Split: {train_ratio*100:.0f}% train, {test_ratio*100:.0f}% test (no validation - using CV)")
    print("="*50)
    
    # validate ratios
    if abs(train_ratio + test_ratio - 1.0) > 0.01:
        print("[ERROR] Ratios must sum to 1.0")
        return False
    
    if not os.path.exists(source_folder):
        print(f"[ERROR] Source folder not found: {source_folder}")
        return False
    
    # create output directories
    train_dir = os.path.join(output_folder, "train")
    test_dir = os.path.join(output_folder, "test")
    
    for d in [train_dir, test_dir]:
        os.makedirs(d, exist_ok=True)
    
    # get all class folders
    class_folders = sorted([
        d for d in os.listdir(source_folder) 
        if os.path.isdir(os.path.join(source_folder, d))
    ])
    
    if not class_folders:
        print("[ERROR] No class folders found in source")
        return False
    
    print(f"\n[INFO] Found {len(class_folders)} classes")
    
    total_stats = {"train": 0, "test": 0}
    
    for class_name in tqdm(class_folders, desc="Splitting classes"):
        class_path = os.path.join(source_folder, class_name)
        
        # get all images in this class
        images = [
            f for f in os.listdir(class_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if len(images) < 3:
            print(f"\n[WARN] {class_name}: only {len(images)} images, need at least 3 for splitting")
            continue
        
        # shuffle and split
        random.shuffle(images)
        
        n_total = len(images)
        n_train = max(1, int(n_total * train_ratio))
        n_test = n_total - n_train
        
        # ensure at least 1 in each set
        if n_test < 1:
            n_test = 1
            n_train = n_total - n_test
        
        train_images = images[:n_train]
        test_images = images[n_train:]
        
        # create class directories and copy files
        for split_name, split_images in [("train", train_images), ("test", test_images)]:
            split_class_dir = os.path.join(output_folder, split_name, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            
            for img in split_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_class_dir, img)
                shutil.copy2(src, dst)
                total_stats[split_name] += 1
    
    print("\n" + "="*50)
    print("  SPLIT COMPLETE")
    print("="*50)
    print(f"  Train: {total_stats['train']} images → {train_dir}")
    print(f"  Test:  {total_stats['test']} images → {test_dir}")
    print("="*50)
    print("\n[NEXT STEP] Run data_augmentation.py to augment ONLY the train set")
    
    return True


if __name__ == "__main__":
    import argparse
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    parser = argparse.ArgumentParser(description='Split dataset into train/test (prevents data leakage)')
    parser.add_argument('--source', '-s', default=os.path.join(project_root, SOURCE_DATASET), help='Source dataset folder')
    parser.add_argument('--output', '-o', default=os.path.join(project_root, OUTPUT_BASE), help='Output folder for splits')
    parser.add_argument('--train', type=float, default=TRAIN_RATIO, help='Training set ratio (default: 0.8)')
    parser.add_argument('--test', type=float, default=TEST_RATIO, help='Test set ratio (default: 0.2)')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed for reproducibility')
    parser.add_argument('--clean', action='store_true', help='Remove existing output folder before splitting')
    
    args = parser.parse_args()
    
    if args.clean and os.path.exists(args.output):
        print(f"[INFO] Removing existing output folder: {args.output}")
        shutil.rmtree(args.output)
    
    split_dataset(
        source_folder=args.source,
        output_folder=args.output,
        train_ratio=args.train,
        test_ratio=args.test,
        seed=args.seed
    )
