"""
Feature extraction pipeline for TuroArnis ML
Extracts features from train/test splits and saves to CSVs
Run this after split_dataset.py and data_augmentation.py
Note: No validation set - RF/XGB use cross-validation internally
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)

# paths
TRAIN_FOLDER = os.path.join(project_root, 'dataset_split', 'train_aug')
TEST_FOLDER = os.path.join(project_root, 'dataset_split', 'test')

CSV_TRAIN = os.path.join(project_root, 'features_train.csv')
CSV_TEST = os.path.join(project_root, 'features_test.csv')


def run_extraction(feature_mode='angles'):
    """
    extract features from train_aug and test folders
    saves to features_train.csv and features_test.csv
    note: no validation set - RF/XGB use cross-validation internally
    
    modes:
    - angles: 58 body features (MediaPipe only)
    - coordinates: 99 body features (MediaPipe only)
    - combined: 72 features (58 body + 14 stick) - MediaPipe + YOLO
    """
    print("\n" + "="*60)
    print("  FEATURE EXTRACTION PIPELINE")
    print("="*60)
    print(f"  Mode: {feature_mode.upper()}")
    
    if feature_mode == 'combined':
        print("  [Body: 58 features] + [Stick: 14 features] = 72 total")
    elif feature_mode == 'angles':
        print("  [Body angles: 58 features]")
    else:
        print("  [Body coordinates: 99 features]")
    print("="*60)
    
    # check folders exist
    folders = [
        ("Train (augmented)", TRAIN_FOLDER),
        ("Test", TEST_FOLDER)
    ]
    
    missing = []
    for name, path in folders:
        if not os.path.exists(path):
            missing.append(f"  - {name}: {path}")
    
    if missing:
        print("\n[ERROR] Missing folders:")
        print("\n".join(missing))
        print("\nRun these commands first:")
        print("  1. python training/split_dataset.py")
        print("  2. python training/data_augmentation.py")
        return False
    
    # import appropriate extraction function
    if feature_mode == 'combined':
        from feature_extraction_combined import extract_features_from_dataset
        extract_func = lambda folder, csv_path: extract_features_from_dataset(folder, csv_path)
    else:
        from feature_extraction import extract_features_from_dataset
        extract_func = lambda folder, csv_path: extract_features_from_dataset(folder, csv_path, feature_mode)
    
    # extract from each folder
    results = {}
    
    print("\n[1/2] Extracting from TRAIN set (augmented)...")
    df_train = extract_func(TRAIN_FOLDER, CSV_TRAIN)
    results['train'] = len(df_train)
    
    print("\n[2/2] Extracting from TEST set (clean)...")
    df_test = extract_func(TEST_FOLDER, CSV_TEST)
    results['test'] = len(df_test)
    
    # summary
    print("\n" + "="*60)
    print("  EXTRACTION COMPLETE")
    print("="*60)
    print(f"  Mode:          {feature_mode.upper()}")
    print(f"  Features:      {len(df_train.columns) - 1}")  # -1 for class column
    print(f"  Train samples: {results['train']:,}")
    print(f"  Test samples:  {results['test']:,}")
    print(f"  Total:         {sum(results.values()):,}")
    print("="*60)
    print("\n  Output files:")
    print(f"    {CSV_TRAIN}")
    print(f"    {CSV_TEST}")
    print("\n[NEXT] Run training:")
    print("  python training/training.py      (DNN)")
    print("  python training/training_alt.py  (RF/XGB)")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from split dataset')
    parser.add_argument('--mode', '-m', default='angles', 
                        choices=['angles', 'coordinates', 'combined'],
                        help='Feature mode: angles (58), coordinates (99), or combined (72 = body+stick)')
    
    args = parser.parse_args()
    run_extraction(args.mode)
