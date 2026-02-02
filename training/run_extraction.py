"""
Feature extraction pipeline for TuroArnis ML
Extracts features from pre-split datasets and saves to CSVs
Run this after split_dataset.py and data_augmentation.py
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)

from feature_extraction import extract_features_from_dataset

# paths
TRAIN_FOLDER = os.path.join(project_root, 'dataset_split', 'train_aug')
VAL_FOLDER = os.path.join(project_root, 'dataset_split', 'val')
TEST_FOLDER = os.path.join(project_root, 'dataset_split', 'test')

CSV_TRAIN = os.path.join(project_root, 'features_train.csv')
CSV_VAL = os.path.join(project_root, 'features_val.csv')
CSV_TEST = os.path.join(project_root, 'features_test.csv')


def run_extraction(feature_mode='angles'):
    """
    extract features from train_aug, val, and test folders
    saves to features_train.csv, features_val.csv, features_test.csv
    """
    print("\n" + "="*60)
    print("  FEATURE EXTRACTION PIPELINE")
    print("="*60)
    print(f"  Mode: {feature_mode.upper()}")
    print("="*60)
    
    # check folders exist
    folders = [
        ("Train (augmented)", TRAIN_FOLDER),
        ("Validation", VAL_FOLDER),
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
    
    # extract from each folder
    results = {}
    
    print("\n[1/3] Extracting from TRAIN set (augmented)...")
    df_train = extract_features_from_dataset(TRAIN_FOLDER, CSV_TRAIN, feature_mode)
    results['train'] = len(df_train)
    
    print("\n[2/3] Extracting from VALIDATION set (clean)...")
    df_val = extract_features_from_dataset(VAL_FOLDER, CSV_VAL, feature_mode)
    results['val'] = len(df_val)
    
    print("\n[3/3] Extracting from TEST set (clean)...")
    df_test = extract_features_from_dataset(TEST_FOLDER, CSV_TEST, feature_mode)
    results['test'] = len(df_test)
    
    # summary
    print("\n" + "="*60)
    print("  EXTRACTION COMPLETE")
    print("="*60)
    print(f"  Train samples: {results['train']:,}")
    print(f"  Val samples:   {results['val']:,}")
    print(f"  Test samples:  {results['test']:,}")
    print(f"  Total:         {sum(results.values()):,}")
    print("="*60)
    print("\n  Output files:")
    print(f"    {CSV_TRAIN}")
    print(f"    {CSV_VAL}")
    print(f"    {CSV_TEST}")
    print("\n[NEXT] Run training:")
    print("  python training/training.py      (DNN)")
    print("  python training/training_alt.py  (RF/XGB)")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from split dataset')
    parser.add_argument('--mode', '-m', default='angles', choices=['angles', 'coordinates'],
                        help='Feature mode: angles (58 features) or coordinates (99 features)')
    
    args = parser.parse_args()
    run_extraction(args.mode)
