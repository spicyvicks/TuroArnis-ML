"""
Experiment 2: Stronger Regularization + 2x Synthetic Training

Uses Plan E hyperparameters:
- HIDDEN_DIM = 96
- DROPOUT = 0.7
- WEIGHT_DECAY = 1e-3
- BATCH_SIZE = 32
- NUM_LAYERS = 3
- 2x synthetic factor

Creates a temporary copy of the training script with modified constants,
runs it, then cleans up.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ORIGINAL_SCRIPT = Path("hybrid_classifier/4e_train_hybrid_gcn_v2_with_synthetic.py")
TEMP_SCRIPT = Path("hybrid_classifier/_temp_train_regularized.py")

def create_modified_script():
    """Create a copy of the training script with Plan E hyperparameters."""
    content = ORIGINAL_SCRIPT.read_text()
    
    # Replace hyperparameter constants
    replacements = [
        ("HIDDEN_DIM = 128", "HIDDEN_DIM = 96"),
        ("DROPOUT = 0.5", "DROPOUT = 0.7"),
        ("WEIGHT_DECAY = 5e-5", "WEIGHT_DECAY = 1e-3"),
        ("BATCH_SIZE = 64", "BATCH_SIZE = 32"),
        ("NUM_LAYERS = 3", "NUM_LAYERS = 3"),  # Keep same, just explicit
    ]
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new, 1)
            print(f"  Replaced: {old} -> {new}")
        else:
            print(f"  WARNING: Could not find: {old}")
    
    # Also modify model save path to distinguish from original
    content = content.replace(
        'model_name = f"model_{view_suffix}_with_synthetic_{synthetic_factor}x"',
        'model_name = f"model_{view_suffix}_with_synthetic_{synthetic_factor}x_regularized"'
    )
    content = content.replace(
        'history_path = HISTORY_DIR / f"history_{view_suffix}_with_synthetic_{synthetic_factor}x.json"',
        'history_path = HISTORY_DIR / f"history_{view_suffix}_with_synthetic_{synthetic_factor}x_regularized.json"'
    )
    
    TEMP_SCRIPT.write_text(content)
    print(f"\nCreated temporary training script: {TEMP_SCRIPT}")


def run_training():
    """Run the modified training script."""
    print("\n" + "="*60)
    print("Running Experiment 2: Stronger Regularization + 2x Synthetic")
    print("="*60)
    print("Hyperparameters:")
    print("  HIDDEN_DIM = 96")
    print("  DROPOUT = 0.7")
    print("  WEIGHT_DECAY = 1e-3")
    print("  BATCH_SIZE = 32")
    print("  NUM_LAYERS = 3")
    print("  SYNTHETIC_FACTOR = 2")
    print("="*60 + "\n")
    
    result = subprocess.run([
        sys.executable,
        str(TEMP_SCRIPT),
        "--viewpoint", "front",
        "--synthetic_factor", "2",
        "--epochs", "150",
        "--patience", "20"
    ])
    
    return result.returncode


def cleanup():
    """Remove temporary script."""
    if TEMP_SCRIPT.exists():
        TEMP_SCRIPT.unlink()
        print(f"\nCleaned up temporary script: {TEMP_SCRIPT}")


def main():
    try:
        create_modified_script()
        exit_code = run_training()
        
        if exit_code == 0:
            print("\n" + "="*60)
            print("EXPERIMENT 2 COMPLETE")
            print("="*60)
            print("\nModel saved as:")
            print("  hybrid_classifier/models/model_front_with_synthetic_2x_regularized.pth")
            print("\nNext steps:")
            print("  1. Evaluate with: python evaluate_real_only_v2.py")
            print("  2. Compare against 1x (32.5%) and 3x (40.6%) baselines")
            print("  3. If still <50%, consider Plan B (hand landmarks)")
            print("="*60)
        else:
            print(f"\nTraining exited with code {exit_code}")
            
    finally:
        cleanup()


if __name__ == "__main__":
    main()
