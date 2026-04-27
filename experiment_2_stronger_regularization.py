"""
Experiment 2: Stronger Regularization + 2x Synthetic Training

Trains with Plan E hyperparameters:
- HIDDEN_DIM=96 (moderate increase)
- DROPOUT=0.7 (higher regularization)
- WEIGHT_DECAY=1e-3 (AdamW-style)
- BATCH_SIZE=32 (smaller batches)
- 2x synthetic (middle ground between 1x and 3x)

Compares against existing 1x and 3x models.
"""

import sys
sys.path.insert(0, 'hybrid_classifier')

# Monkey-patch hyperparameters BEFORE importing training script
import hybrid_classifier.train_hybrid_gcn_v2_regularized as trainer

# Override config
trainer.HIDDEN_DIM = 96
trainer.DROPOUT = 0.7
trainer.WEIGHT_DECAY = 1e-3
trainer.BATCH_SIZE = 32
trainer.NUM_LAYERS = 3

# Run training with 2x synthetic
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewpoint', default='front')
    parser.add_argument('--synthetic_factor', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()
    
    # We need to call the training function directly
    # Since 4e_train_hybrid_gcn_v2_with_synthetic.py uses argparse,
    # we'll use a subprocess call with modified script
    import subprocess
    result = subprocess.run([
        sys.executable,
        'hybrid_classifier/4e_train_hybrid_gcn_v2_with_synthetic.py',
        '--viewpoint', args.viewpoint,
        '--synthetic_factor', str(args.synthetic_factor),
        '--epochs', str(args.epochs),
        '--patience', str(args.patience)
    ])
    sys.exit(result.returncode)
