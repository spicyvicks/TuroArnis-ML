"""
Batch training script for all advanced baseline models
Trains Transformer, EdgeConv, and ST-GCN for all viewpoints
"""
import subprocess
import sys
from pathlib import Path

VIEWPOINTS = ['front', 'left', 'right']
MODELS = {
    'pure_gcn': {
        'script': 'baseline_comparison/pure_gcn/train_pure_gcn.py',
        'name': 'Pure GCN (No Expert Features)'
    },
    'transformer': {
        'script': 'baseline_comparison/transformer/train_transformer.py',
        'name': 'Pose Transformer'
    },
    'edgeconv': {
        'script': 'baseline_comparison/edgeconv/train_edgeconv.py',
        'name': 'EdgeConv (Dynamic Graph CNN)'
    },
    'stgcn': {
        'script': 'baseline_comparison/stgcn/train_stgcn.py',
        'name': 'ST-GCN (Spatio-Temporal GCN)'
    }
}

def train_model(model_key, viewpoint, epochs=150):
    """Train a single model for a specific viewpoint"""
    model_info = MODELS[model_key]
    print(f"\n{'='*80}")
    print(f"Training {model_info['name']} - Viewpoint: {viewpoint}")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable,
        model_info['script'],
        '--viewpoint', viewpoint,
        '--epochs', str(epochs)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ {model_info['name']} ({viewpoint}) completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {model_info['name']} ({viewpoint}) failed with error: {e}")
        return False
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch train advanced baseline models")
    parser.add_argument('--models', nargs='+', choices=list(MODELS.keys()) + ['all'],
                        default=['all'],
                        help='Models to train (default: all)')
    parser.add_argument('--viewpoints', nargs='+', choices=VIEWPOINTS + ['all'],
                        default=['all'],
                        help='Viewpoints to train (default: all)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs (default: 150)')
    
    args = parser.parse_args()
    
    # Determine which models to train
    models_to_train = list(MODELS.keys()) if 'all' in args.models else args.models
    
    # Determine which viewpoints to train
    viewpoints_to_train = VIEWPOINTS if 'all' in args.viewpoints else args.viewpoints
    
    print(f"\n{'='*80}")
    print(f"BATCH TRAINING ADVANCED BASELINES")
    print(f"{'='*80}")
    print(f"Models: {', '.join(models_to_train)}")
    print(f"Viewpoints: {', '.join(viewpoints_to_train)}")
    print(f"Epochs: {args.epochs}")
    print(f"{'='*80}\n")
    
    results = {}
    
    for model_key in models_to_train:
        results[model_key] = {}
        for viewpoint in viewpoints_to_train:
            success = train_model(model_key, viewpoint, args.epochs)
            results[model_key][viewpoint] = 'SUCCESS' if success else 'FAILED'
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*80}\n")
    
    for model_key, viewpoint_results in results.items():
        print(f"{MODELS[model_key]['name']}:")
        for viewpoint, status in viewpoint_results.items():
            status_symbol = '✓' if status == 'SUCCESS' else '✗'
            print(f"  {status_symbol} {viewpoint}: {status}")
        print()

if __name__ == "__main__":
    main()
