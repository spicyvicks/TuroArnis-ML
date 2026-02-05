"""
Export all model metadata to a single consolidated file.
Creates models/model_registry.json with all model information.
"""
import os
import sys
import json
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

MODELS_DIR = os.path.join(project_root, 'models')
OUTPUT_FILE = os.path.join(MODELS_DIR, 'model_registry.json')


def export_all_metadata():
    """Read all model metadata and export to a single registry file."""
    
    print("\n" + "="*60)
    print("  EXPORTING MODEL METADATA")
    print("="*60)
    
    registry = {
        'exported_at': datetime.now().isoformat(),
        'total_models': 0,
        'models': []
    }
    
    if not os.path.exists(MODELS_DIR):
        print(f"[ERROR] Models directory not found: {MODELS_DIR}")
        return None
    
    # Collect all model metadata
    for item in sorted(os.listdir(MODELS_DIR)):
        item_path = os.path.join(MODELS_DIR, item)
        
        # Skip if not a directory or doesn't start with 'v'
        if not os.path.isdir(item_path) or not item.startswith('v'):
            continue
        
        metadata_path = os.path.join(item_path, 'metadata.json')
        
        if not os.path.exists(metadata_path):
            print(f"  [SKIP] {item}: no metadata.json")
            continue
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Extract key information
            model_entry = {
                'version': item,
                'model_type': metadata.get('model_type', 'unknown'),
                'trained_at': metadata.get('trained_at', 'unknown'),
                'test_accuracy': metadata.get('test_accuracy', 0),
                'cv_accuracy': metadata.get('cv_accuracy', 0),
                'n_features_in': metadata.get('n_features_in', 0),
                'num_features': metadata.get('num_features', 0),
                'feature_selection_used': metadata.get('feature_selection_used', False),
                'stick_features_protected': metadata.get('stick_features_protected', False),
                'hyperparameters': metadata.get('hyperparameters', {}),
                'class_names': metadata.get('class_names', []),
            }
            
            # Check for selected features file
            selected_features_path = os.path.join(item_path, 'selected_features.json')
            if os.path.exists(selected_features_path):
                with open(selected_features_path, 'r') as f:
                    selected = json.load(f)
                model_entry['selected_features_count'] = len(selected)
            
            registry['models'].append(model_entry)
            print(f"  ✓ {item} ({model_entry['model_type']}) - Test: {model_entry['test_accuracy']*100:.2f}%")
            
        except Exception as e:
            print(f"  [ERROR] {item}: {e}")
    
    registry['total_models'] = len(registry['models'])
    
    # Sort by test accuracy (descending)
    registry['models'].sort(key=lambda x: x['test_accuracy'], reverse=True)
    
    # Add rankings
    for i, model in enumerate(registry['models'], 1):
        model['rank'] = i
    
    # Save to file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print("\n" + "="*60)
    print(f"  EXPORTED {registry['total_models']} MODELS")
    print("="*60)
    print(f"  Output: {OUTPUT_FILE}")
    
    # Print summary table
    print("\n  TOP 10 MODELS BY TEST ACCURACY:")
    print("  " + "-"*70)
    print(f"  {'Rank':<5} {'Version':<35} {'Type':<15} {'Test Acc':<10} {'Features'}")
    print("  " + "-"*70)
    
    for model in registry['models'][:10]:
        print(f"  {model['rank']:<5} {model['version']:<35} {model['model_type']:<15} {model['test_accuracy']*100:>6.2f}%    {model['n_features_in']}")
    
    print("  " + "-"*70)
    
    # Summary by model type
    print("\n  SUMMARY BY MODEL TYPE:")
    type_stats = {}
    for model in registry['models']:
        mtype = model['model_type']
        if mtype not in type_stats:
            type_stats[mtype] = {'count': 0, 'best_acc': 0, 'total_acc': 0}
        type_stats[mtype]['count'] += 1
        type_stats[mtype]['total_acc'] += model['test_accuracy']
        type_stats[mtype]['best_acc'] = max(type_stats[mtype]['best_acc'], model['test_accuracy'])
    
    for mtype, stats in type_stats.items():
        avg_acc = stats['total_acc'] / stats['count'] if stats['count'] > 0 else 0
        print(f"  - {mtype}: {stats['count']} models, best={stats['best_acc']*100:.2f}%, avg={avg_acc*100:.2f}%")
    
    return registry


if __name__ == "__main__":
    export_all_metadata()
