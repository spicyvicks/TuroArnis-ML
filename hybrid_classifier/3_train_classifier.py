"""
Step 3: Train Hybrid Classifier
Trains a Random Forest or Neural Network on hybrid features
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json

# Config
FEATURES_DIR = Path("hybrid_classifier/hybrid_features")
OUTPUT_DIR = Path("hybrid_classifier/models")
MODEL_TYPE = "random_forest"  # or "neural_network"

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]


def train_model(model_type='random_forest', viewpoint_filter=None):
    """Train classifier on hybrid features"""
    
    # Determine file suffix based on viewpoint
    suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
    
    # Load data
    train_file = FEATURES_DIR / f"train_features{suffix}.pt"
    test_file = FEATURES_DIR / f"test_features{suffix}.pt"
    
    if not train_file.exists():
        print(f"Error: {train_file} not found!")
        print("Run 2_generate_hybrid_features.py first")
        return
    
    train_data = torch.load(train_file)
    test_data = torch.load(test_file)
    
    X_train = train_data['features'].numpy()
    y_train = train_data['labels'].numpy()
    
    X_test = test_data['features'].numpy()
    y_test = test_data['labels'].numpy()
    
    viewpoint_str = f" ({viewpoint_filter} only)" if viewpoint_filter else ""
    print(f"Training{viewpoint_str}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    # Train model
    if model_type == 'random_forest':
        print("\nTraining Random Forest...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
    elif model_type == 'neural_network':
        print("\nTraining Neural Network...")
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Train-Test Gap: {train_acc - test_acc:.4f}")
    
    print("\nPer-Class Performance:")
    print(classification_report(y_test, test_pred, target_names=CLASS_NAMES, zero_division=0))
    
    # Per-viewpoint accuracy
    print("\nPer-Viewpoint Accuracy:")
    for viewpoint in ['front', 'left', 'right']:
        mask = np.array([v == viewpoint for v in test_data['viewpoints']])
        if mask.sum() > 0:
            vp_acc = accuracy_score(y_test[mask], test_pred[mask])
            print(f"  {viewpoint}: {vp_acc:.4f}")
    
    # Save model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / f"{model_type}_model.pkl"
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Save metadata
    metadata = {
        'model_type': model_type,
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'num_features': int(X_train.shape[1]),
        'num_classes': len(CLASS_NAMES),
        'class_names': CLASS_NAMES
    }
    
    with open(OUTPUT_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='random_forest',
                        choices=['random_forest', 'neural_network'])
    parser.add_argument('--viewpoint', type=str, default=None,
                        choices=['front', 'left', 'right'],
                        help='Train on specific viewpoint only (default: all)')
    args = parser.parse_args()
    
    train_model(args.model_type, args.viewpoint)
