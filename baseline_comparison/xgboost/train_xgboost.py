"""
XGBoost Baseline for Arnis Pose Classification
Flattens node features (35 nodes × 6 features = 210-dim vector)
"""
import xgboost as xgb
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from pathlib import Path

CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]

FEATURES_DIR = Path("hybrid_classifier/hybrid_features_v4")
OUTPUT_DIR = Path("baseline_comparison/xgboost/results")

def load_data(viewpoint_filter=None):
    """Load and flatten node features from hybrid_features_v3"""
    suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
    
    train_data = torch.load(FEATURES_DIR / f"train_features{suffix}.pt")
    test_data = torch.load(FEATURES_DIR / f"test_features{suffix}.pt")
    
    # Flatten: (N, 35, 6) -> (N, 210)
    X_train = train_data['node_features'].numpy().reshape(train_data['node_features'].shape[0], -1)
    y_train = train_data['labels'].numpy()
    
    X_test = test_data['node_features'].numpy().reshape(test_data['node_features'].shape[0], -1)
    y_test = test_data['labels'].numpy()
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def compute_class_weights(y_train):
    """Compute class weights for imbalanced dataset"""
    class_counts = np.bincount(y_train)
    total = len(y_train)
    weights = total / (len(CLASS_NAMES) * class_counts)
    return weights

def train_xgboost(viewpoint_filter=None, use_grid_search=False):
    """Train XGBoost classifier"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("XGBoost Baseline Training")
    print("="*60)
    
    # Load data
    X_train, y_train, X_test, y_test = load_data(viewpoint_filter)
    
    # Compute class weights
    class_weights = compute_class_weights(y_train)
    sample_weights = class_weights[y_train]
    
    if use_grid_search:
        print("\nRunning Grid Search...")
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
        
        base_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(CLASS_NAMES),
            random_state=42
        )
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train, sample_weight=sample_weights)
        model = grid_search.best_estimator_
        print(f"\nBest params: {grid_search.best_params_}")
    else:
        print("\nTraining with default hyperparameters...")
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(CLASS_NAMES),
            max_depth=4,              # Reduced from 6 to 4 to prevent overfitting
            learning_rate=0.05,       # Reduced from 0.1 for better generalization
            n_estimators=150,         # Reduced from 200
            subsample=0.7,            # Increased regularization
            colsample_bytree=0.7,     # Increased regularization
            reg_alpha=0.1,            # L1 regularization
            reg_lambda=1.0,           # L2 regularization
            random_state=42
        )
        
        model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"\n{'='*60}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"{'='*60}\n")
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=CLASS_NAMES, zero_division=0))
    
    # Save model
    suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
    model_path = OUTPUT_DIR / f"xgboost_model{suffix}.json"
    model.save_model(str(model_path))
    print(f"\nModel saved to {model_path}")
    
    # Visualizations
    plot_confusion_matrix(y_test, y_pred_test, suffix)
    plot_feature_importance(model, suffix)
    
    # Save results
    results = {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'classification_report': classification_report(y_test, y_pred_test, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    }
    
    import json
    results_path = OUTPUT_DIR / f"results{suffix}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")

def plot_confusion_matrix(y_true, y_pred, suffix=""):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[c.replace('_correct', '').replace('_', ' ').title() for c in CLASS_NAMES],
                yticklabels=[c.replace('_correct', '').replace('_', ' ').title() for c in CLASS_NAMES])
    plt.title('XGBoost - Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"confusion_matrix{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

def plot_feature_importance(model, suffix=""):
    """Plot top 20 most important features"""
    importance = model.feature_importances_
    
    # Get top 20 features
    top_indices = np.argsort(importance)[-20:]
    top_importance = importance[top_indices]
    
    # Create feature names (node_X_feature_Y format)
    feature_names = []
    for idx in top_indices:
        node_idx = idx // 6
        feat_idx = idx % 6
        feat_name = ['x', 'y', 'z', 'vis', 'dist', 'angle'][feat_idx]
        feature_names.append(f"Node{node_idx}_{feat_name}")
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_importance)), top_importance)
    plt.yticks(range(len(top_importance)), feature_names)
    plt.xlabel('Feature Importance')
    plt.title('XGBoost - Top 20 Feature Importances')
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"feature_importance{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train XGBoost baseline")
    parser.add_argument('--viewpoint', type=str, required=True,
                        choices=['front', 'left', 'right'],
                        help='Viewpoint to train on (required)')
    parser.add_argument('--grid_search', action='store_true',
                        help='Use grid search for hyperparameter tuning')
    
    args = parser.parse_args()
    train_xgboost(args.viewpoint, args.grid_search)
