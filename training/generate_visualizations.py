"""
Visualization generator for RF/XGBoost models
Includes feature selection analysis plots
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

def plot_feature_selection_analysis(feature_selection_data, save_path):
    """Plot showing which features were kept vs dropped during feature selection"""
    if not feature_selection_data:
        return
    
    # Sort by importance
    sorted_data = sorted(feature_selection_data, key=lambda x: x['importance'], reverse=True)
    
    features = [d['feature'] for d in sorted_data]
    importances = [d['importance'] for d in sorted_data]
    selected = [d['selected'] for d in sorted_data]
    
    # Create color array
    colors = ['#2ecc71' if s else '#e74c3c' for s in selected]
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(features) * 0.25)))
    
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importances, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=8)
    ax.invert_yaxis()
    
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Feature Selection Analysis\n(Green = Kept, Red = Dropped)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Selected Features'),
        Patch(facecolor='#e74c3c', label='Dropped Features')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Add count annotation
    kept_count = sum(selected)
    dropped_count = len(selected) - kept_count
    ax.text(0.98, 0.02, f'Kept: {kept_count} | Dropped: {dropped_count}', 
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {os.path.basename(save_path)}")

def plot_final_feature_importance(final_importance_data, save_path, top_n=20):
    """Plot feature importances from the final trained model (after feature selection)"""
    if not final_importance_data:
        return
    
    # Sort by importance and take top N
    sorted_data = sorted(final_importance_data, key=lambda x: x['importance'], reverse=True)[:top_n]
    
    features = [d['feature'] for d in sorted_data]
    importances = [d['importance'] for d in sorted_data]
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    
    y_pos = np.arange(len(features))
    plt.barh(y_pos, importances, color=colors)
    plt.yticks(y_pos, features)
    plt.gca().invert_yaxis()
    
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {len(features)} Feature Importances (Final Model)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {os.path.basename(save_path)}")

def plot_confusion_matrix_heatmap(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('Actual Class', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {os.path.basename(save_path)}")

def plot_class_accuracy(y_true, y_pred, class_names, save_path):
    """Plot per-class accuracy as a bar chart"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Sort by accuracy
    sorted_indices = np.argsort(per_class_acc)
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_acc = per_class_acc[sorted_indices]
    
    plt.figure(figsize=(12, max(6, len(class_names) * 0.3)))
    
    # Color gradient based on accuracy
    colors = plt.cm.RdYlGn(sorted_acc)
    
    y_pos = np.arange(len(sorted_classes))
    bars = plt.barh(y_pos, sorted_acc * 100, color=colors, edgecolor='white')
    
    plt.yticks(y_pos, sorted_classes, fontsize=9)
    plt.xlabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.xlim(0, 105)
    plt.grid(axis='x', alpha=0.3)
    
    # Add accuracy labels
    for i, (acc, bar) in enumerate(zip(sorted_acc, bars)):
        plt.text(acc * 100 + 1, i, f'{acc*100:.1f}%', va='center', fontsize=8)
    
    # Add mean accuracy line
    mean_acc = np.mean(per_class_acc) * 100
    plt.axvline(x=mean_acc, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.1f}%')
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {os.path.basename(save_path)}")

def plot_feature_correlation(X, feature_names, save_path, top_n=30):
    """Plot correlation matrix of top features"""
    if len(feature_names) > top_n:
        feature_names = feature_names[:top_n]
        X = X[:, :top_n]
    
    corr_matrix = np.corrcoef(X.T)
    
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                xticklabels=feature_names, yticklabels=feature_names,
                cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
    
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {os.path.basename(save_path)}")

def generate_visualizations_for_model(version_dir):
    print(f"\n{'='*60}")
    print(f"  GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    print(f"  Model: {os.path.basename(version_dir)}")
    print(f"{'='*60}\n")
    
    # Load metadata
    metadata_path = os.path.join(version_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        print("[ERROR] metadata.json not found")
        return False
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    model_type = metadata.get('model_type', 'unknown')
    if model_type not in ['random_forest', 'xgboost']:
        print(f"[ERROR] Only Random Forest and XGBoost models supported (found: {model_type})")
        return False
    
    # Load model
    model_suffix = 'rf' if model_type == 'random_forest' else 'xgb'
    model_path = os.path.join(version_dir, f'model_{model_suffix}.joblib')
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return False
    
    print("[INFO] Loading model...")
    model = joblib.load(model_path)
    
    # Load label encoder
    encoder_path = os.path.join(version_dir, 'label_encoder.joblib')
    label_encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
    class_names = list(label_encoder.classes_) if label_encoder else metadata.get('class_names', [])
    
    # Load scaler
    scaler_path = os.path.join(version_dir, 'scaler.joblib')
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    # Load feature selection data
    feature_selection_path = os.path.join(version_dir, 'feature_selection_data.json')
    feature_selection_data = None
    if os.path.exists(feature_selection_path):
        with open(feature_selection_path, 'r') as f:
            feature_selection_data = json.load(f)
    
    # Load final importance data
    final_importance_path = os.path.join(version_dir, 'final_importance_data.json')
    final_importance_data = None
    if os.path.exists(final_importance_path):
        with open(final_importance_path, 'r') as f:
            final_importance_data = json.load(f)
    
    # Load selected features
    selected_features_path = os.path.join(version_dir, 'selected_features.json')
    selected_features = None
    if os.path.exists(selected_features_path):
        with open(selected_features_path, 'r') as f:
            selected_features = json.load(f)
    
    # Generate plots
    print("\n[INFO] Generating visualizations...")
    
    # 1. Feature Selection Analysis (if available)
    if feature_selection_data:
        plot_feature_selection_analysis(
            feature_selection_data, 
            os.path.join(version_dir, 'feature_selection_analysis.png')
        )
    
    # 2. Final Feature Importance (if available)
    if final_importance_data:
        plot_final_feature_importance(
            final_importance_data,
            os.path.join(version_dir, 'feature_importance.png')
        )
    
    # 3. Load test data for confusion matrix
    test_csv = os.path.join(project_root, 'features_test.csv')
    if os.path.exists(test_csv):
        print("[INFO] Loading test data...")
        test_df = pd.read_csv(test_csv).dropna()
        
        # Filter to known classes
        test_df = test_df[test_df['class'].isin(class_names)]
        
        X_test = test_df.drop('class', axis=1).values
        y_test_labels = test_df['class'].values
        
        # Encode labels
        y_test = label_encoder.transform(y_test_labels)
        
        # Get feature names from CSV
        feature_names = [col for col in test_df.columns if col != 'class']
        
        # Apply scaler if available
        if scaler:
            X_test = scaler.transform(X_test)
        
        # Apply feature selection if used
        if selected_features:
            # Get indices of selected features
            selected_indices = [feature_names.index(f) for f in selected_features if f in feature_names]
            X_test = X_test[:, selected_indices]
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Plot confusion matrix
        plot_confusion_matrix_heatmap(
            y_test, y_pred, class_names,
            os.path.join(version_dir, 'confusion_matrix.png')
        )
        
        # Plot per-class accuracy
        plot_class_accuracy(
            y_test, y_pred, class_names,
            os.path.join(version_dir, 'per_class_accuracy.png')
        )
        
        # Plot feature correlation (on selected features)
        if selected_features and len(selected_features) <= 40:
            plot_feature_correlation(
                X_test, selected_features if selected_features else feature_names,
                os.path.join(version_dir, 'feature_correlation.png')
            )
        
        # Generate classification report
        report = classification_report(y_test, y_pred, target_names=class_names)
        report_path = os.path.join(version_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Classification Report for {metadata['version']}\n")
            f.write(f"{'='*60}\n\n")
            f.write(report)
        print(f"[INFO] Saved: classification_report.txt")
    else:
        print(f"[WARN] Test CSV not found: {test_csv}")
    
    print(f"\n{'='*60}")
    print("  VISUALIZATION GENERATION COMPLETE")
    print(f"{'='*60}\n")
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualizations for trained RF/XGBoost models')
    parser.add_argument('--version', '-v', type=str, help='Model version directory name')
    parser.add_argument('--all', '-a', action='store_true', help='Generate for all RF/XGBoost models')
    
    args = parser.parse_args()
    
    models_dir = os.path.join(project_root, 'models')
    
    if args.all:
        versions = [d for d in os.listdir(models_dir) 
                   if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('v')]
        
        for version in versions:
            version_dir = os.path.join(models_dir, version)
            metadata_path = os.path.join(version_dir, 'metadata.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                model_type = metadata.get('model_type', '')
                if model_type in ['random_forest', 'xgboost']:
                    generate_visualizations_for_model(version_dir)
        
    elif args.version:
        version_dir = os.path.join(models_dir, args.version)
        if not os.path.exists(version_dir):
            print(f"[ERROR] Version directory not found: {version_dir}")
            return
        
        generate_visualizations_for_model(version_dir)
    
    else:
        print("[ERROR] Specify --version <name> or --all")
        parser.print_help()

if __name__ == "__main__":
    main()
