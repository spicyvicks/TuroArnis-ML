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

def plot_gridsearch_results(cv_results, save_path, n_iter=None):
    mean_scores = cv_results['mean_test_score']
    std_scores = cv_results['std_test_score']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(mean_scores))
    ax.plot(x, mean_scores, 'b-o', label='Mean CV Score', markersize=4)
    ax.fill_between(x, mean_scores - std_scores, mean_scores + std_scores, 
                     alpha=0.2, color='b', label='±1 std')
    
    best_idx = np.argmax(mean_scores)
    ax.plot(best_idx, mean_scores[best_idx], 'r*', markersize=15, label=f'Best ({mean_scores[best_idx]:.4f})')
    
    ax.set_xlabel('Parameter Combination Index', fontsize=12)
    ax.set_ylabel('CV Score (Accuracy)', fontsize=12)
    title = f'{"Randomized" if n_iter else "Grid"} Search Cross-Validation Results'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {os.path.basename(save_path)}")

def plot_feature_importance(model, feature_names, save_path, top_n=20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    
    plt.barh(range(top_n), importances[indices], color=colors)
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
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

def plot_parameter_importance(cv_results, param_grid, save_path):
    results_df = pd.DataFrame(cv_results)
    
    # Analyze each parameter's impact
    param_importance = {}
    for param_name in param_grid.keys():
        param_col = f'param_{param_name}'
        if param_col in results_df.columns:
            # Group by parameter value and get mean score variance
            grouped = results_df.groupby(param_col)['mean_test_score']
            variance = grouped.std().mean()
            param_importance[param_name] = variance
    
    if not param_importance:
        print("[WARN] No parameter importance data available")
        return
    
    # Sort and plot
    params = list(param_importance.keys())
    importances = [param_importance[p] for p in params]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(params)))
    plt.bar(params, importances, color=colors)
    plt.xlabel('Hyperparameter', fontsize=12)
    plt.ylabel('Score Variance (Impact)', fontsize=12)
    plt.title('Hyperparameter Sensitivity Analysis', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {os.path.basename(save_path)}")

def get_feature_names(csv_path):
    df = pd.read_csv(csv_path, nrows=0)
    return df.columns[1:].tolist()  # Skip 'class' column

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
    
    # Map model type to correct filename suffix
    model_suffix = 'rf' if model_type == 'random_forest' else 'xgb'
    
    # Load model
    model_path = os.path.join(version_dir, f'model_{model_suffix}.joblib')
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return False
    
    print("[INFO] Loading model...")
    model = joblib.load(model_path)
    
    # Check if it's a grid search object
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    if isinstance(model, (GridSearchCV, RandomizedSearchCV)):
        grid_search = model
        best_model = model.best_estimator_
    else:
        grid_search = None
        best_model = model
    
    # Load label encoder and scaler
    encoder_path = os.path.join(version_dir, 'label_encoder.joblib')
    scaler_path = os.path.join(version_dir, 'scaler.joblib')
    
    label_encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    class_names = list(label_encoder.classes_) if label_encoder else None
    
    # Load test data from CSV
    csv_path = metadata.get('csv_path')
    if not csv_path or not os.path.exists(csv_path):
        print(f"[WARN] CSV not found, looking for default...")
        csv_path = os.path.join(project_root, 'arnis_poses_angles.csv')
    
    if not os.path.exists(csv_path):
        print("[ERROR] Training CSV not found, cannot generate some plots")
        feature_names = None
        y_test, y_pred = None, None
    else:
        print(f"[INFO] Loading data from: {os.path.basename(csv_path)}")
        data = pd.read_csv(csv_path).dropna()
        
        # Get feature names
        feature_names = get_feature_names(csv_path)
        
        # Extract test data
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        X = data.iloc[:, 1:].values
        y_labels = data.iloc[:, 0].values
        
        le = LabelEncoder()
        y = le.fit_transform(y_labels)
        
        # Reproduce same split (70/10/20)
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        if scaler:
            sc = StandardScaler()
            sc.fit(X_temp)
            X_test = sc.transform(X_test)
        
        # Get predictions
        y_pred = best_model.predict(X_test)
    
    # Generate plots
    print("\n[INFO] Generating visualizations...")
    
    # 1. GridSearch Results (if available)
    if grid_search is not None:
        cv_results = grid_search.cv_results_
        n_iter = metadata.get('n_iter') if 'n_iter' in metadata else None
        plot_gridsearch_results(cv_results, os.path.join(version_dir, 'gridsearch_results.png'), n_iter)
        
        # 4. Parameter Importance
        param_grid = metadata.get('param_grid', {})
        if param_grid:
            plot_parameter_importance(cv_results, param_grid, os.path.join(version_dir, 'param_importance.png'))
    else:
        print("[WARN] Model is not a GridSearch object, skipping CV plots")
    
    # 2. Feature Importance
    if feature_names is not None:
        plot_feature_importance(best_model, feature_names, os.path.join(version_dir, 'feature_importance.png'))
    
    # 3. Confusion Matrix
    if y_test is not None and y_pred is not None and class_names is not None:
        plot_confusion_matrix_heatmap(y_test, y_pred, class_names, os.path.join(version_dir, 'confusion_matrix.png'))
    
    print(f"\n{'='*60}")
    print("  VISUALIZATION GENERATION COMPLETE")
    print(f"{'='*60}\n")
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualizations for trained RF/XGBoost models')
    parser.add_argument('--version', '-v', type=str, help='Model version directory name (e.g., v018_ang3_rf)')
    parser.add_argument('--all', '-a', action='store_true', help='Generate for all RF/XGBoost models')
    
    args = parser.parse_args()
    
    models_dir = os.path.join(project_root, 'models')
    
    if args.all:
        # Generate for all models
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
