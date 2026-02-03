"""
Random Forest and XGBoost training for pose classification
Alternative architectures to DNN.
Consumes: features_train.csv, features_test.csv (validation omitted)
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import threading

# Thread-safe progress bar logic (kept from original)
class GridSearchProgress:
    def __init__(self, total_fits, desc="Grid Search"):
        self.total = total_fits
        self.pbar = tqdm(total=total_fits, desc=f"  {desc}", 
                         bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        self.count = 0
        self._lock = threading.Lock()
    
    def update(self):
        with self._lock:
            self.count += 1
            self.pbar.update(1)
    
    def close(self):
        self.pbar.close()

_progress_tracker = None

def _scoring_with_progress(estimator, X, y):
    global _progress_tracker
    score = accuracy_score(y, estimator.predict(X))
    if _progress_tracker:
        _progress_tracker.update()
    return score

# optional: xgboost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARN] XGBoost not installed. Run: pip install xgboost")

# paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

csv_train_file = os.path.join(project_root, 'features_train.csv')
csv_test_file = os.path.join(project_root, 'features_test.csv')
models_dir = os.path.join(project_root, 'models')

def get_next_version(models_dir):
    existing = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('v')]
    if not existing:
        return 1
    versions = []
    for d in existing:
        try:
            versions.append(int(d.split('_')[0][1:]))
        except:
            pass
    return max(versions) + 1 if versions else 1

def load_and_prepare_data():
    """Load data from train and test CSVs (validation omitted for RF/XGB)"""
    print("  Loading CSVs...")
    if not os.path.exists(csv_train_file):
        print(f"[ERROR] Missing {csv_train_file}. Run run_extraction.py first.")
        sys.exit(1)
        
    train = pd.read_csv(csv_train_file).dropna()
    test = pd.read_csv(csv_test_file).dropna()
    
    # Get feature names (all columns except 'class')
    feature_names = [col for col in train.columns if col != 'class']
    
    # Fit encoder on training classes
    le = LabelEncoder()
    le.fit(train['class'].values)
    class_names = list(le.classes_)
    
    def prep(df):
        df = df[df['class'].isin(class_names)]
        X = df.drop('class', axis=1).values
        y = le.transform(df['class'].values)
        return X, y
        
    X_train, y_train = prep(train)
    X_test, y_test = prep(test)
    
    print(f"  Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(feature_names)}")
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return (X_train, y_train), (X_test, y_test), scaler, le, class_names, feature_names

def select_features(X_train, y_train, X_test, feature_names, keep_ratio=0.6):
    """Auto-select top features using RF importance. Returns reduced X and selected feature info."""
    print("\n  [STEP 1] AUTO-FEATURE SELECTION")
    print(f"  Original features: {len(feature_names)}")
    
    # Quick RF to get importances
    quick_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    quick_rf.fit(X_train, y_train)
    
    importances = quick_rf.feature_importances_
    n_keep = max(10, int(len(feature_names) * keep_ratio))  # Keep at least 10 features
    
    # Get indices of top features
    top_indices = np.argsort(importances)[::-1][:n_keep]
    top_indices_sorted = np.sort(top_indices)  # Keep original order for consistency
    
    # Select features
    X_train_sel = X_train[:, top_indices_sorted]
    X_test_sel = X_test[:, top_indices_sorted]
    selected_names = [feature_names[i] for i in top_indices_sorted]
    
    # Build importance ranking for all features (for visualization)
    all_importance_data = []
    for i, (name, imp) in enumerate(zip(feature_names, importances)):
        all_importance_data.append({
            'feature': name,
            'importance': float(imp),
            'selected': i in top_indices_sorted,
            'rank': int(np.where(np.argsort(importances)[::-1] == i)[0][0]) + 1
        })
    
    # Show dropped features
    dropped_count = len(feature_names) - n_keep
    print(f"  Kept: {n_keep} | Dropped: {dropped_count} (bottom {100-int(keep_ratio*100)}%)")
    
    # Show top 5 kept and bottom 5 dropped
    sorted_by_imp = sorted(all_importance_data, key=lambda x: x['importance'], reverse=True)
    print("  Top 5 kept:")
    for f in sorted_by_imp[:5]:
        print(f"    - {f['feature']}: {f['importance']:.4f}")
    print("  Bottom 5 dropped:")
    for f in sorted_by_imp[-5:]:
        if not f['selected']:
            print(f"    - {f['feature']}: {f['importance']:.4f}")
    
    return X_train_sel, X_test_sel, selected_names, all_importance_data

def train_random_forest(data_pack, use_feature_selection=True, keep_ratio=0.6):
    (X_train, y_train), (X_test, y_test), scaler, le, class_names, feature_names = data_pack
    
    print("\n" + "="*50)
    print("  RANDOM FOREST TRAINING")
    print("="*50)
    
    # Feature Selection Step
    all_importance_data = None
    selected_feature_names = feature_names
    if use_feature_selection and len(feature_names) > 15:
        X_train, X_test, selected_feature_names, all_importance_data = select_features(
            X_train, y_train, X_test, feature_names, keep_ratio
        )
    
    print(f"\n  [STEP 2] GRID/RANDOM SEARCH (on {len(selected_feature_names)} features)")
    
    from sklearn.model_selection import RandomizedSearchCV
    
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    
    # Expanded grid
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 15, 20, 25],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    # Use RandomizedSearch to keep it efficient despite larger grid
    n_iter = 20
    cv_folds = 5
    total_fits = n_iter * cv_folds
    
    print(f"  Random Search: {total_fits} fits (n_iter={n_iter}, cv={cv_folds})...")
    
    global _progress_tracker
    _progress_tracker = GridSearchProgress(total_fits, desc="RF Random Search")
    
    grid_search = RandomizedSearchCV(
        rf_base, param_grid, n_iter=n_iter, cv=cv_folds, scoring=_scoring_with_progress, 
        n_jobs=2, verbose=0, random_state=42
    )
    
    try:
        grid_search.fit(X_train, y_train)
    finally:
        _progress_tracker.close()
        _progress_tracker = None
        
    print(f"\n  Best Params: {grid_search.best_params_}")
    print(f"  Best CV Acc: {grid_search.best_score_:.2%}")
    
    model = grid_search.best_estimator_
    
    # Evaluate on pure Test set
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  Test Acc:    {test_acc:.2%}")
    
    # Show top 10 features from final model
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("\n  Top 10 Features (final model):")
        for f in range(min(10, len(indices))):
            print(f"    {f+1}. {selected_feature_names[indices[f]]}: {importances[indices[f]]:.4f}")
    
    # Build final importance data for selected features
    final_importance_data = []
    for i, name in enumerate(selected_feature_names):
        final_importance_data.append({
            'feature': name,
            'importance': float(model.feature_importances_[i])
        })
    
    save_model(
        model, scaler, le, class_names, test_acc, 'random_forest', 
        grid_search.best_params_, selected_feature_names,
        all_importance_data, final_importance_data
    )

def train_xgboost(data_pack, use_feature_selection=True, keep_ratio=0.6):
    if not HAS_XGBOOST: return
    (X_train, y_train), (X_test, y_test), scaler, le, class_names, feature_names = data_pack

    print("\n" + "="*50)
    print("  XGBOOST TRAINING")
    print("="*50)
    
    # Feature Selection Step
    all_importance_data = None
    selected_feature_names = feature_names
    if use_feature_selection and len(feature_names) > 15:
        X_train, X_test, selected_feature_names, all_importance_data = select_features(
            X_train, y_train, X_test, feature_names, keep_ratio
        )
    
    print(f"\n  [STEP 2] RANDOM SEARCH (on {len(selected_feature_names)} features)")
    
    from sklearn.model_selection import RandomizedSearchCV
    
    xgb_base = xgb.XGBClassifier(objective='multi:softmax', num_class=len(class_names), 
                                 random_state=42, n_jobs=1, verbosity=0, use_label_encoder=False)
    
    # Expanded grid
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    n_iter = 20
    cv_folds = 5
    total_fits = n_iter * cv_folds
    
    print(f"  Random Search: {total_fits} fits (n_iter={n_iter}, cv={cv_folds})...")
    
    global _progress_tracker
    _progress_tracker = GridSearchProgress(total_fits, desc="XGB Search")
    
    search = RandomizedSearchCV(
        xgb_base, param_grid, n_iter=n_iter, cv=cv_folds, 
        scoring=_scoring_with_progress, n_jobs=2, verbose=0, random_state=42
    )
    
    try:
        search.fit(X_train, y_train)
    finally:
        _progress_tracker.close()
        _progress_tracker = None
        
    print(f"\n  Best Params: {search.best_params_}")
    print(f"  Best CV Acc: {search.best_score_:.2%}")
    
    model = search.best_estimator_
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  Test Acc:    {test_acc:.2%}")
    
    # Show top 10 features from final model
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("\n  Top 10 Features (final model):")
        for f in range(min(10, len(indices))):
            print(f"    {f+1}. {selected_feature_names[indices[f]]}: {importances[indices[f]]:.4f}")
    
    # Build final importance data for selected features
    final_importance_data = []
    for i, name in enumerate(selected_feature_names):
        final_importance_data.append({
            'feature': name,
            'importance': float(model.feature_importances_[i])
        })
    
    save_model(
        model, scaler, le, class_names, test_acc, 'xgboost', 
        search.best_params_, selected_feature_names,
        all_importance_data, final_importance_data
    )

def save_model(model, scaler, le, class_names, test_acc, model_type, params,
               selected_features=None, feature_selection_data=None, final_importance_data=None):
    version_num = get_next_version(models_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = f"v{version_num:03d}_{timestamp}_{model_type}"
    version_dir = os.path.join(models_dir, version_name)
    os.makedirs(version_dir, exist_ok=True)
    
    ext = 'rf' if model_type == 'random_forest' else 'xgb'
    model_path = os.path.join(version_dir, f'model_{ext}.joblib')
    encoder_path = os.path.join(version_dir, 'label_encoder.joblib')
    scaler_path = os.path.join(version_dir, 'scaler.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)
    joblib.dump(scaler, scaler_path)
    
    # Save selected feature indices for inference
    if selected_features:
        with open(os.path.join(version_dir, 'selected_features.json'), 'w') as f:
            json.dump(selected_features, f, indent=2)
    
    # Save feature selection data for visualization
    if feature_selection_data:
        with open(os.path.join(version_dir, 'feature_selection_data.json'), 'w') as f:
            json.dump(feature_selection_data, f, indent=2)
    
    # Save final importance data for visualization
    if final_importance_data:
        with open(os.path.join(version_dir, 'final_importance_data.json'), 'w') as f:
            json.dump(final_importance_data, f, indent=2)
    
    metadata = {
        'version': version_name,
        'model_type': model_type,
        'trained_at': datetime.now().isoformat(),
        'test_accuracy': float(test_acc),
        'hyperparameters': params,
        'class_names': class_names,
        'num_features': len(selected_features) if selected_features else None,
        'feature_selection_used': feature_selection_data is not None
    }
    with open(os.path.join(version_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\n  [SAVED] {version_name}")
    
    # Active Model Logic
    active_path = os.path.join(models_dir, 'active_model.json')
    should_update = True
    if os.path.exists(active_path):
        try:
            with open(active_path, 'r') as f: cur = json.load(f)
            if test_acc <= cur.get('test_accuracy', 0): should_update = False
        except: pass
        
    if should_update:
        active_rec = {
            'version': version_name,
            'path': version_dir,
            'model_path': model_path,
            'model_type': model_type,
            'encoder_path': encoder_path,
            'scaler_path': scaler_path,
            'selected_features_path': os.path.join(version_dir, 'selected_features.json') if selected_features else None,
            'test_accuracy': float(test_acc),
            'set_at': datetime.now().isoformat()
        }
        with open(active_path, 'w') as f: json.dump(active_rec, f, indent=2)
        print(f"  [ACTIVE] Set as active model.")

if __name__ == "__main__":
    os.makedirs(models_dir, exist_ok=True)
    print("\nSelect model to train:")
    print("  1. Random Forest")
    print("  2. XGBoost")
    c = input("\nChoice: ").strip()
    
    data_pack = load_and_prepare_data()
    
    if c == '1': train_random_forest(data_pack)
    elif c == '2': train_xgboost(data_pack)
    else: print("Invalid choice")
