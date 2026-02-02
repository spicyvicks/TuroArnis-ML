"""
Random Forest and XGBoost training for pose classification
Alternative architectures to DNN.
Consumes: features_train.csv, features_val.csv, features_test.csv
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
csv_val_file = os.path.join(project_root, 'features_val.csv')
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
    """Load data from the 3 CSVs and prepare X, y"""
    print("  Loading CSVs...")
    if not os.path.exists(csv_train_file):
        print(f"[ERROR] Missing {csv_train_file}. Run run_extraction.py first.")
        sys.exit(1)
        
    train = pd.read_csv(csv_train_file).dropna()
    val = pd.read_csv(csv_val_file).dropna()
    test = pd.read_csv(csv_test_file).dropna()
    
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
    X_val, y_val = prep(val)
    X_test, y_test = prep(test)
    
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, le, class_names

def train_random_forest(data_pack):
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, le, class_names = data_pack
    
    print("\n" + "="*50)
    print("  RANDOM FOREST TRAINING")
    print("="*50)
    
    # GridSearch needs a single training set, usually we combine Train+Val for CV, or just use Train
    # Here we will use Train + Val for the GridSearch to maximize data
    X_full = np.vstack([X_train, X_val])
    y_full = np.hstack([y_train, y_val])
    
    from sklearn.model_selection import GridSearchCV
    
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
    }
    
    n_combinations = np.prod([len(v) for v in param_grid.values()])
    cv_folds = 3
    total_fits = n_combinations * cv_folds
    
    print(f"\n  Grid Search: {total_fits} fits...")
    
    global _progress_tracker
    _progress_tracker = GridSearchProgress(total_fits, desc="RF Grid Search")
    
    grid_search = GridSearchCV(
        rf_base, param_grid, cv=cv_folds, scoring=_scoring_with_progress, 
        n_jobs=2, verbose=0
    )
    
    try:
        grid_search.fit(X_full, y_full)
    finally:
        _progress_tracker.close()
        _progress_tracker = None
        
    print(f"\n  Best Params: {grid_search.best_params_}")
    print(f"  Best CV Acc: {grid_search.best_score_:.2%}")
    
    model = grid_search.best_estimator_
    
    # Evaluate on pure Test set
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  Test Acc:    {test_acc:.2%}")
    
    save_model(model, scaler, le, class_names, test_acc, 'random_forest', grid_search.best_params_)

def train_xgboost(data_pack):
    if not HAS_XGBOOST: return
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, le, class_names = data_pack

    print("\n" + "="*50)
    print("  XGBOOST TRAINING")
    print("="*50)
    
    X_full = np.vstack([X_train, X_val])
    y_full = np.hstack([y_train, y_val])
    
    from sklearn.model_selection import RandomizedSearchCV
    
    xgb_base = xgb.XGBClassifier(objective='multi:softmax', num_class=len(class_names), 
                                 random_state=42, n_jobs=1, verbosity=0, use_label_encoder=False)
    
    param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8]
    }
    
    n_iter = 20
    cv_folds = 3
    total_fits = n_iter * cv_folds
    
    print(f"\n  Random Search: {total_fits} fits...")
    
    global _progress_tracker
    _progress_tracker = GridSearchProgress(total_fits, desc="XGB Search")
    
    search = RandomizedSearchCV(
        xgb_base, param_grid, n_iter=n_iter, cv=cv_folds, 
        scoring=_scoring_with_progress, n_jobs=2, verbose=0, random_state=42
    )
    
    try:
        search.fit(X_full, y_full)
    finally:
        _progress_tracker.close()
        _progress_tracker = None
        
    print(f"\n  Best Params: {search.best_params_}")
    print(f"  Best CV Acc: {search.best_score_:.2%}")
    
    model = search.best_estimator_
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  Test Acc:    {test_acc:.2%}")
    
    save_model(model, scaler, le, class_names, test_acc, 'xgboost', search.best_params_)

def save_model(model, scaler, le, class_names, test_acc, model_type, params):
    version_num = get_next_version(models_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = f"v{version_num:03d}_{timestamp}_{model_type}" # e.g. v005_20240101_random_forest
    version_dir = os.path.join(models_dir, version_name)
    os.makedirs(version_dir, exist_ok=True)
    
    ext = 'rf' if model_type == 'random_forest' else 'xgb'
    model_path = os.path.join(version_dir, f'model_{ext}.joblib')
    encoder_path = os.path.join(version_dir, 'label_encoder.joblib')
    scaler_path = os.path.join(version_dir, 'scaler.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)
    joblib.dump(scaler, scaler_path)
    
    metadata = {
        'version': version_name,
        'model_type': model_type,
        'trained_at': datetime.now().isoformat(),
        'test_accuracy': float(test_acc),
        'hyperparameters': params,
        'class_names': class_names
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
