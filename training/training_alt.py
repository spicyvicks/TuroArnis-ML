"""
Random Forest and XGBoost training for pose classification
Alternative architectures to DNN
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import threading


class GridSearchProgress:
    """Thread-safe progress tracker for GridSearchCV/RandomizedSearchCV"""
    def __init__(self, total_fits, desc="Grid Search"):
        self.total = total_fits
        self.pbar = tqdm(total=total_fits, desc=f"  {desc}", 
                         bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        self.count = 0
        self._lock = threading.Lock()  # Thread-safe lock for n_jobs > 1
    
    def update(self):
        with self._lock:
            self.count += 1
            self.pbar.update(1)
    
    def close(self):
        self.pbar.close()


# Global progress tracker
_progress_tracker = None


def _scoring_with_progress(estimator, X, y):
    """Custom scoring function that updates progress bar (thread-safe)
    
    Note: This function has the signature (estimator, X, y) which means it
    should be passed directly to GridSearchCV, NOT wrapped with make_scorer.
    """
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

def get_next_version(models_dir):
    """get next version number"""
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

def train_random_forest(csv_path, models_dir, model_name=None):
    """train random forest classifier"""
    print("\n" + "="*50)
    print("  RANDOM FOREST TRAINING")
    print("="*50)
    
    # load data
    data = pd.read_csv(csv_path).dropna()
    X = data.iloc[:, 1:].values
    y_labels = data.iloc[:, 0].values
    
    # encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    class_names = list(label_encoder.classes_)
    
    print(f"\n  Classes: {len(class_names)}")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    
    # split data: 70% train, 10% val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp)
    
    print(f"  Split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    
    # scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # combine train and val for grid search
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.hstack([y_train, y_val])
    
    # enhanced hyperparameters with grid search
    global _progress_tracker
    from sklearn.model_selection import GridSearchCV
    
    # base model
    rf_base = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        oob_score=True,                # out-of-bag score
        bootstrap=True                 # use bootstrap sampling
    )
    
    # parameter grid for tuning (30 combinations = 90 fits)
    param_grid = {
        'n_estimators': [200, 300, 400],        # 3 options
        'max_depth': [10, 15, 20],              # 3 options
        'min_samples_split': [2, 5],            # 2 options
        'criterion': ['gini']                   # 1 option (gini is faster)
    }
    # Total: 3 × 3 × 2 × 1 = 18 combinations... let's add more
    # Actually: using 5 × 3 × 2 = 30 combinations
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],  # 5 options
        'max_depth': [10, 15, 20],                  # 3 options
        'min_samples_split': [2, 5],                # 2 options
    }
    # Total: 5 × 3 × 2 = 30 combinations × 3 folds = 90 fits
    
    # calculate total fits for progress bar
    n_combinations = np.prod([len(v) for v in param_grid.values()])
    cv_folds = 3
    total_fits = n_combinations * cv_folds  # 24 × 3 = 72 fits
    
    print("\n  Performing Grid Search...")
    print(f"  Testing {n_combinations} parameter combinations ({total_fits} total fits)")
    print()
    
    # initialize progress tracker
    _progress_tracker = GridSearchProgress(total_fits, desc="RF Grid Search")
    
    # grid search with cross-validation and progress tracking
    grid_search = GridSearchCV(
        rf_base,
        param_grid,
        cv=cv_folds,
        scoring=_scoring_with_progress,  # custom scorer with progress
        n_jobs=2,                         # parallelism with thread-safe progress
        verbose=0                         # disable default verbose
    )
    
    try:
        grid_search.fit(X_train_full, y_train_full)
    finally:
        _progress_tracker.close()
        _progress_tracker = None
    
    print(f"\n  Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"    {param}: {value}")
    print(f"  Best CV Score: {grid_search.best_score_*100:.2f}%")
    
    # use best model
    model = grid_search.best_estimator_
    
    # evaluate on test set (not used in grid search)
    train_acc = accuracy_score(y_train_full, model.predict(X_train_full))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    val_acc = grid_search.best_score_  # use CV score as val
    
    print(f"\n  Train Accuracy: {train_acc*100:.2f}%")
    print(f"  CV Accuracy:    {val_acc*100:.2f}%")
    print(f"  Test Accuracy:  {test_acc*100:.2f}%")
    
    # save model
    version_num = get_next_version(models_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_name:
        version_name = f"v{version_num:03d}_{model_name}_rf"
    else:
        version_name = f"v{version_num:03d}_{timestamp}_rf"
    version_dir = os.path.join(models_dir, version_name)
    os.makedirs(version_dir, exist_ok=True)
    
    model_path = os.path.join(version_dir, 'model_rf.joblib')
    encoder_path = os.path.join(version_dir, 'label_encoder.joblib')
    scaler_path = os.path.join(version_dir, 'scaler.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    joblib.dump(scaler, scaler_path)
    
    # save metadata
    metadata = {
        'version': version_name,
        'model_type': 'random_forest',
        'trained_at': datetime.now().isoformat(),
        'test_accuracy': float(test_acc),
        'val_accuracy': float(val_acc),
        'num_classes': len(class_names),
        'num_features': X.shape[1],
        'train_samples': len(X_train),
        'class_names': class_names,
        'hyperparameters': {
            'n_estimators': 200,
            'max_depth': 20
        }
    }
    with open(os.path.join(version_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  Saved to: {version_name}")
    
    # update active model if better
    update_active_model(models_dir, version_name, version_dir, model_path, encoder_path, scaler_path, test_acc, 'random_forest')
    
    return test_acc, version_name


def train_xgboost(csv_path, models_dir, model_name=None):
    """train xgboost classifier"""
    if not HAS_XGBOOST:
        print("\n[ERROR] XGBoost not installed. Run: pip install xgboost")
        return None, None
    
    print("\n" + "="*50)
    print("  XGBOOST TRAINING")
    print("="*50)
    
    # load data
    data = pd.read_csv(csv_path).dropna()
    X = data.iloc[:, 1:].values
    y_labels = data.iloc[:, 0].values
    
    # encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    class_names = list(label_encoder.classes_)
    
    print(f"\n  Classes: {len(class_names)}")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    
    # split data: 70% train, 10% val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp)
    
    print(f"  Split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    
    # scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # combine train and val for grid search
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.hstack([y_train, y_val])
    
    # enhanced hyperparameters with grid search
    global _progress_tracker
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    
    # base model - IMPORTANT: n_jobs=1 to avoid conflict with sklearn parallelization
    xgb_base = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(class_names),
        random_state=42,
        n_jobs=1,               # Use 1 here, sklearn handles parallelization
        verbosity=0,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    # parameter grid for tuning (expanded for better search)
    param_grid = {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'min_child_weight': [1, 2, 3, 5],
        'gamma': [0, 0.05, 0.1, 0.2],
        'reg_alpha': [0, 0.01, 0.1, 0.5],
        'reg_lambda': [0.5, 1.0, 1.5, 2.0]
    }
    
    # calculate total fits for progress bar
    n_iter = 30                           # Reduced for faster training
    cv_folds = 3
    total_fits = n_iter * cv_folds        # 30 × 3 = 90 fits
    
    print("\n  Performing Randomized Search...")
    print(f"  Testing {n_iter} random combinations ({total_fits} total fits)")
    print()
    
    # initialize progress tracker
    _progress_tracker = GridSearchProgress(total_fits, desc="XGB Random Search")
    
    random_search = RandomizedSearchCV(
        xgb_base,
        param_grid,
        n_iter=n_iter,
        cv=cv_folds,
        scoring=_scoring_with_progress,  # custom scorer with progress
        n_jobs=2,                         # parallelism with thread-safe progress
        verbose=0,                        # disable default verbose
        random_state=42
    )
    
    try:
        random_search.fit(X_train_full, y_train_full)
    finally:
        _progress_tracker.close()
        _progress_tracker = None
    
    print(f"\n  Best parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"    {param}: {value}")
    print(f"  Best CV Score: {random_search.best_score_*100:.2f}%")
    
    # use best model
    model = random_search.best_estimator_
    
    # evaluate on test set
    train_acc = accuracy_score(y_train_full, model.predict(X_train_full))
    val_acc = random_search.best_score_
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"\n  Train Accuracy: {train_acc*100:.2f}%")
    print(f"  CV Accuracy:    {val_acc*100:.2f}%")
    print(f"  Test Accuracy:  {test_acc*100:.2f}%")
    
    # save model
    version_num = get_next_version(models_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_name:
        version_name = f"v{version_num:03d}_{model_name}_xgb"
    else:
        version_name = f"v{version_num:03d}_{timestamp}_xgb"
    version_dir = os.path.join(models_dir, version_name)
    os.makedirs(version_dir, exist_ok=True)
    
    model_path = os.path.join(version_dir, 'model_xgb.joblib')
    encoder_path = os.path.join(version_dir, 'label_encoder.joblib')
    scaler_path = os.path.join(version_dir, 'scaler.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    joblib.dump(scaler, scaler_path)
    
    # save metadata
    metadata = {
        'version': version_name,
        'model_type': 'xgboost',
        'trained_at': datetime.now().isoformat(),
        'test_accuracy': float(test_acc),
        'val_accuracy': float(val_acc),
        'num_classes': len(class_names),
        'num_features': X.shape[1],
        'train_samples': len(X_train),
        'class_names': class_names,
        'hyperparameters': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1
        }
    }
    with open(os.path.join(version_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  Saved to: {version_name}")
    
    # update active model if better
    update_active_model(models_dir, version_name, version_dir, model_path, encoder_path, scaler_path, test_acc, 'xgboost')
    
    return test_acc, version_name


def update_active_model(models_dir, version_name, version_dir, model_path, encoder_path, scaler_path, test_acc, model_type):
    """update active model only if accuracy is higher"""
    active_model_path = os.path.join(models_dir, 'active_model.json')
    
    should_set_active = True
    if os.path.exists(active_model_path):
        with open(active_model_path, 'r') as f:
            current_active = json.load(f)
        current_metadata_path = os.path.join(current_active['path'], 'metadata.json')
        if os.path.exists(current_metadata_path):
            with open(current_metadata_path, 'r') as f:
                current_metadata = json.load(f)
            current_acc = current_metadata.get('test_accuracy', 0)
            if test_acc <= current_acc:
                should_set_active = False
                print(f"\n  New accuracy ({test_acc*100:.2f}%) <= current ({current_acc*100:.2f}%)")
                print(f"  Keeping {current_active['version']} as active")
    
    if should_set_active:
        active_config = {
            'version': version_name,
            'path': version_dir,
            'model_path': model_path,
            'model_type': model_type,
            'encoder_path': encoder_path,
            'scaler_path': scaler_path,
            'test_accuracy': float(test_acc),
            'set_at': datetime.now().isoformat()
        }
        with open(active_model_path, 'w') as f:
            json.dump(active_config, f, indent=2)
        print(f"\n  [OK] Set as active model ({test_acc*100:.2f}%)")


if __name__ == "__main__":
    models_dir = os.path.join(project_root, 'models')
    csv_path = os.path.join(project_root, 'arnis_poses_angles.csv')
    
    os.makedirs(models_dir, exist_ok=True)
    
    print("\nSelect model to train:")
    print("  1. Random Forest")
    print("  2. XGBoost")
    
    choice = input("\nChoice: ").strip()
    
    if choice == '1':
        train_random_forest(csv_path, models_dir)
    elif choice == '2':
        train_xgboost(csv_path, models_dir)
    else:
        print("Invalid choice")
