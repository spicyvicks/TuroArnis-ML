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

def load_and_prepare_data(use_cross_validation=False):
    """Load data from train and test CSVs.
    
    Args:
        use_cross_validation: If True, combines train+test for cross-validation
    """
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
    
    if use_cross_validation:
        # Combine train + test for cross-validation
        combined = pd.concat([train, test], ignore_index=True)
        X_all, y_all = prep(combined)
        print(f"  Combined data: {len(X_all)} samples | Features: {len(feature_names)}")
        print(f"  Using 5-fold stratified cross-validation")
        
        # Scale all data
        scaler = StandardScaler()
        X_all = scaler.fit_transform(X_all)
        
        # Return None for test set to indicate CV mode
        return (X_all, y_all), (None, None), scaler, le, class_names, feature_names
    else:
        X_train, y_train = prep(train)
        X_test, y_test = prep(test)
        
        print(f"  Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(feature_names)}")
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return (X_train, y_train), (X_test, y_test), scaler, le, class_names, feature_names

def select_features(X_train, y_train, X_test, feature_names, keep_ratio=0.6, protect_stick=True):
    """Auto-select top features using RF importance. Returns reduced X and selected feature info.
    
    Args:
        protect_stick: If True, stick features are always kept regardless of importance.
    """
    print("\n  [STEP 1] AUTO-FEATURE SELECTION")
    print(f"  Original features: {len(feature_names)}")
    
    # Protected stick features (always kept regardless of importance)
    PROTECTED_STICK_FEATURES = [
        'stick_detected', 'stick_length_norm', 'stick_angle',
        'grip_x_norm', 'grip_y_norm', 'tip_x_norm', 'tip_y_norm',
        'grip_to_holding_wrist', 'holding_hand',
        'tip_vs_shoulder_y', 'tip_vs_hip_x', 'tip_vs_hip_y',
        'stick_conf', 'keypoint_conf',
        # Strike target features
        'tip_to_left_eye', 'tip_to_right_eye', 'tip_to_crown',
        'tip_to_chest', 'tip_to_solar',
        'tip_to_left_knee', 'tip_to_right_knee',
        'stick_dir_x', 'stick_dir_y', 'tip_height_vs_body',
        # Laterality features (critical for left/right differentiation)
        'dominant_arm', 'dominant_elbow_height', 'dominant_wrist_forward',
        'dominant_hand_height', 'active_wrist_x', 'arm_ext_diff'
    ]
    
    # Find indices of protected features (only if protect_stick is enabled)
    protected_indices = set()
    if protect_stick:
        for i, name in enumerate(feature_names):
            if name in PROTECTED_STICK_FEATURES:
                protected_indices.add(i)
        print(f"  Protected stick features: {len(protected_indices)}")
    else:
        print(f"  Stick feature protection: DISABLED")
    
    # Quick RF to get importances
    quick_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    quick_rf.fit(X_train, y_train)
    
    importances = quick_rf.feature_importances_
    
    # Calculate how many non-protected features to keep
    n_non_protected = len(feature_names) - len(protected_indices)
    n_keep_non_protected = max(10, int(n_non_protected * keep_ratio))
    
    # Get indices of top non-protected features
    non_protected_indices = [i for i in range(len(feature_names)) if i not in protected_indices]
    non_protected_importances = [(i, importances[i]) for i in non_protected_indices]
    non_protected_importances.sort(key=lambda x: x[1], reverse=True)
    top_non_protected = set([idx for idx, _ in non_protected_importances[:n_keep_non_protected]])
    
    # Combine protected + top non-protected
    final_indices = protected_indices | top_non_protected
    final_indices_sorted = np.sort(list(final_indices))
    
    # Select features
    X_train_sel = X_train[:, final_indices_sorted]
    X_test_sel = X_test[:, final_indices_sorted]
    selected_names = [feature_names[i] for i in final_indices_sorted]
    
    # Build importance ranking for all features (for visualization)
    all_importance_data = []
    for i, (name, imp) in enumerate(zip(feature_names, importances)):
        all_importance_data.append({
            'feature': name,
            'importance': float(imp),
            'selected': i in final_indices_sorted,
            'protected': i in protected_indices,
            'rank': int(np.where(np.argsort(importances)[::-1] == i)[0][0]) + 1
        })
    
    # Show summary
    dropped_count = len(feature_names) - len(final_indices_sorted)
    print(f"  Kept: {len(final_indices_sorted)} ({len(protected_indices)} protected + {len(top_non_protected)} by importance)")
    print(f"  Dropped: {dropped_count} (bottom {100-int(keep_ratio*100)}% of non-protected)")
    
    # Show top 5 kept and bottom 5 dropped
    sorted_by_imp = sorted(all_importance_data, key=lambda x: x['importance'], reverse=True)
    print("  Top 5 by importance:")
    for f in sorted_by_imp[:5]:
        prot = " [PROTECTED]" if f['protected'] else ""
        print(f"    - {f['feature']}: {f['importance']:.4f}{prot}")
    print("  Bottom 5 dropped:")
    dropped_features = [f for f in sorted_by_imp if not f['selected']]
    for f in dropped_features[-5:]:
        print(f"    - {f['feature']}: {f['importance']:.4f}")
    
    return X_train_sel, X_test_sel, selected_names, all_importance_data

def train_random_forest(data_pack, use_feature_selection=True, keep_ratio=0.6, protect_stick=True, use_cv=False, hybrid_mode=False):
    (X_train, y_train), (X_test, y_test), scaler, le, class_names, feature_names = data_pack
    
    print("\n" + "="*50)
    print("  RANDOM FOREST TRAINING")
    if use_cv:
        print("  Mode: CROSS-VALIDATION (5-fold on all data)")
    elif hybrid_mode:
        print("  Mode: HYBRID (CV on train + held-out test)")
    else:
        print("  Mode: TRAIN/TEST SPLIT")
    print("="*50)
    
    # Store original feature count
    n_features_original = X_train.shape[1]
    
    # Feature Selection Step
    all_importance_data = None
    selected_feature_names = feature_names
    X_selected = X_train
    X_test_selected = X_test
    if use_feature_selection and len(feature_names) > 15:
        if use_cv:
            # CV mode: select on all data
            X_selected, _, selected_feature_names, all_importance_data = select_features(
                X_train, y_train, X_train, feature_names, keep_ratio, protect_stick
            )
        else:
            # Split or hybrid mode: select on train, apply to test
            X_train, X_test_selected, selected_feature_names, all_importance_data = select_features(
                X_train, y_train, X_test, feature_names, keep_ratio, protect_stick
            )
            X_selected = X_train
    
    print(f"\n  [STEP 2] GRID/RANDOM SEARCH (on {len(selected_feature_names)} features)")
    
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
    
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
    
    # Use stratified k-fold for consistency
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    grid_search = RandomizedSearchCV(
        rf_base, param_grid, n_iter=n_iter, cv=skf, scoring=_scoring_with_progress, 
        n_jobs=2, verbose=0, random_state=42
    )
    
    try:
        grid_search.fit(X_selected, y_train)
    finally:
        _progress_tracker.close()
        _progress_tracker = None
        
    print(f"\n  Best Params: {grid_search.best_params_}")
    print(f"  Best CV Acc: {grid_search.best_score_:.2%}")
    
    model = grid_search.best_estimator_
    
    # Final evaluation
    if use_cv:
        # For CV mode, do a final 5-fold cross-validation with the best model
        print("\n  [STEP 3] FINAL CROSS-VALIDATION (5-fold)")
        final_cv_scores = cross_val_score(model, X_selected, y_train, cv=skf, n_jobs=-1)
        cv_acc = final_cv_scores.mean()
        cv_std = final_cv_scores.std()
        print(f"  CV Accuracy:  {cv_acc:.2%} ± {cv_std:.2%}")
        print(f"  Fold scores:  {[f'{s:.2%}' for s in final_cv_scores]}")
        
        # Train final model on ALL data
        print("\n  [STEP 4] TRAINING FINAL MODEL ON ALL DATA")
        model.fit(X_selected, y_train)
        test_acc = cv_acc  # Use CV accuracy as the reported accuracy
    elif hybrid_mode:
        # HYBRID: CV on training data + final evaluation on held-out test
        print("\n  [STEP 3] CROSS-VALIDATION ON TRAINING DATA (5-fold)")
        final_cv_scores = cross_val_score(model, X_selected, y_train, cv=skf, n_jobs=-1)
        cv_acc = final_cv_scores.mean()
        cv_std = final_cv_scores.std()
        print(f"  CV Accuracy:  {cv_acc:.2%} ± {cv_std:.2%}")
        print(f"  Fold scores:  {[f'{s:.2%}' for s in final_cv_scores]}")
        
        # Evaluate on held-out test set
        print("\n  [STEP 4] HELD-OUT TEST SET EVALUATION")
        if X_test_selected is not None and len(X_test_selected) > 0:
            test_acc = accuracy_score(y_test, model.predict(X_test_selected))
            print(f"  Test Acc:     {test_acc:.2%} (on {len(y_test)} held-out samples)")
            print(f"\n  📊 SUMMARY:")
            print(f"     CV Accuracy (reliable):  {cv_acc:.2%} ± {cv_std:.2%}")
            print(f"     Test Accuracy (verify):  {test_acc:.2%}")
            # Use CV accuracy as the primary metric (more reliable)
            test_acc = cv_acc
        else:
            test_acc = cv_acc
    else:
        # Traditional train/test split evaluation
        test_acc = accuracy_score(y_test, model.predict(X_test_selected if use_feature_selection else X_test))
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
        all_importance_data, final_importance_data, n_features_original
    )

def train_xgboost(data_pack, use_feature_selection=True, keep_ratio=0.6, protect_stick=True, use_cv=False, hybrid_mode=False):
    if not HAS_XGBOOST: return
    (X_train, y_train), (X_test, y_test), scaler, le, class_names, feature_names = data_pack

    print("\n" + "="*50)
    print("  XGBOOST TRAINING")
    if use_cv:
        print("  Mode: CROSS-VALIDATION (5-fold on all data)")
    elif hybrid_mode:
        print("  Mode: HYBRID (CV on train + held-out test)")
    else:
        print("  Mode: TRAIN/TEST SPLIT")
    print("="*50)
    
    # Store original feature count
    n_features_original = X_train.shape[1]
    
    # Feature Selection Step
    all_importance_data = None
    selected_feature_names = feature_names
    X_selected = X_train
    X_test_selected = X_test
    if use_feature_selection and len(feature_names) > 15:
        if use_cv:
            X_selected, _, selected_feature_names, all_importance_data = select_features(
                X_train, y_train, X_train, feature_names, keep_ratio, protect_stick
            )
        else:
            X_train, X_test_selected, selected_feature_names, all_importance_data = select_features(
                X_train, y_train, X_test, feature_names, keep_ratio, protect_stick
            )
            X_selected = X_train
    
    print(f"\n  [STEP 2] RANDOM SEARCH (on {len(selected_feature_names)} features)")
    
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
    
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
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    search = RandomizedSearchCV(
        xgb_base, param_grid, n_iter=n_iter, cv=skf, 
        scoring=_scoring_with_progress, n_jobs=2, verbose=0, random_state=42
    )
    
    try:
        search.fit(X_selected, y_train)
    finally:
        _progress_tracker.close()
        _progress_tracker = None
        
    print(f"\n  Best Params: {search.best_params_}")
    print(f"  Best CV Acc: {search.best_score_:.2%}")
    
    model = search.best_estimator_
    
    # Final evaluation
    if use_cv:
        print("\n  [STEP 3] FINAL CROSS-VALIDATION (5-fold)")
        final_cv_scores = cross_val_score(model, X_selected, y_train, cv=skf, n_jobs=-1)
        cv_acc = final_cv_scores.mean()
        cv_std = final_cv_scores.std()
        print(f"  CV Accuracy:  {cv_acc:.2%} ± {cv_std:.2%}")
        print(f"  Fold scores:  {[f'{s:.2%}' for s in final_cv_scores]}")
        
        print("\n  [STEP 4] TRAINING FINAL MODEL ON ALL DATA")
        model.fit(X_selected, y_train)
        test_acc = cv_acc
    elif hybrid_mode:
        # HYBRID: CV on training data + final evaluation on held-out test
        print("\n  [STEP 3] CROSS-VALIDATION ON TRAINING DATA (5-fold)")
        final_cv_scores = cross_val_score(model, X_selected, y_train, cv=skf, n_jobs=-1)
        cv_acc = final_cv_scores.mean()
        cv_std = final_cv_scores.std()
        print(f"  CV Accuracy:  {cv_acc:.2%} ± {cv_std:.2%}")
        print(f"  Fold scores:  {[f'{s:.2%}' for s in final_cv_scores]}")
        
        # Evaluate on held-out test set
        print("\n  [STEP 4] HELD-OUT TEST SET EVALUATION")
        if X_test_selected is not None and len(X_test_selected) > 0:
            test_acc = accuracy_score(y_test, model.predict(X_test_selected))
            print(f"  Test Acc:     {test_acc:.2%} (on {len(y_test)} held-out samples)")
            print(f"\n  📊 SUMMARY:")
            print(f"     CV Accuracy (reliable):  {cv_acc:.2%} ± {cv_std:.2%}")
            print(f"     Test Accuracy (verify):  {test_acc:.2%}")
            test_acc = cv_acc  # Use CV accuracy as the primary metric
        else:
            test_acc = cv_acc
    else:
        test_acc = accuracy_score(y_test, model.predict(X_test_selected if use_feature_selection else X_test))
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
        all_importance_data, final_importance_data, n_features_original
    )

def save_model(model, scaler, le, class_names, test_acc, model_type, params,
               selected_features=None, feature_selection_data=None, final_importance_data=None, n_features_in=None):
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
        'n_features_in': n_features_in,  # Total features before selection
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
    
    # Ask about feature selection
    print("\nFeature Selection:")
    print("  1. Use feature selection (keep 60% of features, protect stick features)")
    print("  2. Use all features (no selection)")
    fs_choice = input("\nChoice [1/2]: ").strip()
    use_feature_selection = (fs_choice != '2')
    
    if use_feature_selection:
        print("✓ Feature selection enabled (60% kept, stick features protected)")
    else:
        print("✓ Using all features")
    
    data_pack = load_and_prepare_data()
    
    if c == '1': train_random_forest(data_pack, use_feature_selection=use_feature_selection)
    elif c == '2': train_xgboost(data_pack, use_feature_selection=use_feature_selection)
    else: print("Invalid choice")
