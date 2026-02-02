

import os
import sys
import json
import shutil
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

MODELS_DIR = os.path.join(project_root, 'models')


class EnsembleClassifier:
    
    def __init__(self, model_versions=None, voting='soft', weights=None, verbose=True):
        self.voting = voting
        self.weights = weights
        self.verbose = verbose
        self.models = []
        self.model_info = []
        self.scaler = None
        self.label_encoder = None
        

        if model_versions is None:
            model_versions = self._auto_select_models()
        
        self._load_models(model_versions)
        

        if self.weights is not None:
            if len(self.weights) != len(self.models):
                raise ValueError(f"Number of weights ({len(self.weights)}) must match number of models ({len(self.models)})")

            self.weights = np.array(self.weights) / np.sum(self.weights)
        else:

            self.weights = np.ones(len(self.models)) / len(self.models)
    
    def _auto_select_models(self):
        if not os.path.exists(MODELS_DIR):
            raise ValueError(f"Models directory not found: {MODELS_DIR}")
        

        model_groups = {
            'random_forest': [],
            'xgboost': []
        }
        
        for item in os.listdir(MODELS_DIR):
            item_path = os.path.join(MODELS_DIR, item)
            if os.path.isdir(item_path) and item.startswith('v'):
                metadata_path = os.path.join(item_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    model_type = metadata.get('model_type', 'dnn')
                    accuracy = metadata.get('test_accuracy', 0)
                    

                    if model_type != 'dnn' and model_type in model_groups:
                        model_groups[model_type].append({
                            'name': item,
                            'accuracy': accuracy,
                            'metadata': metadata
                        })
        

        selected = []
        for model_type, models in model_groups.items():
            if models:

                models.sort(key=lambda x: x['accuracy'], reverse=True)
                best = models[0]
                selected.append(best['name'])
                if self.verbose:
                    print(f"[AUTO-SELECT] {model_type.upper()}: {best['name']} (Accuracy: {best['accuracy']*100:.2f}%)")
        
        if not selected:
            raise ValueError("No models found in models directory")
        
        return selected
    
    def _load_models(self, model_versions):
        print(f"\n{'='*60}")
        print(f"  LOADING ENSEMBLE MODELS ({self.voting.upper()} VOTING)")
        print(f"{'='*60}")
        
        for version_name in model_versions:
            version_path = os.path.join(MODELS_DIR, version_name)
            metadata_path = os.path.join(version_path, 'metadata.json')
            
            if not os.path.exists(metadata_path):
                print(f"[WARN] Skipping {version_name}: metadata.json not found")
                continue
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            model_type = metadata.get('model_type', 'dnn')
            
            if model_type == 'dnn':
                model_path = os.path.join(version_path, 'model.keras')
                if os.path.exists(model_path):
                    import tensorflow as tf
                    model = tf.keras.models.load_model(model_path)
                else:
                    print(f"[WARN] Skipping {version_name}: model.keras not found")
                    continue
            else:
                if model_type == 'random_forest':
                    model_path = os.path.join(version_path, 'model_rf.joblib')
                elif model_type == 'xgboost':
                    model_path = os.path.join(version_path, 'model_xgb.joblib')
                else:
                    model_path = os.path.join(version_path, 'model.joblib')
                
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                else:
                    print(f"[WARN] Skipping {version_name}: {os.path.basename(model_path)} not found")
                    continue
            
            if self.scaler is None:
                scaler_path = os.path.join(version_path, 'scaler.joblib')
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
            
            if self.label_encoder is None:
                encoder_path = os.path.join(version_path, 'label_encoder.joblib')
                if os.path.exists(encoder_path):
                    self.label_encoder = joblib.load(encoder_path)
            
            self.models.append(model)
            self.model_info.append({
                'name': version_name,
                'type': model_type,
                'accuracy': metadata.get('test_accuracy', 0)
            })
            
            if self.verbose:
                acc = metadata.get('test_accuracy', 0) * 100
                print(f"  ✓ Loaded {version_name} ({model_type.upper()}) - Accuracy: {acc:.2f}%")
        
        print(f"{'='*60}")
        print(f"  Total models loaded: {len(self.models)}")
        print(f"{'='*60}\n")
        
        if len(self.models) == 0:
            raise ValueError("No models could be loaded")
        
        if self.scaler is None or self.label_encoder is None:
            raise ValueError("Could not load scaler or label encoder")
    
    def predict_proba(self, X):
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Collect probabilities from all models
        all_probas = []
        
        for i, model in enumerate(self.models):
            model_type = self.model_info[i]['type']
            
            if model_type == 'dnn':
                # DNN outputs probabilities directly
                proba = model.predict(X_scaled, verbose=0)
            else:
                # Random Forest and XGBoost both have predict_proba method
                proba = model.predict_proba(X_scaled)
            
            all_probas.append(proba)
        
        # Weight and average
        all_probas = np.array(all_probas)  # Shape: (n_models, n_samples, n_classes)
        
        # Apply weights
        weighted_probas = np.zeros_like(all_probas[0])
        for i in range(len(self.models)):
            weighted_probas += self.weights[i] * all_probas[i]
        
        return weighted_probas
    
    def predict(self, X):
        if self.voting == 'soft':
            # Get averaged probabilities and take argmax
            probas = self.predict_proba(X)
            predictions_encoded = np.argmax(probas, axis=1)
        else:
            # Hard voting: majority vote
            X_scaled = self.scaler.transform(X)
            
            # Collect predictions from all models
            all_predictions = []
            
            for i, model in enumerate(self.models):
                model_type = self.model_info[i]['type']
                
                if model_type == 'dnn':
                    proba = model.predict(X_scaled, verbose=0)
                    pred = np.argmax(proba, axis=1)
                else:
                    pred = model.predict(X_scaled)
                
                all_predictions.append(pred)
            
            all_predictions = np.array(all_predictions)  # Shape: (n_models, n_samples)
            
            # Majority vote for each sample
            predictions_encoded = []
            for sample_idx in range(all_predictions.shape[1]):
                votes = all_predictions[:, sample_idx]
                # Count votes (using bincount)
                vote_counts = np.bincount(votes, weights=self.weights)
                predictions_encoded.append(np.argmax(vote_counts))
            
            predictions_encoded = np.array(predictions_encoded)
        
        # Decode to original labels
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        return predictions
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions)
        
        return accuracy, predictions, report
    
    def get_model_contributions(self, X):
        X_scaled = self.scaler.transform(X)
        contributions = []
        
        for i, model in enumerate(self.models):
            model_type = self.model_info[i]['type']
            
            if model_type == 'dnn':
                proba = model.predict(X_scaled, verbose=0)
            else:
                proba = model.predict(X_scaled)
            
            pred_encoded = np.argmax(proba, axis=1)
            pred_labels = self.label_encoder.inverse_transform(pred_encoded)
            confidence = np.max(proba, axis=1)
            
            contributions.append({
                'model': self.model_info[i]['name'],
                'type': model_type,
                'predictions': pred_labels,
                'confidence': confidence,
                'weight': self.weights[i]
            })
        
        return contributions


def evaluate_ensemble(csv_path=None, model_versions=None, voting='soft', weights=None):
    # FORCE USE OF features_test.csv
    csv_path = os.path.join(project_root, 'features_test.csv')
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] Test data not found: {csv_path}")
        return None, 0
    
    print(f"\n[INFO] Loading test dataset...")
    df = pd.read_csv(csv_path)
    
    # Extract features and labels
    X_test = df.drop('class', axis=1).values
    y_test = df['class'].values
    
    print(f"[INFO] Test set: {len(X_test)} samples\n")
    
    # Create ensemble
    ensemble = EnsembleClassifier(
        model_versions=model_versions,
        voting=voting,
        weights=weights,
        verbose=True
    )
    
    # Evaluate ensemble
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE EVALUATION")
    print(f"{'='*60}")
    
    accuracy, predictions, report = ensemble.evaluate(X_test, y_test)
    
    print(f"\n📊 ENSEMBLE ACCURACY: {accuracy*100:.2f}%")
    print(f"\nVoting Strategy: {voting.upper()}")
    if weights:
        print(f"Weights: {weights}")
    print(f"\n{'-'*60}")
    print("CLASSIFICATION REPORT:")
    print(f"{'-'*60}")
    print(report)
    
    # Compare with individual models
    print(f"\n{'='*60}")
    print(f"  INDIVIDUAL MODEL COMPARISON")
    print(f"{'='*60}")
    
    for i, info in enumerate(ensemble.model_info):
        model_acc = info['accuracy'] * 100
        improvement = (accuracy - info['accuracy']) * 100
        print(f"  {info['name']} ({info['type'].upper()})")
        print(f"    - Individual Accuracy: {model_acc:.2f}%")
        print(f"    - Ensemble Improvement: {improvement:+.2f}%")
        print()
    
    print(f"{'='*60}\n")
    
    return ensemble, accuracy


def optimize_ensemble_weights(csv_path=None, model_versions=None, voting='soft'):
    # Default CSV path
    if csv_path is None:
        csv_path = os.path.join(project_root, 'features_val.csv')

    print(f"\n{'='*60}")
    print(f"  OPTIMIZING ENSEMBLE WEIGHTS")
    print(f"{'='*60}\n")
    
    print(f"[INFO] Loading validation and test data...")
    val_path = os.path.join(project_root, 'features_val.csv')
    test_path = os.path.join(project_root, 'features_test.csv')
    
    if not os.path.exists(val_path) or not os.path.exists(test_path):
        print(f"[ERROR] CSV files not found. Run run_extraction.py first.")
        return None, 0, None
        
    df_val = pd.read_csv(val_path)
    X_val = df_val.drop('class', axis=1).values
    y_val = df_val['class'].values
    
    df_test = pd.read_csv(test_path)
    X_test = df_test.drop('class', axis=1).values
    y_test = df_test['class'].values
    
    print(f"[INFO] Val: {len(X_val)}, Test: {len(X_test)} samples\n")
    
    # Create base ensemble to get model list
    base_ensemble = EnsembleClassifier(
        model_versions=model_versions,
        voting=voting,
        weights=None,
        verbose=True
    )
    
    num_models = len(base_ensemble.models)
    
    if num_models < 2:
        print("[ERROR] Need at least 2 models for ensemble")
        return None, 0, None
    
    # Grid search for weights
    print(f"\n{'='*60}")
    print(f"  GRID SEARCH ({num_models} models)")
    print(f"{'='*60}\n")
    
    best_weights = None
    best_val_acc = 0
    
    # Generate weight combinations
    from itertools import product
    
    if num_models == 2:
        # For 2 models: try weights from 0.1 to 0.9 in steps of 0.1
        weight_range = [i/10 for i in range(1, 10)]
        combinations = [(w, 1-w) for w in weight_range]
    else:
        # For 3+ models: coarser grid to avoid explosion
        weight_range = [0.2, 0.3, 0.4, 0.5]
        combinations = []
        for combo in product(weight_range, repeat=num_models-1):
            last_weight = 1.0 - sum(combo)
            if 0.1 <= last_weight <= 0.6:  # Last weight should also be reasonable
                combinations.append(combo + (last_weight,))
    
    print(f"[INFO] Testing {len(combinations)} weight combinations...\n")
    
    for weights in combinations:
        # Create ensemble with these weights
        ensemble = EnsembleClassifier(
            model_versions=model_versions,
            voting=voting,
            weights=list(weights),
            verbose=False
        )
        
        # Evaluate on validation set
        val_acc, _, _ = ensemble.evaluate(X_val, y_val)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = weights
    
    print(f"{'='*60}")
    print(f"  OPTIMIZATION RESULTS")
    print(f"{'='*60}\n")
    print(f"Best weights: {[f'{w:.2f}' for w in best_weights]}")
    print(f"Validation accuracy: {best_val_acc*100:.2f}%\n")
    
    # Test with optimal weights
    final_ensemble = EnsembleClassifier(
        model_versions=model_versions,
        voting=voting,
        weights=list(best_weights),
        verbose=False
    )
    
    test_acc, _, test_report = final_ensemble.evaluate(X_test, y_test)
    
    print(f"\n📊 TEST ACCURACY (optimized): {test_acc*100:.2f}%\n")
    print(f"{'='*60}\n")
    
    return list(best_weights), test_acc, final_ensemble


def save_ensemble_model(model_versions, weights, voting, accuracy, csv_path, models_dir, name_suffix=None):
    # Get next version number
    from model_manager import get_next_version_number
    version_num = get_next_version_number()
    
    # Create version name
    if name_suffix:
        version_name = f"v{version_num:03d}_ensemble_{name_suffix}"
    else:
        version_name = f"v{version_num:03d}_ensemble"
    
    version_path = os.path.join(models_dir, version_name)
    os.makedirs(version_path, exist_ok=True)
    
    # Save ensemble configuration
    ensemble_config = {
        'model_versions': model_versions,
        'weights': weights,
        'voting': voting,
        'created_at': datetime.now().isoformat()
    }
    
    config_path = os.path.join(version_path, 'ensemble_config.json')
    with open(config_path, 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    
    # Save metadata (compatible with model manager)
    metadata = {
        'model_type': 'ensemble',
        'test_accuracy': accuracy,
        'trained_at': datetime.now().isoformat(),
        'num_classes': None,  # Will be filled from component model
        'train_samples': None,
        'csv_used': os.path.basename(csv_path),
        'component_models': model_versions,
        'voting_strategy': voting
    }
    
    # Copy scaler and encoder from first component model
    first_model_path = os.path.join(models_dir, model_versions[0])
    
    # Copy scaler
    src_scaler = os.path.join(first_model_path, 'scaler.joblib')
    dst_scaler = os.path.join(version_path, 'scaler.joblib')
    if os.path.exists(src_scaler):
        shutil.copy(src_scaler, dst_scaler)
    
    # Copy label encoder
    src_encoder = os.path.join(first_model_path, 'label_encoder.joblib')
    dst_encoder = os.path.join(version_path, 'label_encoder.joblib')
    if os.path.exists(src_encoder):
        shutil.copy(src_encoder, dst_encoder)
        
        # Get num_classes from encoder
        encoder = joblib.load(dst_encoder)
        metadata['num_classes'] = len(encoder.classes_)
    
    # Save metadata
    metadata_path = os.path.join(version_path, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE MODEL SAVED")
    print(f"{'='*60}")
    print(f"  Version: {version_name}")
    print(f"  Path: {version_path}")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Models: {', '.join([m.split('_')[0] for m in model_versions])}")
    print(f"{'='*60}\n")
    
    return version_name, version_path


def create_ensemble_model():
    print("\n" + "="*60)
    print("  CREATE ENSEMBLE MODEL")
    print("="*60)
    
    # Get available non-ensemble models
    available_models = []
    if os.path.exists(MODELS_DIR):
        for item in os.listdir(MODELS_DIR):
            item_path = os.path.join(MODELS_DIR, item)
            if os.path.isdir(item_path) and item.startswith('v'):
                metadata_path = os.path.join(item_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Exclude existing ensembles and DNN models - only RF and XGBoost allowed
                    model_type = metadata.get('model_type', 'dnn')
                    if model_type != 'ensemble' and model_type != 'dnn':
                        available_models.append({
                            'name': item,
                            'type': model_type,
                            'accuracy': metadata.get('test_accuracy', 0)
                        })
    
    if len(available_models) < 2:
        print("[ERROR] Need at least 2 trained models (RF/XGBoost) to create ensemble")
        return
    
    # Sort by accuracy
    available_models.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\nAvailable models:")
    for i, m in enumerate(available_models, 1):
        print(f"  {i}. {m['name']} ({m['type'].upper()}) - {m['accuracy']*100:.2f}%")
    
    print("\nOptions:")
    print("  1. Auto-select best RF + XGBoost")
    print("  2. Manually select models")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    model_versions = None
    
    if choice == '2':
        # Manual selection
        indices = input("Enter model numbers separated by commas (e.g. 1,3): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in indices.split(',')]
            model_versions = [available_models[i]['name'] for i in indices if 0 <= i < len(available_models)]
            
            if len(model_versions) < 2:
                print("[ERROR] Must select at least 2 models")
                return
        except:
            print("[ERROR] Invalid input")
            return
    
    # Voting strategy
    voting = input("\nVoting strategy (soft/hard) [default=soft]: ").strip().lower()
    if voting not in ['soft', 'hard']:
        voting = 'soft'
    
    # CSV path
    csv_choice = input("\nUse angles or coordinates features? (angles/coordinates) [default=angles]: ").strip().lower()
    if csv_choice == 'coordinates':
        csv_path = os.path.join(project_root, 'arnis_poses_coordinates.csv')
    else:
        csv_path = os.path.join(project_root, 'arnis_poses_angles.csv')
    
    # Weight optimization
    optimize = input("\nOptimize weights using grid search? (y/n) [default=y]: ").strip().lower()
    
    if optimize != 'n':
        # Optimize weights
        weights, accuracy, ensemble = optimize_ensemble_weights(csv_path, model_versions, voting)
        if weights is None:
            print("[ERROR] Weight optimization failed")
            return
        # Extract model versions from ensemble
        model_versions = [info['name'] for info in ensemble.model_info]
    else:
        # Use equal weights
        ensemble, accuracy = evaluate_ensemble(csv_path, model_versions, voting, None)
        if ensemble is None:
            return
        weights = list(ensemble.weights)
        model_versions = [info['name'] for info in ensemble.model_info]
    
    # Ask for name suffix
    name_suffix = input("\nEnter name for this ensemble (or press Enter to skip): ").strip()
    if name_suffix:
        name_suffix = name_suffix.replace(' ', '_').replace('-', '_')
        name_suffix = ''.join(c for c in name_suffix if c.isalnum() or c == '_')
    else:
        name_suffix = None
    
    # Save ensemble
    version_name, version_path = save_ensemble_model(
        model_versions, weights, voting, accuracy, 
        csv_path, MODELS_DIR, name_suffix
    )
    
    # Ask if set as active
    set_active = input("\nSet this ensemble as active model? (y/n) [default=n]: ").strip().lower()
    if set_active == 'y':
        from model_manager import set_active_model
        set_active_model(version_name)


def interactive_ensemble():
    """
    Interactive menu for ensemble model
    """
    print("\n" + "="*60)
    print("  ENSEMBLE MODEL EVALUATION")
    print("="*60)
    
    # Get available models
    available_models = []
    if os.path.exists(MODELS_DIR):
        for item in os.listdir(MODELS_DIR):
            item_path = os.path.join(MODELS_DIR, item)
            if os.path.isdir(item_path) and item.startswith('v'):
                metadata_path = os.path.join(item_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    available_models.append({
                        'name': item,
                        'type': metadata.get('model_type', 'dnn'),
                        'accuracy': metadata.get('test_accuracy', 0)
                    })
    
    if not available_models:
        print("[ERROR] No models found. Train models first.")
        return
    
    # Sort by accuracy
    available_models.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\nAvailable models:")
    for i, m in enumerate(available_models, 1):
        print(f"  {i}. {m['name']} ({m['type'].upper()}) - {m['accuracy']*100:.2f}%")
    
    print("\nOptions:")
    print("  1. Auto-select best model of each type")
    print("  2. Manually select models")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    model_versions = None
    weights = None
    
    if choice == '2':
        # Manual selection
        indices = input("Enter model numbers separated by commas (e.g. 1,3,5): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in indices.split(',')]
            model_versions = [available_models[i]['name'] for i in indices if 0 <= i < len(available_models)]
            
            # Ask for weights
            use_weights = input("Use custom weights? (y/n) [default=n]: ").strip().lower()
            if use_weights == 'y':
                weights_str = input(f"Enter {len(model_versions)} weights separated by commas: ").strip()
                weights = [float(x.strip()) for x in weights_str.split(',')]
        except:
            print("[ERROR] Invalid input. Using auto-select.")
            model_versions = None
    
    # Voting strategy
    voting = input("\nVoting strategy (soft/hard) [default=soft]: ").strip().lower()
    if voting not in ['soft', 'hard']:
        voting = 'soft'
    
    # CSV path
    csv_choice = input("\nUse angles or coordinates features? (angles/coordinates) [default=angles]: ").strip().lower()
    if csv_choice == 'coordinates':
        csv_path = os.path.join(project_root, 'arnis_poses_coordinates.csv')
    else:
        csv_path = os.path.join(project_root, 'arnis_poses_angles.csv')
    
    # Run evaluation
    evaluate_ensemble(csv_path, model_versions, voting, weights)


def generate_ensemble_visualizations(version_path, csv_path=None):
    """
    Generate visualizations for a saved ensemble model
    
    Creates:
    1. Confusion Matrix
    2. Model Contribution Chart (weights)
    3. Performance Comparison Chart
    
    Args:
        version_path: Path to ensemble model directory
        csv_path: Path to CSV with features (default: from metadata or angles)
    """
    print(f"\n{'='*60}")
    print(f"  GENERATING ENSEMBLE VISUALIZATIONS")
    print(f"{'='*60}")
    print(f"  Ensemble: {os.path.basename(version_path)}")
    print(f"{'='*60}\n")
    
    # Load ensemble configuration
    config_path = os.path.join(version_path, 'ensemble_config.json')
    metadata_path = os.path.join(version_path, 'metadata.json')
    
    if not os.path.exists(config_path) or not os.path.exists(metadata_path):
        print("[ERROR] Ensemble configuration files not found")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get CSV path
    if csv_path is None:
        csv_path = metadata.get('csv_used', 'arnis_poses_angles.csv')
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(project_root, csv_path)
    
    # Override old CSV references to new pipeline file
    if 'arnis_poses' in csv_path or not os.path.exists(csv_path):
        csv_path = os.path.join(project_root, 'features_test.csv')
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        print(f"[HINT] Run 'python training/run_extraction.py' to generate features.")
        return False
    
    # Load TEST data directly for visualization

    print(f"[INFO] Loading data from: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)
    
    # If it's the new split files, X_test is just the content
    if 'features_test.csv' in os.path.basename(csv_path):
        X_test = df.drop('class', axis=1).values
        y_test = df['class'].values
    else:
        # Legacy fallback
        X = df.drop('class', axis=1).values
        y = df['class'].values
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load ensemble
    print("[INFO] Loading ensemble model...")
    ensemble = EnsembleClassifier(
        model_versions=config['model_versions'],
        voting=config['voting'],
        weights=config['weights'],
        verbose=False
    )
    
    # Get predictions
    print("[INFO] Generating predictions...")
    y_pred = ensemble.predict(X_test)
    
    # Import plotting libraries
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 1. Confusion Matrix
    print("[INFO] Creating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    class_names = ensemble.label_encoder.classes_
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('Actual Class', fontsize=12)
    plt.title('Ensemble Model - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(version_path, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: confusion_matrix.png")
    
    # 2. Model Contribution Chart (Weights)
    print("[INFO] Creating model contribution chart...")
    model_names = [info['name'] for info in ensemble.model_info]
    model_types = [info['type'].upper() for info in ensemble.model_info]
    weights = ensemble.weights
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(model_names)))
    bars = ax.bar(range(len(model_names)), weights, color=colors)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Weight (Contribution)', fontsize=12)
    ax.set_title('Ensemble Model Contributions', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([f"{name}\n({mtype})" for name, mtype in zip(model_names, model_types)], 
                       rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(version_path, 'model_contributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: model_contributions.png")
    
    # 3. Performance Comparison
    print("[INFO] Creating performance comparison chart...")
    ensemble_acc = metadata.get('test_accuracy', 0)
    individual_accs = [info['accuracy'] for info in ensemble.model_info]
    all_accs = individual_accs + [ensemble_acc]
    all_labels = [f"{name}\n({mtype})" for name, mtype in zip(model_names, model_types)] + ['Ensemble']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#3498db'] * len(individual_accs) + ['#e74c3c']  # Blue for individuals, red for ensemble
    bars = ax.bar(range(len(all_labels)), [acc * 100 for acc in all_accs], color=colors)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', label='Individual Models'),
                      Patch(facecolor='#e74c3c', label='Ensemble')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(version_path, 'performance_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: performance_comparison.png")
    
    # 4. Error Distribution by Class
    print("[INFO] Creating error distribution chart...")
    
    # Calculate per-class error rates for ensemble
    ensemble_errors = {}
    for class_name in class_names:
        class_mask = y_test == class_name
        class_preds = y_pred[class_mask]
        error_rate = 1 - np.mean(class_preds == class_name)
        ensemble_errors[class_name] = error_rate
    
    # Get individual model predictions
    individual_errors = {info['name']: {} for info in ensemble.model_info}
    for i, info in enumerate(ensemble.model_info):
        model = ensemble.models[i]
        model_type = info['type']
        
        # Get predictions from individual model
        X_test_scaled = ensemble.scaler.transform(X_test)
        if model_type == 'dnn':
            proba = model.predict(X_test_scaled, verbose=0)
            y_pred_model = np.argmax(proba, axis=1)
        else:
            y_pred_model = model.predict(X_test_scaled)
        
        # Convert to labels
        y_pred_model_labels = ensemble.label_encoder.inverse_transform(y_pred_model)
        
        # Calculate per-class errors
        for class_name in class_names:
            class_mask = y_test == class_name
            class_preds = y_pred_model_labels[class_mask]
            error_rate = 1 - np.mean(class_preds == class_name)
            individual_errors[info['name']][class_name] = error_rate
    
    # Plot error distribution
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(class_names))
    width = 0.8 / (len(ensemble.model_info) + 1)
    
    # Plot individual models
    for i, (model_name, errors) in enumerate(individual_errors.items()):
        error_values = [errors[c] * 100 for c in class_names]
        offset = (i - len(individual_errors)/2) * width
        ax.bar(x + offset, error_values, width, label=model_name, alpha=0.7)
    
    # Plot ensemble
    ensemble_error_values = [ensemble_errors[c] * 100 for c in class_names]
    offset = (len(individual_errors) - len(individual_errors)/2) * width
    ax.bar(x + offset, ensemble_error_values, width, label='Ensemble', 
           color='#e74c3c', alpha=0.9, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Error Rate (%)', fontsize=12)
    ax.set_title('Error Distribution by Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(os.path.join(version_path, 'error_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: error_distribution.png")
    
    # 5. Classification Report (save as both text and visual table)
    print("[INFO] Generating classification report...")
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    
    report_text = classification_report(y_test, y_pred, target_names=class_names)
    
    # Save to text file
    report_path = os.path.join(version_path, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("  ENSEMBLE MODEL CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Ensemble: {os.path.basename(version_path)}\n")
        f.write(f"Voting Strategy: {config['voting']}\n")
        f.write(f"Component Models: {', '.join(config['model_versions'])}\n")
        f.write(f"Overall Accuracy: {metadata.get('test_accuracy', 0)*100:.2f}%\n\n")
        f.write("="*60 + "\n")
        f.write("Per-Class Metrics:\n")
        f.write("="*60 + "\n\n")
        f.write(report_text)
        f.write("\n" + "="*60 + "\n")
    
    print(f"  ✓ Saved: classification_report.txt")
    
    # Create visual table
    print("[INFO] Creating classification report table...")
    
    # Get metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=class_names, zero_division=0
    )
    
    # Calculate accuracy per class
    accuracy_per_class = []
    for i, class_name in enumerate(class_names):
        class_mask = y_test == class_name
        class_preds = y_pred[class_mask]
        acc = np.mean(class_preds == class_name) if class_mask.sum() > 0 else 0
        accuracy_per_class.append(acc)
    
    # Create table data
    table_data = []
    for i, class_name in enumerate(class_names):
        # Shorten class names for display
        display_name = class_name.replace('_correct', '').replace('_', ' ').title()
        if len(display_name) > 25:
            display_name = display_name[:22] + '...'
        
        table_data.append([
            display_name,
            f'{precision[i]:.3f}',
            f'{recall[i]:.3f}',
            f'{f1[i]:.3f}',
            f'{accuracy_per_class[i]:.3f}',
            f'{int(support[i])}'
        ])
    
    # Add overall metrics
    overall_accuracy = metadata.get('test_accuracy', 0)
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    table_data.append([
        '---',
        '---',
        '---',
        '---',
        '---',
        '---'
    ])
    table_data.append([
        'Weighted Avg',
        f'{weighted_precision:.3f}',
        f'{weighted_recall:.3f}',
        f'{weighted_f1:.3f}',
        f'{overall_accuracy:.3f}',
        f'{int(support.sum())}'
    ])
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(class_names) * 0.5 + 2)))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'Support'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.35, 0.13, 0.13, 0.13, 0.13, 0.13])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(6):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(table_data) + 1):
        for j in range(6):
            cell = table[(i, j)]
            if i == len(table_data) or i == len(table_data) - 1:
                # Separator and weighted avg row
                cell.set_facecolor('#E7E6E6')
                if i == len(table_data):
                    cell.set_text_props(weight='bold')
            elif i % 2 == 0:
                cell.set_facecolor('#F2F2F2')
            else:
                cell.set_facecolor('white')
    
    # Add title
    title_text = f'Classification Report - {os.path.basename(version_path)}\n'
    title_text += f'Voting: {config["voting"]} | Overall Accuracy: {overall_accuracy*100:.2f}%'
    plt.title(title_text, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(version_path, 'classification_report.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: classification_report.png")
    
    # Print summary to console
    print(f"\n{'-'*60}")
    print("CLASSIFICATION REPORT SUMMARY:")
    print(f"{'-'*60}")
    print(report_text)
    
    print(f"\n{'='*60}")
    print("  VISUALIZATION GENERATION COMPLETE")
    print(f"{'='*60}\n")
    
    return True


if __name__ == "__main__":
    interactive_ensemble()
