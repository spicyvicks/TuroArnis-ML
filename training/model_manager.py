import os
import sys
import json
import shutil
import subprocess
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

MODELS_DIR = os.path.join(project_root, 'models')
ACTIVE_MODEL_FILE = os.path.join(MODELS_DIR, 'active_model.json')

def get_all_model_versions():
    versions = []
    if not os.path.exists(MODELS_DIR):
        return versions
    
    for item in os.listdir(MODELS_DIR):
        item_path = os.path.join(MODELS_DIR, item)
        if os.path.isdir(item_path) and item.startswith('v'):
            metadata_path = os.path.join(item_path, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                versions.append({
                    'name': item,
                    'path': item_path,
                    **metadata
                })
    
    # sort by version number
    versions.sort(key=lambda x: x['name'], reverse=True)
    return versions

def get_next_version_number():
    versions = get_all_model_versions()
    if not versions:
        return 1
    
    # extract version numbers
    max_version = 0
    for v in versions:
        try:
            num = int(v['name'].split('_')[0][1:])
            max_version = max(max_version, num)
        except:
            pass
    return max_version + 1

def get_active_model():
    if os.path.exists(ACTIVE_MODEL_FILE):
        with open(ACTIVE_MODEL_FILE, 'r') as f:
            return json.load(f)
    return None

def set_active_model(version_name):
    version_path = os.path.join(MODELS_DIR, version_name)
    if not os.path.exists(version_path):
        print(f"[ERROR] Version {version_name} not found")
        return False
    
    # Check metadata for model type
    model_type = 'dnn'
    meta_path = os.path.join(version_path, 'metadata.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                model_type = meta.get('model_type', 'dnn')
        except:
            pass
            
    # Determine correct model filename
    if model_type == 'random_forest':
        model_file = 'model_rf.joblib'
    elif model_type == 'xgboost':
        model_file = 'model_xgb.joblib'
    else:
        model_file = 'model.keras'
        
    active_config = {
        'version': version_name,
        'path': version_path,
        'model_path': os.path.join(version_path, model_file),
        'model_type': model_type,
        'encoder_path': os.path.join(version_path, 'label_encoder.joblib'),
        'scaler_path': os.path.join(version_path, 'scaler.joblib'),
        'set_at': datetime.now().isoformat()
    }
    
    with open(ACTIVE_MODEL_FILE, 'w') as f:
        json.dump(active_config, f, indent=2)
    
    print(f"[OK] Active model set to: {version_name} ({model_type.upper()})")
    return True

def list_models():
    versions = get_all_model_versions()
    active = get_active_model()
    active_version = active['version'] if active else None
    
    if not versions:
        print("\n[INFO] No model versions found.")
        print("       Run option 1 to train a new model.")
        return
    
    print("\n" + "="*70)
    print("  AVAILABLE MODEL VERSIONS")
    print("="*70)
    print(f"{'Version':<25} {'Accuracy':<12} {'Date':<20} {'Active'}")
    print("-"*70)
    
    for v in versions:
        is_active = "  ★" if v['name'] == active_version else ""
        acc = f"{v.get('test_accuracy', 0)*100:.1f}%" if v.get('test_accuracy') else "N/A"
        date = v.get('trained_at', 'Unknown')[:19].replace('T', ' ')
        print(f"{v['name']:<25} {acc:<12} {date:<20} {is_active}")
    
    print("="*70)

def train_new_model():
    print("\n" + "="*40)
    print("   TRAIN NEW MODEL")
    print("="*40)
    
    # 1. Extraction Phase
    print("\n[STEP 1] Data Preparation")
    print("Do you want to re-extract features? (Recommended if you added new images)")
    print("  1. Yes, run extraction pipeline")
    print("  2. No, use existing CSVs")
    ext_choice = input("Choice [1]: ").strip()
    
    if ext_choice != '2':
        print("\nSelect Feature Mode:")
        print("  1. Angles (58 body features) - Body only")
        print("  2. Coordinates (99 body features) - Body only")
        print("  3. Combined (72 features) - Body + Stick [RECOMMENDED]")
        m_choice = input("Choice [3]: ").strip()
        if m_choice == '1':
            mode = 'angles'
        elif m_choice == '2':
            mode = 'coordinates'
        else:
            mode = 'combined'
        
        print(f"\n[INFO] Running extraction ({mode})...")
        try:
            subprocess.run([sys.executable, 'training/run_extraction.py', '--mode', mode], 
                         cwd=project_root, check=True)
        except subprocess.CalledProcessError:
            print("\n[ERROR] Extraction failed. Aborting.")
            input("Press Enter...")
            return

    # 2. Training Phase
    print("\n[STEP 2] Model Training")
    print("Select Architecture:")
    print("  1. Dense Neural Network (DNN)")
    print("  2. Random Forest")
    print("  3. XGBoost")
    
    arch = input("Choice [1]: ").strip()
    
    # Ask about feature selection and CV for RF/XGB
    use_feature_selection = True
    protect_stick = True
    use_cv = False
    hybrid_mode = False
    if arch in ['2', '3']:
        print("\nEvaluation Method:")
        print("  1. Cross-validation only (5-fold, uses all data, most reliable)")
        print("  2. Train/Test split only (faster, small test set)")
        print("  3. HYBRID: CV on train + final test on held-out set (best of both)")
        eval_choice = input("Choice [3]: ").strip()
        
        if eval_choice == '1':
            use_cv = True
            hybrid_mode = False
            print("✓ Using 5-fold cross-validation on all data")
        elif eval_choice == '2':
            use_cv = False
            hybrid_mode = False
            print("✓ Using train/test split")
        else:  # Default to hybrid (option 3)
            use_cv = False
            hybrid_mode = True
            print("✓ HYBRID: CV for tuning + held-out test for final accuracy")
        
        print("\nFeature Selection:")
        print("  1. Use all features (no selection)")
        print("  2. Feature selection WITH protection (keep stick + laterality features)")
        print("  3. Feature selection WITHOUT protection (drop any low-importance)")
        fs_choice = input("Choice [1]: ").strip()
        
        if fs_choice == '1' or fs_choice == '':
            use_feature_selection = False
            print("✓ Using all features")
        elif fs_choice == '2':
            use_feature_selection = True
            protect_stick = True
            print("✓ Feature selection enabled, important features protected")
        elif fs_choice == '3':
            use_feature_selection = True
            protect_stick = False
            print("✓ Feature selection enabled, no protection")
        else:
            use_feature_selection = False
            print("✓ Using all features")
    
    if arch == '2':
        # Random Forest
        try:
            sys.path.append(current_dir) # ensure training_alt is importable
            from training_alt import load_and_prepare_data, train_random_forest
            print("\n[INFO] Loading data for Random Forest...")
            data = load_and_prepare_data(use_cross_validation=use_cv)
            train_random_forest(data, use_feature_selection=use_feature_selection, protect_stick=protect_stick, use_cv=use_cv, hybrid_mode=hybrid_mode)
        except SystemExit:
            print("[INFO] Training aborted.")
        except Exception as e:
            print(f"[ERROR] Failed to run RF training: {e}")
            
    elif arch == '3':
        # XGBoost
        try:
            sys.path.append(current_dir)
            from training_alt import load_and_prepare_data, train_xgboost
            print("\n[INFO] Loading data for XGBoost...")
            data = load_and_prepare_data(use_cross_validation=use_cv)
            train_xgboost(data, use_feature_selection=use_feature_selection, protect_stick=protect_stick, use_cv=use_cv, hybrid_mode=hybrid_mode)
        except SystemExit:
            print("[INFO] Training aborted.")
        except Exception as e:
            print(f"[ERROR] Failed to run XGB training: {e}")
            
    elif arch in ['1', '']:
        # DNN
        try:
            subprocess.run([sys.executable, 'training/training.py'], 
                         cwd=project_root, check=True)
        except subprocess.CalledProcessError:
            print("\n[ERROR] DNN Training failed.")
    else:
        print("[ERROR] Invalid selection")
        
    input("\nPress Enter to continue...")

def generate_analysis():
    versions = get_all_model_versions()
    
    if not versions:
        print("\n[ERROR] No models found. Train a model first.")
        return
    
    print("\n" + "="*60)
    print("  GENERATE MODEL ANALYSIS")
    print("="*60)
    print("\nSelect a model:")
    for i, v in enumerate(versions, 1):
        model_type = v.get('model_type', 'unknown').upper()
        acc = f"{v.get('test_accuracy', 0)*100:.1f}%" if v.get('test_accuracy') else "N/A"
        print(f"  {i}. {v['name']} ({model_type}) - {acc}")
    
    try:
        choice = int(input("\nEnter number: ")) - 1
        if 0 <= choice < len(versions):
            selected = versions[choice]
            model_type = selected.get('model_type', 'dnn')
            
            print(f"\n[INFO] Generating analysis for {selected['name']} ({model_type.upper()})...")
            
            if model_type == 'dnn':
                # Generate classification report for DNN
                model_path = os.path.join(selected['path'], 'model.keras')
                if not os.path.exists(model_path):
                    print(f"[ERROR] Model file not found: {model_path}")
                    return
                
                sys.path.insert(0, os.path.join(project_root, 'tools'))
                from get_classification_report import generate_classification_report  # type: ignore
                generate_classification_report()
                
            elif model_type in ['random_forest', 'xgboost']:
                # Generate visualizations for RF/XGBoost
                from generate_visualizations import generate_visualizations_for_model
                generate_visualizations_for_model(selected['path'])
                
            elif model_type == 'ensemble':
                # Generate visualizations for ensemble
                from ensemble_model import generate_ensemble_visualizations
                generate_ensemble_visualizations(selected['path'])
                
            else:
                print(f"[WARN] Unknown model type: {model_type}")
        else:
            print("[ERROR] Invalid selection")
    except ValueError:
        print("[ERROR] Invalid input")

def compare_models():
    versions = get_all_model_versions()
    
    if len(versions) < 2:
        print("\n[ERROR] Need at least 2 models to compare.")
        return
    
    print("\nSelect first model:")
    for i, v in enumerate(versions, 1):
        print(f"  {i}. {v['name']}")
    
    try:
        choice1 = int(input("Enter number: ")) - 1
        choice2 = int(input("Enter second model number: ")) - 1
        
        if 0 <= choice1 < len(versions) and 0 <= choice2 < len(versions):
            v1, v2 = versions[choice1], versions[choice2]
            
            print("\n" + "="*50)
            print("  MODEL COMPARISON")
            print("="*50)
            print(f"{'Metric':<20} {v1['name']:<15} {v2['name']:<15}")
            print("-"*50)
            
            acc1 = f"{v1.get('test_accuracy', 0)*100:.1f}%" if v1.get('test_accuracy') else "N/A"
            acc2 = f"{v2.get('test_accuracy', 0)*100:.1f}%" if v2.get('test_accuracy') else "N/A"
            print(f"{'Test Accuracy':<20} {acc1:<15} {acc2:<15}")
            
            classes1 = v1.get('num_classes', 'N/A')
            classes2 = v2.get('num_classes', 'N/A')
            print(f"{'Classes':<20} {classes1:<15} {classes2:<15}")
            
            samples1 = v1.get('train_samples', 'N/A')
            samples2 = v2.get('train_samples', 'N/A')
            print(f"{'Train Samples':<20} {samples1:<15} {samples2:<15}")
            
            print("="*50)
        else:
            print("[ERROR] Invalid selection")
    except ValueError:
        print("[ERROR] Invalid input")

def delete_model():
    versions = get_all_model_versions()
    active = get_active_model()
    
    if not versions:
        print("\n[ERROR] No models to delete.")
        return
    
    print("\nSelect a model to delete:")
    for i, v in enumerate(versions, 1):
        is_active = " (ACTIVE)" if active and v['name'] == active['version'] else ""
        acc = f"{v.get('test_accuracy', 0)*100:.1f}%" if v.get('test_accuracy') else "N/A"
        print(f"  {i}. {v['name']} - {acc}{is_active}")
    
    try:
        choice = int(input("\nEnter number (0 to cancel): "))
        if choice == 0:
            return
        
        choice -= 1
        if 0 <= choice < len(versions):
            selected = versions[choice]
            
            if active and selected['name'] == active['version']:
                print("[ERROR] Cannot delete active model. Set another model as active first.")
                return
            
            confirm = input(f"Delete {selected['name']}? (yes/no): ")
            if confirm.lower() == 'yes':
                shutil.rmtree(selected['path'])
                print(f"[OK] Deleted {selected['name']}")
            else:
                print("[INFO] Cancelled")
        else:
            print("[ERROR] Invalid selection")
    except ValueError:
        print("[ERROR] Invalid input")

def set_active_model_menu():
    versions = get_all_model_versions()
    
    if not versions:
        print("\n[ERROR] No models found.")
        return
    
    active = get_active_model()
    
    print("\nSelect a model to set as active:")
    for i, v in enumerate(versions, 1):
        is_current = " (current)" if active and v['name'] == active['version'] else ""
        acc = f"{v.get('test_accuracy', 0)*100:.1f}%" if v.get('test_accuracy') else "N/A"
        print(f"  {i}. {v['name']} - {acc}{is_current}")
    
    try:
        choice = int(input("\nEnter number: ")) - 1
        if 0 <= choice < len(versions):
            set_active_model(versions[choice]['name'])
        else:
            print("[ERROR] Invalid selection")
    except ValueError:
        print("[ERROR] Invalid input")


def create_ensemble_menu():
    from ensemble_model import create_ensemble_model
    create_ensemble_model()

def evaluate_ensemble_menu():
    from ensemble_model import interactive_ensemble
    interactive_ensemble()

def evaluate_existing_model():
    """Evaluate an existing model with detailed metrics"""
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    versions = get_all_model_versions()
    
    if not versions:
        print("\n[ERROR] No models found.")
        return
    
    # Filter to only RF and XGBoost models (not DNN for now)
    rf_xgb_versions = [v for v in versions if v.get('model_type') in ['random_forest', 'xgboost']]
    
    print("\n" + "="*60)
    print("   EVALUATE EXISTING MODEL")
    print("="*60)
    print("\nSelect a model to evaluate (0 to go back):")
    for i, v in enumerate(rf_xgb_versions, 1):
        acc = f"{v.get('test_accuracy', 0)*100:.1f}%" if v.get('test_accuracy') else "N/A"
        model_type = v.get('model_type', 'unknown')
        n_features = v.get('n_features_in', v.get('num_features', '?'))
        print(f"  {i}. {v['name']} ({model_type}) - {acc} - {n_features} features")
    
    if not rf_xgb_versions:
        print("\n[ERROR] No RF/XGBoost models found. Train one first.")
        return
    
    try:
        choice_input = input("\nEnter number (0 to go back): ").strip()
        if choice_input == '0' or choice_input.lower() == 'b':
            return
        choice = int(choice_input) - 1
        if not (0 <= choice < len(rf_xgb_versions)):
            print("[ERROR] Invalid selection")
            return
    except ValueError:
        print("[ERROR] Invalid input")
        return
    
    selected = rf_xgb_versions[choice]
    model_path = selected['path']
    model_type = selected.get('model_type', 'random_forest')
    
    print(f"\n[INFO] Loading model: {selected['name']}")
    
    # Load the model
    ext = 'rf' if model_type == 'random_forest' else 'xgb'
    model = joblib.load(os.path.join(model_path, f'model_{ext}.joblib'))
    scaler = joblib.load(os.path.join(model_path, 'scaler.joblib'))
    le = joblib.load(os.path.join(model_path, 'label_encoder.joblib'))
    
    # Load selected features if available
    selected_features = None
    sf_path = os.path.join(model_path, 'selected_features.json')
    if os.path.exists(sf_path):
        with open(sf_path, 'r') as f:
            selected_features = json.load(f)
    
    # Ask evaluation mode
    print("\nEvaluation Mode (0 to go back):")
    print("  1. Test set only (quick)")
    print("  2. Cross-validation on training data (reliable)")
    print("  3. Both (comprehensive)")
    eval_mode = input("Choice [3]: ").strip()
    if eval_mode == '0' or eval_mode.lower() == 'b':
        return
    if not eval_mode:
        eval_mode = '3'
    
    # Load data
    csv_train = os.path.join(project_root, 'features_train.csv')
    csv_test = os.path.join(project_root, 'features_test.csv')
    
    train_df = pd.read_csv(csv_train).dropna()
    test_df = pd.read_csv(csv_test).dropna()
    
    feature_names = [c for c in train_df.columns if c != 'class']
    
    # Check feature compatibility
    model_n_features = selected.get('n_features_in')
    if model_n_features and model_n_features != len(feature_names):
        print(f"\n[WARN] Model trained on {model_n_features} features, CSV has {len(feature_names)}")
        print("       Results may be inaccurate. Consider retraining.")
    
    # Apply feature selection if model used it
    if selected_features:
        print(f"[INFO] Applying feature selection ({len(selected_features)} features)")
        train_df = train_df[['class'] + [f for f in selected_features if f in train_df.columns]]
        test_df = test_df[['class'] + [f for f in selected_features if f in test_df.columns]]
        feature_names = [c for c in train_df.columns if c != 'class']
    
    X_train = train_df.drop('class', axis=1).values
    y_train = le.transform(train_df['class'].values)
    X_test = test_df.drop('class', axis=1).values
    y_test = le.transform(test_df['class'].values)
    
    # Scale
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*60)
    print("   EVALUATION RESULTS")
    print("="*60)
    
    # Test set evaluation
    if eval_mode != '2':
        print("\n📊 TEST SET EVALUATION")
        print("-"*40)
        y_pred = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"   Accuracy: {test_acc*100:.2f}% ({sum(y_test == y_pred)}/{len(y_test)} correct)")
        
        # Per-class accuracy
        print("\n   Per-Class Performance:")
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        for cls in sorted(le.classes_):
            recall = report[cls]['recall']
            support = int(report[cls]['support'])
            status = "✓" if recall >= 0.6 else ("⚠" if recall >= 0.4 else "✗")
            print(f"   {status} {cls:35} {recall*100:5.1f}% ({support} samples)")
    
    # Cross-validation
    if eval_mode != '1':
        print("\n📊 CROSS-VALIDATION (5-fold on training data)")
        print("-"*40)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, n_jobs=-1)
        print(f"   CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
        print(f"   Fold scores: {[f'{s*100:.1f}%' for s in cv_scores]}")
    
    # Confusion matrix summary
    if eval_mode != '2':
        print("\n📊 MOST CONFUSED PAIRS")
        print("-"*40)
        cm = confusion_matrix(y_test, y_pred)
        confused_pairs = []
        for i in range(len(le.classes_)):
            for j in range(len(le.classes_)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append((le.classes_[i], le.classes_[j], cm[i, j], cm[i].sum()))
        
        confused_pairs.sort(key=lambda x: x[2]/max(x[3], 1), reverse=True)
        for true_cls, pred_cls, count, total in confused_pairs[:5]:
            pct = count / total * 100 if total > 0 else 0
            print(f"   {true_cls[:20]:20} → {pred_cls[:20]:20} ({pct:.0f}%)")
    
    print("\n" + "="*60)
    
    # Ask about visualizations
    print("\nGenerate visualizations? (0 to go back to menu)")
    print("  1. Yes, show me the charts!")
    print("  2. No thanks")
    viz_choice = input("Choice [1]: ").strip()
    
    if viz_choice == '0' or viz_choice.lower() == 'b':
        return
    
    if viz_choice != '2' and eval_mode != '2':
        generate_evaluation_visualizations(
            model, X_test_scaled, y_test, y_pred, le, 
            selected['name'], test_acc, report, cm,
            cv_scores if eval_mode != '1' else None,
            eval_mode
        )
    
    # Option to view model details
    print("\nView this model's details & config? (0 to go back to menu)")
    print("  1. Yes")
    print("  2. No")
    details_choice = input("Choice [2]: ").strip()
    
    if details_choice == '0' or details_choice.lower() == 'b':
        return
    
    if details_choice == '1':
        view_model_details(preselected_model=selected)
    else:
        input("\nPress Enter to continue...")

def generate_evaluation_visualizations(model, X_test, y_test, y_pred, le, model_name, test_acc, report, cm, cv_scores=None, eval_mode='1'):
    """Generate and display evaluation visualizations"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Create output directory
    viz_dir = os.path.join(project_root, 'reports', 'evaluations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Evaluation mode abbreviations
    mode_abbrev = {'1': 'test', '2': 'cv', '3': 'hybrid'}
    eval_abbrev = mode_abbrev.get(eval_mode, 'eval')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up the figure with multiple subplots
    n_plots = 3 if cv_scores is not None else 2
    fig = plt.figure(figsize=(16, 5 * ((n_plots + 1) // 2)))
    
    # 1. Confusion Matrix Heatmap
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Shorten class names for display
    short_names = [c.replace('_correct', '').replace('_', ' ')[:15] for c in le.classes_]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=short_names, yticklabels=short_names, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title(f'Confusion Matrix\n{model_name}')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax1.yaxis.get_majorticklabels(), rotation=0)
    
    # 2. Per-Class Accuracy Bar Chart
    ax2 = fig.add_subplot(2, 2, 2)
    
    class_accs = []
    for cls in le.classes_:
        class_accs.append((cls.replace('_correct', '').replace('_', ' ')[:18], report[cls]['recall']))
    
    class_accs.sort(key=lambda x: x[1])
    names, accs = zip(*class_accs)
    colors = ['#e74c3c' if a < 0.4 else '#f39c12' if a < 0.6 else '#27ae60' for a in accs]
    
    bars = ax2.barh(names, [a * 100 for a in accs], color=colors)
    ax2.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax2.axvline(x=test_acc * 100, color='blue', linestyle='-', alpha=0.7, label=f'Overall: {test_acc*100:.1f}%')
    ax2.set_xlabel('Accuracy (%)')
    ax2.set_title('Per-Class Accuracy')
    ax2.legend(loc='lower right')
    ax2.set_xlim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{acc*100:.0f}%', va='center', fontsize=8)
    
    # 3. CV Fold Scores (if available)
    if cv_scores is not None:
        ax3 = fig.add_subplot(2, 2, 3)
        fold_names = [f'Fold {i+1}' for i in range(len(cv_scores))]
        colors = ['#3498db' if s >= cv_scores.mean() else '#95a5a6' for s in cv_scores]
        bars = ax3.bar(fold_names, cv_scores * 100, color=colors)
        ax3.axhline(y=cv_scores.mean() * 100, color='green', linestyle='--', 
                   label=f'Mean: {cv_scores.mean()*100:.1f}%')
        ax3.axhline(y=(cv_scores.mean() - cv_scores.std()) * 100, color='orange', 
                   linestyle=':', alpha=0.7)
        ax3.axhline(y=(cv_scores.mean() + cv_scores.std()) * 100, color='orange', 
                   linestyle=':', alpha=0.7, label=f'±{cv_scores.std()*100:.1f}%')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Cross-Validation Fold Scores')
        ax3.legend()
        ax3.set_ylim(0, 100)
        
        for bar, score in zip(bars, cv_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{score*100:.1f}%', ha='center', fontsize=9)
    
    # 4. Feature Importance (top 15)
    if hasattr(model, 'feature_importances_'):
        ax4 = fig.add_subplot(2, 2, 4)
        
        # Get feature names from the model's training
        importances = model.feature_importances_
        n_features = len(importances)
        
        # Create generic feature indices if we don't have names
        indices = np.argsort(importances)[-15:]  # Top 15
        
        ax4.barh(range(len(indices)), importances[indices], color='#9b59b6')
        ax4.set_yticks(range(len(indices)))
        ax4.set_yticklabels([f'Feature {i}' for i in indices])
        ax4.set_xlabel('Importance')
        ax4.set_title('Top 15 Feature Importances')
    
    plt.suptitle(f'Model Evaluation: {model_name}\nTest Accuracy: {test_acc*100:.1f}%', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(viz_dir, f'{eval_abbrev}_{model_name}_{timestamp}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualizations saved to: {save_path}")
    
    # Show the plot
    plt.show()
    
    return save_path

def view_model_details(preselected_model=None):
    """View detailed configuration of a specific model and optionally generate visualizations"""
    models = get_all_model_versions()
    if not models:
        print("[ERROR] No models found")
        return
    
    # If model was preselected (from evaluate flow), use it directly
    if preselected_model:
        selected = preselected_model
    else:
        print("\n" + "="*50)
        print("   VIEW MODEL DETAILS & ANALYSIS")
        print("="*50)
        
        # Show model selection
        print("\nSelect a model (0 to go back):")
        for i, m in enumerate(models, 1):
            active = " (active)" if m.get('is_active') else ""
            model_type = m.get('model_type', 'unknown').upper()
            acc = f"{m.get('test_accuracy', 0)*100:.1f}%" if m.get('test_accuracy') else "N/A"
            print(f"  {i}. {m['name']} ({model_type}) - {acc}{active}")
        
        try:
            choice_input = input("\nEnter number (0 to go back): ").strip()
            if choice_input == '0' or choice_input.lower() == 'b':
                return
            choice = int(choice_input) - 1
            if choice < 0 or choice >= len(models):
                print("[ERROR] Invalid selection")
                return
        except ValueError:
            print("[ERROR] Please enter a number")
            return
        
        selected = models[choice]
    
    model_dir = selected['path']
    
    # Load metadata
    metadata_path = os.path.join(model_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        print(f"[ERROR] No metadata found for {selected['name']}")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    model_type = metadata.get('model_type', 'unknown')
    
    # Display details
    print("\n" + "="*50)
    print(f"   📋 MODEL: {selected['name']}")
    print("="*50)
    
    print(f"\n🏷️  TYPE: {model_type.upper()}")
    print(f"📅 Trained: {metadata.get('trained_at', 'unknown')[:19]}")
    print(f"🎯 Test Accuracy: {metadata.get('test_accuracy', 0)*100:.1f}%")
    print(f"📊 Features: {metadata.get('num_features', metadata.get('n_features_in', 'unknown'))}")
    print(f"🔧 Feature Selection: {'Yes' if metadata.get('feature_selection_used') else 'No'}")
    
    # Hyperparameters
    if 'hyperparameters' in metadata:
        print("\n⚙️  HYPERPARAMETERS:")
        print("-"*40)
        for key, value in metadata['hyperparameters'].items():
            print(f"   {key}: {value}")
    
    # Class names
    if 'class_names' in metadata:
        print(f"\n📂 CLASSES ({len(metadata['class_names'])}):")
        print("-"*40)
        for i, cls in enumerate(metadata['class_names'], 1):
            short_name = cls.replace('_correct', '').replace('_', ' ')
            print(f"   {i:2}. {short_name}")
    
    # Check for additional files
    print("\n📁 MODEL FILES:")
    print("-"*40)
    for f in os.listdir(model_dir):
        size = os.path.getsize(os.path.join(model_dir, f))
        size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
        print(f"   • {f} ({size_str})")
    
    # Check for selected features
    features_path = os.path.join(model_dir, 'selected_features.json')
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            features = json.load(f)
        if isinstance(features, list) and len(features) > 0:
            print(f"\n🎯 SELECTED FEATURES ({len(features)}):")
            print("-"*40)
            for feat in features[:10]:
                print(f"   • {feat}")
            if len(features) > 10:
                print(f"   ... and {len(features)-10} more")
    
    print("\n" + "="*50)
    
    # Option to generate visualizations
    print("\nGenerate visualizations for this model? (0 to go back)")
    print("  1. Yes, generate charts")
    print("  2. No, return to menu")
    viz_choice = input("Choice [2]: ").strip()
    
    if viz_choice == '0' or viz_choice.lower() == 'b':
        return
    
    if viz_choice == '1':
        print(f"\n[INFO] Generating analysis for {selected['name']} ({model_type.upper()})...")
        
        if model_type == 'dnn':
            # Generate classification report for DNN
            model_path = os.path.join(model_dir, 'model.keras')
            if not os.path.exists(model_path):
                print(f"[ERROR] Model file not found: {model_path}")
                return
            
            sys.path.insert(0, os.path.join(project_root, 'tools'))
            from get_classification_report import generate_classification_report  # type: ignore
            generate_classification_report()
            
        elif model_type in ['random_forest', 'xgboost']:
            # Generate visualizations for RF/XGBoost
            from generate_visualizations import generate_visualizations_for_model
            generate_visualizations_for_model(model_dir)
            
        elif model_type == 'ensemble':
            # Generate visualizations for ensemble
            from ensemble_model import generate_ensemble_visualizations
            generate_ensemble_visualizations(model_dir)
            
        else:
            print(f"[WARN] Unknown model type: {model_type}")
    
    input("\nPress Enter to continue...")

def main_menu():
    while True:
        print("\n" + "="*40)
        print("   TUROARNIS MODEL MANAGER")
        print("="*40)
        print("  1. Train new model")
        print("  2. Evaluate existing model")
        print("  3. View model details & analysis")
        print("  4. List all models")
        print("  5. Set active model")
        print("  6. Compare models")
        print("  7. Delete a model")
        print("  8. Create ensemble model")
        print("  9. Evaluate ensemble model")
        print("  0. Exit")
        print("="*40)
        
        try:
            choice = input("Enter choice (0-9): ").strip()
            
            if choice == '1':
                train_new_model()
            elif choice == '2':
                evaluate_existing_model()
            elif choice == '3':
                view_model_details()
            elif choice == '4':
                list_models()
            elif choice == '5':
                set_active_model_menu()
            elif choice == '6':
                compare_models()
            elif choice == '7':
                delete_model()
            elif choice == '8':
                create_ensemble_menu()
            elif choice == '9':
                evaluate_ensemble_menu()
            elif choice == '0':
                print("\n[INFO] Goodbye!")
                break
            else:
                print("[ERROR] Invalid choice")
        except KeyboardInterrupt:
            print("\n\n[INFO] Interrupted. Goodbye!")
            break
            break

if __name__ == "__main__":
    main_menu()
