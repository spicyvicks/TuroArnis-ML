import os
import sys
import json
import shutil
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
    
    active_config = {
        'version': version_name,
        'path': version_path,
        'model_path': os.path.join(version_path, 'model.keras'),
        'encoder_path': os.path.join(version_path, 'label_encoder.joblib'),
        'scaler_path': os.path.join(version_path, 'scaler.joblib'),
        'set_at': datetime.now().isoformat()
    }
    
    with open(ACTIVE_MODEL_FILE, 'w') as f:
        json.dump(active_config, f, indent=2)
    
    print(f"[OK] Active model set to: {version_name}")
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
    
    # ask for architecture
    print("\nSelect model architecture:")
    print("  1. DNN (Dense Neural Network) - current default")
    print("  2. Random Forest - often better for tabular data")
    print("  3. XGBoost - gradient boosting, often best accuracy")
    
    arch_choice = input("\nEnter choice (1-3) [default=1]: ").strip()
    
    # common settings for all architectures
    model_name = None
    feature_mode = 'angles'
    do_extraction = False
    
    # feature mode selection (for all architectures)
    print("\nSelect feature mode:")
    print("  1. Angles (33 features) - joint angles + positions")
    print("  2. Coordinates (99 features) - raw landmark coordinates")
    mode_choice = input("Enter choice (1 or 2) [default=1]: ").strip()
    if mode_choice == '2':
        feature_mode = 'coordinates'
    
    # set csv path based on feature mode
    if feature_mode == 'coordinates':
        csv_filename = 'arnis_poses_coordinates.csv'
    else:
        csv_filename = 'arnis_poses_angles.csv'
    
    csv_path = os.path.join(project_root, csv_filename)
    
    # check if CSV exists and ask about extraction
    if os.path.exists(csv_path):
        print(f"\n[INFO] Found existing CSV: {csv_filename}")
        print("  1. Use existing CSV (skip extraction)")
        print("  2. Re-extract features from images")
        extract_choice = input("Enter choice (1 or 2) [default=1]: ").strip()
        if extract_choice == '2':
            do_extraction = True
    else:
        print(f"\n[INFO] CSV not found: {csv_filename}")
        print("[INFO] Will extract features from images...")
        do_extraction = True
    
    # model name (for RF and XGBoost)
    if arch_choice in ['2', '3']:
        print("\nEnter a name for this model (for organization):")
        print("  Examples: 'test1', 'aug_data', 'final'")
        model_name = input("Name (or press Enter to skip): ").strip()
        if model_name:
            model_name = model_name.replace(' ', '_').replace('-', '_')
            model_name = ''.join(c for c in model_name if c.isalnum() or c == '_')
    
    # perform extraction if needed
    if do_extraction:
        from feature_extraction import extract_features_from_dataset
        dataset_path = os.path.join(project_root, 'dataset_aug')
        
        if not os.path.exists(dataset_path):
            print(f"[ERROR] Dataset folder not found: {dataset_path}")
            return
        
        extract_features_from_dataset(dataset_path, csv_path, feature_mode)
    
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    if arch_choice == '2':
        # Random Forest
        print(f"\n[INFO] Training Random Forest with {feature_mode.upper()} features...")
        from training_alt import train_random_forest
        
        acc, version = train_random_forest(csv_path, models_dir, model_name)
        if acc:
            print(f"\n[OK] Random Forest training complete! Accuracy: {acc*100:.2f}%")
        return
        
    elif arch_choice == '3':
        # XGBoost
        print(f"\n[INFO] Training XGBoost with {feature_mode.upper()} features...")
        from training_alt import train_xgboost, HAS_XGBOOST
        
        if not HAS_XGBOOST:
            print("[ERROR] XGBoost not installed. Run: pip install xgboost")
            return
        
        acc, version = train_xgboost(csv_path, models_dir, model_name)
        if acc:
            print(f"\n[OK] XGBoost training complete! Accuracy: {acc*100:.2f}%")
        return
    
    # DNN training (default)
    print(f"\n[INFO] Using DNN with {feature_mode.upper()} mode")
    print("[INFO] Running training script...\n")
    
    import subprocess
    training_script = os.path.join(current_dir, 'training.py')
    
    # set feature mode via environment variable
    env = os.environ.copy()
    env['FEATURE_MODE'] = feature_mode
    
    # run training.py as subprocess with mode
    result = subprocess.run(
        [sys.executable, training_script],
        cwd=project_root,
        env=env
    )
    
    if result.returncode == 0:
        print("\n[OK] Training completed successfully!")
    else:
        print(f"\n[ERROR] Training failed with code {result.returncode}")

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
                from get_classification_report import generate_classification_report
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

def main_menu():
    while True:
        print("\n" + "="*40)
        print("   TUROARNIS MODEL MANAGER")
        print("="*40)
        print("  1. Train new model")
        print("  2. Generate analysis (reports/visualizations)")
        print("  3. List all models")
        print("  4. Set active model")
        print("  5. Compare models")
        print("  6. Delete a model")
        print("  7. Create ensemble model")
        print("  8. Evaluate ensemble model")
        print("  9. Exit")
        print("="*40)
        
        try:
            choice = input("Enter choice (1-9): ").strip()
            
            if choice == '1':
                train_new_model()
            elif choice == '2':
                generate_analysis()
            elif choice == '3':
                list_models()
            elif choice == '4':
                set_active_model_menu()
            elif choice == '5':
                compare_models()
            elif choice == '6':
                delete_model()
            elif choice == '7':
                create_ensemble_menu()
            elif choice == '8':
                evaluate_ensemble_menu()
            elif choice == '9':
                print("\n[INFO] Goodbye!")
                break
            else:
                print("[ERROR] Invalid choice")
        except KeyboardInterrupt:
            print("\n\n[INFO] Interrupted. Goodbye!")
            break

if __name__ == "__main__":
    main_menu()
