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
    
    if arch == '2':
        # Random Forest
        try:
            sys.path.append(current_dir) # ensure training_alt is importable
            from training_alt import load_and_prepare_data, train_random_forest
            print("\n[INFO] Loading data for Random Forest...")
            data = load_and_prepare_data()
            train_random_forest(data)
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
            data = load_and_prepare_data()
            train_xgboost(data)
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
