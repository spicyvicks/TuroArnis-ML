"""
Analyze why test set accuracy is so different from CV accuracy.
Helps identify domain shift, class imbalance, or specific failure patterns.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from collections import Counter

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def analyze_test_failures():
    """Detailed analysis of what's failing in the test set"""
    
    # Load data
    train_df = pd.read_csv(os.path.join(project_root, 'features_train.csv')).dropna()
    test_df = pd.read_csv(os.path.join(project_root, 'features_test.csv')).dropna()
    
    print("="*60)
    print("   TEST SET FAILURE ANALYSIS")
    print("="*60)
    
    # 1. Class distribution comparison
    print("\n📊 CLASS DISTRIBUTION")
    print("-"*50)
    train_counts = Counter(train_df['class'])
    test_counts = Counter(test_df['class'])
    
    print(f"{'Class':<35} {'Train':>8} {'Test':>8} {'Ratio':>8}")
    print("-"*60)
    for cls in sorted(train_counts.keys()):
        tr = train_counts[cls]
        te = test_counts.get(cls, 0)
        ratio = te / tr if tr > 0 else 0
        flag = "⚠️" if te < 10 else ""
        print(f"{cls:<35} {tr:>8} {te:>8} {ratio:>7.2f} {flag}")
    
    print(f"\nTotal: Train={len(train_df)}, Test={len(test_df)}")
    
    # 2. Feature distribution comparison
    print("\n\n📊 FEATURE DISTRIBUTION SHIFT")
    print("-"*50)
    print("Features with biggest train→test shift:")
    
    feature_cols = [c for c in train_df.columns if c != 'class']
    shifts = []
    
    for col in feature_cols:
        train_mean = train_df[col].mean()
        test_mean = test_df[col].mean()
        train_std = train_df[col].std()
        
        # Normalized shift (how many std devs different)
        if train_std > 0:
            shift = abs(test_mean - train_mean) / train_std
        else:
            shift = 0
        shifts.append((col, shift, train_mean, test_mean))
    
    shifts.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Feature':<40} {'Shift':>8} {'Train μ':>10} {'Test μ':>10}")
    print("-"*70)
    for col, shift, tr_mean, te_mean in shifts[:15]:
        flag = "🔴" if shift > 1.0 else ("🟡" if shift > 0.5 else "")
        print(f"{col:<40} {shift:>7.2f}σ {tr_mean:>10.3f} {te_mean:>10.3f} {flag}")
    
    # 3. Stick detection comparison
    print("\n\n📊 STICK DETECTION COMPARISON")
    print("-"*50)
    if 'stick_detected' in feature_cols:
        train_stick = train_df['stick_detected'].mean() * 100
        test_stick = test_df['stick_detected'].mean() * 100
        print(f"   Train stick detection rate: {train_stick:.1f}%")
        print(f"   Test stick detection rate:  {test_stick:.1f}%")
        if abs(train_stick - test_stick) > 20:
            print("   ⚠️ BIG DIFFERENCE - stick detection may be failing on test set!")
    
    # 4. Load a model and analyze per-class failures
    print("\n\n📊 PER-CLASS FAILURE ANALYSIS")
    print("-"*50)
    
    # Find latest model
    models_dir = os.path.join(project_root, 'models')
    model_folders = [d for d in os.listdir(models_dir) 
                     if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('v')]
    
    if not model_folders:
        print("No models found")
        return
    
    model_folders.sort()
    latest = model_folders[-1]
    model_path = os.path.join(models_dir, latest)
    
    print(f"Using model: {latest}")
    
    # Load model components
    try:
        if os.path.exists(os.path.join(model_path, 'model_rf.joblib')):
            model = joblib.load(os.path.join(model_path, 'model_rf.joblib'))
        else:
            model = joblib.load(os.path.join(model_path, 'model_xgb.joblib'))
        scaler = joblib.load(os.path.join(model_path, 'scaler.joblib'))
        le = joblib.load(os.path.join(model_path, 'label_encoder.joblib'))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load selected features if any
    sf_path = os.path.join(model_path, 'selected_features.json')
    selected_features = None
    if os.path.exists(sf_path):
        with open(sf_path, 'r') as f:
            selected_features = json.load(f)
    
    # Prepare data
    X_test = test_df.drop('class', axis=1).values
    y_test = le.transform(test_df['class'].values)
    
    X_test_scaled = scaler.transform(X_test)
    
    if selected_features:
        indices = [feature_cols.index(f) for f in selected_features if f in feature_cols]
        X_test_scaled = X_test_scaled[:, indices]
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Analyze failures per class
    print(f"\n{'True Class':<25} {'Accuracy':>8} {'Most Confused With':<25} {'Count':>6}")
    print("-"*70)
    
    worst_classes = []
    for i, cls in enumerate(le.classes_):
        mask = y_test == i
        if mask.sum() == 0:
            continue
        
        cls_pred = y_pred[mask]
        correct = (cls_pred == i).sum()
        total = mask.sum()
        acc = correct / total
        
        # Find most confused class
        wrong_preds = cls_pred[cls_pred != i]
        if len(wrong_preds) > 0:
            confused_with = Counter(wrong_preds).most_common(1)[0]
            confused_cls = le.classes_[confused_with[0]]
            confused_count = confused_with[1]
        else:
            confused_cls = "-"
            confused_count = 0
        
        status = "🔴" if acc < 0.3 else ("🟡" if acc < 0.5 else "🟢")
        short_cls = cls.replace('_correct', '').replace('_', ' ')
        short_confused = confused_cls.replace('_correct', '').replace('_', ' ')
        print(f"{status} {short_cls:<23} {acc*100:>6.1f}% → {short_confused:<23} ({confused_count})")
        
        worst_classes.append((cls, acc, confused_cls, confused_count))
    
    # Summary
    worst_classes.sort(key=lambda x: x[1])
    print("\n\n📊 SUMMARY")
    print("-"*50)
    print("Worst 3 classes:")
    for cls, acc, confused, count in worst_classes[:3]:
        print(f"   • {cls}: {acc*100:.0f}% (confused with {confused})")
    
    # Recommendations
    print("\n\n💡 RECOMMENDATIONS")
    print("-"*50)
    
    # Check for systematic issues
    stick_shift = abs(train_df['stick_detected'].mean() - test_df['stick_detected'].mean()) if 'stick_detected' in feature_cols else 0
    
    if stick_shift > 0.2:
        print("1. STICK DETECTION ISSUE")
        print("   Your test set has very different stick detection rate.")
        print("   → Check if test images have different lighting/angles")
        print("   → May need to retrain stick detector with more diverse data")
    
    big_shifts = [s for s in shifts if s[1] > 1.0]
    if len(big_shifts) > 5:
        print("\n2. DOMAIN SHIFT DETECTED")
        print(f"   {len(big_shifts)} features have >1σ shift between train and test")
        print("   → Test set may be from different source (camera, person, setting)")
        print("   → Consider: are train/test images visually similar?")
    
    left_right_confusion = any('left' in w[0] and 'right' in w[2] for w in worst_classes[:5])
    if left_right_confusion:
        print("\n3. LEFT/RIGHT CONFUSION")
        print("   Model confuses left and right moves")
        print("   → Test subject may be mirrored or facing different direction")
        print("   → Consider adding image flipping augmentation")
    
    print("\n")

if __name__ == "__main__":
    analyze_test_failures()
