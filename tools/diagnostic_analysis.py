"""
Comprehensive diagnostic analysis of model failures.
Identifies which classes are failing, how, and why.
Goal: Find gaps for 80% accuracy improvement.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

MODELS_DIR = os.path.join(project_root, 'models')
DATA_DIR = project_root


def load_best_model():
    """Load v048 (best model) and its metadata."""
    model_path = os.path.join(MODELS_DIR, 'v048_20260205_202345_xgboost')
    
    model = joblib.load(os.path.join(model_path, 'model_xgb.joblib'))
    scaler = joblib.load(os.path.join(model_path, 'scaler.joblib'))
    encoder = joblib.load(os.path.join(model_path, 'label_encoder.joblib'))
    
    with open(os.path.join(model_path, 'metadata.json')) as f:
        metadata = json.load(f)
    
    with open(os.path.join(model_path, 'selected_features.json')) as f:
        selected_features = json.load(f)
    
    return model, scaler, encoder, metadata, selected_features


def load_data():
    """Load test data."""
    csv_path = os.path.join(DATA_DIR, 'features_test.csv')
    df = pd.read_csv(csv_path)
    
    X = df.drop('class', axis=1).values
    y = df['class'].values
    feature_names = [col for col in df.columns if col != 'class']
    
    return X, y, feature_names, df


def analyze_class_performance(model, scaler, encoder, X, y, selected_features, feature_names, metadata):
    """Analyze per-class accuracy - use training approach."""
    print("\n" + "="*80)
    print("  CLASS-LEVEL PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Load the ACTUAL training data that was used
    # This ensures we have the exact same features the model expects
    csv_train = os.path.join(DATA_DIR, 'features_train.csv')
    df_train = pd.read_csv(csv_train)
    train_feature_names = [col for col in df_train.columns if col != 'class']
    
    # Get indices of selected features in training data
    selected_indices = [train_feature_names.index(f) for f in selected_features if f in train_feature_names]
    
    # Now apply to test data using same features
    X_test_selected = X[:, selected_indices]
    
    # Scale using the scaler
    X_scaled = scaler.transform(X_test_selected.reshape(len(X_test_selected), len(selected_indices)))
    y_pred_encoded = model.predict(X_scaled)
    y_pred = encoder.inverse_transform(y_pred_encoded)
    
    # Get probabilities for confidence analysis
    y_proba = model.predict_proba(X_scaled)
    y_conf = np.max(y_proba, axis=1)
    
    # Per-class analysis
    classes = np.unique(y)
    class_results = []
    
    print("\n{:<40} {:<10} {:<10} {:<15}".format("CLASS", "COUNT", "ACCURACY", "AVG CONFIDENCE"))
    print("-" * 75)
    
    for cls in sorted(classes):
        mask = y == cls
        correct = (y_pred[mask] == cls).sum()
        total = mask.sum()
        accuracy = correct / total if total > 0 else 0
        avg_conf = y_conf[mask].mean()
        
        class_results.append({
            'class': cls,
            'count': total,
            'correct': correct,
            'wrong': total - correct,
            'accuracy': accuracy,
            'avg_confidence': avg_conf
        })
        
        status = "✓" if accuracy >= 0.8 else "✗"
        print("{:<40} {:<10} {:<10.2%} {:<15.2%}".format(
            f"{status} {cls}", total, accuracy, avg_conf
        ))
    
    df_results = pd.DataFrame(class_results)
    df_results = df_results.sort_values('accuracy')
    
    print("\n" + "-" * 75)
    print(f"Overall Accuracy: {(y_pred == y).sum() / len(y):.2%}")
    print(f"Classes below 80%: {(df_results['accuracy'] < 0.8).sum()}/{len(classes)}")
    print(f"Classes below 50%: {(df_results['accuracy'] < 0.5).sum()}/{len(classes)}")
    
    return df_results, y_pred, y_conf


def analyze_confusions(y, y_pred):
    """Analyze confusion patterns."""
    print("\n" + "="*80)
    print("  CONFUSION ANALYSIS - WHICH CLASSES ARE CONFUSED WITH EACH OTHER?")
    print("="*80)
    
    # Build confusion patterns
    confusions = []
    
    classes = np.unique(y)
    for true_cls in sorted(classes):
        mask = y == true_cls
        preds_for_this = y_pred[mask]
        
        for pred_cls in sorted(classes):
            if pred_cls != true_cls:
                count = (preds_for_this == pred_cls).sum()
                if count > 0:
                    confusions.append({
                        'true': true_cls,
                        'predicted': pred_cls,
                        'count': count,
                        'pct': count / mask.sum() * 100
                    })
    
    df_conf = pd.DataFrame(confusions).sort_values('count', ascending=False)
    
    print("\nTop 15 Confusion Patterns:")
    print("{:<40} {:<40} {:<8} {:<8}".format("TRUE CLASS", "PREDICTED AS", "COUNT", "% OF TRUE"))
    print("-" * 100)
    
    for _, row in df_conf.head(15).iterrows():
        print("{:<40} {:<40} {:<8} {:<8.1f}%".format(
            row['true'][:40], row['predicted'][:40], int(row['count']), row['pct']
        ))
    
    return df_conf


def analyze_feature_patterns(X, y, feature_names, selected_features, df_test_results):
    """Analyze feature differences between hard vs easy classes."""
    print("\n" + "="*80)
    print("  FEATURE ANALYSIS - WHY ARE SOME CLASSES HARDER?")
    print("="*80)
    
    # Get worst performing classes
    worst_classes = df_test_results.nsmallest(3, 'accuracy')['class'].values
    best_classes = df_test_results.nlargest(3, 'accuracy')['class'].values
    
    selected_indices = [feature_names.index(f) for f in selected_features if f in feature_names]
    
    print(f"\n🔴 WORST CLASSES (hardest to classify):")
    for cls in worst_classes:
        mask = y == cls
        X_cls = X[mask][:, selected_indices]
        print(f"\n  {cls} (accuracy: {df_test_results[df_test_results['class']==cls]['accuracy'].values[0]:.1%})")
        print(f"    - Samples: {mask.sum()}")
        print(f"    - Avg feature std: {X_cls.std(axis=0).mean():.4f}")
        print(f"    - Feature variance: {X_cls.var(axis=0).mean():.4f}")
    
    print(f"\n🟢 BEST CLASSES (easiest to classify):")
    for cls in best_classes:
        mask = y == cls
        X_cls = X[mask][:, selected_indices]
        print(f"\n  {cls} (accuracy: {df_test_results[df_test_results['class']==cls]['accuracy'].values[0]:.1%})")
        print(f"    - Samples: {mask.sum()}")
        print(f"    - Avg feature std: {X_cls.std(axis=0).mean():.4f}")
        print(f"    - Feature variance: {X_cls.var(axis=0).mean():.4f}")


def analyze_feature_importance(model, feature_names, selected_features):
    """Show which features the model relies on most."""
    print("\n" + "="*80)
    print("  FEATURE IMPORTANCE - WHAT MATTERS TO THE MODEL?")
    print("="*80)
    
    importance = model.feature_importances_
    
    # Map back to feature names
    selected_indices = [feature_names.index(f) for f in selected_features if f in feature_names]
    importance_with_names = [(selected_features[i], importance[i]) for i in range(len(selected_features))]
    importance_with_names.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 20 Most Important Features:")
    print("{:<40} {:<15}".format("FEATURE", "IMPORTANCE"))
    print("-" * 55)
    
    for feat, imp in importance_with_names[:20]:
        bar = "█" * int(imp * 100)
        print("{:<40} {:<15} {}".format(feat, f"{imp:.4f}", bar))
    
    print("\nBottom 10 Least Important Features (candidates for removal):")
    print("{:<40} {:<15}".format("FEATURE", "IMPORTANCE"))
    print("-" * 55)
    
    for feat, imp in importance_with_names[-10:]:
        bar = "█" * int(imp * 100)
        print("{:<40} {:<15} {}".format(feat, f"{imp:.4f}", bar))
    
    return importance_with_names


def analyze_data_distribution(df_full, y):
    """Analyze train/test distribution."""
    print("\n" + "="*80)
    print("  DATA DISTRIBUTION ANALYSIS")
    print("="*80)
    
    class_counts = pd.Series(y).value_counts()
    
    print("\nTest Set Class Distribution:")
    print("{:<40} {:<10} {:<8}".format("CLASS", "COUNT", "% OF TEST"))
    print("-" * 60)
    
    for cls in sorted(class_counts.index):
        count = class_counts[cls]
        pct = count / len(y) * 100
        print("{:<40} {:<10} {:<8.1f}%".format(cls, count, pct))
    
    # Check for imbalance
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"\nImbalance Ratio (max/min): {imbalance_ratio:.2f}x")
    if imbalance_ratio > 2:
        print("⚠️  SEVERE CLASS IMBALANCE - this could be a major issue!")


def generate_summary_report(df_results, df_conf, importance_with_names):
    """Generate summary of findings."""
    print("\n" + "="*80)
    print("  SUMMARY & RECOMMENDATIONS FOR 80% ACCURACY")
    print("="*80)
    
    failing_classes = df_results[df_results['accuracy'] < 0.5]
    print(f"\n🔴 CRITICAL ISSUES ({len(failing_classes)} classes below 50%):")
    for _, row in failing_classes.iterrows():
        print(f"   - {row['class']}: {row['accuracy']:.1%} accuracy ({row['wrong']} misclassified)")
    
    print(f"\n📊 KEY FINDINGS:")
    print(f"   - Best class accuracy: {df_results['accuracy'].max():.1%}")
    print(f"   - Worst class accuracy: {df_results['accuracy'].min():.1%}")
    print(f"   - Average class accuracy: {df_results['accuracy'].mean():.1%}")
    print(f"   - Overall test accuracy: {df_results['correct'].sum() / df_results['count'].sum():.1%}")
    
    print(f"\n💡 TO REACH 80% ACCURACY, YOU NEED TO:")
    
    target_correct = int(0.8 * df_results['count'].sum())
    current_correct = df_results['correct'].sum()
    needed = target_correct - current_correct
    
    print(f"   - Gain {needed} more correct predictions ({needed/df_results['count'].sum()*100:.1f}% of test set)")
    print(f"   - Currently: {current_correct}/{df_results['count'].sum()} = {current_correct/df_results['count'].sum():.1%}")
    
    # Prioritize improvements
    print(f"\n🎯 PRIORITY IMPROVEMENTS:")
    
    # Find most confused pairs
    worst_confusion = df_conf.iloc[0]
    print(f"\n   1. FIX: {worst_confusion['true']} ← → {worst_confusion['predicted']}")
    print(f"      - {int(worst_confusion['count'])} misclassifications ({worst_confusion['pct']:.1f}% of {worst_confusion['true']})")
    print(f"      - Likely feature overlap between these classes")
    
    # Feature-based insights
    low_importance_feats = [f for f, i in importance_with_names if i < 0.001]
    print(f"\n   2. FEATURE CLEANUP: {len(low_importance_feats)} features with near-zero importance")
    print(f"      - Retraining without noise might help")
    
    print(f"\n   3. CLASS-SPECIFIC MODELS:")
    print(f"      - Current: 1 model for all 13 classes")
    print(f"      - Better: Hierarchical (stance/block/thrust) → type → direction")
    
    print(f"\n   4. AUGMENTATION STRATEGY:")
    print(f"      - Current: Uniform noise + flip augmentation")
    print(f"      - Better: SMOTE for minority classes + hard negative mining")


def main():
    print("\n" + "="*80)
    print("  COMPREHENSIVE MODEL DIAGNOSTIC")
    print("  Goal: Understand path to 80% accuracy")
    print("="*80)
    
    # Load everything
    model, scaler, encoder, metadata, selected_features = load_best_model()
    X, y, feature_names, df_full = load_data()
    
    # Run analyses
    df_results, y_pred, y_conf = analyze_class_performance(model, scaler, encoder, X, y, selected_features, feature_names, metadata)
    df_conf = analyze_confusions(y, y_pred)
    analyze_feature_patterns(X, y, feature_names, selected_features, df_results)
    importance_with_names = analyze_feature_importance(model, feature_names, selected_features)
    analyze_data_distribution(df_full, y)
    generate_summary_report(df_results, df_conf, importance_with_names)
    
    print("\n" + "="*80)
    print("  DIAGNOSTIC COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
