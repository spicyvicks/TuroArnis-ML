import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Load best model (v048: 50.4% accuracy)
# This model was trained on 92 features (before we added palm orientation features)
models_dir = os.path.join(project_root, 'models')
model_dir = os.path.join(models_dir, 'v048_20260205_202345_xgboost')

print(f"Loading best model: v048_20260205_202345_xgboost (50.4% test accuracy)")

# Load model files - v048 uses model_xgb.joblib for the actual model
model = joblib.load(os.path.join(model_dir, 'model_xgb.joblib'))
scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))

with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
    metadata = json.load(f)

# Load selected features
with open(os.path.join(model_dir, 'selected_features.json'), 'r') as f:
    selected_features = json.load(f)

# Load test data
test_csv = os.path.join(project_root, 'features_test.csv')
df_test = pd.read_csv(test_csv)

all_columns = [col for col in df_test.columns if col != 'class']

print(f"\nModel expects: {metadata['n_features_in']} features")
print(f"Current CSV has: {len(all_columns)} features")
print(f"Model uses {len(selected_features)} selected features")

# Solution: Model was trained on the first 92 features (original set)
# Current CSV has 110 features (added palm orientation, target ratios, closest indicators)
# Use only the first 92 features which existed during training

original_feature_count = 92
feature_names_subset = all_columns[:original_feature_count]

print(f"Using only first {original_feature_count} features (original feature set before augmentation)")

# Build mapping from selected feature names to indices
selected_indices = []
for f in selected_features:
    if f in feature_names_subset:
        selected_indices.append(feature_names_subset.index(f))

print(f"Found {len(selected_indices)} selected features in original feature set")

# Extract and scale test data
X_test = df_test[feature_names_subset].values
y_test = df_test['class'].values

X_test_scaled = scaler.transform(X_test)
X_test_model = X_test_scaled[:, selected_indices]

# Predict and analyze
y_pred = model.predict(X_test_model)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"OVERALL ACCURACY: {accuracy:.2%}")
print(f"{'='*60}")

# Per-class analysis
classes = sorted(df_test['class'].unique())
print(f"\nPer-class accuracy ({len(classes)} classes):\n")

class_accuracies = []
for cls in classes:
    mask = y_test == cls
    if mask.sum() == 0:
        continue
    cls_acc = accuracy_score(y_test[mask], y_pred[mask])
    cls_count = mask.sum()
    class_accuracies.append((cls, cls_acc, cls_count))

# Sort by accuracy
class_accuracies.sort(key=lambda x: x[1])

print(f"{'Class Name':<30} {'Accuracy':>10} {'Count':>8}")
print("-" * 50)
for cls, acc, count in class_accuracies:
    print(f"{cls:<30} {acc:>9.1%} {count:>8}")

# Identify worst performing classes
print(f"\n\nBOTTOM 5 WORST CLASSES:")
print("-" * 50)
for cls, acc, count in class_accuracies[:5]:
    mask = y_test == cls
    wrong_preds = y_pred[mask]
    # Count what they were predicted as
    unique, counts = np.unique(wrong_preds, return_counts=True)
    wrong_as = {}
    for u, c in zip(unique, counts):
        if u != cls:
            wrong_as[u] = c
    
    wrong_str = ", ".join([f"{k}({v})" for k,v in sorted(wrong_as.items(), key=lambda x: -x[1])[:3]])
    print(f"{cls}: {acc:.1%} accuracy, often confused as: {wrong_str}")

# Confusion matrix for bottom classes
print(f"\n\nCONFUSION MATRIX (Top-Left: Bottom 5 worst classes):")
print("-" * 50)

bottom_5_classes = [x[0] for x in class_accuracies[:5]]
mask = np.isin(y_test, bottom_5_classes)
y_test_bottom = y_test[mask]
y_pred_bottom = y_pred[mask]

cm = confusion_matrix(y_test_bottom, y_pred_bottom, labels=bottom_5_classes)

header = 'Actual \\ Predicted'
print(f"{header:<20}", end="")
for cls in bottom_5_classes:
    print(f"{cls[:8]:>10}", end="")
print()

for i, actual_cls in enumerate(bottom_5_classes):
    print(f"{actual_cls:<20}", end="")
    for j in range(len(bottom_5_classes)):
        print(f"{cm[i,j]:>10}", end="")
    print()

print(f"\n\nKEY FINDINGS:")
print("-" * 50)
print(f"• Overall test accuracy: {accuracy:.2%}")
print(f"• Range: {min(x[1] for x in class_accuracies):.1%} to {max(x[1] for x in class_accuracies):.1%}")
print(f"• Best class: {class_accuracies[-1][0]} ({class_accuracies[-1][1]:.1%})")
print(f"• Worst class: {class_accuracies[0][0]} ({class_accuracies[0][1]:.1%})")
print(f"• To reach 80% overall: need improvement on bottom {sum(1 for _, acc, _ in class_accuracies if acc < 0.7)} classes")
