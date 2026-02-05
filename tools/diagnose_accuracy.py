"""Diagnose accuracy issues - find which classes are confused"""
import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('features_train.csv').dropna()
test = pd.read_csv('features_test.csv').dropna()

feature_cols = [c for c in train.columns if c != 'class']
X_train = train[feature_cols].values
y_train = train['class'].values
X_test = test[feature_cols].values
y_test = test['class'].values

# Encode labels
le = LabelEncoder()
le.fit(y_train)
y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("="*70)
print("ACCURACY DIAGNOSIS")
print("="*70)

# 1. Cross-validation on training data
print("\n1. CROSS-VALIDATION (5-fold on training data)")
print("   This shows expected accuracy range:")
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
cv_scores = cross_val_score(rf, X_train_scaled, y_train_enc, cv=StratifiedKFold(5, shuffle=True, random_state=42))
print(f"   CV Accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
print(f"   Range: {cv_scores.min()*100:.1f}% - {cv_scores.max()*100:.1f}%")

# 2. Train and evaluate
print("\n2. TRAINING FINAL MODEL...")
rf.fit(X_train_scaled, y_train_enc)
train_acc = rf.score(X_train_scaled, y_train_enc)
test_acc = rf.score(X_test_scaled, y_test_enc)
print(f"   Train accuracy: {train_acc*100:.1f}%")
print(f"   Test accuracy:  {test_acc*100:.1f}%")

if train_acc > 0.95 and test_acc < 0.6:
    print("   ⚠️  OVERFITTING DETECTED: High train, low test accuracy")

# 3. Per-class accuracy
print("\n3. PER-CLASS ACCURACY (which classes are failing?):")
y_pred = rf.predict(X_test_scaled)
report = classification_report(y_test_enc, y_pred, target_names=le.classes_, output_dict=True)

class_stats = []
for cls in le.classes_:
    stats = report[cls]
    n_samples = int(stats['support'])
    acc = stats['recall']  # recall = accuracy for that class
    class_stats.append((cls, acc, n_samples))

# Sort by accuracy (worst first)
class_stats.sort(key=lambda x: x[1])
print("\n   Class                          Accuracy  Samples")
print("   " + "-"*50)
for cls, acc, n in class_stats:
    bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
    status = "❌" if acc < 0.4 else ("⚠️" if acc < 0.6 else "✓")
    print(f"   {cls:32} {acc*100:5.1f}%  {n:3d} {status}")

# 4. Confusion matrix - find most confused pairs
print("\n4. MOST CONFUSED CLASS PAIRS:")
cm = confusion_matrix(y_test_enc, y_pred)
confused_pairs = []
for i in range(len(le.classes_)):
    for j in range(len(le.classes_)):
        if i != j and cm[i, j] > 0:
            confused_pairs.append((le.classes_[i], le.classes_[j], cm[i, j], cm[i].sum()))

confused_pairs.sort(key=lambda x: x[2]/x[3], reverse=True)
print("\n   True Class → Predicted As (% of true class samples)")
print("   " + "-"*60)
for true_cls, pred_cls, count, total in confused_pairs[:10]:
    pct = count / total * 100
    print(f"   {true_cls[:25]:25} → {pred_cls[:25]:25} ({pct:.0f}%)")

# 5. Feature importance for stick vs body
print("\n5. FEATURE IMPORTANCE (Stick vs Body):")
importances = rf.feature_importances_
stick_features = [c for c in feature_cols if any(s in c for s in ['stick', 'grip', 'tip_'])]
body_features = [c for c in feature_cols if c not in stick_features]

stick_imp = sum(importances[feature_cols.index(f)] for f in stick_features)
body_imp = sum(importances[feature_cols.index(f)] for f in body_features)

print(f"   Body features ({len(body_features)}): {body_imp*100:.1f}% importance")
print(f"   Stick features ({len(stick_features)}): {stick_imp*100:.1f}% importance")

# Top 10 features
print("\n   Top 10 most important features:")
feat_imp = list(zip(feature_cols, importances))
feat_imp.sort(key=lambda x: x[1], reverse=True)
for i, (feat, imp) in enumerate(feat_imp[:10], 1):
    is_stick = "🔸" if feat in stick_features else "  "
    print(f"   {i:2d}. {is_stick} {feat:30} {imp*100:.2f}%")

# 6. Suggestions
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

if test_acc < 0.5:
    print("\n⚠️  Test accuracy below 50% suggests:")
    
    # Check for similar classes
    similar_pairs = [p for p in confused_pairs if p[2]/p[3] > 0.2]
    if similar_pairs:
        print("\n   1. MERGE SIMILAR CLASSES:")
        for true_cls, pred_cls, count, total in similar_pairs[:5]:
            print(f"      - '{true_cls}' often confused with '{pred_cls}'")
    
    # Check class imbalance
    train_counts = train['class'].value_counts()
    if train_counts.max() / train_counts.min() > 2:
        print(f"\n   2. CLASS IMBALANCE: {train_counts.max()}/{train_counts.min()} = {train_counts.max()/train_counts.min():.1f}x difference")
        print(f"      Largest: {train_counts.idxmax()} ({train_counts.max()} samples)")
        print(f"      Smallest: {train_counts.idxmin()} ({train_counts.min()} samples)")
    
    # Small test set
    if len(test) < 300:
        print(f"\n   3. SMALL TEST SET: Only {len(test)} samples ({len(test)//len(le.classes_)} per class avg)")
        print("      Consider using cross-validation for more reliable estimates")

print("\n   4. TRY GROUPING SIMILAR POSES:")
print("      - Group all 'thrust' moves together")
print("      - Group all 'block' moves together")
print("      - Reduces from 13 → fewer classes, higher accuracy")
