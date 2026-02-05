"""Quick data analysis script"""
import pandas as pd
import numpy as np

train = pd.read_csv('features_train.csv')
test = pd.read_csv('features_test.csv')

print("="*60)
print("DATA ANALYSIS")
print("="*60)

# Stick detection rate
print(f"\n1. STICK DETECTION RATE:")
print(f"   Train: {(train['stick_detected']>0).mean()*100:.1f}%")
print(f"   Test:  {(test['stick_detected']>0).mean()*100:.1f}%")

# Per-class stick detection
print(f"\n2. STICK DETECTION BY CLASS (Train):")
for cls in sorted(train['class'].unique()):
    subset = train[train['class'] == cls]
    rate = (subset['stick_detected'] > 0).mean() * 100
    print(f"   {cls}: {rate:.0f}%")

# Feature variance check
print(f"\n3. LOW VARIANCE FEATURES (potentially useless):")
feature_cols = [c for c in train.columns if c != 'class']
low_var = []
for col in feature_cols:
    var = train[col].var()
    if var < 0.001:
        low_var.append((col, var))
if low_var:
    for col, var in low_var[:10]:
        print(f"   {col}: variance={var:.6f}")
else:
    print("   None detected")

# Check for NaN or constant values in stick features
print(f"\n4. STICK FEATURE ANALYSIS:")
stick_features = ['stick_detected', 'stick_length_norm', 'stick_angle', 
                  'grip_x_norm', 'grip_y_norm', 'tip_x_norm', 'tip_y_norm',
                  'stick_conf', 'keypoint_conf']
for feat in stick_features:
    if feat in train.columns:
        non_zero = (train[feat] != 0).mean() * 100
        mean_val = train[feat].mean()
        std_val = train[feat].std()
        print(f"   {feat}: non-zero={non_zero:.1f}%, mean={mean_val:.3f}, std={std_val:.3f}")

# Check if test set has different characteristics
print(f"\n5. TRAIN vs TEST FEATURE DISTRIBUTION:")
print("   (Large differences = potential data leakage or distribution shift)")
for feat in ['stick_angle', 'stick_length_norm', 'right_elbow_angle', 'left_elbow_angle']:
    if feat in train.columns:
        train_mean = train[feat].mean()
        test_mean = test[feat].mean()
        diff = abs(train_mean - test_mean)
        print(f"   {feat}: train={train_mean:.3f}, test={test_mean:.3f}, diff={diff:.3f}")

# Confusion analysis - which classes are similar?
print(f"\n6. SAMPLES PER CLASS:")
print(f"   Train total: {len(train)}")
print(f"   Test total: {len(test)}")
print(f"   Train/Test ratio: {len(train)/len(test):.1f}x")
