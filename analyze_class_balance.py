"""Analyze per-class accuracy for front view model"""
import json
import numpy as np

# Load training features to get class distribution
import torch

data = torch.load('hybrid_classifier/hybrid_features_v3/train_features_front.pt')
labels = data['labels'].numpy()

print('=== TRAINING DATA CLASS DISTRIBUTION ===')
class_counts = np.bincount(labels, minlength=13)
CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct', 'neutral'
]

for i, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
    pct = count / len(labels) * 100
    marker = " <-- ZERO-STICK" if i in [0, 1, 2, 3, 12] else ""
    print(f'{i:2d}: {name:35s} {count:4d} ({pct:5.1f}%){marker}')

print(f'\nTotal samples: {len(labels)}')

# Check zero-stick samples
data_test = torch.load('hybrid_classifier/hybrid_features_v3/test_features_front.pt')
test_labels = data_test['labels'].numpy()
test_counts = np.bincount(test_labels, minlength=13)

print('\n=== TEST DATA CLASS DISTRIBUTION ===')
for i, (name, count) in enumerate(zip(CLASS_NAMES, test_counts)):
    pct = count / len(test_labels) * 100
    marker = " <-- ZERO-STICK" if i in [0, 1, 2, 3, 12] else ""
    print(f'{i:2d}: {name:35s} {count:3d} ({pct:5.1f}%){marker}')

print(f'\nTotal test samples: {len(test_labels)}')

# Summary
zero_stick_train = sum(class_counts[i] for i in [0, 1, 2, 3, 12])
zero_stick_test = sum(test_counts[i] for i in [0, 1, 2, 3, 12])

print(f'\n=== ZERO-STICK CLASSES SUMMARY (0,1,2,3,12) ===')
print(f'Training: {zero_stick_train}/{len(labels)} ({zero_stick_train/len(labels)*100:.1f}%)')
print(f'Test:     {zero_stick_test}/{len(test_labels)} ({zero_stick_test/len(test_labels)*100:.1f}%)')

# Note: To get per-class accuracy, we would need to load the model and run evaluation
# with sklearn.metrics.classification_report or similar
print('\nNote: Load model and run detailed evaluation to get per-class accuracy breakdown')
