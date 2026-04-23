"""Analyze front model training history"""
import json
import numpy as np

# Load history
with open('hybrid_classifier/models/history_front.json', 'r') as f:
    h = json.load(f)

# Find best validation epoch
best_epoch = np.argmax(h['val_acc'])
best_val = h['val_acc'][best_epoch]
best_train = h['train_acc'][best_epoch]

print('=== FRONT MODEL TRAINING SUMMARY ===')
print(f'Best validation accuracy: {best_val*100:.1f}% at epoch {best_epoch+1}')
print(f'Training accuracy at best epoch: {best_train*100:.1f}%')
print(f'Gap at best epoch: {(best_train-best_val)*100:.1f}%')
print()
print(f"Final epoch: {len(h['val_acc'])}")
print(f"Final train: {h['train_acc'][-1]*100:.1f}% | Final val: {h['val_acc'][-1]*100:.1f}%")
print()

# Show last few epochs progression
print('Recent validation progression:')
for i in range(max(0, len(h['val_acc'])-5), len(h['val_acc'])):
    print(f"  Epoch {i+1}: val={h['val_acc'][i]*100:.1f}%, train={h['train_acc'][i]*100:.1f}%")

# Training curve summary
print()
print('=== TRAINING CURVE SUMMARY ===')
epochs = len(h['val_acc'])
print(f"Total epochs: {epochs}")
print(f"Best val: {max(h['val_acc'])*100:.1f}% at epoch {np.argmax(h['val_acc'])+1}")
print(f"Val range: {min(h['val_acc'])*100:.1f}% - {max(h['val_acc'])*100:.1f}%")
print(f"Train range: {min(h['train_acc'])*100:.1f}% - {max(h['train_acc'])*100:.1f}%")
