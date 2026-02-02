import os
import csv
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from experiment_manager import CustomExperimentManager

# paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

csv_train_file = os.path.join(project_root, 'features_train.csv')
csv_val_file = os.path.join(project_root, 'features_val.csv')
csv_test_file = os.path.join(project_root, 'features_test.csv')
models_dir = os.path.join(project_root, 'models')

def plot_training_history(history, save_path, plt):
    plt.figure(figsize=(15, 6))
    # acc
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.ylim(0, 1.05) 
    # loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Model Loss Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n[INFO] Training history plot (linear graph) saved to: {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, plt, sns, confusion_matrix):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('Actual Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] Confusion matrix plot saved to: {save_path}")
    plt.close()

def get_next_version():
    existing = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('v')]
    if not existing:
        return 1
    versions = []
    for d in existing:
        try:
            versions.append(int(d.split('_')[0][1:]))
        except:
            pass
    return max(versions) + 1 if versions else 1

class ExperimentLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, experiment_manager):
        super().__init__()
        self.exp = experiment_manager
        
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            self.exp.log_metrics(
                epoch=epoch + 1,
                train_accuracy=logs.get('accuracy', 0),
                train_loss=logs.get('loss', 0),
                val_accuracy=logs.get('val_accuracy', 0),
                val_loss=logs.get('val_loss', 0)
            )

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  DENSE NEURAL NETWORK TRAINING")
    print("="*50)

    # 1. Check Data
    for csv_file in [csv_train_file, csv_val_file, csv_test_file]:
        if not os.path.exists(csv_file):
            print(f"[ERROR] CSV file not found: {csv_file}")
            print("Run 'python training/run_extraction.py' first.")
            sys.exit(1)

    print("[INFO] Loading pre-split datasets...")
    train_data = pd.read_csv(csv_train_file).dropna()
    val_data = pd.read_csv(csv_val_file).dropna()
    test_data = pd.read_csv(csv_test_file).dropna()
    
    if train_data.empty:
        print("[CRITICAL] Training data is empty!")
        sys.exit(1)

    # 2. Prepare Data
    label_encoder = LabelEncoder()
    # Fit only on training data classes to ensure consistency
    label_encoder.fit(train_data['class'].values)
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)
    
    # helper to process split
    def prepare_split(df, name):
        # filter unknown classes (e.g. if test set has a class not in train set - rare but possible)
        df = df[df['class'].isin(class_names)]
        X = df.drop('class', axis=1).values
        y = label_encoder.transform(df['class'].values)
        return X, y

    X_train, y_train = prepare_split(train_data, 'Train')
    X_val, y_val = prepare_split(val_data, 'Val')
    X_test, y_test = prepare_split(test_data, 'Test')

    num_features = X_train.shape[1]
    print(f"  - Features: {num_features}")
    print(f"  - Classes: {num_classes} {class_names}")
    print(f"  - Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # 3. Scale Features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # 4. Setup Training
    os.makedirs(models_dir, exist_ok=True)
    version_num = get_next_version()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = f"v{version_num:03d}_{timestamp}"
    version_dir = os.path.join(models_dir, version_name)
    os.makedirs(version_dir, exist_ok=True)
    
    model_save_path = os.path.join(version_dir, 'model.keras')
    encoder_path = os.path.join(version_dir, 'label_encoder.joblib')
    scaler_path = os.path.join(version_dir, 'scaler.joblib')

    exp = CustomExperimentManager(
        experiment_name="pose_classifier_dnn",
        description="DNN Training on split dataset",
        base_dir=os.path.join(project_root, 'experiments')
    )
    
    # Log config
    exp.log_config({
        "model_architecture": "Dense Neural Network",
        "input_features": num_features,
        "classes": num_classes,
        "train_samples": len(X_train),
        "learning_rate": 0.001,
        "epochs": 500,
        "batch_size": 16
    })

    # 5. Build Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-5)
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((num_features,)),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.1),
        
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # 6. Train
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=50, monitor='val_accuracy', restore_best_weights=True, mode='max'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=1e-6, mode='min'),
        ExperimentLoggerCallback(exp)
    ]
    
    print("\n[INFO] Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 7. Evaluate
    print("\n" + "="*50)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")
    
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    # 8. Save Artifacts
    history_plot_path = os.path.join(version_dir, 'training_history.png')
    plot_training_history(history, history_plot_path, plt)
    
    cm_plot_path = os.path.join(version_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, class_names, cm_plot_path, plt, sns, confusion_matrix)
    
    print(f"\n[INFO] Saving model to {version_dir}...")
    model.save(model_save_path)
    joblib.dump(label_encoder, encoder_path)
    joblib.dump(scaler, scaler_path)
    
    # Save Metadata
    metadata = {
        'version': version_name,
        'model_type': 'dnn',
        'trained_at': datetime.now().isoformat(),
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'num_classes': num_classes,
        'num_features': num_features,
        'class_names': class_names
    }
    with open(os.path.join(version_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
        
    # Active Model Logic
    active_model_path = os.path.join(models_dir, 'active_model.json')
    should_set_active = True
    if os.path.exists(active_model_path):
        try:
            with open(active_model_path, 'r') as f:
                current = json.load(f)
            if test_acc <= current.get('test_accuracy', 0):
                should_set_active = False
                print(f"[INFO] New model ({test_acc:.2%}) <= Current active ({current.get('test_accuracy', 0):.2%}). Not updating active.")
        except:
            pass
            
    if should_set_active:
        active_config = {
            'version': version_name,
            'path': version_dir,
            'model_path': model_save_path,
            'encoder_path': encoder_path,
            'scaler_path': scaler_path,
            'test_accuracy': float(test_acc),
            'model_type': 'dnn',
            'set_at': datetime.now().isoformat()
        }
        with open(active_model_path, 'w') as f:
            json.dump(active_config, f, indent=2)
        print(f"[INFO] Set as ACTIVE model.")

    exp.finalize(status="completed", notes=f"Test Accuracy: {test_acc:.2%}")
    print("\n[SUCCESS] Training finished.")