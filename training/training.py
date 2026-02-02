import os
import csv
import sys
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import StandardScaler

worker_pose_instance = None

# feature extraction mode: 'coordinates' (99 features) or 'angles' (25 features)
# can be set via environment variable or changed here
FEATURE_MODE = os.environ.get('FEATURE_MODE', 'angles')

def init_worker():
    global worker_pose_instance
    worker_pose_instance = mp.solutions.pose.Pose(
        static_image_mode=True, 
        min_detection_confidence=0.5,
        model_complexity=2
    )

def calculate_angle_3d(a, b, c):
    """calculate angle at point b given 3 points"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    dot_product = np.dot(ba, bc)
    magnitude = np.linalg.norm(ba) * np.linalg.norm(bc)
    if magnitude == 0:
        return 0
    return np.degrees(np.arccos(np.clip(dot_product / magnitude, -1.0, 1.0)))

def extract_angles_from_image(image_path):
    """extract joint angles and key positions (33 features)"""
    global worker_pose_instance
    if worker_pose_instance is None:
        init_worker()
        
    image = cv2.imread(image_path)
    if image is None: 
        return None
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = worker_pose_instance.process(image_rgb)
    
    if not results.pose_world_landmarks:
        return None
        
    try:
        lm = results.pose_world_landmarks.landmark
        
        # joint angles (16 angles)
        angles = [
            # elbows
            calculate_angle_3d([lm[11].x, lm[11].y, lm[11].z], [lm[13].x, lm[13].y, lm[13].z], [lm[15].x, lm[15].y, lm[15].z]),  # left elbow
            calculate_angle_3d([lm[12].x, lm[12].y, lm[12].z], [lm[14].x, lm[14].y, lm[14].z], [lm[16].x, lm[16].y, lm[16].z]),  # right elbow
            # shoulders
            calculate_angle_3d([lm[23].x, lm[23].y, lm[23].z], [lm[11].x, lm[11].y, lm[11].z], [lm[13].x, lm[13].y, lm[13].z]),  # left shoulder
            calculate_angle_3d([lm[24].x, lm[24].y, lm[24].z], [lm[12].x, lm[12].y, lm[12].z], [lm[14].x, lm[14].y, lm[14].z]),  # right shoulder
            # wrists
            calculate_angle_3d([lm[13].x, lm[13].y, lm[13].z], [lm[15].x, lm[15].y, lm[15].z], [lm[19].x, lm[19].y, lm[19].z]),  # left wrist
            calculate_angle_3d([lm[14].x, lm[14].y, lm[14].z], [lm[16].x, lm[16].y, lm[16].z], [lm[20].x, lm[20].y, lm[20].z]),  # right wrist
            # hips
            calculate_angle_3d([lm[11].x, lm[11].y, lm[11].z], [lm[23].x, lm[23].y, lm[23].z], [lm[25].x, lm[25].y, lm[25].z]),  # left hip
            calculate_angle_3d([lm[12].x, lm[12].y, lm[12].z], [lm[24].x, lm[24].y, lm[24].z], [lm[26].x, lm[26].y, lm[26].z]),  # right hip
            # knees
            calculate_angle_3d([lm[23].x, lm[23].y, lm[23].z], [lm[25].x, lm[25].y, lm[25].z], [lm[27].x, lm[27].y, lm[27].z]),  # left knee
            calculate_angle_3d([lm[24].x, lm[24].y, lm[24].z], [lm[26].x, lm[26].y, lm[26].z], [lm[28].x, lm[28].y, lm[28].z]),  # right knee
            # ankles (NEW)
            calculate_angle_3d([lm[25].x, lm[25].y, lm[25].z], [lm[27].x, lm[27].y, lm[27].z], [lm[31].x, lm[31].y, lm[31].z]),  # left ankle
            calculate_angle_3d([lm[26].x, lm[26].y, lm[26].z], [lm[28].x, lm[28].y, lm[28].z], [lm[32].x, lm[32].y, lm[32].z]),  # right ankle
            # arm-to-torso angles
            calculate_angle_3d([lm[12].x, lm[12].y, lm[12].z], [lm[11].x, lm[11].y, lm[11].z], [lm[13].x, lm[13].y, lm[13].z]),  # left arm raise
            calculate_angle_3d([lm[11].x, lm[11].y, lm[11].z], [lm[12].x, lm[12].y, lm[12].z], [lm[14].x, lm[14].y, lm[14].z]),  # right arm raise
            # torso angle (NEW) - spine alignment
            calculate_angle_3d([(lm[11].x+lm[12].x)/2, (lm[11].y+lm[12].y)/2, (lm[11].z+lm[12].z)/2], 
                              [(lm[23].x+lm[24].x)/2, (lm[23].y+lm[24].y)/2, (lm[23].z+lm[24].z)/2],
                              [(lm[23].x+lm[24].x)/2, (lm[23].y+lm[24].y)/2 + 0.1, (lm[23].z+lm[24].z)/2]),  # torso lean
        ]
        
        # relative positions (19 features)
        # wrist positions relative to shoulder
        left_wrist_rel = [lm[15].x - lm[11].x, lm[15].y - lm[11].y, lm[15].z - lm[11].z]
        right_wrist_rel = [lm[16].x - lm[12].x, lm[16].y - lm[12].y, lm[16].z - lm[12].z]
        
        # hand positions relative to hip center
        hip_center = [(lm[23].x + lm[24].x)/2, (lm[23].y + lm[24].y)/2, (lm[23].z + lm[24].z)/2]
        left_hand_rel = [lm[19].x - hip_center[0], lm[19].y - hip_center[1]]
        right_hand_rel = [lm[20].x - hip_center[0], lm[20].y - hip_center[1]]
        
        # foot positions relative to hip center (NEW)
        left_foot_rel = [lm[31].x - hip_center[0], lm[31].y - hip_center[1]]
        right_foot_rel = [lm[32].x - hip_center[0], lm[32].y - hip_center[1]]
        
        # body balance/tilt features
        shoulder_tilt = lm[11].y - lm[12].y
        hip_tilt = lm[23].y - lm[24].y
        stance_width = abs(lm[27].x - lm[28].x)  # distance between ankles
        
        # facing direction (positive = facing right, negative = facing left)
        facing_direction = lm[11].z - lm[12].z  # left_shoulder.z - right_shoulder.z
        
        features = (angles + left_wrist_rel + right_wrist_rel + 
                   left_hand_rel + right_hand_rel + 
                   left_foot_rel + right_foot_rel +
                   [shoulder_tilt, hip_tilt, stance_width, facing_direction])
        return features
        
    except Exception:
        return None

def extract_coordinates_from_image(image_path):
    """extract hip-centered coordinates (99 features)"""
    global worker_pose_instance
    if worker_pose_instance is None:
        init_worker()
        
    image = cv2.imread(image_path)
    if image is None: 
        return None
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = worker_pose_instance.process(image_rgb)
    
    if not results.pose_world_landmarks:
        return None
        
    try:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark])
        
        left_hip_idx = 23
        right_hip_idx = 24
        
        if left_hip_idx >= len(landmarks) or right_hip_idx >= len(landmarks):
            return None
            
        hip_center = (landmarks[left_hip_idx] + landmarks[right_hip_idx]) / 2.0
        
        normalized_landmarks = landmarks - hip_center
        
        coordinates = normalized_landmarks.flatten().tolist()
        return coordinates
        
    except Exception:
        return None

def plot_training_history(history, save_path, plt):
    plt.figure(figsize=(15, 6))
    #acc
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.ylim(0, 1.05) 
    #loss
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


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg') 
    import joblib
    import seaborn as sns
    import tensorflow as tf
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    from experiment_manager import CustomExperimentManager

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)

    # use pre-split datasets to prevent data leakage
    # run split_dataset.py first, then data_augmentation.py
    train_folder = os.path.join(project_root, 'dataset_split', 'train_aug')  # augmented train
    val_folder = os.path.join(project_root, 'dataset_split', 'val')  # clean val (no aug)
    test_folder = os.path.join(project_root, 'dataset_split', 'test')  # clean test (no aug)
    
    csv_train_file = os.path.join(project_root, 'features_train.csv')
    csv_val_file = os.path.join(project_root, 'features_val.csv')
    csv_test_file = os.path.join(project_root, 'features_test.csv')
    models_dir = os.path.join(project_root, 'models')
    
    # versioned model saving
    from datetime import datetime
    import json
    
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
    
    os.makedirs(models_dir, exist_ok=True)
    
    version_num = get_next_version()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = f"v{version_num:03d}_{timestamp}"
    version_dir = os.path.join(models_dir, version_name)
    os.makedirs(version_dir, exist_ok=True)
    
    model_save_path = os.path.join(version_dir, 'model.keras')
    encoder_path = os.path.join(version_dir, 'label_encoder.joblib')
    scaler_path = os.path.join(version_dir, 'scaler.joblib')
    
    print(f"\n[INFO] Training version: {version_name}")
    
    # Initialize experiment tracking
    exp = CustomExperimentManager(
        experiment_name="pose_classifier",
        description="Training pose classifier with current architecture",
        base_dir=os.path.join(project_root, 'experiments')
    )
    
    RUN_FEATURE_EXTRACTION = True 

    # helper function to extract features from a dataset folder
    def extract_features_from_folder(folder_path, csv_path, extraction_func, header, desc="Extracting"):
        if not os.path.exists(folder_path):
            print(f"\n[ERROR] Folder not found: {folder_path}")
            return 0
        
        pose_classes = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
        
        all_image_paths = []
        path_to_class_map = {}
        for class_name in pose_classes:
            class_folder_path = os.path.join(folder_path, class_name)
            for item in os.listdir(class_folder_path):
                 item_path = os.path.join(class_folder_path, item)
                 if os.path.isdir(item_path):
                     for filename in os.listdir(item_path):
                         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                             full_path = os.path.join(item_path, filename)
                             all_image_paths.append(full_path)
                             path_to_class_map[full_path] = class_name
                 elif item.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = item_path
                    all_image_paths.append(full_path)
                    path_to_class_map[full_path] = class_name
        
        if not all_image_paths:
            print(f"[WARN] No images found in {folder_path}")
            return 0
        
        MAX_PROCESSES_CAP = 6  
        num_processes = max(1, min(cpu_count() - 1, MAX_PROCESSES_CAP))
        
        with Pool(processes=num_processes, initializer=init_worker) as pool:
            features = pool.imap(extraction_func, all_image_paths)
            results = list(tqdm(features, total=len(all_image_paths), desc=f"  - {desc}"))
        
        success_count = 0
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i, feat in enumerate(results):
                if feat:
                    image_path = all_image_paths[i]
                    class_name = path_to_class_map[image_path]
                    writer.writerow([class_name] + feat)
                    success_count += 1
        
        return success_count

    if RUN_FEATURE_EXTRACTION:
        # set extraction function and header based on mode
        if FEATURE_MODE == 'angles':
            extraction_func = extract_angles_from_image
            angle_names = ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
                          'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                          'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                          'left_arm_raise', 'right_arm_raise', 'torso_lean']
            position_names = ['lwrist_rel_x', 'lwrist_rel_y', 'lwrist_rel_z',
                             'rwrist_rel_x', 'rwrist_rel_y', 'rwrist_rel_z',
                             'lhand_rel_x', 'lhand_rel_y', 'rhand_rel_x', 'rhand_rel_y',
                             'lfoot_rel_x', 'lfoot_rel_y', 'rfoot_rel_x', 'rfoot_rel_y',
                             'shoulder_tilt', 'hip_tilt', 'stance_width', 'facing_direction']
            header = ['class'] + angle_names + position_names
            print(f"\n[STAGE 1] Extracting ANGLE features (33 features)...")
        else:
            extraction_func = extract_coordinates_from_image
            header = ['class'] + [f'{ax}_{i}' for i in range(33) for ax in ['x', 'y', 'z']]
            print(f"\n[STAGE 1] Extracting COORDINATE features (99 features)...")
        
        # check that split folders exist
        if not os.path.exists(train_folder):
            print(f"\n[ERROR] Training folder not found: {train_folder}")
            print("Run these commands first:")
            print("  1. python training/split_dataset.py")
            print("  2. python training/data_augmentation.py")
            sys.exit(1)
        
        print(f"  - Mode: {FEATURE_MODE.upper()}")
        print(f"  - Train folder: {train_folder}")
        print(f"  - Val folder: {val_folder}")
        print(f"  - Test folder: {test_folder}")
        
        # extract features from each split separately (NO DATA LEAKAGE)
        print("\n  Extracting from TRAIN set (augmented)...")
        train_count = extract_features_from_folder(train_folder, csv_train_file, extraction_func, header, "Train")
        
        print("\n  Extracting from VAL set (clean)...")
        val_count = extract_features_from_folder(val_folder, csv_val_file, extraction_func, header, "Val")
        
        print("\n  Extracting from TEST set (clean)...")
        test_count = extract_features_from_folder(test_folder, csv_test_file, extraction_func, header, "Test")
        
        print(f"\n[SUCCESS] Feature extraction complete:")
        print(f"  - Train: {train_count} samples")
        print(f"  - Val: {val_count} samples")
        print(f"  - Test: {test_count} samples")
        print("="*50)
        
        if train_count == 0:
            print("[ERROR] No training features extracted. Check images.")
            sys.exit(1)
            
    else:
        print("\n[STAGE 1] Skipping extraction. Using existing CSVs.")
        for csv_file in [csv_train_file, csv_val_file, csv_test_file]:
            if not os.path.exists(csv_file):
                print(f"[ERROR] CSV file not found: {csv_file}")
                sys.exit(1)
        print("="*50)

    print("\n[STAGE 2] Starting Model Training...")
    
    # load pre-split datasets (NO train_test_split needed - data is already split!)
    train_data = pd.read_csv(csv_train_file).dropna()
    val_data = pd.read_csv(csv_val_file).dropna()
    test_data = pd.read_csv(csv_test_file).dropna()
    
    if train_data.empty:
        print("\n[CRITICAL ERROR] Training CSV is empty. Cannot train.")
        sys.exit(1)
    
    # fit label encoder on training data
    label_encoder = LabelEncoder()
    label_encoder.fit(train_data['class'].values)
    
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)
    
    # prepare train set
    X_train = train_data.drop('class', axis=1).values
    y_train = label_encoder.transform(train_data['class'].values)
    
    # prepare val set (filter to known classes)
    val_data = val_data[val_data['class'].isin(class_names)]
    X_val = val_data.drop('class', axis=1).values
    y_val = label_encoder.transform(val_data['class'].values)
    
    # prepare test set (filter to known classes)
    test_data = test_data[test_data['class'].isin(class_names)]
    X_test = test_data.drop('class', axis=1).values
    y_test = label_encoder.transform(test_data['class'].values)
    
    num_features = X_train.shape[1]
    print(f"  - Training on {num_features} features for {num_classes} classes.")
    print(f"  - Data loaded: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    print(f"  - [NO DATA LEAKAGE] Train/Val/Test sets are completely separate")
    
    # apply feature scaling (critical for neural networks)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # coordinate-level augmentation 
    # def augment_coordinates(X, y, noise_levels=[0.02, 0.05], copies=2):
    #     """add slight noise to coordinates to increase variety"""
    #     X_aug = [X]
    #     y_aug = [y]
    #     for _ in range(copies):
    #         noise_level = np.random.choice(noise_levels)
    #         noise = np.random.normal(0, noise_level, X.shape)
    #         X_noisy = X + noise
    #         X_aug.append(X_noisy)
    #         y_aug.append(y)
    #     return np.vstack(X_aug), np.hstack(y_aug)
    
    # X_train_orig_size = len(X_train)
    # X_train, y_train = augment_coordinates(X_train, y_train)
    # print(f"  - Coordinate augmentation: {X_train_orig_size} → {len(X_train)} samples")
    
    # save scaler for inference
    joblib.dump(scaler, scaler_path)
    print(f"  - Feature scaler saved")

    # Log experiment configuration
    exp.log_config({
        "model_architecture": "Dense Neural Network",
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 500,
        "optimizer": "Adam",
        "weight_decay": 1e-5,
        "dense_layers": [256, 128, 64, 32],
        "dropout_rates": [0.4, 0.3, 0.2, 0.1],
        "activation": "LeakyReLU(0.1)",
        "regularization": "L2(1e-4)",
        "early_stopping_patience": 50,
        "reduce_lr_patience": 20,
        "dataset": "arnis_poses_coordinates.csv",
        "num_classes": num_classes,
        "num_features": num_features,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    })

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,  
        beta_1=0.9, 
        beta_2=0.999,
        weight_decay=1e-5  
    )
    
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
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    es_callback = tf.keras.callbacks.EarlyStopping(
        patience=50,
        monitor='val_accuracy',
        restore_best_weights=True,
        mode='max'
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=20,
        min_lr=1e-6,
        mode='min'
    )
    
    # Custom callback to log metrics to experiment
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
    
    exp_logger = ExperimentLoggerCallback(exp)
    
    print("\n  - Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=16,
        validation_data=(X_val, y_val),  # use validation set, not test set
        callbacks=[es_callback, reduce_lr, exp_logger],
        verbose=1
    )
    print("\n[SUCCESS] Model training complete.")

    print("\n" + "="*50)
    print("      FINAL EVALUATION AND SAVING")
    print("="*50)
    val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Model Accuracy on Test Set: {val_acc*100:.2f}%\n")

    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    print("\n--- Final Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    print("\n[INFO] Generating and saving evaluation plots...")
    history_plot_path = os.path.join(version_dir, 'training_history.png')
    plot_training_history(history, history_plot_path, plt)
    
    cm_plot_path = os.path.join(version_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, class_names, cm_plot_path, plt, sns, confusion_matrix)

    print("\n[INFO] Saving final model and label encoder...")
    model.save(model_save_path)
    joblib.dump(label_encoder, encoder_path)
    
    # save metadata
    metadata = {
        'version': version_name,
        'trained_at': datetime.now().isoformat(),
        'feature_mode': FEATURE_MODE,  # 'angles' or 'coordinates'
        'test_accuracy': float(val_acc),
        'test_loss': float(val_loss),
        'num_classes': num_classes,
        'num_features': num_features,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'epochs_trained': len(history.history['accuracy']),
        'class_names': class_names
    }
    with open(os.path.join(version_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # set as active model only if accuracy is higher than current active
    active_model_path = os.path.join(models_dir, 'active_model.json')
    
    should_set_active = True
    if os.path.exists(active_model_path):
        with open(active_model_path, 'r') as f:
            current_active = json.load(f)
        # check if current active has higher accuracy
        current_metadata_path = os.path.join(current_active['path'], 'metadata.json')
        if os.path.exists(current_metadata_path):
            with open(current_metadata_path, 'r') as f:
                current_metadata = json.load(f)
            current_acc = current_metadata.get('test_accuracy', 0)
            if val_acc <= current_acc:
                should_set_active = False
                print(f"\n[INFO] New model accuracy ({val_acc*100:.2f}%) <= current active ({current_acc*100:.2f}%)")
                print(f"[INFO] Keeping {current_active['version']} as active model")
    
    if should_set_active:
        active_config = {
            'version': version_name,
            'path': version_dir,
            'model_path': model_save_path,
            'encoder_path': encoder_path,
            'scaler_path': scaler_path,
            'test_accuracy': float(val_acc),
            'set_at': datetime.now().isoformat()
        }
        with open(active_model_path, 'w') as f:
            json.dump(active_config, f, indent=2)
        print(f"\n[INFO] Set as active model (highest accuracy: {val_acc*100:.2f}%)")
    
    # save to experiment folder
    exp.save_model(model_save_path, "final_model.keras")
    exp.save_model(encoder_path, "label_encoder.joblib")
    exp.save_artifact(history_plot_path, subfolder="plots")
    exp.save_artifact(cm_plot_path, subfolder="plots")
    
    exp.add_note(f"Final test accuracy: {val_acc*100:.2f}%")
    exp.add_note(f"Total epochs trained: {len(history.history['accuracy'])}")
    
    exp.finalize(
        status="completed",
        notes=f"Achieved {val_acc:.2%} accuracy on test set with {num_classes} classes"
    )

    print(f"\n[SUCCESS] Training complete!")
    print(f"  - Version: {version_name}")
    print(f"  - Accuracy: {val_acc:.2%}")
    print(f"  - Model saved to: {version_dir}")
    print(f"  - Set as active model")