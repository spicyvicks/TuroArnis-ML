import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import joblib
from sklearn.preprocessing import LabelEncoder

# add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from training.geopose.weapon_geometry import correct_weapon_geometry
from training.geopose.graph_builder import ArnisGraphBuilder

# mediapipe import with protobuf fix
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import mediapipe as mp
mp_pose = mp.solutions.pose

def process_dataset(dataset_root, save_dir):
    # path to stick detector
    stick_model_path = os.path.join(project_root, 'runs', 'pose', 'arnis_stick_detector', 'weights', 'best.pt')
    if not os.path.exists(stick_model_path):
        print(f"Error: Stick model not found at {stick_model_path}")
        return

    print("Initializing models...")
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)
    stick_model = YOLO(stick_model_path)
    graph_builder = ArnisGraphBuilder()
    
    data_list = []
    dataset_path = os.path.join(project_root, dataset_root)
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found at {dataset_path}")
        return

    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    class_map = {cls: i for i, cls in enumerate(classes)}
    
    print(f"Classes found ({len(classes)}): {classes}")
    
    # save LabelEncoder
    save_path = os.path.join(project_root, save_dir)
    os.makedirs(save_path, exist_ok=True)
    le = LabelEncoder()
    le.fit(classes)
    joblib.dump(le, os.path.join(save_path, 'label_encoder.joblib'))
    
    for cls_name in classes:
        cls_dir = os.path.join(dataset_path, cls_name)
        
        label = class_map[cls_name]
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.png'))]
        
        print(f"Processing {cls_name} ({len(images)} images)...")
        
        for img_name in tqdm(images, desc=cls_name):
            img_path = os.path.join(cls_dir, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            h, w = img.shape[:2]
            
            # mp pose
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            if not results.pose_landmarks: continue
            
            mp_landmarks = results.pose_landmarks.landmark
            
            # Convert MP (33) to COCO (17) format
            # [Nose, LEye, REye, LEar, REar, LSho, RSho, LElb, RElb, LWri, RWri, LHip, RHip, LKnee, RKnee, LAnk, RAnk]
            # MP Indices: 0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
            coco_indices = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            
            body_kpts = []
            for idx in coco_indices:
                lm = mp_landmarks[idx]
                body_kpts.append([lm.x * w, lm.y * h])
            body_kpts = np.array(body_kpts, dtype=np.float32) # (17, 2)
            
            # yolo stick
            stick_results = stick_model(img, verbose=False)
            
            grip = np.array([0,0], dtype=np.float32)
            tip = np.array([0,0], dtype=np.float32)
            
            # extract stick keypoints
            found_stick = False
            if stick_results and stick_results[0].keypoints is not None:
                kpts_data = stick_results[0].keypoints.data
                if len(kpts_data) > 0:
                    kpts = kpts_data[0].cpu().numpy()
                    if len(kpts) >= 2:
                        grip = kpts[0][:2]
                        tip = kpts[1][:2]
                        found_stick = True
            
            # geometry correction
            # Using COCO indices: LWrist=9, RWrist=10, LElbow=7, RElbow=8
            l_wrist = body_kpts[9]
            r_wrist = body_kpts[10]
            l_elbow = body_kpts[7]
            r_elbow = body_kpts[8]
            
            # only correct if stick is detected (check for non-zero coordinates if confidence missing)
            # YOLO kpts usually include confidence, but we sliced it.
            # Let's assume if found_stick is True, we have points.
            if found_stick:
                d_l = np.linalg.norm(grip - l_wrist)
                d_r = np.linalg.norm(grip - r_wrist)
                if d_l < d_r:
                    wrist, elbow = l_wrist, l_elbow
                else:
                    wrist, elbow = r_wrist, r_elbow
                    
                corrected_tip = correct_weapon_geometry(grip, tip, wrist, elbow)
                tip = corrected_tip
            
            # Prepare Weapon Keypoints (2, 2)
            weapon_kpts = np.array([grip, tip], dtype=np.float32)
            
            # build graph
            # Pass numpy arrays: body (17,2), weapon (2,2)
            data = graph_builder.build_graph(body_kpts, weapon_kpts, label=label)
            data_list.append(data)
    
    pose.close()
    
    # split
    total = len(data_list)
    if total == 0:
        print("No samples processed!")
        return

    # Stratified Split (Manual implementation or shuffle)
    indices = torch.randperm(total)
    split = int(0.8 * total)
    train_indices = indices[:split]
    test_indices = indices[split:]
    
    train_data = [data_list[i] for i in train_indices]
    test_data = [data_list[i] for i in test_indices]
    
    torch.save(train_data, os.path.join(save_path, 'train_graphs.pt'))
    torch.save(test_data, os.path.join(save_path, 'test_graphs.pt'))
    
    # save dataset info
    import json
    info = {
        'num_train': len(train_data),
        'num_test': len(test_data),
        'num_classes': len(classes),
        'classes': classes
    }
    with open(os.path.join(save_path, 'dataset_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Saved {len(train_data)} train and {len(test_data)} test graphs to {save_path}")

if __name__ == '__main__':
    process_dataset('dataset', 'data/processed')
