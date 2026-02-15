"""
Real-Time Inference Script for Spatial GCN
Processes webcam feed with temporal smoothing using deque buffer
"""

import cv2
import torch
import mediapipe as mp
import numpy as np
from collections import deque
from pathlib import Path
from ultralytics import YOLO
from torch_geometric.data import Data, Batch

import sys
sys.path.append('models')
from spatial_gcn import SpatialGCN


# Class names
CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]

# Skeleton edges (same as in create_graph_dataset.py)
SKELETON_EDGES = [
    (11, 12), (12, 11), (11, 23), (23, 11), (12, 24), (24, 12),
    (23, 24), (24, 23), (11, 13), (13, 11), (13, 15), (15, 13),
    (12, 14), (14, 12), (14, 16), (16, 14), (23, 25), (25, 23),
    (25, 27), (27, 25), (24, 26), (26, 24), (26, 28), (28, 26),
    (0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 7), (7, 3),
    (15, 33), (33, 15), (16, 33), (33, 16), (33, 34), (34, 33),
]


class RealtimeGCNInference:
    """Real-time pose classification with temporal smoothing."""
    
    def __init__(
        self,
        model_path,
        stick_detector_path='runs/pose/arnis_stick_detector/weights/best.pt',
        buffer_size=10,
        device='cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load GCN model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = SpatialGCN(
            in_channels=3,
            hidden_channels=64,
            num_classes=len(CLASS_NAMES),
            dropout=0.5
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✓ Loaded GCN model from {model_path}")
        
        # MediaPipe pose
        mp_pose = mp.solutions.pose
        self.pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Stick detector
        self.stick_detector = YOLO(stick_detector_path)
        print(f"✓ Loaded stick detector from {stick_detector_path}")
        
        # Prediction buffer for temporal smoothing
        self.prediction_buffer = deque(maxlen=buffer_size)
        
        # MediaPipe drawing
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp_pose
    
    def extract_pose_keypoints(self, frame):
        """Extract pose keypoints from frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
        
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.visibility])
        
        return np.array(keypoints, dtype=np.float32), results
    
    def detect_stick(self, frame):
        """Detect stick in frame."""
        results = self.stick_detector(frame, verbose=False)
        
        if len(results[0].boxes) > 0:
            bbox = results[0].boxes[0].xyxy[0].cpu().numpy()
            conf = results[0].boxes[0].conf[0].cpu().numpy()
            
            h, w = frame.shape[:2]
            stick_top = [bbox[0] / w, bbox[1] / h, float(conf)]
            stick_bottom = [bbox[2] / w, bbox[3] / h, float(conf)]
            
            return np.array([stick_top, stick_bottom], dtype=np.float32)
        else:
            return np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], dtype=np.float32)
    
    def build_graph(self, pose_keypoints, stick_nodes):
        """Build graph from keypoints and stick."""
        node_features = np.vstack([pose_keypoints, stick_nodes])
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t().contiguous()
        
        graph = Data(x=x, edge_index=edge_index)
        return graph
    
    def predict(self, frame):
        """Predict pose class for frame."""
        # Extract pose
        pose_result = self.extract_pose_keypoints(frame)
        if pose_result is None:
            return None, 0.0, None
        
        pose_keypoints, pose_landmarks = pose_result
        
        # Detect stick
        stick_nodes = self.detect_stick(frame)
        
        # Build graph
        graph = self.build_graph(pose_keypoints, stick_nodes)
        
        # Predict
        batch = Batch.from_data_list([graph]).to(self.device)
        
        with torch.no_grad():
            logits = self.model(batch.x, batch.edge_index, batch.batch)
            probs = torch.softmax(logits, dim=-1)[0]
        
        return probs.cpu(), pose_landmarks, stick_nodes
    
    def get_stable_prediction(self, frame):
        """Get temporally smoothed prediction."""
        probs, pose_landmarks, stick_nodes = self.predict(frame)
        
        if probs is None:
            return None, 0.0, None, None
        
        # Add to buffer
        self.prediction_buffer.append(probs)
        
        # Average predictions
        if len(self.prediction_buffer) > 0:
            avg_probs = torch.stack(list(self.prediction_buffer)).mean(dim=0)
            final_class = avg_probs.argmax().item()
            confidence = avg_probs[final_class].item()
        else:
            final_class = probs.argmax().item()
            confidence = probs[final_class].item()
        
        return final_class, confidence, pose_landmarks, stick_nodes
    
    def draw_results(self, frame, class_idx, confidence, pose_landmarks, stick_nodes):
        """Draw pose, stick, and prediction on frame."""
        # Draw pose landmarks
        if pose_landmarks is not None:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_landmarks.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        
        # Draw stick bounding box
        if stick_nodes[0, 2] > 0.0:  # If stick detected
            h, w = frame.shape[:2]
            x1, y1 = int(stick_nodes[0, 0] * w), int(stick_nodes[0, 1] * h)
            x2, y2 = int(stick_nodes[1, 0] * w), int(stick_nodes[1, 1] * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Stick: {stick_nodes[0, 2]:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        # Draw prediction
        if class_idx is not None:
            class_name = CLASS_NAMES[class_idx].replace('_correct', '').replace('_', ' ').title()
            
            # Background rectangle
            cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)
            
            # Text
            cv2.putText(
                frame,
                f"Pose: {class_name}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"Confidence: {confidence:.2%}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return frame
    
    def run(self, camera_id=0):
        """Run real-time inference on webcam."""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        print("\n" + "=" * 60)
        print("Real-Time Pose Classification")
        print("=" * 60)
        print("Press 'q' to quit")
        print("=" * 60 + "\n")
        
        fps_buffer = deque(maxlen=30)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Start timer
            start_time = cv2.getTickCount()
            
            # Get prediction
            class_idx, confidence, pose_landmarks, stick_nodes = self.get_stable_prediction(frame)
            
            # Draw results
            frame = self.draw_results(frame, class_idx, confidence, pose_landmarks, stick_nodes)
            
            # Calculate FPS
            end_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (end_time - start_time)
            fps_buffer.append(fps)
            avg_fps = np.mean(fps_buffer)
            
            # Draw FPS
            cv2.putText(
                frame,
                f"FPS: {avg_fps:.1f}",
                (frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Show frame
            cv2.imshow('Arnis Pose Classification', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Inference stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time GCN inference")
    parser.add_argument(
        '--model',
        type=str,
        default='models/gcn_checkpoints/best_model.pth',
        help='Path to trained GCN model'
    )
    parser.add_argument(
        '--stick_detector',
        type=str,
        default='runs/pose/arnis_stick_detector/weights/best.pt',
        help='Path to stick detector'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera ID'
    )
    parser.add_argument(
        '--buffer_size',
        type=int,
        default=10,
        help='Prediction buffer size for temporal smoothing'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Please train the model first using train_gcn.py")
        exit(1)
    
    # Run inference
    inference = RealtimeGCNInference(
        model_path=args.model,
        stick_detector_path=args.stick_detector,
        buffer_size=args.buffer_size
    )
    
    inference.run(camera_id=args.camera)
