
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import matplotlib.pyplot as plt

# --- CONFIG ---
STICK_MODEL_PATH = "runs/pose/arnis_stick_detector/weights/best.pt"
TEST_IMAGE = "dataset_split/train/front/neutral_stance/1.jpg"
OUTPUT_FILE = "graph_visualization.jpg"

# --- MEDIAPIPE SETUP ---
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

def visualize_graph():
    # 1. Load Image
    img = cv2.imread(TEST_IMAGE)
    if img is None:
        print(f"Error: Could not read {TEST_IMAGE}")
        return
    h, w, _ = img.shape
    
    # Copy for drawing
    vis_img = img.copy()

    # 2. Extract Body Pose
    print("Extracting body pose...")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(img_rgb)
    
    body_kpts = []
    if results.pose_landmarks:
        # Draw standard landmarks
        mp_drawing.draw_landmarks(
            vis_img, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS
        )
        
        # Store for connecting to stick
        for lm in results.pose_landmarks.landmark:
            body_kpts.append([int(lm.x * w), int(lm.y * h)])
    else:
        print("No body pose detected!")
        return

    # 3. Extract Stick
    print("Extracting stick...")
    stick_model = YOLO(STICK_MODEL_PATH)
    stick_results = stick_model(TEST_IMAGE, verbose=False)[0]
    
    stick_kpts = []
    if stick_results.keypoints is not None and len(stick_results.keypoints.data) > 0:
        kpts = stick_results.keypoints.data[0].cpu().numpy()
        
        grip_x, grip_y = int(kpts[0][0]), int(kpts[0][1])
        tip_x, tip_y = int(kpts[1][0]), int(kpts[1][1])
        
        stick_kpts = [ (grip_x, grip_y), (tip_x, tip_y) ]
        
        # Draw Stick (Thick Red Line)
        cv2.line(vis_img, (grip_x, grip_y), (tip_x, tip_y), (0, 0, 255), 4)
        cv2.circle(vis_img, (grip_x, grip_y), 8, (0, 255, 0), -1) # Grip = Green
        cv2.circle(vis_img, (tip_x, tip_y), 8, (0, 0, 255), -1)   # Tip = Red
        
        # 4. Draw Graph Connections (Wrist -> Stick)
        # Left Wrist: 15, Right Wrist: 16
        if len(body_kpts) > 16:
            left_wrist = body_kpts[15]
            right_wrist = body_kpts[16]
            
            # Draw blue dashed lines (simulated) connecting wrists to stick grip
            # (In standard OpenCV we just draw lines)
            cv2.line(vis_img, tuple(left_wrist), (grip_x, grip_y), (255, 255, 0), 2)
            cv2.line(vis_img, tuple(right_wrist), (grip_x, grip_y), (255, 255, 0), 2)
            
            cv2.putText(vis_img, "Wrist-Grip Connection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Save
    cv2.imwrite(OUTPUT_FILE, vis_img)
    print(f"Visualization saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    visualize_graph()
