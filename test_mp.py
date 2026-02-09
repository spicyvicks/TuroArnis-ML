import mediapipe as mp
print(f"MediaPipe version: {mp.__version__}")
try:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    print("Pose initialized successfully")
except Exception as e:
    print(f"Error initializing Pose: {e}")
