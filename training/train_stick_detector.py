from ultralytics import YOLO
import torch

def train_stick_detector():
    """
    Train a YOLOv8-Pose model to detect arnis sticks (2 keypoints: grip and tip).
    """
    # 1. Load the model
    # Using 'yolov8n-pose.pt' (nano) for speed, or 'yolov8s-pose.pt' (small) for better accuracy
    print("Loading YOLOv8-Pose model...")
    model = YOLO('yolov8n-pose.pt')  

    # 2. Train the model
    print("Starting training...")
    try:
        results = model.train(
            data='dataset_stick/data.yaml',  # Path to dataset config
            epochs=100,                      # 100 epochs should be sufficient
            imgsz=640,                       # Image size
            batch=16,                        # Batch size
            project='runs/pose',             # Output project directory
            name='arnis_stick_detector_v2',  # Experiment name (all lowercase/underscores to be safe)
            exist_ok=True,                   # Overwrite if exists
            device=0 if torch.cuda.is_available() else 'cpu', # Use GPU if available
            patience=15,                     # Early stopping
            augment=True,                    # Default augmentation
        )
        print("Training completed successfully!")
        print(f"Best model saved to: {results.save_dir}/weights/best.pt")
        
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == '__main__':
    # Ensure dataset_stick/data.yaml exists and is correct before running
    train_stick_detector()
