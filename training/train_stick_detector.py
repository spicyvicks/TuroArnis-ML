"""
Train YOLOv8 model for arnis stick detection with keypoints.

The model will detect:
- Stick bounding box
- Stick keypoints (grip point and tip point)

Dataset should be in YOLOv8 format from Roboflow.
"""

from ultralytics import YOLO
import os

def train_stick_detector(data_yaml_path, epochs=100, img_size=640, batch_size=16):
    """
    Train YOLOv8 pose model for stick detection with keypoints.
    
    Args:
        data_yaml_path: Path to data.yaml file from Roboflow dataset
        epochs: Number of training epochs
        img_size: Image size for training
        batch_size: Batch size for training
    """
    
    # Use YOLOv8n-pose as base model (smallest, fastest)
    # Available options: yolov8n-pose.pt, yolov8s-pose.pt, yolov8m-pose.pt, yolov8l-pose.pt, yolov8x-pose.pt
    model = YOLO('yolov8n-pose.pt')
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='arnis_stick_detector',
        patience=20,  # Early stopping patience
        save=True,
        device='cpu',  # Use CPU (change to 0 for GPU if available)
        workers=4,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=True,  # Single class: stick
        rect=False,
        cos_lr=True,  # Cosine learning rate scheduler
        close_mosaic=10,  # Close mosaic augmentation in last 10 epochs
        amp=True,  # Automatic Mixed Precision
        fraction=1.0,  # Train on 100% of data
        profile=False,
        freeze=None,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,  # Box loss gain
        cls=0.5,  # Class loss gain
        dfl=1.5,  # Distribution Focal Loss gain
        pose=12.0,  # Pose loss gain (important for keypoint detection)
        kobj=1.0,  # Keypoint objectness loss gain
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,  # HSV-Saturation augmentation
        hsv_v=0.4,  # HSV-Value augmentation
        degrees=0.0,  # Rotation augmentation (degrees)
        translate=0.1,  # Translation augmentation
        scale=0.5,  # Scale augmentation
        shear=0.0,  # Shear augmentation
        perspective=0.0,  # Perspective augmentation
        flipud=0.0,  # Flip up-down augmentation
        fliplr=0.5,  # Flip left-right augmentation
        mosaic=1.0,  # Mosaic augmentation
        mixup=0.0,  # Mixup augmentation
        copy_paste=0.0,  # Copy-paste augmentation
    )
    
    # Validate the model
    metrics = model.val()
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best model saved to: runs/pose/arnis_stick_detector/weights/best.pt")
    print(f"Last model saved to: runs/pose/arnis_stick_detector/weights/last.pt")
    print(f"\nValidation Metrics:")
    print(f"Box mAP50: {metrics.box.map50:.4f}")
    print(f"Box mAP50-95: {metrics.box.map:.4f}")
    print(f"Pose mAP50: {metrics.pose.map50:.4f}")
    print(f"Pose mAP50-95: {metrics.pose.map:.4f}")
    
    return model


def test_model(model_path, test_image_path):
    """
    Test the trained model on a single image.
    
    Args:
        model_path: Path to trained model weights
        test_image_path: Path to test image
    """
    model = YOLO(model_path)
    results = model(test_image_path)
    
    # Display results
    for result in results:
        # Show image with detections
        result.show()
        
        # Print detection info
        if result.keypoints is not None:
            print(f"\nDetected {len(result.boxes)} stick(s)")
            for i, (box, kpts) in enumerate(zip(result.boxes, result.keypoints)):
                print(f"\nStick {i+1}:")
                print(f"  Confidence: {box.conf.item():.3f}")
                print(f"  Bbox: {box.xyxy.tolist()}")
                print(f"  Keypoints (x, y, conf):")
                for j, kpt in enumerate(kpts.data[0]):
                    print(f"    Point {j+1}: ({kpt[0]:.1f}, {kpt[1]:.1f}, {kpt[2]:.3f})")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Train: python training/train_stick_detector.py <path_to_data.yaml>")
        print("  Test:  python training/train_stick_detector.py <model_path> <test_image_path>")
        print("\nExample:")
        print("  python training/train_stick_detector.py arnis_stick/data.yaml")
        print("  python training/train_stick_detector.py runs/pose/arnis_stick_detector/weights/best.pt test_image.jpg")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        # Training mode
        data_yaml = sys.argv[1]
        if not os.path.exists(data_yaml):
            print(f"Error: data.yaml not found at {data_yaml}")
            print("Please download your Roboflow dataset in YOLOv8 format")
            sys.exit(1)
        
        print(f"Starting training with dataset: {data_yaml}")
        train_stick_detector(data_yaml)
        
    elif len(sys.argv) == 3:
        # Testing mode
        model_path = sys.argv[1]
        test_image = sys.argv[2]
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            sys.exit(1)
        if not os.path.exists(test_image):
            print(f"Error: Test image not found at {test_image}")
            sys.exit(1)
        
        print(f"Testing model: {model_path}")
        print(f"On image: {test_image}")
        test_model(model_path, test_image)
