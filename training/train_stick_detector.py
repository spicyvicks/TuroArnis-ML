"""Train YOLOv8 model for arnis stick detection with keypoints.

Research-Backed Best Practices Applied:
1. Confidence Thresholding (75%): Predictions below 0.75 are rejected during inference
   - Ref: Guo et al. (2017) "On Calibration of Modern Neural Networks"
2. Conservative Augmentation: Moderate augmentation to avoid overfitting to augmented distribution
   - Ref: Shorten & Khoshgoftaar (2019) - monitor val_loss for signs of over-augmentation
3. Early Stopping: patience=20 prevents overfitting when val_loss plateaus
"""

from ultralytics import YOLO
import os
import sys
import torch
import shutil

# Set up paths relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)  # Add training folder to path for imports

from experiment_manager import CustomExperimentManager

# ============================================================================
# CONFIGURATION - Research-backed defaults
# ============================================================================
CONFIDENCE_THRESHOLD = 0.75  # Reject predictions below 75% (Guo et al., 2017)
KEYPOINT_CONFIDENCE_THRESHOLD = 0.5  # Keypoints need at least 50% confidence
AUGMENTATION_INTENSITY = "moderate"  # Options: "light", "moderate", "heavy"

def get_augmentation_params(intensity="moderate"):
    """
    Get augmentation parameters based on intensity level.
    
    Research note (Shorten & Khoshgoftaar, 2019):
    - Too much augmentation can cause overfitting to augmented distribution
    - Monitor validation loss - if it increases while training loss decreases, reduce intensity
    
    Args:
        intensity: "light", "moderate", or "heavy"
    
    Returns:
        Dictionary of augmentation parameters
    """
    presets = {
        "light": {
            "hsv_h": 0.01, "hsv_s": 0.4, "hsv_v": 0.3,
            "degrees": 10.0, "translate": 0.05, "scale": 0.3,
            "fliplr": 0.5, "mosaic": 0.5, "mixup": 0.0,
        },
        "moderate": {
            "hsv_h": 0.015, "hsv_s": 0.5, "hsv_v": 0.4,
            "degrees": 15.0, "translate": 0.1, "scale": 0.4,
            "fliplr": 0.5, "mosaic": 0.8, "mixup": 0.0,
        },
        "heavy": {
            "hsv_h": 0.02, "hsv_s": 0.7, "hsv_v": 0.5,
            "degrees": 20.0, "translate": 0.15, "scale": 0.5,
            "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.1,
        }
    }
    return presets.get(intensity, presets["moderate"])


def train_stick_detector(data_yaml_path, epochs=50, img_size=416, batch_size=8, 
                         augmentation_intensity="moderate"):
    """
    Train YOLOv8 pose model for stick detection with keypoints.
    
    Args:
        data_yaml_path: Path to data.yaml file from Roboflow dataset
        epochs: Number of training epochs (default: 50 for faster CPU training)
        img_size: Image size for training (default: 416 for faster CPU training)
        batch_size: Batch size for training (default: 8 for CPU memory efficiency)
        augmentation_intensity: "light", "moderate", or "heavy" (default: moderate)
                               Research suggests monitoring val_loss for over-augmentation
    """
    
    # Initialize experiment manager
    exp = CustomExperimentManager(
        experiment_name="stick_detector",
        description=f"Stick detection training: {epochs} epochs, imgsz={img_size}"
    )
    
    # Detect device
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    
    # Use YOLOv8n-pose as base model (smallest, fastest)
    # Available options: yolov8n-pose.pt, yolov8s-pose.pt, yolov8m-pose.pt, yolov8l-pose.pt, yolov8x-pose.pt
    model = YOLO('yolov8n-pose.pt')
    
    # Get augmentation parameters
    aug_params = get_augmentation_params(augmentation_intensity)
    
    # Log config with research notes
    exp.log_config({
        "model": "yolov8n-pose",
        "epochs": epochs,
        "imgsz": img_size,
        "batch_size": batch_size,
        "device": str(device),
        "data_yaml": data_yaml_path,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "keypoint_confidence_threshold": KEYPOINT_CONFIDENCE_THRESHOLD,
        "augmentation_intensity": augmentation_intensity,
        "augmentation_params": aug_params,
        "research_notes": {
            "confidence_threshold": "Guo et al. (2017) - neural nets often overconfident",
            "augmentation": "Shorten & Khoshgoftaar (2019) - monitor val_loss for over-augmentation",
            "early_stopping": "patience=15 to prevent overfitting",
            "cpu_optimized": "Using imgsz=416, epochs=50, batch=8 for faster local training"
        }
    })
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=os.path.join(project_root, 'runs', 'pose'),
        name='arnis_stick_detector',
        patience=15,  # Early stopping patience (faster convergence check)
        save=True,
        device=device,
        workers=2,  # Reduced workers for CPU
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=True,  # Single class: stick
        rect=False,
        cos_lr=True,  # Cosine learning rate scheduler
        close_mosaic=10,  # Close mosaic augmentation in last 10 epochs
        amp=(device != 'cpu'),  # Automatic Mixed Precision
        fraction=1.0,  # Train on 100% of data
        exist_ok=True,  # Overwrite previous run to maintain consistent path
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
        # Augmentation params from research-backed presets (Shorten & Khoshgoftaar, 2019)
        hsv_h=aug_params["hsv_h"],
        hsv_s=aug_params["hsv_s"],
        hsv_v=aug_params["hsv_v"],
        degrees=aug_params["degrees"],
        translate=aug_params["translate"],
        scale=aug_params["scale"],
        shear=0.0,  # Keep zero - stick detection needs straight lines
        perspective=0.0,  # Keep zero - perspective distorts keypoint labels
        flipud=0.0,  # Keep zero - sticks aren't typically upside down
        fliplr=aug_params["fliplr"],
        mosaic=aug_params["mosaic"],
        mixup=aug_params["mixup"],
        copy_paste=0.0,  # Keep zero - causes label confusion for keypoints
    )
    
    # Validate the model
    metrics = model.val()
    
    # Log metrics to experiment manager
    exp.log_metrics(
        box_map50=metrics.box.map50,
        box_map=metrics.box.map,
        pose_map50=metrics.pose.map50,
        pose_map=metrics.pose.map
    )
    
    # Copy model weights to experiment folder
    best_weights = os.path.join(project_root, 'runs', 'pose', 'arnis_stick_detector', 'weights', 'best.pt')
    if os.path.exists(best_weights):
        exp.save_model(best_weights, "best_stick_detector.pt")
        
    exp.finalize(f"Completed {epochs} epochs. Box mAP50: {metrics.box.map50:.4f}, Pose mAP50: {metrics.pose.map50:.4f}")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best model saved to: {best_weights}")
    print(f"Last model saved to: {os.path.join(project_root, 'runs', 'pose', 'arnis_stick_detector', 'weights', 'last.pt')}")
    print(f"Experiment log: {exp.experiment_dir}")
    print(f"\nValidation Metrics:")
    print(f"Box mAP50: {metrics.box.map50:.4f}")
    print(f"Box mAP50-95: {metrics.box.map:.4f}")
    print(f"Pose mAP50: {metrics.pose.map50:.4f}")
    print(f"Pose mAP50-95: {metrics.pose.map:.4f}")
    
    return model


def test_model(model_path, test_image_path, conf_threshold=None, kpt_conf_threshold=None):
    """
    Test the trained model on a single image with confidence thresholding.
    
    Research note (Guo et al., 2017):
    - Modern neural networks are often overconfident
    - We apply a 75% confidence threshold by default to reduce false positives
    
    Args:
        model_path: Path to trained model weights
        test_image_path: Path to test image
        conf_threshold: Minimum box confidence (default: 0.75 per research)
        kpt_conf_threshold: Minimum keypoint confidence (default: 0.5)
    """
    if conf_threshold is None:
        conf_threshold = CONFIDENCE_THRESHOLD
    if kpt_conf_threshold is None:
        kpt_conf_threshold = KEYPOINT_CONFIDENCE_THRESHOLD
        
    model = YOLO(model_path)
    # Apply confidence threshold during inference
    results = model(test_image_path, conf=conf_threshold)
    
    # Display results
    for result in results:
        # Show image with detections
        result.show()
        
        # Print detection info
        if result.keypoints is not None:
            valid_detections = 0
            print(f"\n[INFO] Confidence threshold: {conf_threshold:.0%} (Guo et al., 2017)")
            print(f"[INFO] Keypoint threshold: {kpt_conf_threshold:.0%}")
            
            for i, (box, kpts) in enumerate(zip(result.boxes, result.keypoints)):
                box_conf = box.conf.item()
                
                # Check keypoint confidence
                kpts_data = kpts.data[0]
                if len(kpts_data) >= 2:
                    grip_conf = kpts_data[0][2].item()
                    tip_conf = kpts_data[1][2].item()
                    
                    if grip_conf < kpt_conf_threshold or tip_conf < kpt_conf_threshold:
                        print(f"\n[REJECTED] Stick {i+1}: Keypoint confidence too low")
                        print(f"  Grip conf: {grip_conf:.3f}, Tip conf: {tip_conf:.3f}")
                        continue
                
                valid_detections += 1
                print(f"\n[ACCEPTED] Stick {valid_detections}:")
                print(f"  Box Confidence: {box_conf:.1%}")
                print(f"  Bbox: {box.xyxy.tolist()}")
                print(f"  Keypoints (x, y, conf):")
                for j, kpt in enumerate(kpts_data):
                    kpt_name = "Grip" if j == 0 else "Tip"
                    print(f"    {kpt_name}: ({kpt[0]:.1f}, {kpt[1]:.1f}, {kpt[2]:.1%})")
            
            print(f"\n[SUMMARY] {valid_detections} valid detection(s) above threshold")
    
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
