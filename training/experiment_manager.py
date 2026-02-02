
import os
import json
import csv
from datetime import datetime
import shutil
import yaml


class CustomExperimentManager:
    """
    Simple experiment tracking with folders and CSV logs.
    
    Usage:
        exp = CustomExperimentManager(
            experiment_name="pose_classifier",
            description="Testing new augmentation strategy"
        )
        exp.log_config({
            "learning_rate": 0.001,
            "batch_size": 32
        })
        exp.log_metrics(epoch=1, train_acc=0.85, val_acc=0.82)
        exp.save_model("model.keras", "final_model.keras")
        exp.finalize(notes="Best model so far")
    """
    
    def __init__(self, experiment_name, description="", base_dir="experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of experiment type (e.g., "pose_classifier", "stick_detector")
            description: What makes this experiment different
            base_dir: Root directory for all experiments
        """
        self.experiment_name = experiment_name
        self.description = description
        self.base_dir = base_dir
        
        # Create timestamped experiment folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{timestamp}"
        self.experiment_dir = os.path.join(base_dir, self.experiment_id)
        
        # Create folder structure
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "logs"), exist_ok=True)
        
        # Initialize metrics CSV
        self.metrics_file = os.path.join(self.experiment_dir, "metrics.csv")
        self.metrics_headers = []
        
        # Log experiment info
        self.info = {
            "experiment_id": self.experiment_id,
            "experiment_name": experiment_name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "status": "running"
        }
        self._save_info()
        
        print(f"\n{'='*60}")
        print(f"🧪 Experiment Started: {self.experiment_id}")
        print(f"📁 Location: {self.experiment_dir}")
        print(f"{'='*60}\n")
        
    def log_config(self, config):
        """
        Save experiment configuration (hyperparameters, settings).
        
        Args:
            config: Dictionary of configuration parameters
        """
        config_path = os.path.join(self.experiment_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Config saved: config.yaml")
        
    def log_metrics(self, epoch=None, **metrics):
        """
        Log training metrics to CSV.
        
        Args:
            epoch: Epoch number (optional)
            **metrics: Key-value pairs of metrics (train_acc=0.95, val_loss=0.15)
        """
        # Add epoch if provided
        if epoch is not None:
            metrics = {"epoch": epoch, **metrics}
            
        # Create CSV headers on first call
        if not self.metrics_headers:
            self.metrics_headers = list(metrics.keys())
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.metrics_headers)
                writer.writeheader()
        
        # Append metrics
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics_headers)
            writer.writerow(metrics)
            
        # Print summary
        metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                 for k, v in metrics.items()])
        print(f"📊 Metrics logged: {metrics_str}")
        
    def save_model(self, source_path, name=None):
        """
        Copy model to experiment folder.
        
        Args:
            source_path: Path to model file
            name: Optional new name (default: keeps original name)
        """
        if not os.path.exists(source_path):
            print(f"⚠️  Model not found: {source_path}")
            return
            
        dest_name = name if name else os.path.basename(source_path)
        dest_path = os.path.join(self.experiment_dir, "models", dest_name)
        shutil.copy2(source_path, dest_path)
        print(f"💾 Model saved: models/{dest_name}")
        
    def save_artifact(self, source_path, subfolder=None):
        """
        Copy any file (plot, log, dataset info) to experiment.
        
        Args:
            source_path: Path to file
            subfolder: Optional subfolder ("plots", "logs", etc.)
        """
        if not os.path.exists(source_path):
            print(f"⚠️  File not found: {source_path}")
            return
            
        if subfolder:
            dest_dir = os.path.join(self.experiment_dir, subfolder)
            os.makedirs(dest_dir, exist_ok=True)
        else:
            dest_dir = self.experiment_dir
            
        dest_path = os.path.join(dest_dir, os.path.basename(source_path))
        shutil.copy2(source_path, dest_path)
        
        relative_path = os.path.relpath(dest_path, self.experiment_dir)
        print(f"📎 Artifact saved: {relative_path}")
        
    def add_note(self, note):
        """Add text note to experiment log"""
        log_path = os.path.join(self.experiment_dir, "notes.txt")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, 'a') as f:
            f.write(f"[{timestamp}] {note}\n")
        print(f"📝 Note added")
        
    def finalize(self, status="completed", notes=None):
        """
        Mark experiment as complete.
        
        Args:
            status: "completed", "failed", or "stopped"
            notes: Optional final notes
        """
        self.info["status"] = status
        self.info["completed_at"] = datetime.now().isoformat()
        if notes:
            self.info["final_notes"] = notes
        self._save_info()
        
        print(f"\n{'='*60}")
        print(f"✅ Experiment {status.upper()}: {self.experiment_id}")
        if notes:
            print(f"📝 Notes: {notes}")
        print(f"{'='*60}\n")
        
    def _save_info(self):
        """Internal: Save experiment info"""
        info_path = os.path.join(self.experiment_dir, "experiment_info.json")
        with open(info_path, 'w') as f:
            json.dump(self.info, f, indent=2)


class ExperimentComparator:
    """Compare results across multiple experiments"""
    
    def __init__(self, base_dir="experiments"):
        self.base_dir = base_dir
        
    def list_experiments(self, experiment_name=None):
        """
        List all experiments, optionally filtered by name.
        
        Returns:
            List of (folder_path, info_dict) tuples
        """
        if not os.path.exists(self.base_dir):
            return []
            
        experiments = []
        for folder in os.listdir(self.base_dir):
            folder_path = os.path.join(self.base_dir, folder)
            if os.path.isdir(folder_path):
                info_path = os.path.join(folder_path, "experiment_info.json")
                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                    if experiment_name is None or info.get("experiment_name") == experiment_name:
                        experiments.append((folder_path, info))
        
        # Sort by creation time (newest first)
        experiments.sort(key=lambda x: x[1].get("created_at", ""), reverse=True)
        return experiments
    
    def print_summary(self, experiment_name=None):
        """Print a summary of all experiments"""
        experiments = self.list_experiments(experiment_name)
        
        if not experiments:
            print("No experiments found")
            return
            
        print(f"\n{'='*80}")
        print(f"Experiment Summary" + (f" - {experiment_name}" if experiment_name else ""))
        print(f"{'='*80}")
        
        for folder_path, info in experiments:
            folder_name = os.path.basename(folder_path)
            status = info.get("status", "unknown")
            desc = info.get("description", "No description")
            created = info.get("created_at", "unknown")[:19].replace("T", " ")
            
            status_icon = "✅" if status == "completed" else "❌" if status == "failed" else "🔄"
            
            print(f"\n{status_icon} {folder_name}")
            print(f"   Status: {status}")
            print(f"   Created: {created}")
            print(f"   Description: {desc}")
            
            # Show best metrics if available
            metrics_file = os.path.join(folder_path, "metrics.csv")
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        if rows:
                            last_row = rows[-1]
                            print(f"   Metrics: {', '.join([f'{k}={v}' for k, v in last_row.items()])}")
                except:
                    pass
        
        print(f"\n{'='*80}\n")
    
    def compare_metrics(self, experiment_ids, metric_name):
        """
        Compare a specific metric across experiments.
        
        Args:
            experiment_ids: List of experiment folder names
            metric_name: Name of metric to compare (e.g., "val_accuracy")
        """
        print(f"\n📊 Comparing: {metric_name}")
        print("="*60)
        
        results = []
        for exp_id in experiment_ids:
            metrics_file = os.path.join(self.base_dir, exp_id, "metrics.csv")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows and metric_name in rows[0]:
                        values = [float(row[metric_name]) for row in rows if metric_name in row]
                        best_value = max(values)
                        final_value = values[-1]
                        results.append((exp_id, best_value, final_value))
                    else:
                        print(f"{exp_id}: Metric not found")
            else:
                print(f"{exp_id}: No metrics file")
        
        # Sort by best value
        results.sort(key=lambda x: x[1], reverse=True)
        
        for exp_id, best, final in results:
            print(f"{exp_id}")
            print(f"  Best: {best:.4f} | Final: {final:.4f}")
        
        print("="*60)


# Example usage for pose classifier
def example_pose_classifier_training():
    """Example: How to use custom tracking for pose classifier"""
    
    exp = CustomExperimentManager(
        experiment_name="pose_classifier",
        description="Testing dropout=0.5 and batch_size=64"
    )
    
    # Log configuration
    exp.log_config({
        "model": "Dense Neural Network",
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 50,
        "dropout": 0.5,
        "optimizer": "Adam",
        "dataset": "arnis_poses_v2.csv",
        "num_classes": 15,
        "input_features": 99
    })
    
    # Simulate training
    for epoch in range(5):  # In real code, use your actual epochs
        # Your training code here...
        train_acc = 0.85 + epoch * 0.02
        val_acc = 0.82 + epoch * 0.015
        train_loss = 0.5 - epoch * 0.05
        val_loss = 0.55 - epoch * 0.045
        
        exp.log_metrics(
            epoch=epoch + 1,
            train_accuracy=train_acc,
            val_accuracy=val_acc,
            train_loss=train_loss,
            val_loss=val_loss
        )
    
    # Save model and artifacts
    # exp.save_model("models/arnis_coordinates_classifier.keras", "final_model.keras")
    # exp.save_artifact("training_history.png", subfolder="plots")
    # exp.save_artifact("confusion_matrix.png", subfolder="plots")
    
    # Add notes
    exp.add_note("Increased dropout improved generalization")
    exp.add_note("Best model at epoch 45")
    
    # Finalize
    exp.finalize(status="completed", notes="Achieved 95% validation accuracy")


# Example usage for stick detector
def example_stick_detector_training():
    """Example: How to use custom tracking for YOLO stick detector"""
    
    exp = CustomExperimentManager(
        experiment_name="stick_detector",
        description="Increased pose_loss_gain to 15.0"
    )
    
    # Log YOLO configuration
    exp.log_config({
        "model": "yolov8n-pose",
        "epochs": 100,
        "img_size": 640,
        "batch_size": 16,
        "patience": 20,
        "pose_loss_gain": 15.0,
        "box_loss_gain": 7.5,
        "optimizer": "auto",
        "dataset": "roboflow_sticks_v3"
    })
    
    # After training, log final metrics
    exp.log_metrics(
        box_map50=0.87,
        box_map50_95=0.74,
        pose_map50=0.81,
        pose_map50_95=0.68
    )
    
    # Save trained model
    # exp.save_model("runs/pose/arnis_stick_detector/weights/best.pt", "best.pt")
    # exp.save_model("runs/pose/arnis_stick_detector/weights/last.pt", "last.pt")
    
    # Save training plots
    # exp.save_artifact("runs/pose/arnis_stick_detector/results.png", subfolder="plots")
    
    exp.finalize(status="completed", notes="Best mAP so far")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Custom Experiment Tracking System")
    print("="*60)
    print("\n📦 Dependencies:")
    print("   - No external packages needed!")
    print("   - Uses built-in Python libraries only")
    print("\n📁 Folder Structure:")
    print("   experiments/")
    print("     pose_classifier_20260112_143022/")
    print("       ├── experiment_info.json")
    print("       ├── config.yaml")
    print("       ├── metrics.csv")
    print("       ├── notes.txt")
    print("       ├── models/")
    print("       │   └── final_model.keras")
    print("       ├── plots/")
    print("       │   ├── training_history.png")
    print("       │   └── confusion_matrix.png")
    print("       └── logs/")
    print("\n📊 Features:")
    print("   - Simple CSV metrics tracking")
    print("   - YAML config files")
    print("   - Timestamped folders")
    print("   - Easy to backup and share")
    print("\n🚀 Usage:")
    print("   python example_pose_classifier_training()")
    print("   python example_stick_detector_training()")
    print("\n📋 View Experiments:")
    print("   comparator = ExperimentComparator()")
    print("   comparator.print_summary('pose_classifier')")
    print("="*60)
    
    # Run example
    print("\nRunning example...")
    example_pose_classifier_training()
