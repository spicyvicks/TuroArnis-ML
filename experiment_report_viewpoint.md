# Experiment Report: Viewpoint-Specific Training (Branch: viewpoint-tuning)

**Date**: 2026-02-09
**Objective**: Improve pose classification accuracy by training specialized models for each camera viewpoint (Front, Left, Right).

## 1. Methodologies Implemented

### A. Feature Extraction (`extract_features_enriched.py`)
- **Engine**: MediaPipe BlazePose (Full).
- **Features**: 99 total (33 body keypoints + engineered angles + stick features).
- **Viewpoint Parsing**: Automatically inferred `front`, `left`, `right` metadata from directory structure.
- **Stick Logic**: Heuristic-based connection (Wrist -> Stick Tip) combined with Object Detection.

### B. Data Augmentation
- **3D Rotation (Train Only)**: Generated 5 synthetic views per image by treating the 3D skeleton as a point cloud and rotating it ±15 degrees.
- **Offline Mixup (Train Only)**: Generated 3x data by interpolating features between random pairs of the same class (Beta distribution).
- **Test Set**: Kept "Pure" (No augmentation) to ensure realistic evaluation.

### C. Training Strategy (`training_viewpoint_tuned.py`)
- **Algorithm**: XGBoost (Selected over Random Forest).
- **Hyperparameter Tuning**: "Goldilocks" Search (RandomizedSearchCV) optimizing:
  - `max_depth`: [4, 5, 6]
  - `learning_rate`: [0.03, 0.05, 0.1]
  - `subsample`/`colsample`: [0.6 - 0.8]
- **Splitting**: Models trained on `dataset_augmented`, evaluated on `dataset_split/test`.

## 2. Results Summary

| Viewpoint | Model | Train Acc (CV) | Test Acc | TTA Acc | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Front** | XGBoost | 97.24% | **36.10%** | 35.71% | High overfitting. TTA yielded no improvement. |
| **Left** | XGBoost | ~96.5% | **42.00%** | N/A | Better generalization than Front. |
| **Right** | XGBoost | 96.85% | **43.38%** | N/A | Best performing viewpoint. |

**Aggregated Performance**: ~40.5% Average Accuracy.

## 3. Analysis & Findings

### Critical Issues
1.  **Massive Overfitting**: The gap between Train (~97%) and Test (~40%) indicates the model is memorizing the augmented training data rather than learning robust features.
2.  **Stick Detection Failure**: Classification reports show **0.00 Recall** for classes heavily reliant on stick orientation (e.g., `left_elbow_block`, `left_chest_thrust`). This suggests the stick detector is failing on the test set, confusing the model.

### Successes
1.  **Viewpoint Separation**: Side views (Left/Right) outperform Front view, likely because limb extension is more visible.
2.  **Pipeline Stability**: The end-to-end pipeline (Extract -> Augment -> Tune -> Train) is robust and error-free.

## 4. Recommendations for Next Methodology

1.  **Fix Stick Detector**: The #1 priority. If the stick isn't detected, accuracy is capped at ~40%.
2.  **Switch to MoveNet?**: Viable for speed (2-3x faster) on weaker hardware, but requires re-extraction.
3.  **Graph Neural Networks (GCN)**: A GCN (ST-GCN) might capture motion/structure better than flat features, but requires re-tooling the pipeline for graph data.
