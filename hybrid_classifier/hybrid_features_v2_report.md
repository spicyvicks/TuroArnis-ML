# Hybrid Features V2: Technical Documentation

## 1. Overview
The specialized machine learning pipeline designed to classify Arnis stances by combining **geometric node features** (spatial Graph Convolutional Network inputs) with **global hybrid features** (expert-engineered geometric similarity scores).

This hybrid approach leverages the strengths of two distinct methodologies:
1.  **Graph Neural Networks (GCN)**: Captures the spatial relationships and connectivity of the body and stick keypoints.
2.  **Expert Rule-Based Features**: Encodes domain knowledge about Arnis stances (e.g., specific angles, stick positions) as similarity scores against "ideal" reference poses.

## 2. Technology Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Language** | Python 3.9+ | Core programming language |
| **Deep Learning** | PyTorch, PyTorch Geometric | Model architecture, training, and graph processing |
| **Pose Estimation** | MediaPipe Pose | Extracting 33-point body skeleton keypoints |
| **Object Detection** | YOLOv8 (Ultralytics) | Detecting the Arnis stick (grip and tip points) |
| **Image Processing** | OpenCV | Image loading and preprocessing |
| **Data Processing** | NumPy, Pandas | Feature engineering and data manipulation |

## 3. The Pipeline

### Step 1: Reference Feature Extraction
**Script:** `1_extract_reference_features.py`
**Input:** High-quality reference images (5 per class/viewpoint) in `reference_poses/`
**Output:** `feature_templates.json`

The system first establishes a "ground truth" for geometric features. For each class and viewpoint (Front, Left, Right), it:
1.  Extracts geometric measurements (angles, relative distances, heights).
2.  Computes the **mean** and **standard deviation** for each feature.
3.  Saves these statistics as "templates" to benchmarking future inputs.

### Step 2: Feature Generation (V2)
**Script:** `2b_generate_node_hybrid_features.py`
**Input:** Raw Dataset Images
**Output:** `train_features.pt`, `test_features.pt` in `hybrid_classifier/hybrid_features_v2/`
**Artifacts:** `feature_templates.json`

This step transforms raw images into the specific tensor formats required by the HybridGCN model.

#### A. Pre-processing
1.  **Stick Detection**: A custom stick detector created in YOLOv8 format predicts the bounding box and keypoints (Grip, Tip) of the Arnis stick.
2.  **Pose Estimation**: MediaPipe predicts 33 body landmarks (x, y, z, visibility).
3.  **Graph Construction**: The body and stick are merged into a single **35-node graph** (33 body nodes + 2 stick nodes).

#### B. Node Features (Local)
For each of the 35 nodes, a 6-dimensional feature vector is generated:
1.  **x** (normalized coordinate)
2.  **y** (normalized coordinate)
3.  **z** (depth coordinate)
4.  **visibility** (confidence score)
5.  **dist_to_hip_3d** (Euclidean distance to the hip center)
6.  **angle_from_hip** (2D angle relative to the hip center)

**Shape:** `[Num_Nodes (35), Feature_Dim (6)]`

#### C. Hybrid Features (Global)
The system calculates global geometric descriptors for the entire pose and compares them to the templates from Step 1 using a **Gaussian Similarity Function**:

$$ Similarity = \exp \left( -0.5 \left( \frac{value - \mu}{\sigma} \right)^2 \right) $$

Features include:
- **Angles**: Elbows, shoulders, knees (degrees).
- **Heights**: Wrists, elbows, stick tip/grip relative to hip.
- **Relative Positions**: Stick tip vs. nose/shoulder, hand separation distance.
- **Stick Orientation**: Angle and directional vectors (dx, dy).

**Shape:** `[Hybrid_Feature_Dim]` (Varies based on template, typically ~20-30 scores)

### Step 3: Model Training
**Script:** `4c_train_hybrid_gcn_v2.py`
**Architecture:** `HybridGCN`

The model effectively fuses two parallel processing streams:

1.  **GCN Stream (Spatial)**:
    - Input: Node Features + Learnable Node Embeddings (Dim 8).
    - Layers: 3x `GCNConv` layers with generic ReLU activation and Batch Normalization.
    - Pooling: `global_mean_pool` aggregates node representations into a graph vector.

2.  **MLP Stream (Expert)**:
    - Input: Hybrid Similarity Scores.
    - Layers: 2x Linear layers with BatchNorm and ReLU.
    - Purpose: Encodes how closely the pose matches known "ideal" templates.

3.  **Fusion & Classification**:
    - The outputs of the GCN and MLP streams are **concatenated**.
    - A final Fusion MLP maps the combined features to class probabilities.

**Data Augmentation:**
- **Random Scaling**: Scales inputs (0.85x - 1.15x) to simulate different camera distances.
- **Jittering**: Adds Gaussian noise to coordinates to improve robustness.

### Step 4: Evaluation
**Script:** `8_compare_models.py`

Evaluation focuses on accuracy across different viewpoint scenarios.

1.  **Specialist Ensemble Approach**:
    - Trains three separate models: Front, Left, Right.
    - During inference, the system selects the model corresponding to the camera's viewpoint.
    - **Pros**: Higher accuracy per specific view.
    - **Cons**: Requires knowing the viewpoint beforehand.

2.  **Merged Model Approach**:
    - Trains a single unified model on all data.
    - **Pros**: Simpler deployment, no viewpoint selection needed.
    - **Cons**: Slightly lower accuracy due to diverse input distributions.

**Key Metrics**:
- Overall Accuracy
- Per-Viewpoint Accuracy
- Confusion Matrix (via `5_analyze_hybrid_gcn.py` logic)

## 4. Evaluation Method Details

The evaluation process (`8_compare_models.py`) performs a rigorous A/B test:
1.  **Load Test Data**: Loads the unseen `test_features.pt`.
2.  **Run Specialist Ensemble**:
    - Iterates through test samples.
    - Routes 'front' samples to the Front Model, 'left' to Left Model, etc.
    - Computes aggregate accuracy.
3.  **Run Merged Model**:
    - Passes all samples through the single Merged Model.
    - Computes aggregate accuracy.
4.  **Comparison**:
    - Calculates the accuracy delta.
    - Declares a "winner" (Specialist vs. Merged) based on the performance gap.

## 5. Summary of Architecture

```mermaid
graph TD
    Img[Input Image] --> YOLO[Stick Detection]
    Img --> MP[MediaPipe Pose]
    
    YOLO --> Nodes[35 Keypoints]
    MP --> Nodes
    
    Nodes --> NodeFeat[Node Features (x,y,z,vis,dist,ang)]
    Nodes --> GlobalFeat[Global Geometric Metrics]
    
    GlobalFeat --> Template{Gaussian Comparison vs Templates}
    Template --> HybridScores[Hybrid Similarity Scores]
    
    subgraph HybridGCN Model
        NodeFeat --> GCN[GCN Layers (Spatial)]
        NodeEmbed[Node Identity Emb] --> GCN
        
        HybridScores --> MLP[MLP Layers (Expert Context)]
        
        GCN --> Pool[Global Pooling]
        Pool --> Concat((Concatenate))
        MLP --> Concat
        
        Concat --> Fusion[Fusion MLP]
        Fusion --> Class[Softmax Classification]
    end
```
