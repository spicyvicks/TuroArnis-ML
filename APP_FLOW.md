# TuroArnis System Architecture & Workflow

This document details the complete flow of the TuroArnis ecosystem, covering both the **Machine Learning Pipeline** (where models are built) and the **Main Application** (where they are used).

---

## 🏗️ PART 1: The Machine Learning Pipeline (`TuroArnis-ML`)

The ML repository acts as the "factory" for the AI models. Its central controller is **`model_manager.py`**, a CLI (Command Line Interface) tool that orchestrates the entire lifecycle.

### 1. The Controller (`model_manager.py`)
When you run this script, it initializes the environment:
- **Scans `models/`**: Looks for directories starting with `v` (e.g., `v1_dnn`, `v2_rf`).
- **Reads `metadata.json`**: For each model, it loads accuracy stats, training dates, and architecture types.
- **Checks `active_model.json`**: Identifies which model is currently "Production Ready".

### 2. Workflow A: Data Preparation
**Goal**: Convert raw images into mathematical features.
1. **Input**: Images are stored in `dataset/` (organized by class folders).
2. **Augmentation**: `split_dataset.py` ensures strict train/test splitting *before* augmentation to prevent data leakage.
3. **Extraction** (`run_extraction.py`):
   - Iterates through every image.
   - **Body**: Runs **MediaPipe Pose** to get 33 3D landmarks.
   - **Stick**: Runs **YOLOv8** to detect the arnis stick endpoints.
   - **Calculation**: Computes ~72 features, including:
     - Joint angles (Elbow, Shoulder, Knee).
     - Stick-relative vectors (Angle of stick vs. arm).
   - **Output**: Saves `features_train.csv` and `features_test.csv`.

### 3. Workflow B: Model Training
**Goal**: Teach the AI to recognize poses from the CSV data.
The manager allows selecting between architectures:
*   **DNN (Deep Neural Network)**: Good for complex non-linear relationships.
*   **Random Forest / XGBoost**: Often superior for tabular data (like our feature CSVs).

**The Training Process:**
1.  **Loading**: Reads CSVs, handles `NaN` values, and encodes labels.
2.  **Feature Selection**: Optionally drops "noise" features to improve speed/accuracy.
3.  **Cross-Validation**: Splits training data into 5 "folds" to ensure the model isn't just memorizing data.
4.  **Artifact Generation**:
    *   Saves the Model (`model.keras` or `model.joblib`).
    *   Saves the Scaler (`scaler.joblib`) – *Crucial for normalizing inputs*.
    *   Saves the Encoder (`label_encoder.joblib`) – *Converts "0" back to "Strike"*.
    *   Saves `metadata.json` – Detailed config log.

### 4. Workflow C: Evaluation & Deployment
- **Analysis**: Generates confusion matrices to see which moves get confused (e.g., "Temple Strike" vs "Eye Poke").
- **Activation**: When you select **"Set Active Model"**, it updates `models/active_model.json`. This file acts as a pointer that the Main App reads.

---

## 📱 PART 2: The Main Application (`TuroArnis`)

The Main App is the end-user GUI. Its "Brain" is the model trained in Part 1.

### 1. Initialization Phase (`main_app.py`)
1.  **Splash Screen**: UI Loading (assets/logos).
2.  **Resource Loading**:
    *   Locates the **Active Model** path using the `active_model.json` pointer.
    *   Loads the **Stick Detector** (YOLO weights).
    *   Initializes the Database Connection (`turoarnis.db`).
3.  **User Login**: Checks `users` table or prompts for a new profile.

### 2. The Real-Time Loop (`video_loop`)
This loop runs ~30 times per second on a separate thread to keep the UI responsive.

#### Step 1: Vision Processing
*   **Capture**: Grabs frame from webcam.
*   **Optimization**: Resizes to `480x360` (Analysis is faster on smaller images).
*   **Smart Scheduling**:
    *   **Post Estimation** (MediaPipe): Runs *Every Frame*.
    *   **Stick Detection** (YOLO): Runs *Every 4th Frame* (Resource heavy, so we skip frames and interpolate).
    *   **Classification** (ML Model): Runs *Every 8th Frame* (To avoid flickering predictions).

#### Step 2: Pose Analysis (`pose_analyzer.py`)
1.  **Feature Extraction**: Extracts the **EXACT** same features as the training phase (Angles + Stick vectors).
2.  **Normalization**: Applies the `scaler.joblib` from the trained model.
3.  **Prediction**: The model outputs a probability (e.g., "Right Temple Strike: 92%").
4.  **Smoothing**: Uses a rolling average buffer to prevent the prediction from jittering.

#### Step 3: Feedback Engine (`feedback_analyzer.py`)
*   **Context**: Knows the user wants to practice "Form X".
*   **Comparison**: Checks the predicted pose vs. the target pose.
*   **Correction**: If the user is close but imperfect, it calculates specific corrections (e.g., "Straighten your arm", "Lower your stance").

#### Step 4: UI Rendering
*   **Overlays**: Draws the skeleton (Green=Good, Red=Bad) and Stick lines.
*   **HUD**: Updates Confidence Bar, FPS Counter, and Feedback Toast.
*   **Canvas**: Converts the OpenCV image to Tkinter format for display.

### 3. Session & Data Tracking
*   **State Machine**: To count as a "Correct Rep", you must hold the correct pose for **10 consecutive frames**. This prevents accidental triggers.
*   **Database**: Every session saves:
    *   User ID / Timestamp.
    *   Accuracy % (Correct Frames / Total Frames).
    *   Specific errors made.

---

## 🔄 Data Flow Summary

1.  **Images** ➔ **features.csv** (via Extraction)
2.  **features.csv** ➔ **Model** (via Training)
3.  **Model** ➔ **Active Config** (via Manager)
4.  **Active Config** ➔ **Main App** (via JSON Loader)
5.  **Main App** ➔ **User Feedback** (via Real-time Inference)
