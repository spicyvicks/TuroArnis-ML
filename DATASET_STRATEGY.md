# TuroArnis ML Dataset & Training Strategy

This document outlines the machine learning strategy for the TuroArnis application to ensure high accuracy, real-world robustness, and valid performance metrics.

---

## 1. The Core Philosophy: "Correct-Only Training"

**Concept**: We train the model ONLY on perfectly executed moves. We teach the model *what "Right" looks like*, not what "Wrong" looks like.

### Why?
Training on "Incorrect" data confuses the model because "Bad Form" often looks geometrically similar to "Good Form". Instead, we rely on:
1.  **Classification**: The model identifies the *intent* (e.g., "This looks like a Crown Thrust").
2.  **Validation**: The application (or a secondary logic layer) checks if the confidence is high enough or if specific angles meet the "Correct" criteria.

---

## 2. Dataset Structure

The dataset is organized to support this philosophy and prevents "forced choice" errors.

### `dataset/` (The "Golden" Training Set)
Contains **ONLY** validated, correct examples.
*   `crown_thrust_correct/`
*   `left_temple_block_correct/`
*   ... (All 12 strikes/blocks)
*   **`neutral_stance/`** (Critical Addition)

### `incorrect/` (The Negative Test Set)
Contains improper form, sloppy execution, and mistakes.
*   **DO NOT TRAIN ON THIS.**
*   **Usage**: Run these images through the trained model.
    *   **Success**: Model predicts with **Low Confidence** (< 60%) or misclassifies as `neutral_stance`.
    *   **Failure**: Model predicts "Crown Thrust" with **High Confidence** (99%). This indicates the model is "Overfitting" or too lenient.

---

## 3. The "Neutral Stance" Class

**Purpose**: To solve the "Forced Choice" problem. Without this class, the AI is forced to classify *everything* (tying shoes, walking, standing still) as one of the 12 martial arts moves.

**What to Collect**:
1.  **Handa (Ready Position)**: Standing with sticks, waiting.
2.  **Pugay (Respect)**: Standing straight.
3.  **Idle/Relaxed**: Arms at sides, checking phone, fixing hair.
4.  **Transitions**: Blurry frames of moving between positions.

---

## 4. The Anti-Leakage Workflow

We use a strict **Split-First, Augment-Second** pipeline to ensure the model is never tested on data it has seen during training.

### Step 1: Split (`training/split_dataset.py`)
Randomly divides the raw `dataset/` folder into three isolated buckets:
*   **Train (70%)**: The only data the model learns from.
*   **Val (10%)**: Used during training to tune hyperparameters (stop early if not improving).
*   **Test (20%)**: The "Final Exam". **NEVER** accessed during training.

### Step 2: Augment (`training/data_augmentation.py`)
Applies rotation, scaling, and lighting changes **ONLY** to the `Train` bucket. 
*   **Input**: `dataset_split/train`
*   **Output**: `dataset_split/train_aug`
*   *Note: Valid and Test sets remain pure/clean to represent real-world camera input.*

### Step 3: Train (`training/training.py`)
Trains the model (Neural Network, Random Forest, or XGBoost) using the isolated buckets.
*   Trains on: `dataset_split/train_aug`
*   Validates on: `dataset_split/val`
*   Tests on: `dataset_split/test`

---

## 5. Deployment logic

When using the model in the actual App (`main_app.py`):

1.  **Prediction**: Get the class and probability from the model.
    ```python
    prediction, probability = model.predict(frame)
    ```

2.  **Thresholding**:
    *   **If Class == "Neutral Stance"**: Do nothing (State: "Waiting").
    *   **If Probability < 75%**: Do nothing (State: "Unsure/Bad Form").
    *   **If Probability >= 75%**: Register the move.

3.  **Heuristic Validation (Optional but Recommended)**:
    *   Even if the model says "Crown Thrust (90%)", check the elbow angle using code.
    *   `if angle < 140: return "Feedback: Extend Arm Fully"`

---

## Summary Checklist for New Data
- [ ] Ensure new images are perfectly executed (if for `dataset/`).
- [ ] Place "bad form" images in `incorrect/` for testing.
- [ ] Always run `split_dataset.py` first when adding new data.
- [ ] Capture `neutral_stance` data if false positives occur frequently.

---

## 6. Advanced: Training on Incorrect Data vs. Heuristics

There are two main ways to handle feedback for "Bad Form". This project prioritizes **Option 2 (Hybrid)**.

### Option 1: Explicit Error Training (Not Recommended)
Training classes like `Crown Thrust (Good)` vs `Crown Thrust (Bad Elbow)`.
*   **Pros**: precise feedback ("Fix your elbow").
*   **Cons**:
    *   **Infinite Mistakes**: There are million ways to be wrong. You cannot train for all of them.
    *   **Decision Boundary**: "Good" and "Almost Good" are mathematically very close, leading to model confusion.
    *   **Data Heavy**: requires massive amounts of "bad" data.

### Option 2: The Hybrid "Sandwich" Strategy (Recommended)
This combines the stability of ML with the precision of rule-based logic.

**Phase 1: ML Identification** (What are they doing?)
*   Model trained ONLY on `Correct` + `Neutral` data.
*   **Input**: User Frame.
*   **Output**: "Crown Thrust" (95%).

**Phase 2: Code Grading** (Are they doing it well?)
*   Since we now know it's a "Crown Thrust", we apply specific math rules in code.
*   **Example Rule**:
    ```python
    if detected_class == "crown_thrust_correct":
        # Calculate elbow angle using landmarks 11, 13, 15
        angle = calculate_angle(shoulder, elbow, wrist)
        
        if angle < 140:
             return "Feedback: Straighten your arm more!"
        if angle > 170:
             return "Feedback: Don't hyperextend!"
        return "Good Form!"
    ```

**Why this wins**:
1.  **Robust**: Math (`angle < 140`) is absolute. It doesn't "hallucinate."
2.  **Lean Data**: You don't need 1,000 photos of bent elbows. You just need the logic.
3.  **Adjustable**: You can make grading stricter (e.g., change 140 to 150) by changing one number in code, without retraining the whole model.
