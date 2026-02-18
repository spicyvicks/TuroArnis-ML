# Hybrid GCN Training Workflow (Updated Stick Detection)

Since we updated the stick detection logic to match the live app, you must regenerate all features before retraining.

## 1. Regenerate Training Features
Run this to process the **training dataset** with the new shin-based stick detection.
```bash
python hybrid_classifier/2b_generate_node_hybrid_features.py
```
*(This processes all viewpoints: front, left, right)*

## 2. Regenerate Test Features
Run this to process the **test dataset** with the new logic.
```bash
python hybrid_classifier/2c_extract_test_features.py
```
*(This processes all viewpoints: front, left, right)*

## 3. Train the Models
Train the Hybrid GCN V2 models for all viewpoints.
```bash
python hybrid_classifier/4c_train_hybrid_gcn_v2.py
```
*(This trains 3 separate models: front, left, right)*

## 4. Evaluate Performance
Run evaluation for each viewpoint individually to generate confusion matrices and reports.

**Left Viewpoint:**
```bash
python hybrid_classifier/4d_evaluate_model.py --viewpoint left
```

**Right Viewpoint:**
```bash
python hybrid_classifier/4d_evaluate_model.py --viewpoint right
```

**Front Viewpoint:**
```bash
python hybrid_classifier/4d_evaluate_model.py --viewpoint front
```
