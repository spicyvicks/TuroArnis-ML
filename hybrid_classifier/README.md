# Hybrid Rule-Based ML Classifier

This pipeline combines rule-based feature engineering with machine learning to classify Arnis stances.

## Folder Structure

```
hybrid_classifier/
├── 1_extract_reference_features.py  # Analyze reference images
├── 2_generate_hybrid_features.py    # Convert dataset to hybrid features
├── 3_train_classifier.py            # Train classifier
├── feature_templates.json           # Generated feature statistics
├── hybrid_features/                 # Generated feature tensors
└── models/                          # Trained models

reference_poses/
├── front/
│   ├── crown_thrust_correct/        # Put 5 reference images here
│   ├── left_chest_thrust_correct/
│   └── ... (13 classes total)
├── left/
│   └── ... (same 13 classes)
└── right/
    └── ... (same 13 classes)
```

## Workflow

### Step 1: Select Reference Images
Manually copy **5 high-quality images** per class per viewpoint into `reference_poses/`.

**Total needed:** 13 classes × 3 viewpoints × 5 images = **195 images**

**Selection criteria:**
- Clear, well-executed stance
- No occlusion
- Good lighting
- Representative of the class

### Step 2: Extract Feature Templates
```bash
python hybrid_classifier/1_extract_reference_features.py
```

This will:
- Extract geometric features (angles, heights, distances) from reference images
- Compute mean/std for each feature per class/viewpoint
- Save to `hybrid_classifier/feature_templates.json`

### Step 3: Generate Hybrid Features
```bash
python hybrid_classifier/2_generate_hybrid_features.py
```

This will:
- Process all images in `dataset_split/`
- Compute geometric features for each image
- Convert to similarity scores using Gaussian matching against templates
- Save feature tensors to `hybrid_classifier/hybrid_features/`

### Step 4: Train Classifier
```bash
# Random Forest (recommended)
python hybrid_classifier/3_train_classifier.py --model_type random_forest

# Or Neural Network
python hybrid_classifier/3_train_classifier.py --model_type neural_network
```

This will:
- Train classifier on hybrid features
- Report accuracy and per-class performance
- Save model to `hybrid_classifier/models/`

## Expected Performance

Based on the hybrid approach, we expect:
- **60-80% accuracy** (vs. 25% with pure GCN)
- Better interpretability (can see which geometric features matter)
- Faster training (no graph convolutions needed)

## How It Works

1. **Reference images** define "ideal" geometric signatures for each stance
2. **Geometric features** (20+ measurements) are extracted from every image
3. **Similarity scores** are computed: how close is this pose to the reference?
4. **ML classifier** learns which combinations of similarities predict each class

This is still **machine learning** because the model learns decision boundaries from data, but the features are engineered using domain knowledge.
