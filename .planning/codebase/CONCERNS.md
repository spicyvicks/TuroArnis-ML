# Codebase Concerns - Arnis Pose Classification ML

**Analysis Date:** 2025-04-18

**Project:** Graph Convolutional Network (GCN) for Arnis martial arts pose classification  
**Architecture:** HybridGCN V2 with node features + global hybrid features  
**Primary Issue:** Models misclassify correct poses, showing significant train-inference gaps

---

## 1. Critical Issues (Immediate Misclassification Causes)

### 1.1 Training-Inference Feature Extraction Mismatch

**Problem:** Production feature extraction does NOT use Method 4 stick correction, while training does.

| Component | Stick Correction | File |
|-----------|------------------|------|
| Training | Method 4 (pinky snap + shin-based length) | `hybrid_classifier/2b_generate_node_hybrid_features.py` lines 59-161 |
| Production | None - raw YOLO output | `deployment_package/src/feature_extraction.py` lines 69-80 |

**Impact:** This is the PRIMARY cause of misclassification. In deployment:
- Stick length not normalized to body proportions
- Grip anchor not snapped to MediaPipe wrist/pinky landmarks
- No foreshortening detection (threshold: 40px)
- Features differ significantly from training templates

**Fix Approach:** Port `apply_stick_method4_correction()` from training to `deployment_package/src/feature_extraction.py`

---

### 1.2 Severe Overfitting in Left View Model

**Problem:** Left specialist model shows catastrophic overfitting with training stopped early.

**Evidence (from `hybrid_classifier/models/history_left.json`):**
```
Epoch 12: Train Acc = 67.3%, Test Acc = 46.3%
Gap: 21% (exceeds overfitting_threshold = 20%)
Training stopped after only 12 epochs
```

**Root Causes:**
- Very limited left-view test data (only 192 images in test set)
- Class imbalance amplified (left-side poses underrepresented)
- Dropout 0.5 insufficient for the small dataset size

**Impact:** Left viewpoint predictions unreliable. Correct left-side poses likely misclassified.

---

### 1.3 Different Models Used in Training vs Inference

**Problem:** Training uses HybridGCN V2 but inference scripts use older SpatialGCN.

| Script | Model Class | File |
|--------|-------------|------|
| Training | HybridGCN (node + hybrid features) | `hybrid_classifier/4c_train_hybrid_gcn_v2.py` |
| Real-time Inference | SpatialGCN (node features only) | `inference_realtime_gcn.py` lines 54-60 |

**Impact:** Inference completely ignores the 30-dimension hybrid similarity features that comprise ~50% of model capacity. This invalidates all training optimizations.

**Fix Approach:** Update `inference_realtime_gcn.py` to load HybridGCN and compute hybrid features using template matching.

---

## 2. Data Pipeline Problems

### 2.1 Extreme Class Imbalance

**Distribution in front view training data:**
| Class | Count | % of Total |
|-------|-------|------------|
| right_elbow_block_correct | 248 | 13.3% |
| neutral | 200 | 10.7% |
| left_knee_block_correct | 196 | 10.5% |
| ... | ... | ... |
| crown_thrust_correct | 42 | 2.2% |
| left_eye_thrust_correct | 41 | 2.2% |
| left_elbow_block_correct | 39 | 2.1% |
| left_chest_thrust_correct | 37 | 2.0% |

**Ratio:** 6.7:1 between most/least represented classes

**Impact:** Model biased toward right-side blocks. Left thrusts and crown thrusts systematically misclassified.

**Current Mitigation:** `compute_class_weights()` uses inverse frequency weighting, but with insufficient absolute samples for minority classes, weighting alone cannot compensate.

---

### 2.2 Inadequate Data Augmentation for Left Poses

**Problem:** Augmentation creates flipped versions (50% of augmented data), but flipping logic has issues.

**In `0c_augment_training_data.py`:**
- 3 augmented copies per original: 1 non-flipped, 2 flipped
- Left-right poses become ambiguous when flipped
- Feature template negation for horizontal features (`mirror_features`) may not be properly applied during training

**Evidence:** Left view model performs significantly worse than front/right, suggesting augmentation artifacts.

---

### 2.3 Template-Based Feature Drift

**Problem:** Hybrid features computed from templates may not match real-time pose variations.

**In `hybrid_classifier/1_extract_reference_features.py` lines 287-319:**
- Templates computed from single reference pose per class
- Standard deviations (std) very small for some features
- Gaussian similarity `exp(-0.5 * ((value-mean)/std)^2)` assigns near-zero scores to valid but non-reference poses

**Impact:** Correct poses that deviate slightly from reference template get near-zero hybrid similarity scores, losing the discriminative signal.

---

## 3. Architecture Limitations

### 3.1 Dropout Rate Too Low for Hidden Dimension

**Current Config:**
- Hidden dim: 256
- Dropout: 0.5
- Node embedding: 8 dimensions
- Total trainable params: ~1.4M per model

**Problem:** With only ~1,871 training images (front view) and 256 hidden dimensions, model has massive capacity to memorize rather than generalize.

**Evidence:** Front model train_acc 84% vs test_acc 73% (gap: 11%) after 53 epochs suggests memorization.

**Recommendation:** Increase dropout to 0.7 for left/right models, or reduce hidden dim to 128.

---

### 3.2 BatchNorm in Small Batch Training

**Problem:** Training uses batch_size=32 with `drop_last=True`, but test sets are small (<200 images).

**Impact:** BatchNorm statistics computed on small batches have high variance. During inference with batch_size=1 (real-time), running stats may not match training distribution.

**Evidence:** Test accuracy oscillates significantly in training history (e.g., front model: 70.7% → 73.6% → 70.7% between epochs 7-9).

---

### 3.3 Node Embedding Dimension Inappropriateness

**Current:** 8-dim learnable embeddings for 35 nodes

**Issue:** With only 35 unique nodes, 8-dim embeddings create 280 additional parameters that must be learned from sparse data. Embeddings don't capture meaningful anatomical relationships (e.g., left wrist should be closer to left elbow than right wrist in embedding space, but this isn't enforced).

**Recommendation:** Replace learned embeddings with fixed anatomical encodings, or increase to 16-dim with orthogonality constraints.

---

## 4. Inference-Specific Mismatches

### 4.1 Skeleton Edge Definition Inconsistency

**Training edges (`4c_train_hybrid_gcn_v2.py` lines 31-37):**
```python
SKELETON_EDGES = [
    (11, 12), (12, 11), (11, 23), ...
    (15, 33), (33, 15), (16, 33), (33, 16), (33, 34), (34, 33)
]
```

**Inference edges (`inference_realtime_gcn.py` lines 30-37):**
```python
SKELETON_EDGES = [
    (11, 12), ..., (3, 7), (7, 3),  # EXTRA HEAD EDGES
    (15, 33), (33, 15), (16, 33), (33, 16), (33, 34), (34, 33)
]
```

**Impact:** Graph structure differs between training and inference. Edge index tensor shape mismatch could cause silent feature corruption or crashes.

---

### 4.2 Viewpoint Detection Dependency

**Problem:** Production code requires explicit viewpoint selection, but no automatic detection exists.

**In `deployment_package/docs/implementation_plan.md` lines 9-19:**
- Three specialist models (front/left/right) require manual selection
- Auto-detection using shoulder_width/torso ratio proposed but not implemented
- Real-time viewpoint changes (user moving) not handled

**Impact:** Wrong viewpoint model selection causes immediate misclassification regardless of pose quality.

---

### 4.3 Missing Confidence Thresholding

**Problem:** No confidence filtering in real-time inference.

**In `inference_realtime_gcn.py`:**
- Raw softmax output used directly
- No minimum confidence threshold (e.g., 0.7)
- Low-confidence predictions (<0.5) still displayed as valid

**Evidence:** Evaluation shows accuracy drops sharply below 0.7 confidence threshold (see threshold sensitivity analysis in `4d_evaluate_model.py`).

---

## 5. Prioritized Recommendations for Retraining

### Priority 1: Fix Feature Extraction Consistency (CRITICAL)

**Before ANY retraining:**
1. Port Method 4 stick correction to `deployment_package/src/feature_extraction.py`
2. Implement identical preprocessing in both training and inference
3. Add unit test to verify feature vector equality between pipelines

**Verification:** Extract features from same image through both pipelines, assert `np.allclose()`

---

### Priority 2: Balance Dataset (HIGH)

**Before retraining:**
1. Collect additional samples for underrepresented classes:
   - crown_thrust_correct: need +80 samples (current: 42)
   - left_chest_thrust_correct: need +85 samples (current: 37)
   - left_eye_thrust_correct: need +80 samples (current: 41)
   - left_elbow_block_correct: need +85 samples (current: 39)

2. OR implement stratified sampling with class-specific augmentation

---

### Priority 3: Update Inference Architecture (HIGH)

1. Rewrite `inference_realtime_gcn.py` to use HybridGCN
2. Add hybrid feature computation pipeline
3. Add viewpoint classifier or manual selector with UI

---

### Priority 4: Regularization Improvements (MEDIUM)

**Architecture changes for retraining:**
```python
# Recommended config adjustments
dropout = 0.7  # Increased from 0.5
hidden_dim = 128  # Reduced from 256
embedding_dim = 16  # Increased with orthogonality constraint
patience = 30  # Increased from 20
overfitting_threshold = 0.15  # Reduced from 0.20
```

---

### Priority 5: Template Feature Enhancement (MEDIUM)

1. Compute templates from multiple reference poses per class (not just one)
2. Increase std values by 20% to allow more pose variation
3. Add per-feature importance weighting

---

### Priority 6: Training Monitoring (LOW)

1. Add per-class accuracy tracking during training
2. Log confusion matrix every 10 epochs
3. Implement early stopping based on worst-class accuracy, not just aggregate

---

## 6. Test Coverage Gaps

| Component | Coverage | Risk |
|-----------|----------|------|
| Feature extraction parity | None | High - training vs inference mismatch |
| Stick correction accuracy | Manual only | High - no automated geometric validation |
| Per-class accuracy | Confusion matrix only | Medium - aggregated metrics hide class bias |
| Real-time performance | None | Medium - FPS and latency untested |
| Cross-viewpoint robustness | Manual verification | Medium - no automated test suite |

**Missing Tests:**
- `test_feature_extraction_parity.py`: Verify training and inference pipelines produce identical features
- `test_stick_correction.py`: Geometric validation of stick length ratios
- `test_class_balance.py`: Verify minimum samples per class before training

---

## Summary: Root Cause of "Correct Poses Misclassified"

The primary cause is a **train-inference feature extraction mismatch** where:
1. Training uses sophisticated stick correction (Method 4)
2. Inference uses raw YOLO output
3. This creates systematic feature vector differences

Secondary causes:
1. Severe underfitting on left view (insufficient data + early stopping)
2. Class imbalance causing bias toward right-side blocks
3. Missing hybrid features in real-time inference entirely

**Immediate action:** Fix stick correction in production before retraining any models.

---

*Concerns audit: 2025-04-18*
