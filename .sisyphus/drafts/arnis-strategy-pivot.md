# Consultation Draft: TuroArnis ML Strategy Pivot

## Current Situation (from SESSION_SUMMARY.md)
- **Achieved**: 50.4% test accuracy (up from 45%)
- **Target**: ≥47% ✅ EXCEEDED
- **Overfitting gap**: 27.5% (down from 35%)
- **Models trained**: 22
- **Features**: 110 total
- **Best model**: v048 (XGBoost)

## User's Frustration
- Feels training sessions have been disappointing
- No significant increase in testing (though data shows 5.4% improvement)
- Wants to abandon current approach
- Interested in methodology from Microsoft document

## Microsoft/Kimi Methodology (from convo with kimi.txt)

### Core Architecture: GeoPose-Net
**Graph Neural Network (GAT) + CNN Fusion**

**Stream 1: Spatial Graph Attention Network (SGAT)**
- Nodes: 19 (17 COCO body + 2 weapon keypoints)
- Edges: Skeleton + weapon attachment + semantic Arnis-specific edges
- Learns spatial relationships directly from topology, not hand-crafted angles

**Stream 2: Lightweight Visual Encoder**
- MobileNetV3-Small pretrained on ImageNet
- Processes 128x128 ROI around weapon grip
- Adds visual context to graph features

**Hierarchical Classification Head:**
1. Stance Family (4 classes: Forward, Back, Horse, Cat)
2. Weapon Chamber Position (3 classes: High/Heaven, Middle, Low)
3. Specific Pose (12 classes) - conditioned on above

**Key Innovations:**
1. **Geometric Rigidity Constraint**: Enforces stick-to-forearm ratio (1.35x) and angle alignment
2. **Multi-Task Learning**: Classification + weapon geometry reconstruction
3. **ArcFace Loss**: Metric learning for small dataset (1,200 samples)
4. **Hard Negative Mining**: Synthetic negatives by swapping weapon orientations

**Expected Results:**
- Current RF: ~50-75%
- GCN (pose only): ~82%
- GeoPose-Net (full): **89-93%**

### Implementation Phases
1. **Data Preprocessing**: CSV → Graph format with weapon geometry correction
2. **Model Development**: PyTorch + PyTorch Geometric implementation
3. **Training**: 5-fold CV with ArcFace loss, 100 epochs
4. **Temporal Smoothing**: EMA for video frames
5. **Deployment**: ONNX conversion for desktop

## Test Strategy Decision (CONFIRMED)

### Automated Tests
✅ **Option C: None** - Skip formal tests initially, use training accuracy metrics
- Can add TDD (Option A) or tests-after (Option B) in a few days
- Focus on getting to 80%+ accuracy first

### Agent QA Verification
Every task will include **Agent-Executed QA Scenarios**:
- **Bash**: For training runs, file operations, accuracy checks
- **Python REPL**: For model inference validation
- **interactive_bash**: For CLI training scripts

## Timeline Constraint (CRITICAL)

**User Requirement**: Methodology achievable in **DAYS**, not weeks

### Revised Phased Approach (Aggressive MVP)

**Phase 1: Day 1-2** - Environment + Weapon Geometry Correction
- Install PyTorch + PyTorch Geometric (CPU)
- Implement weapon rigidity correction
- Validate on existing dataset

**Phase 2: Day 3-4** - Basic GCN Classifier
- Simple 2-layer GAT (no visual stream yet)
- Convert CSV to graph format
- Train and validate (target: 75-80%)

**Phase 3: Day 5-6** - Hierarchical Classification
- Add stance family + chamber position heads
- Train full model (target: 80-85%)

**Phase 4: Day 7** - Deployment Prep
- ONNX export
- Integration with existing app
- Final validation

**What We're SKIPPING Initially (can add later):**
- Visual stream (MobileNetV3) - adds complexity, marginal gain for MVP
- ArcFace loss - nice to have but not essential for 80%
- Hard negative mining - advanced technique for later
- Temporal smoothing - your app runs inference every 10 min anyway

### Simplified Architecture for MVP

**Stream 1: Basic GCN**
- 2-layer GATConv (not GATv2)
- 128 hidden dimensions
- Global mean pooling (not attention pooling)
- ~10K parameters (very light)

**Classification: Flat 12-class initially**
- Skip hierarchical for MVP
- Add hierarchy once base GCN works

**This should train in <30 minutes on your CPU**

## User Decisions (CONFIRMED)

### 1. Scope: Full Migration (Option A)
✅ **Replace RF/XGB entirely with GeoPose-Net**
- Phased approach: Weapon geometry correction first, then GCN

### 2. Implementation: Phased (Option C)
✅ **Week 1-2**: Weapon geometry correction on current pipeline
✅ **Week 3-4**: Add GCN classifier
✅ **Week 5**: Full GeoPose-Net with fusion

### 3. Technical Prerequisites
✅ **PyTorch**: Not installed, but can install (CPU-only)
✅ **Hardware**: Intel Core 7 150U @ 1.80 GHz (CPU-only)
✅ **PyTorch Geometric**: Not installed yet
✅ **Python**: 3.11.1
⚠️ **COMPATIBILITY REQUIREMENT**: Must work with existing app via:
   - `active_model.json` pointer system
   - `scaler.joblib` for normalization
   - `label_encoder.joblib` for class encoding
   - `metadata.json` for model info
   - `model.joblib` (or equivalent ONNX)

### 4. Data Questions (ANSWERED)
✅ **12 Arnis poses**: From dataset folder names
✅ **Handedness**: Mostly right-handed
✅ **Missing keypoints**: Only in neutral stance

### 5. Success Criteria (DEFINED)
✅ **Minimum**: 80%
✅ **Target**: >85%
✅ **Training time**: 1 hour per session
✅ **Inference**: Every 10 minutes acceptable (not real-time)
✅ **Deployment**: Desktop application (ONNX export)

## Application Compatibility Requirements (from APP_FLOW.md)

### Expected Artifacts
The new model must produce these files to be compatible with `pose_analyzer.py`:

```
models/v{version}_geopose/
├── model.joblib              # OR model.onnx for deployment
├── scaler.joblib             # Feature normalizer (CRITICAL)
├── label_encoder.joblib      # Class name encoder
├── metadata.json             # Model info
└── selected_features.json    # Feature list (optional but helpful)
```

### Integration Points
1. **Feature Extraction** (`pose_analyzer.py`):
   - Must extract SAME features as training (angles + stick vectors)
   - Uses `scaler.joblib` to normalize features
   
2. **Inference** (`video_loop`):
   - Runs every 8th frame (not real-time requirement)
   - Returns: `prediction, probability = model.predict(features)`
   
3. **Activation** (`model_manager.py`):
   - Updates `active_model.json` with path to new model
   - App loads model from that path

### ONNX Deployment Strategy
Since the new model uses PyTorch, we'll export to ONNX:
```python
# After training
torch.onnx.export(model, dummy_input, "model.onnx")

# App loads with onnxruntime
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
prediction = session.run(None, {"input": features})
```

This maintains compatibility while upgrading the architecture.
