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

## Key Decisions Needed

### 1. Scope: Full Migration vs Hybrid?
**Option A: Full Migration (RECOMMENDED)**
- Replace RF/XGB entirely with GeoPose-Net
- Complete rewrite of training pipeline
- Higher risk but maximum potential gain

**Option B: Hybrid (Conservative)**
- Keep current feature extraction
- Add GCN as secondary classifier
- Ensemble with existing models
- Lower risk, incremental improvement

### 2. Implementation Complexity?
**Option A: Full Implementation (4-6 weeks)**
- Complete PyTorch Geometric pipeline
- All bells and whistles (ArcFace, hard negatives, temporal smoothing)
- Maximum accuracy potential

**Option B: MVP (1-2 weeks)**
- Basic GCN + simple fusion
- Skip hierarchical classification initially
- Validate approach before full investment

### 3. Pre-requisites Check
- [ ] PyTorch installed?
- [ ] CUDA available (or CPU-only)?
- [ ] PyTorch Geometric can be installed?
- [ ] Existing CSV landmarks can be converted?

### 4. Data Questions
- Are the 12 Arnis pose names/descriptions available?
- Is weapon handedness consistent (right-handed)?
- Any missing weapon keypoints in current dataset?

### 5. Success Criteria
- Minimum acceptable accuracy: ___%
- Target accuracy: ___%
- Maximum acceptable training time: ___
- Deployment constraint: Desktop only / Cloud option?
