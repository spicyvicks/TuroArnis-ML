# Arnis GeoPose-Net Migration: RF/XGB to Graph Neural Network

## TL;DR

> **Quick Summary**: Replace Random Forest/XGBoost (50.4% accuracy) with Graph Neural Network architecture achieving 80-93% accuracy. Uses weapon geometry correction + graph attention networks to eliminate keypoint drift and overfitting.
>
> **Deliverables**:
> - PyTorch Geometric training pipeline (`training/geopose/`)
> - Weapon geometry correction module (`training/geopose/weapon_geometry.py`)
> - Graph construction from existing CSV landmarks (`training/geopose/graph_builder.py`)
> - Lightweight GCN model (~10K params, trains in <30 min on CPU)
> - ONNX export for desktop app compatibility
> - 80%+ test accuracy (target: 85%+)
>
> **Estimated Effort**: Medium (7 days, MVP approach)
> **Parallel Execution**: NO - Sequential phases (each depends on previous)
> **Critical Path**: Setup → Geometry Correction → Graph Conversion → GCN Training → ONNX Export

---

## Context

### Original Request
User is frustrated with current RF/XGB training plateau (50.4% test accuracy, 27.5% overfitting gap). Wants to migrate to scientifically-validated methodology from Microsoft document (Kimi conversation) using Graph Neural Networks with geometric constraints.

### Interview Summary

**Key Discussions**:
- **Current State**: 50.4% accuracy achieved, 27.5% CV-test gap, 22 models trained
- **Pain Points**: Weapon keypoint drift (too short/long/wrong angle), overfitting, plateau
- **Hardware**: Intel Core 7 150U @ 1.80 GHz (CPU-only), Python 3.11.1
- **Data**: 1,200 samples (12 classes × 100 samples), right-handed, minimal missing keypoints
- **Timeline**: Needs results in DAYS, not weeks (MVP approach)
- **Success Criteria**: Minimum 80%, target >85%, 1-hour training sessions acceptable
- **Integration**: Must be compatible with existing app via ONNX export

**Technical Decisions**:
- **Scope**: Full migration (replace RF/XGB, not hybrid)
- **Implementation**: Phased MVP (skip visual stream, ArcFace, hard negatives initially)
- **Test Strategy**: None for now (can add TDD later), use training accuracy metrics
- **Simplified Architecture**: Basic 2-layer GAT (not GATv2), flat 12-class initially, skip hierarchy for MVP

### Microsoft/Kimi Methodology Overview

**Core Innovation**: GeoPose-Net with geometric rigidity constraints
- **Problem**: YOLO weapon keypoints drift (length/angle errors)
- **Solution**: Enforce stick-to-forearm ratio (1.35x) and angle alignment (±20°)
- **Architecture**: Graph Attention Network (GAT) + hierarchical classification
- **Expected Accuracy**: 89-93% (vs current 50.4%)

**MVP Simplifications** (for days timeline):
- Skip visual stream (MobileNetV3) - adds complexity, marginal MVP gain
- Skip ArcFace loss - advanced technique for later refinement
- Skip hard negative mining - can add once base model works
- Skip temporal smoothing - app inference every 10 min anyway
- Use flat 12-class classification initially (add hierarchy in Phase 2)

### Application Compatibility Requirements

From `APP_FLOW.md`, the new model must produce:
```
models/v{version}_geopose/
├── model.onnx                 # Main model (ONNX format for PyTorch compatibility)
├── scaler.joblib              # Feature normalizer (REQUIRED by pose_analyzer.py)
├── label_encoder.joblib       # Class encoder
├── metadata.json              # Model metadata
└── selected_features.json     # Feature list
```

Integration via `active_model.json` pointer system.

---

## Work Objectives

### Core Objective
Replace angle-based tabular features (110 features → RF/XGB) with graph-based topology learning (19 nodes + edges → GCN), achieving 80%+ test accuracy while maintaining weapon geometric consistency.

### Concrete Deliverables
1. **Environment Setup**: PyTorch + PyTorch Geometric (CPU) installed
2. **Weapon Geometry Module**: `training/geopose/weapon_geometry.py` - corrects keypoint drift
3. **Graph Builder**: `training/geopose/graph_builder.py` - converts CSV landmarks to PyG Data objects
4. **GCN Model**: `training/geopose/model.py` - 2-layer GAT, ~10K params
5. **Training Script**: `training/geopose/train.py` - training loop with cross-validation
6. **ONNX Export**: `training/geopose/export_onnx.py` - exports for app compatibility
7. **Model Manager Integration**: Update `training/model_manager.py` to support GeoPose models

### Definition of Done
- [ ] Model achieves ≥80% test accuracy on 5-fold CV
- [ ] Weapon geometry correction enforces anatomical validity rate ≥95%
- [ ] Training time <1 hour on Intel Core 7 150U CPU
- [ ] ONNX export loads successfully in app via `onnxruntime`
- [ ] All required artifacts created (`model.onnx`, `scaler.joblib`, `label_encoder.joblib`, `metadata.json`)

### Must Have
- Weapon geometry correction (rigidity constraints)
- Graph construction from existing CSV landmarks
- Functional GCN model with ≥80% accuracy
- ONNX export for app compatibility
- Training completes in <1 hour

### Must NOT Have (Guardrails)
- Visual stream (CNN) in MVP - adds complexity, defer to Phase 2
- Hierarchical classification in MVP - flat 12-class first
- ArcFace loss in MVP - standard cross-entropy sufficient for 80%
- Hard negative mining in MVP - advanced augmentation for later
- Real-time inference optimization (<30ms) - not required (10 min interval acceptable)
- GPU dependency - must work on CPU-only

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO (new PyTorch pipeline)
- **Automated tests**: None for MVP (can add TDD later)
- **Framework**: No unit tests, use training accuracy as verification

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

**Verification Tool by Deliverable Type:**

| Type | Tool | How Agent Verifies |
|------|------|-------------------|
| **Training/Model** | Bash + Python REPL | Run training script, check accuracy output, verify model files created |
| **Data Processing** | Bash + Python REPL | Convert data, validate graph structure, check node/edge counts |
| **Export/Deployment** | Bash + Python REPL | Load ONNX, run inference, verify output shape |

**Each Scenario MUST Follow This Format:**

```
Scenario: [Descriptive name]
  Tool: Bash (training) / Python REPL (model validation)
  Preconditions: [Setup required]
  Steps:
    1. [Exact command with arguments]
    2. [Validation check]
    3. [Assertion with expected value]
  Expected Result: [Concrete outcome]
  Evidence: [Output capture / file verification]
```

**Evidence Requirements:**
- Training logs captured in `.sisyphus/evidence/`
- Model checkpoints verified via Python REPL
- Accuracy metrics saved to file and asserted

---

## Execution Strategy

### Sequential Phases (No Parallel Execution)

Each phase depends on the previous. Complete Phase N before starting Phase N+1.

```
Phase 1: Environment Setup (Day 1)
└── Task 1: Install PyTorch + PyTorch Geometric

Phase 2: Weapon Geometry Correction (Day 1-2)
└── Task 2: Implement weapon_geometry.py
    └── Task 3: Validate geometry correction on dataset

Phase 3: Graph Conversion (Day 2-3)
└── Task 4: Create graph_builder.py
    └── Task 5: Convert CSV to PyG Data format

Phase 4: GCN Model & Training (Day 3-5)
└── Task 6: Implement model.py (2-layer GAT)
    └── Task 7: Create train.py with cross-validation
        └── Task 8: Train and validate (target 80%+)

Phase 5: Deployment Integration (Day 6-7)
└── Task 9: Create ONNX export script
    └── Task 10: Integrate with model_manager.py
        └── Task 11: End-to-end validation
```

### Critical Path
Task 1 → Task 2 → Task 4 → Task 6 → Task 7 → Task 9 → Task 11

### Agent Dispatch Summary

| Phase | Tasks | Recommended Agent |
|-------|-------|-------------------|
| 1 | 1 | delegate_task(category='unspecified-high', load_skills=[], run_in_background=false) |
| 2 | 2-3 | delegate_task(category='unspecified-high', load_skills=[], run_in_background=false) |
| 3 | 4-5 | delegate_task(category='unspecified-high', load_skills=[], run_in_background=false) |
| 4 | 6-8 | delegate_task(category='unspecified-high', load_skills=[], run_in_background=false) |
| 5 | 9-11 | delegate_task(category='unspecified-high', load_skills=[], run_in_background=false) |

---

## TODOs

### Phase 1: Environment Setup (Day 1)

- [ ] 1. Install PyTorch + PyTorch Geometric (CPU)

  **What to do**:
  - Create virtual environment (optional but recommended)
  - Install PyTorch CPU version: `pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cpu`
  - Install PyTorch Geometric: `pip install torch-geometric==2.4.0`
  - Install dependencies: `pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html`
  - Install other requirements: `pip install scikit-learn pandas numpy tqdm onnx onnxruntime`
  - Verify installation with test script

  **Must NOT do**:
  - Install CUDA version (CPU-only requirement)
  - Install heavy dependencies not needed for MVP

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` - Environment setup requires careful dependency management
  - **Skills**: None specifically, but needs Python package management expertise
  - **Skills Evaluated but Omitted**: 
    - `git-master`: Not needed for package installation
    - `frontend-ui-ux`: Not applicable

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Phase 1 starts immediately)
  - **Blocks**: Tasks 2-11 (all depend on PyTorch)
  - **Blocked By**: None (can start immediately)

  **References**:
  - `convo with kimi.txt:9090-9115` - Environment setup commands from Microsoft methodology
  - PyTorch Geometric docs: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
  - CPU wheels: https://data.pyg.org/whl/torch-2.1.0+cpu.html

  **Acceptance Criteria**:
  - [ ] `python -c "import torch; print(torch.__version__)"` → shows 2.1.0
  - [ ] `python -c "import torch_geometric; print(torch_geometric.__version__)"` → shows 2.4.0
  - [ ] `python -c "import torch; print(torch.cuda.is_available())"` → shows False (CPU-only verified)
  - [ ] Can import all required modules without errors

  **Agent-Executed QA Scenario (MANDATORY):**
  ```
  Scenario: Verify PyTorch + PyG Installation
    Tool: Bash
    Preconditions: Clean environment, pip available
    Steps:
      1. Run: python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__); print('PyG:', torch_geometric.__version__); print('CUDA:', torch.cuda.is_available())"
      2. Assert: Output contains "PyTorch: 2.1.0" or compatible version
      3. Assert: Output contains "PyG: 2.4.0" or compatible version
      4. Assert: Output contains "CUDA: False"
      5. Run: python -c "from torch_geometric.nn import GATConv; print('GATConv import OK')"
      6. Assert: Output contains "GATConv import OK"
    Expected Result: All imports successful, CPU-only confirmed
    Evidence: Capture full output to .sisyphus/evidence/task-1-installation.log
  ```

  **Commit**: YES
  - Message: `chore(deps): add PyTorch + PyTorch Geometric for GeoPose-Net`
  - Files: `requirements.txt` (new or updated)
  - Pre-commit: Installation verification script

---

### Phase 2: Weapon Geometry Correction (Day 1-2)

- [ ] 2. Implement Weapon Geometry Correction Module

  **What to do**:
  - Create `training/geopose/weapon_geometry.py`
  - Implement `correct_weapon_geometry()` function based on Microsoft methodology
  - Enforce stick-to-forearm ratio (alpha=1.35)
  - Enforce angle constraint (max_deviation=20° = 0.35 radians)
  - Use spherical interpolation (slerp) for angle correction
  - Handle missing keypoints gracefully

  **Must NOT do**:
  - Modify existing training files (create new module)
  - Assume left-handed users (assume right-handed for MVP)
  - Fail on missing keypoints (return original instead)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` - Mathematical geometry implementation
  - **Skills**: None specifically, needs strong linear algebra/numpy skills
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: Not applicable
    - `git-master`: Commit only after implementation

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Phase 2)
  - **Blocks**: Task 3, 4, 5
  - **Blocked By**: Task 1 (PyTorch must be installed)

  **References**:
  - `convo with kimi.txt:6090-6102` - Mathematical formulation
  - `convo with kimi.txt:6969-7002` - Correction algorithm implementation
  - Formula: L_stick = alpha × ||forearm|| where alpha=1.35
  - Angle constraint: cos⁻¹((tip-grip)·forearm / (||tip-grip|| × ||forearm||)) ≤ 20°

  **Acceptance Criteria**:
  - [ ] Function `correct_weapon_geometry(grip, tip, wrist, elbow, alpha=1.35, max_deviation=0.35)` exists
  - [ ] Returns corrected tip position as numpy array
  - [ ] Enforces length constraint: corrected_length ≈ 1.35 × forearm_length
  - [ ] Enforces angle constraint: angle between stick and forearm ≤ 20°
  - [ ] Handles missing keypoints (returns original if any input is zero/invalid)
  - [ ] Unit test: Valid input → corrected output within constraints
  - [ ] Unit test: Missing keypoint → returns original unchanged

  **Agent-Executed QA Scenario (MANDATORY):**
  ```
  Scenario: Validate Weapon Geometry Correction
    Tool: Python REPL
    Preconditions: weapon_geometry.py exists
    Steps:
      1. Import: from training.geopose.weapon_geometry import correct_weapon_geometry
      2. Create test case:
         - grip = np.array([100, 100])
         - tip = np.array([200, 100])  # 100px stick, horizontal
         - wrist = np.array([100, 150])
         - elbow = np.array([100, 200])  # forearm 50px, vertical
      3. Call: corrected = correct_weapon_geometry(grip, tip, wrist, elbow)
      4. Calculate: stick_length = np.linalg.norm(corrected - grip)
      5. Calculate: forearm_length = np.linalg.norm(wrist - elbow)
      6. Assert: 1.2 ≤ stick_length / forearm_length ≤ 1.5 (ratio constraint)
      7. Create missing keypoint case:
         - grip = np.array([0, 0])  # Missing
      8. Call: corrected = correct_weapon_geometry(grip, tip, wrist, elbow)
      9. Assert: corrected equals original tip (unchanged)
    Expected Result: Geometry constraints enforced, missing data handled gracefully
    Evidence: Output values and assertions to .sisyphus/evidence/task-2-geometry.log
  ```

  **Commit**: YES
  - Message: `feat(geopose): add weapon geometry correction module`
  - Files: `training/geopose/weapon_geometry.py`, `training/geopose/__init__.py`
  - Pre-commit: Run geometry correction validation

---

- [ ] 3. Validate Geometry Correction on Dataset

  **What to do**:
  - Load existing CSV features (`features_train.csv`)
  - Extract weapon keypoints (grip_x, grip_y, tip_x, tip_y, etc.)
  - Apply geometry correction to all samples
  - Calculate anatomical validity rate (% samples with valid stick geometry)
  - Compare before/after statistics
  - Save corrected features to new CSV

  **Must NOT do**:
  - Modify original CSV (create new file)
  - Skip validation (must show improvement)
  - Assume specific column names (inspect CSV first)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` - Data processing and validation
  - **Skills**: None specifically
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: Not applicable

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 4, 5
  - **Blocked By**: Task 2 (geometry module must exist)

  **References**:
  - `features_train.csv` - Existing feature file
  - `training/feature_extraction_combined.py` - Shows weapon feature extraction

  **Acceptance Criteria**:
  - [ ] Script loads `features_train.csv` successfully
  - [ ] Extracts weapon keypoints (identify column names)
  - [ ] Applies correction to all samples with valid weapon data
  - [ ] **Anatomical validity rate ≥95%** (corrected samples)
  - [ ] Saves `features_train_corrected.csv` with corrected coordinates
  - [ ] Logs statistics: before/after angle deviation, length ratios

  **Agent-Executed QA Scenario (MANDATORY):**
  ```
  Scenario: Validate Geometry Correction on Full Dataset
    Tool: Bash + Python REPL
    Preconditions: features_train.csv exists, weapon_geometry.py implemented
    Steps:
      1. Run: python training/geopose/validate_geometry.py
      2. Assert: Script completes without errors
      3. Check output: "Anatomical validity rate: X%"
      4. Assert: X >= 95
      5. Verify file: ls -lh features_train_corrected.csv
      6. Assert: File exists and size > 0
      7. Load and check: python -c "import pandas as pd; df = pd.read_csv('features_train_corrected.csv'); print(f'Samples: {len(df)}, Columns: {len(df.columns)}')"
    Expected Result: ≥95% valid geometry, corrected CSV created
    Evidence: Statistics log to .sisyphus/evidence/task-3-validation.log
  ```

  **Commit**: YES
  - Message: `feat(geopose): validate weapon geometry correction on dataset`
  - Files: `training/geopose/validate_geometry.py`, `features_train_corrected.csv`
  - Pre-commit: Run validation script, check 95% threshold

---

### Phase 3: Graph Conversion (Day 2-3)

- [ ] 4. Create Graph Builder Module

  **What to do**:
  - Create `training/geopose/graph_builder.py`
  - Define COCO skeleton edges (17 body keypoints)
  - Define weapon edges (wrist→grip, elbow→grip, grip→tip)
  - Define semantic edges for Arnis (hand→shoulder, etc.)
  - Implement `build_graph()` function:
    - Input: Keypoints array (19 nodes × 3 coords)
    - Output: PyTorch Geometric `Data` object
  - Compute edge attributes: length, angle (sin/cos), rigidity flag
  - Normalize node features by torso height

  **Must NOT do**:
  - Hardcode edge connections incorrectly (verify COCO format)
  - Skip edge attributes (needed for GAT)
  - Forget batch handling (important for training)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` - Graph construction logic
  - **Skills**: None specifically, needs understanding of PyG Data format

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Phase 3)
  - **Blocks**: Task 5, 6, 7, 8
  - **Blocked By**: Task 3 (corrected features must exist)

  **References**:
  - `convo with kimi.txt:6930-6980` - Graph construction details
  - `convo with kimi.txt:7090-7130` - Edge attribute computation
  - COCO keypoint format: https://github.com/jin-s13/COFE/blob/master/docs/coco-format.md

  **Acceptance Criteria**:
  - [ ] `ArnisGraphBuilder` class exists with proper edge definitions
  - [ ] `build_graph(keypoints, label)` returns `torch_geometric.data.Data` object
  - [ ] Node features: 19 nodes × 3 dims (x, y, confidence), normalized
  - [ ] Edge index: 2 × num_edges tensor (COO format)
  - [ ] Edge attributes: 4 dims per edge (length_norm, sin_angle, cos_angle, rigidity_flag)
  - [ ] Handles batching via `torch_geometric.loader.DataLoader`

  **Agent-Executed QA Scenario (MANDATORY):**
  ```
  Scenario: Validate Graph Construction
    Tool: Python REPL
    Preconditions: graph_builder.py exists
    Steps:
      1. Import: from training.geopose.graph_builder import ArnisGraphBuilder
      2. Create builder: builder = ArnisGraphBuilder()
      3. Create dummy keypoints: np.random.rand(19, 3)
      4. Build graph: data = builder.build_graph(keypoints, label=0)
      5. Assert: isinstance(data, torch_geometric.data.Data)
      6. Assert: data.x.shape == (19, 3)  # 19 nodes, 3 features
      7. Assert: data.edge_index.shape[0] == 2  # COO format
      8. Assert: data.edge_attr.shape[1] == 4  # 4 edge features
      9. Assert: data.y == 0  # Label preserved
      10. Create DataLoader: loader = DataLoader([data, data], batch_size=2)
      11. Iterate: batch = next(iter(loader))
      12. Assert: batch.x.shape[0] == 38  # 2 graphs × 19 nodes
    Expected Result: Valid PyG Data object with correct shapes
    Evidence: Shape values to .sisyphus/evidence/task-4-graph.log
  ```

  **Commit**: YES
  - Message: `feat(geopose): add graph builder for PyTorch Geometric`
  - Files: `training/geopose/graph_builder.py`
  - Pre-commit: Run graph validation

---

- [ ] 5. Convert CSV to PyG Data Format

  **What to do**:
  - Create `training/geopose/convert_dataset.py`
  - Load `features_train_corrected.csv` and `features_test.csv`
  - Extract keypoints for each sample (17 body + 2 weapon)
  - Build graph for each sample using GraphBuilder
  - Save as PyTorch dataset (.pt files)
  - Create train/val/test split
  - Save metadata (class distribution, node/edge counts)

  **Must NOT do**:
  - Lose label information during conversion
  - Create imbalanced splits (maintain stratification)
  - Store as individual files (use torch.save for efficiency)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` - Data pipeline
  - **Skills**: None specifically

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 6, 7, 8
  - **Blocked By**: Task 4 (graph builder must exist)

  **References**:
  - `features_train_corrected.csv` - Input data
  - `convo with kimi.txt:9176-9230` - CSV to graph conversion example

  **Acceptance Criteria**:
  - [ ] Script processes all training samples (should be ~6,180 from SESSION_SUMMARY)
  - [ ] Script processes all test samples (should be ~627)
  - [ ] Creates `data/processed/train_graphs.pt` with list of Data objects
  - [ ] Creates `data/processed/test_graphs.pt`
  - [ ] Creates `data/processed/dataset_info.json` with metadata
  - [ ] Maintains class distribution (stratified)
  - [ ] Processing completes in <5 minutes

  **Agent-Executed QA Scenario (MANDATORY):**
  ```
  Scenario: Convert Full Dataset to Graph Format
    Tool: Bash + Python REPL
    Preconditions: features_train_corrected.csv exists, graph_builder.py implemented
    Steps:
      1. Run: python training/geopose/convert_dataset.py
      2. Assert: Script completes without errors
      3. Verify files: ls -lh data/processed/
      4. Assert: train_graphs.pt exists
      5. Assert: test_graphs.pt exists
      6. Assert: dataset_info.json exists
      7. Load and verify: python -c "
         import torch
         train_data = torch.load('data/processed/train_graphs.pt')
         print(f'Train samples: {len(train_data)}')
         print(f'Sample graph: {train_data[0]}')
         print(f'Num classes: {torch.max(torch.tensor([d.y for d in train_data])) + 1}')
         "
      8. Assert: Train samples > 1000
      9. Assert: Num classes == 12
    Expected Result: Graph dataset created with all samples
    Evidence: Dataset statistics to .sisyphus/evidence/task-5-conversion.log
  ```

  **Commit**: YES
  - Message: `feat(geopose): convert CSV dataset to graph format`
  - Files: `training/geopose/convert_dataset.py`, `data/processed/*.pt`, `data/processed/dataset_info.json`
  - Pre-commit: Run conversion, verify output files

---

### Phase 4: GCN Model & Training (Day 3-5)

- [ ] 6. Implement GCN Model (2-Layer GAT)

  **What to do**:
  - Create `training/geopose/model.py`
  - Implement `GeoPoseNet` class:
    - Layer 1: GATConv(in_channels=3, out_channels=64, heads=4, edge_dim=4)
    - Layer 2: GATConv(in_channels=256, out_channels=128, heads=1, concat=False, edge_dim=4)
    - Global mean pooling
    - Classifier: Linear(128, 12) for 12 classes
  - Include dropout (p=0.2) between layers
  - Implement forward pass handling batching
  - Keep model lightweight (~10K parameters for fast CPU training)

  **Must NOT do**:
  - Add visual stream (MobileNetV3) in MVP - defer to Phase 2
  - Use complex attention pooling - global mean is sufficient
  - Make model too large (will be slow on CPU)
  - Add hierarchical classification yet - flat 12-class for MVP

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain` - Neural network architecture design
  - **Skills**: None specifically, needs PyTorch Geometric expertise

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Phase 4)
  - **Blocks**: Task 7, 8
  - **Blocked By**: Task 5 (graph data must exist)

  **References**:
  - `convo with kimi.txt:7260-7315` - Basic GAT architecture
  - PyTorch Geometric GATConv docs: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.GATConv.html
  - Graph classification example: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py

  **Acceptance Criteria**:
  - [ ] `GeoPoseNet` class exists with GAT layers
  - [ ] `forward(data)` accepts PyG Data object
  - [ ] Output shape: (batch_size, 12) for 12 classes
  - [ ] Model parameter count <20K (lightweight for CPU)
  - [ ] Includes dropout for regularization
  - [ ] Handles batching correctly via `data.batch`

  **Agent-Executed QA Scenario (MANDATORY):**
  ```
  Scenario: Validate GCN Model Architecture
    Tool: Python REPL
    Preconditions: model.py exists, graph data available
    Steps:
      1. Import: from training.geopose.model import GeoPoseNet
      2. Create model: model = GeoPoseNet(num_classes=12)
      3. Count parameters: sum(p.numel() for p in model.parameters())
      4. Assert: Parameter count < 20000
      5. Create dummy batch: data = torch.load('data/processed/train_graphs.pt')[0:2]
      6. Create loader: loader = DataLoader(data, batch_size=2)
      7. Forward pass: output = model(next(iter(loader)))
      8. Assert: output.shape == (2, 12)  # Batch size 2, 12 classes
      9. Test dropout: model.train(); out1 = model(next(iter(loader))); out2 = model(next(iter(loader)))
      10. Assert: not torch.allclose(out1, out2)  # Dropout active in train mode
    Expected Result: Model architecture valid, ~10K params, correct output shape
    Evidence: Parameter count and shapes to .sisyphus/evidence/task-6-model.log
  ```

  **Commit**: YES
  - Message: `feat(geopose): implement 2-layer GAT model`
  - Files: `training/geopose/model.py`
  - Pre-commit: Run model validation, check parameter count

---

- [ ] 7. Create Training Script with Cross-Validation

  **What to do**:
  - Create `training/geopose/train.py`
  - Implement 5-fold stratified cross-validation
  - Training loop with:
    - AdamW optimizer (lr=3e-4, weight_decay=0.01)
    - CosineAnnealingWarmRestarts scheduler (T_0=10)
    - CrossEntropyLoss
    - Early stopping (patience=15 epochs)
  - Track metrics: train loss, val accuracy per fold
  - Save best model per fold
  - Aggregate results across folds
  - Training should complete in <1 hour on your CPU

  **Must NOT do**:
  - Use GPU-specific code (keep CPU-compatible)
  - Train too many epochs (limit to 100 with early stopping)
  - Skip cross-validation (need robust accuracy estimate)
  - Use large batch size (keep small for CPU: 16-32)

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain` - Training loop implementation
  - **Skills**: None specifically

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 8
  - **Blocked By**: Task 6 (model must exist)

  **References**:
  - `convo with kimi.txt:7800-7850` - Training hyperparameters
  - `convo with kimi.txt:7822-7830` - Cross-validation protocol
  - PyTorch training best practices: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

  **Acceptance Criteria**:
  - [ ] `train.py` script exists with CLI arguments
  - [ ] Implements 5-fold stratified cross-validation
  - [ ] Saves models to `models/vXXX_geopose/fold{N}/model.pt`
  - [ ] Logs training metrics to console and file
  - [ ] Supports early stopping
  - [ ] Reports mean ± std accuracy across 5 folds
  - [ ] Training completes in <1 hour on Intel Core 7 150U

  **Agent-Executed QA Scenario (MANDATORY):**
  ```
  Scenario: Validate Training Script
    Tool: Bash
    Preconditions: model.py implemented, graph data exists
    Steps:
      1. Run quick test: python training/geopose/train.py --epochs 2 --folds 2 --quick-test
      2. Assert: Script runs without errors
      3. Verify output: Check console for "Fold 1/2", "Epoch 1/2", "Val Acc: X%"
      4. Verify files: ls models/*_geopose/
      5. Assert: Model checkpoints created
      6. Check logs: cat logs/training.log | head -20
      7. Assert: Training metrics logged
    Expected Result: Training script functional, produces model checkpoints
    Evidence: Training output to .sisyphus/evidence/task-7-training.log
  ```

  **Commit**: YES
  - Message: `feat(geopose): add training script with 5-fold cross-validation`
  - Files: `training/geopose/train.py`, `training/geopose/trainer.py` (if needed)
  - Pre-commit: Run quick training test (2 epochs, 2 folds)

---

- [ ] 8. Train Model and Validate Accuracy

  **What to do**:
  - Run full training: `python training/geopose/train.py`
  - Train for up to 100 epochs with early stopping
  - Monitor validation accuracy across 5 folds
  - Calculate mean ± std accuracy
  - **TARGET: ≥80% mean accuracy, target >85%**
  - If accuracy <80%, debug and iterate (adjust hyperparameters, add augmentation)
  - Select best fold model for export

  **Must NOT do**:
  - Accept <80% accuracy (this is the minimum success criterion)
  - Skip validation on test set
  - Overfit to validation set (use early stopping properly)

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain` - Model training and debugging
  - **Skills**: None specifically

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 9, 10, 11
  - **Blocked By**: Task 7 (training script must work)

  **References**:
  - `SESSION_SUMMARY.md` - Current 50.4% accuracy baseline
  - `convo with kimi.txt:7850-7860` - Expected 89-93% with full method

  **Acceptance Criteria**:
  - [ ] Full 5-fold cross-validation completed
  - [ ] Mean validation accuracy ≥80% (REQUIRED)
  - [ ] Target: Mean accuracy >85%
  - [ ] Test set accuracy calculated (hold-out set)
  - [ ] Training time <1 hour (as per user constraint)
  - [ ] Best model identified and saved
  - [ ] Confusion matrix generated per class

  **Agent-Executed QA Scenario (MANDATORY):**
  ```
  Scenario: Full Model Training and Validation
    Tool: Bash + Python REPL
    Preconditions: Training script works, graph data ready
    Steps:
      1. Run: python training/geopose/train.py
      2. Wait: Training completes (monitor for ~1 hour max)
      3. Check final output: "Mean Val Accuracy: X.XX% ± Y.YY%"
      4. Assert: X.XX >= 80.0  # MINIMUM REQUIREMENT
      5. Log: Save full training output
      6. Verify best model: ls models/*_geopose/best_model/
      7. Load and test: python -c "
         import torch
         model = torch.load('models/vXXX_geopose/best_model/model.pt')
         print(f'Model loaded: {type(model)}')
         "
      8. Assert: Model loads without errors
    Expected Result: Trained model with ≥80% accuracy, ready for export
    Evidence: 
      - Training log: .sisyphus/evidence/task-8-training-full.log
      - Accuracy metrics: .sisyphus/evidence/task-8-accuracy.json
  ```

  **Commit**: YES (after achieving ≥80%)
  - Message: `feat(geopose): train GCN model achieving X% accuracy`
  - Files: `models/vXXX_geopose/`, `logs/training_final.log`
  - Pre-commit: Verify accuracy ≥80%

---

### Phase 5: Deployment Integration (Day 6-7)

- [ ] 9. Create ONNX Export Script

  **What to do**:
  - Create `training/geopose/export_onnx.py`
  - Load trained PyTorch model
  - Create dummy input matching graph structure
  - Export to ONNX format: `torch.onnx.export()`
  - Verify ONNX model with `onnxruntime`
  - Create inference wrapper for app compatibility
  - Save required artifacts:
    - `model.onnx` (main model)
    - `scaler.joblib` (identity scaler or computed scaler)
    - `label_encoder.joblib` (class names)
    - `metadata.json` (model info)

  **Must NOT do**:
  - Export without verification (must test inference)
  - Forget to include preprocessing in export
  - Skip creating scaler/encoder files (app requires them)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` - Model deployment
  - **Skills**: None specifically, needs ONNX expertise

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Phase 5)
  - **Blocks**: Task 10, 11
  - **Blocked By**: Task 8 (trained model must exist)

  **References**:
  - `convo with kimi.txt:7850-7890` - ONNX export and inference
  - `APP_FLOW.md:50-80` - Required artifacts for app
  - PyTorch ONNX tutorial: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

  **Acceptance Criteria**:
  - [ ] `export_onnx.py` script exists
  - [ ] Exports `model.onnx` successfully
  - [ ] ONNX model loads in `onnxruntime`
  - [ ] Inference produces same output as PyTorch model (±0.1%)
  - [ ] Creates `scaler.joblib` (can be identity scaler)
  - [ ] Creates `label_encoder.joblib` with class names
  - [ ] Creates `metadata.json` with model info
  - [ ] All artifacts in correct directory structure

  **Agent-Executed QA Scenario (MANDATORY):**
  ```
  Scenario: Export and Verify ONNX Model
    Tool: Bash + Python REPL
    Preconditions: Trained model exists
    Steps:
      1. Run: python training/geopose/export_onnx.py --model models/vXXX_geopose/best_model/model.pt
      2. Assert: Script completes without errors
      3. Verify files: ls models/vXXX_geopose/
      4. Assert: model.onnx exists
      5. Assert: scaler.joblib exists
      6. Assert: label_encoder.joblib exists
      7. Assert: metadata.json exists
      8. Test inference: python -c "
         import onnxruntime as ort
         import numpy as np
         session = ort.InferenceSession('models/vXXX_geopose/model.onnx')
         print(f'Inputs: {session.get_inputs()}')
         print(f'Outputs: {session.get_outputs()}')
         # Test with dummy input
         dummy_input = np.random.randn(1, 19, 3).astype(np.float32)
         outputs = session.run(None, {'input': dummy_input})
         print(f'Output shape: {outputs[0].shape}')
         "
      9. Assert: ONNX model loads and runs inference
    Expected Result: ONNX model exported and functional
    Evidence: Export log to .sisyphus/evidence/task-9-export.log
  ```

  **Commit**: YES
  - Message: `feat(geopose): add ONNX export for app compatibility`
  - Files: `training/geopose/export_onnx.py`, `models/vXXX_geopose/model.onnx`, etc.
  - Pre-commit: Run export and verify ONNX inference

---

- [ ] 10. Integrate with Model Manager

  **What to do**:
  - Update `training/model_manager.py` to support GeoPose models
  - Add "Train GeoPose Model" option to CLI menu
  - Add GeoPose model to metadata scanning
  - Ensure `active_model.json` can point to GeoPose models
  - Test full workflow: train → export → set active → use in app

  **Must NOT do**:
  - Break existing RF/XGB functionality (keep backward compatible)
  - Skip testing the full workflow
  - Forget to update model type in metadata

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` - CLI integration
  - **Skills**: None specifically

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 11
  - **Blocked By**: Task 9 (ONNX export must work)

  **References**:
  - `training/model_manager.py` - Existing model manager
  - `APP_FLOW.md:10-50` - Model manager workflow

  **Acceptance Criteria**:
  - [ ] `model_manager.py` shows "Train GeoPose GCN" option
  - [ ] Can train GeoPose model from model manager CLI
  - [ ] GeoPose models appear in model list
  - [ ] Can set GeoPose model as active
  - [ ] `active_model.json` correctly points to GeoPose model
  - [ ] Existing RF/XGB options still work

  **Agent-Executed QA Scenario (MANDATORY):**
  ```
  Scenario: Model Manager Integration
    Tool: interactive_bash (CLI testing)
    Preconditions: model_manager.py updated, ONNX model exists
    Steps:
      1. Run: python training/model_manager.py
      2. Send: Check menu displays "Train GeoPose GCN" option
      3. Send: Select option (or verify it exists)
      4. Send: Exit
      5. Verify: Check active_model.json
      6. Assert: JSON structure valid
      7. Test setting active: python -c "
         import json
         with open('models/active_model.json', 'r') as f:
             config = json.load(f)
         print(f'Active model: {config.get(\"model_path\", \"None\")}')
         "
    Expected Result: Model manager supports GeoPose models
    Evidence: CLI interaction to .sisyphus/evidence/task-10-manager.log
  ```

  **Commit**: YES
  - Message: `feat(geopose): integrate with model_manager CLI`
  - Files: `training/model_manager.py`
  - Pre-commit: Test CLI menu displays correctly

---

- [ ] 11. End-to-End Validation

  **What to do**:
  - Run complete workflow: train → export → set active → inference
  - Test inference on sample images
  - Verify weapon geometry correction applied
  - Verify graph construction works
  - Verify ONNX inference matches PyTorch
  - Document any manual steps for app integration
  - Create README for GeoPose pipeline

  **Must NOT do**:
  - Skip testing the full pipeline
  - Leave undocumented manual steps
  - Assume app integration works without testing

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` - Integration testing
  - **Skills**: None specifically

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (final task)
  - **Blocks**: None (end of project)
  - **Blocked By**: Task 10 (model manager integration)

  **References**:
  - `APP_FLOW.md` - Full app workflow
  - `pose_analyzer.py` (in main app) - How model is used

  **Acceptance Criteria**:
  - [ ] Complete workflow tested end-to-end
  - [ ] Inference on test images produces valid predictions
  - [ ] Weapon geometry correction visibly improves predictions
  - [ ] ONNX inference speed acceptable (<1s per frame on CPU)
  - [ ] All artifacts ready for app deployment
  - [ ] README.md created with usage instructions

  **Agent-Executed QA Scenario (MANDATORY):**
  ```
  Scenario: End-to-End Pipeline Validation
    Tool: Bash + Python REPL
    Preconditions: All previous tasks complete
    Steps:
      1. Verify workflow: python training/geopose/test_pipeline.py (if exists)
      2. Or run manually:
         - Train: python training/geopose/train.py --quick-test
         - Export: python training/geopose/export_onnx.py
         - Set active: python training/model_manager.py
      3. Test inference: python -c "
         # Load ONNX and run inference on test sample
         import onnxruntime as ort
         import torch
         # ... test code ...
         print('Inference test PASSED')
         "
      4. Check: All required artifacts exist
      5. Verify: README.md created with instructions
      6. Document: Any limitations or known issues
    Expected Result: Complete pipeline functional, ready for deployment
    Evidence: Pipeline test results to .sisyphus/evidence/task-11-e2e.log
  ```

  **Commit**: YES (final commit)
  - Message: `feat(geopose): complete GeoPose-Net migration, achieving X% accuracy`
  - Files: All new files, updated model_manager.py, README.md
  - Pre-commit: Run full pipeline validation

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `chore(deps): add PyTorch + PyTorch Geometric for GeoPose-Net` | requirements.txt | Installation test |
| 2 | `feat(geopose): add weapon geometry correction module` | weapon_geometry.py | Geometry validation |
| 3 | `feat(geopose): validate weapon geometry correction on dataset` | validate_geometry.py, features_train_corrected.csv | 95% validity check |
| 4 | `feat(geopose): add graph builder for PyTorch Geometric` | graph_builder.py | Graph validation |
| 5 | `feat(geopose): convert CSV dataset to graph format` | convert_dataset.py, data/processed/ | Dataset conversion |
| 6 | `feat(geopose): implement 2-layer GAT model` | model.py | Model validation |
| 7 | `feat(geopose): add training script with 5-fold cross-validation` | train.py, trainer.py | Quick training test |
| 8 | `feat(geopose): train GCN model achieving X% accuracy` | models/vXXX_geopose/ | Accuracy ≥80% |
| 9 | `feat(geopose): add ONNX export for app compatibility` | export_onnx.py, model.onnx | ONNX inference test |
| 10 | `feat(geopose): integrate with model_manager CLI` | model_manager.py | CLI menu test |
| 11 | `feat(geopose): complete GeoPose-Net migration, achieving X% accuracy` | All files, README.md | Full pipeline test |

---

## Success Criteria

### Verification Commands

```bash
# 1. Environment
cd C:\Users\HP\Documents\GitHub\TuroArnis-ML
python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
# Expected: PyTorch version, CUDA: False

# 2. Geometry Correction
python -c "from training.geopose.weapon_geometry import correct_weapon_geometry; print('Geometry module OK')"
# Expected: No errors

# 3. Graph Builder
python -c "from training.geopose.graph_builder import ArnisGraphBuilder; print('Graph builder OK')"
# Expected: No errors

# 4. Model
python -c "from training.geopose.model import GeoPoseNet; model = GeoPoseNet(); print(f'Params: {sum(p.numel() for p in model.parameters())}')"
# Expected: Parameter count <20000

# 5. Training (quick test)
python training/geopose/train.py --epochs 2 --folds 2 --quick-test
# Expected: Completes without errors

# 6. Accuracy (after full training)
python -c "import json; data = json.load(open('models/vXXX_geopose/metadata.json')); print(f'Accuracy: {data[\"accuracy\"]}%')"
# Expected: ≥80%

# 7. ONNX Export
python -c "import onnxruntime as ort; sess = ort.InferenceSession('models/vXXX_geopose/model.onnx'); print('ONNX OK')"
# Expected: No errors

# 8. Model Manager
python training/model_manager.py --list
# Expected: Shows GeoPose models in list
```

### Final Checklist

- [ ] All "Must Have" present
  - [ ] Weapon geometry correction implemented
  - [ ] Graph construction from CSV
  - [ ] GCN model with ≥80% accuracy
  - [ ] ONNX export functional
  - [ ] Training completes in <1 hour
- [ ] All "Must NOT Have" absent
  - [ ] No visual stream in MVP (deferred)
  - [ ] No hierarchical classification yet (deferred)
  - [ ] No GPU dependency
- [ ] App compatibility verified
  - [ ] `model.onnx` created
  - [ ] `scaler.joblib` created
  - [ ] `label_encoder.joblib` created
  - [ ] `metadata.json` created
  - [ ] `active_model.json` can point to GeoPose model

---

**Plan Status:** Ready for execution

Run `/start-work` to begin with the orchestrator.
